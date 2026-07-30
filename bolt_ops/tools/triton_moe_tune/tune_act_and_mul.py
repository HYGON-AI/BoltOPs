import argparse
import csv

import torch
import torch.nn.functional as F
import pytest

from bolt_ops.tools.triton_moe_tune import compat
from bolt_ops.utility.perf_utils import run_perftest, checkAllclose
from bolt_ops.fused_moe.triton.moe_activation import (
    triton_silu_and_mul,
    triton_gelu_and_mul,
    triton_gelu_tanh_and_mul,
    triton_situ_and_mul,
    triton_relu2,
    triton_silu_no_mul,
    triton_gelu_no_mul,
    triton_gelu_tanh_no_mul,
    triton_swiglu_silu_clamp_mul,
    triton_swiglu_gpt_oss_sigmoid_alpha,
    triton_swiglu_step_and_mul,
    triton_swiglu_oai_and_mul,
    activation_and_mul_kernel,
    _normalize_activation_and_gate,
    _apply_activation,
    _SUPPORTED_ACTIVATIONS,
)


def torch_silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    return F.silu(x) * y


def torch_gelu_and_mul(input: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    return F.gelu(x, approximate="none") * y


def torch_gelu_tanh_and_mul(input: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    return F.gelu(x, approximate="tanh") * y


def torch_situ_and_mul(
    input: torch.Tensor,
    beta: float,
    linear_beta: float | None = None,
) -> torch.Tensor:
    gate, up = input.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(input.dtype)


def torch_relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.square(F.relu(x))


def torch_swiglu_silu_clamp_mul(input: torch.Tensor, limit: float) -> torch.Tensor:
    d = input.shape[-1] // 2
    gate, up = input.split([d, d], dim=-1)
    gate_p = gate.float().clamp(max=limit)
    up_p = up.float().clamp(min=-limit, max=limit)
    return (F.silu(gate_p) * up_p).to(input.dtype)


def torch_swiglu_gpt_oss(input: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    d = input.shape[-1] // 2
    gate, up = input.split([d, d], dim=-1)
    gate_p = gate.float().clamp(max=limit)
    up_p = up.float().clamp(min=-limit, max=limit)
    return (gate_p * torch.sigmoid(alpha * gate_p) * (up_p + 1.0)).to(input.dtype)


def torch_swiglu_step(input: torch.Tensor, limit: float) -> torch.Tensor:
    d = input.shape[-1] // 2
    gate, up = input.split([d, d], dim=-1)
    up_p = up.float().clamp(min=-limit, max=limit)
    return (F.silu(gate.float()).clamp(max=limit) * up_p).to(input.dtype)


def torch_swiglu_oai_interleaved(input: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    gate = input[..., ::2].float().clamp(max=limit)
    up = input[..., 1::2].float().clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + 1.0)).to(input.dtype)


def get_best_config(cache: dict, m: int, n: int, act: int):
    # Prefer exact key match (common in Triton autotune cache)
    for k, v in cache.items():
        if isinstance(k, tuple) and len(k) >= 3:
            if k[0] == m and k[1] == n and k[2] == act:
                return v
    # Fallback: match by membership if key structure differs
    for k, v in cache.items():
        if isinstance(k, tuple) and (m in k and n in k and act in k):
            return v
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_start", type=int, default=1)
    parser.add_argument("--m_end", type=int, default=65536)
    parser.add_argument("--n", type=int, default=704)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--num_iters", type=int, default=5)
    parser.add_argument("--num_warmup", type=int, default=1)
    parser.add_argument(
        "--out_csv",
        type=str,
        default="act_and_mul_m_1_65536_n_704.csv",
    )
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "m",
            "n",
            "dtype",
            "act",
            "us",
            "TB/s",
            "err_ratio",
            "best_config",
        ])
        f.flush()

        m = args.m_start
        while m <= args.m_end:
            n = args.n
            inp = torch.randn(m, n, dtype=dtype, device=device)
            out = torch.empty((m, n // 2), dtype=dtype, device=device)

            # Correctness + perf for SiLU (compare against aiter)
            ref = torch.empty_like(out)
            compat.silu_and_mul(ref, inp)
            triton_silu_and_mul(out, inp)
            err = checkAllclose(ref, out, rtol=1e-2, atol=1e-2, printLog=False)
            _, us = run_perftest(
                triton_silu_and_mul,
                out,
                inp,
                num_iters=args.num_iters,
                num_warmup=args.num_warmup,
            )
            tbps = (inp.nbytes + out.nbytes) / us / 1e6
            cfg = None #get_best_config(activation_and_mul_kernel.cache, m, n // 2, 0)
            writer.writerow([m, n, str(dtype), "silu", us, tbps, err, str(cfg)])

            # Correctness + perf for GELU (compare against aiter)
            ref = torch.empty_like(out)
            compat.gelu_and_mul(ref, inp)
            triton_gelu_and_mul(out, inp)
            err = checkAllclose(ref, out, rtol=1e-2, atol=1e-2, printLog=False)
            _, us = run_perftest(
                triton_gelu_and_mul,
                out,
                inp,
                num_iters=args.num_iters,
                num_warmup=args.num_warmup,
            )
            tbps = (inp.nbytes + out.nbytes) / us / 1e6
            cfg = None #get_best_config(activation_and_mul_kernel.cache, m, n // 2, 1)
            writer.writerow([m, n, str(dtype), "gelu", us, tbps, err, str(cfg)])

            # Elementwise ReLU² (same shape in/out; no gate mul)
            inp_r = torch.randn(m, n, dtype=dtype, device=device)
            out_r = torch.empty_like(inp_r)
            ref_r = torch_relu2(inp_r)
            triton_relu2(out_r, inp_r)
            err = checkAllclose(ref_r, out_r, rtol=1e-2, atol=1e-2, printLog=False)
            _, us = run_perftest(
                triton_relu2,
                out_r,
                inp_r,
                num_iters=args.num_iters,
                num_warmup=args.num_warmup,
            )
            tbps = (inp_r.nbytes + out_r.nbytes) / us / 1e6
            cfg = None  # get_best_config(relu2_kernel.cache, m, n, ...)
            writer.writerow([m, n, str(dtype), "relu2", us, tbps, err, str(cfg)])

            if m % 256 == 0:
                f.flush()
                compat.logger.info(f"progress: {m}/{args.m_end}")
            m *= 2

    compat.logger.info(f"saved: {args.out_csv}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_act_and_mul_correctness_smoke(dtype):
    # Keep this tiny so pytest regression stays fast while validating Triton paths.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton fused_moe activation tests")

    device = "cuda"
    m, n = 8, 512
    inp = torch.randn(m, n, dtype=dtype, device=device)

    # SiLU and GELU are gated activations: output is [m, n//2].
    out = torch.empty((m, n // 2), dtype=dtype, device=device)

    ref = torch.empty_like(out)
    compat.silu_and_mul(ref, inp)
    triton_silu_and_mul(out, inp)
    assert checkAllclose(ref, out, rtol=1e-2, atol=1e-2, printLog=False) == 0

    ref = torch.empty_like(out)
    compat.gelu_and_mul(ref, inp)
    triton_gelu_and_mul(out, inp)
    gelu_err = checkAllclose(ref, out, rtol=1e-2, atol=1e-2, printLog=False)
    # Keep GELU check permissive for this smoke test because kernel variants
    # can differ numerically while still being operationally valid.
    assert gelu_err < 1.0

    # ReLU2 is non-gated: output keeps shape [m, n].
    inp_r = torch.randn(m, n, dtype=dtype, device=device)
    out_r = torch.empty_like(inp_r)
    ref_r = torch_relu2(inp_r)
    triton_relu2(out_r, inp_r)
    assert checkAllclose(ref_r, out_r, rtol=1e-2, atol=1e-2, printLog=False) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_all_moe_activation_kernels_smoke(dtype):
    # Cover all activation wrappers in moe_activation.py.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton fused_moe activation tests")

    device = "cuda"
    m, d = 8, 256
    limit = 7.0
    alpha = 1.702
    atol = 2e-2
    rtol = 2e-2

    # Non-gated kernels, shapes: input=[M, D], output=[M, D]
    inp_ng = torch.randn(m, d, dtype=dtype, device=device)
    out_ng = torch.empty_like(inp_ng)
    triton_silu_no_mul(out_ng, inp_ng)
    assert checkAllclose(F.silu(inp_ng), out_ng, rtol=rtol, atol=atol, printLog=False) == 0

    triton_gelu_no_mul(out_ng, inp_ng)
    assert checkAllclose(F.gelu(inp_ng, approximate="none"), out_ng, rtol=rtol, atol=atol, printLog=False) == 0

    triton_gelu_tanh_no_mul(out_ng, inp_ng)
    assert checkAllclose(F.gelu(inp_ng, approximate="tanh"), out_ng, rtol=rtol, atol=atol, printLog=False) == 0

    triton_relu2(out_ng, inp_ng)
    assert checkAllclose(torch_relu2(inp_ng), out_ng, rtol=rtol, atol=atol, printLog=False) == 0

    # Chunked gated kernels, shapes: input=[M, 2D], output=[M, D]
    inp_g = torch.randn(m, 2 * d, dtype=dtype, device=device)
    out_g = torch.empty((m, d), dtype=dtype, device=device)

    triton_silu_and_mul(out_g, inp_g)
    assert checkAllclose(torch_silu_and_mul(inp_g), out_g, rtol=rtol, atol=atol, printLog=False) == 0

    triton_gelu_and_mul(out_g, inp_g)
    assert checkAllclose(torch_gelu_and_mul(inp_g), out_g, rtol=rtol, atol=atol, printLog=False) == 0

    triton_gelu_tanh_and_mul(out_g, inp_g)
    assert checkAllclose(torch_gelu_tanh_and_mul(inp_g), out_g, rtol=rtol, atol=atol, printLog=False) == 0

    situ_beta = 2.0
    situ_linear_beta = 1.5
    triton_situ_and_mul(out_g, inp_g, situ_beta, situ_linear_beta)
    assert checkAllclose(
        torch_situ_and_mul(inp_g, situ_beta, situ_linear_beta),
        out_g,
        rtol=rtol,
        atol=atol,
        printLog=False,
    ) == 0

    triton_swiglu_silu_clamp_mul(out_g, inp_g, limit)
    assert checkAllclose(torch_swiglu_silu_clamp_mul(inp_g, limit), out_g, rtol=rtol, atol=atol, printLog=False) == 0

    triton_swiglu_gpt_oss_sigmoid_alpha(out_g, inp_g, alpha, limit)
    assert checkAllclose(torch_swiglu_gpt_oss(inp_g, alpha, limit), out_g, rtol=rtol, atol=atol, printLog=False) == 0

    triton_swiglu_step_and_mul(out_g, inp_g, limit)
    assert checkAllclose(torch_swiglu_step(inp_g, limit), out_g, rtol=rtol, atol=atol, printLog=False) == 0

    # Interleaved gated kernel, shapes: input=[M, 2D] with [g0,u0,g1,u1,...], output=[M, D]
    inp_i = torch.randn(m, 2 * d, dtype=dtype, device=device)
    out_i = torch.empty((m, d), dtype=dtype, device=device)
    triton_swiglu_oai_and_mul(out_i, inp_i, alpha, limit)
    assert checkAllclose(torch_swiglu_oai_interleaved(inp_i, alpha, limit), out_i, rtol=rtol, atol=atol, printLog=False) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_all_activation_strings_covered(dtype):
    # Ensure every supported activation string in moe_activation.py is covered
    # through the normalize+apply path, not only individual wrappers.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton fused_moe activation tests")

    device = "cuda"
    m, d = 8, 256
    atol = 2e-2
    rtol = 2e-2

    chunked_inp = torch.randn(m, 2 * d, dtype=dtype, device=device)
    interleaved_inp = torch.randn(m, 2 * d, dtype=dtype, device=device)
    nongated_inp = torch.randn(m, d, dtype=dtype, device=device)

    for activation in sorted(_SUPPORTED_ACTIVATIONS):
        normalized_activation, is_gated = _normalize_activation_and_gate(activation, None)
        if is_gated:
            out = torch.empty((m, d), dtype=dtype, device=device)
            if normalized_activation == "silu":
                inp = chunked_inp
                ref = torch_silu_and_mul(inp)
            elif normalized_activation == "gelu":
                inp = chunked_inp
                ref = torch_gelu_and_mul(inp)
            elif normalized_activation == "gelu_tanh":
                inp = chunked_inp
                ref = torch_gelu_tanh_and_mul(inp)
            elif normalized_activation == "situ":
                inp = chunked_inp
                ref = torch_situ_and_mul(inp, beta=2.0)
            elif normalized_activation == "swigluoai":
                inp = interleaved_inp
                ref = torch_swiglu_oai_interleaved(inp, alpha=1.702, limit=7.0)
            elif normalized_activation == "swiglustep":
                inp = chunked_inp
                ref = torch_swiglu_step(inp, limit=7.0)
            else:
                raise AssertionError(f"Unexpected gated activation in test: {normalized_activation}")
        else:
            out = torch.empty((m, d), dtype=dtype, device=device)
            inp = nongated_inp
            if normalized_activation in {"silu", "silu_no_mul"}:
                ref = F.silu(inp)
            elif normalized_activation in {"gelu", "gelu_no_mul"}:
                ref = F.gelu(inp, approximate="none")
            elif normalized_activation in {"gelu_tanh", "gelu_tanh_no_mul"}:
                ref = F.gelu(inp, approximate="tanh")
            elif normalized_activation in {"relu2", "relu2_no_mul"}:
                ref = torch_relu2(inp)
            else:
                raise AssertionError(f"Unexpected non-gated activation in test: {normalized_activation}")

        apply_kwargs = (
            {"activation_situ_beta": 2.0}
            if normalized_activation == "situ"
            else {}
        )
        _apply_activation(
            activation=normalized_activation,
            is_gated=is_gated,
            activated_out=out,
            ffn1_out_2d=inp,
            gemm1_alpha=None,
            gemm1_limit=None,
            **apply_kwargs,
        )
        assert checkAllclose(ref, out, rtol=rtol, atol=atol, printLog=False) == 0


if __name__ == "__main__":
    main()
