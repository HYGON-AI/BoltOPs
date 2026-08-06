"""Unified causal conv1d API: cross-backend correctness, routing, bake-off.

Both Triton ports (SGLang / vLLM) are exercised through the unified entry
points (``causal_conv1d_fn`` / ``causal_conv1d_update``) against the same
PyTorch reference, so feature drift between the two kernels fails loudly.
This is the single test suite for the operator: per-port edge cases
(measured Kimi-K3 shapes, non-contiguous layouts, vLLM varlen decode, large
state cache, hand-computed known values) are folded in here and parametrized
over both backends.

Default ``pytest``: routing rules (CPU) + cross-backend correctness (GPU).
``pytest --bench``: times both backends on the same real-model shapes and
prints the comparison that gates the eventual single-kernel merge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from boltops.generic import (  # noqa: E402
    causal_conv1d_fn,
    causal_conv1d_update,
)
from boltops.generic.triton.causal_conv1d.core import (  # noqa: E402
    PAD_SLOT_ID,
    resolve_fwd_backend,
    resolve_update_backend,
)


def test_causal_conv1d_package_has_compact_module_layout():
    """Routing helpers live in core; kernel variants live in kernels."""
    from boltops.generic.triton.causal_conv1d import kernels  # noqa: E402

    assert hasattr(kernels, "causal_conv1d_fn_sglang")
    assert hasattr(kernels, "causal_conv1d_fn_vllm")

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)

DIM_PER_RANK = 1536  # TP=8: 12 heads x 128
DIM_DECODE = 3 * DIM_PER_RANK
WIDTH = 4
NUM_SLOTS = 33
MAX_DECODE_BATCH = 32

# Measured strides of the conv_state pool view [slots, dim, state_len]:
# pool [slots, 3, 4608] -> transpose(-1, -2) -> (13824, 1, 4608)
MEASURED_STATE_STRIDES = (13824, 1, 4608)


# ---------------------------------------------------------------------------
# PyTorch references (shared oracle for both ports)
# ---------------------------------------------------------------------------


def conv_dtype(weight: torch.Tensor) -> torch.dtype:
    """Oracle compute dtype.

    The kernels accumulate in FP32 and store into ``x``'s dtype, so when the
    checkpoint weight is FP32 the reference computes in FP32 as well.
    """
    return torch.float32 if weight.dtype == torch.float32 else weight.dtype


def silu_if(x: torch.Tensor, activation: Union[str, bool, None]) -> torch.Tensor:
    return F.silu(x) if activation in ("silu", "swish") else x


def causal_conv1d_fn_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Union[str, bool, None] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """Pure-PyTorch reference for varlen causal conv1d fwd."""
    dim, _cu_seqlen = x.shape
    width = weight.shape[1]
    state_len = width - 1
    compute = conv_dtype(weight)
    out = torch.zeros_like(x)
    offsets = query_start_loc.tolist()
    w = weight.unsqueeze(1).to(compute)
    b = bias.to(compute) if bias is not None else None
    for i in range(len(offsets) - 1):
        state_line = cache_indices[i].item() if cache_indices is not None else i
        if state_line == pad_slot_id:
            continue
        bos, eos = offsets[i], offsets[i + 1]
        x_seq = x[:, bos:eos]
        has_init = (
            bool(has_initial_state[i]) if has_initial_state is not None else False
        )
        if has_init:
            state = conv_states[state_line][:, :state_len]
            x_new = torch.cat([state, x_seq], dim=-1).unsqueeze(0)
            out_seq = F.conv1d(x_new.to(compute), w, b, groups=dim)
            combined = torch.cat([state, x_seq], dim=-1)
        else:
            out_seq = F.conv1d(
                x_seq.unsqueeze(0).to(compute), w, b, padding=state_len, groups=dim
            )[..., : x_seq.shape[-1]]
            combined = x_seq
        out[:, bos:eos] = silu_if(out_seq[0], activation)

        if conv_states is not None:
            if combined.shape[-1] >= state_len:
                new_state = combined[:, -state_len:]
            else:
                new_state = F.pad(combined, (state_len - combined.shape[-1], 0))
            conv_states[state_line][:, :state_len].copy_(new_state)
    return out


def causal_conv1d_update_torch(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[str, bool, None] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """Pure-PyTorch reference for causal conv1d update (decode)."""
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = width - 1
    compute = conv_dtype(weight)
    out = torch.zeros_like(x)
    w = weight.unsqueeze(1).to(compute)
    b = bias.to(compute) if bias is not None else None
    for i in range(batch):
        state_line = (
            conv_state_indices[i].item() if conv_state_indices is not None else i
        )
        if state_line == pad_slot_id:
            continue
        state = conv_state[state_line][:, :state_len]
        x_new = torch.cat([state, x[i]], dim=-1).unsqueeze(0)
        conv_state[state_line][:, :state_len].copy_(x_new[0, :, -state_len:])
        out[i] = silu_if(
            F.conv1d(x_new.to(compute), w, b, groups=dim)[0], activation
        )
    return out


def assert_close(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> None:
    """Tolerances used across the causal conv1d suite."""
    rtol, atol = (3e-4, 1e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Backend routing (pure logic, runs on CPU)
# ---------------------------------------------------------------------------


def test_fwd_auto_routing():
    # No backend-specific argument -> vLLM is the default base.
    assert resolve_fwd_backend() == "vllm"
    # SGLang-only signal.
    assert resolve_fwd_backend(seq_lens_cpu=[46, 91]) == "sglang"
    # vLLM-only signals.
    assert resolve_fwd_backend(metadata=object()) == "vllm"
    assert resolve_fwd_backend(block_size_to_align=16) == "vllm"
    assert resolve_fwd_backend(initial_state_idx=object()) == "vllm"
    # Ambiguous mix -> error; explicit backend overrides.
    with pytest.raises(ValueError, match="ambiguous backend"):
        resolve_fwd_backend(seq_lens_cpu=[46], block_size_to_align=16)
    assert resolve_fwd_backend(
        "sglang", seq_lens_cpu=[46], block_size_to_align=16
    ) == "sglang"
    with pytest.raises(ValueError):
        resolve_fwd_backend("tilelang")


def test_update_auto_routing():
    assert resolve_update_backend() == "vllm"
    assert resolve_update_backend(num_accept_tokens=object()) == "sglang"
    assert resolve_update_backend(retrieve_next_token=object()) == "sglang"
    assert resolve_update_backend(cache_seqlens=object()) == "sglang"
    assert resolve_update_backend(num_accepted_tokens=object()) == "vllm"
    assert resolve_update_backend(query_start_loc=object()) == "vllm"
    assert resolve_update_backend(max_query_len=5) == "vllm"
    assert resolve_update_backend(out=object()) == "vllm"
    with pytest.raises(ValueError, match="ambiguous backend"):
        resolve_update_backend(num_accept_tokens=object(), query_start_loc=object())
    assert resolve_update_backend(
        "vllm", num_accept_tokens=object(), query_start_loc=object()
    ) == "vllm"


# ---------------------------------------------------------------------------
# Fixtures (Kimi-K3 scale, same layout as the per-port suites)
# ---------------------------------------------------------------------------


def _make_fwd_inputs(
    dim: int = DIM_PER_RANK,
    width: int = WIDTH,
    x_dtype: torch.dtype = torch.bfloat16,
    weight_dtype: torch.dtype = torch.float32,
    num_seqs: int = 8,
    num_slots: int = NUM_SLOTS,
    with_padding: bool = True,
    seqlen: Optional[int] = None,
):
    torch.manual_seed(42)
    if seqlen is not None:
        # uniform-length scale case: T_total = num_seqs * seqlen exactly
        lengths = [seqlen] * num_seqs
        padding = 0
    else:
        lengths = [46, 91, 137, 182, 57, 203, 118, 76][:num_seqs]
        padding = 2 if with_padding else 0
        if padding:
            lengths = lengths + [41, 88]
    cu_seqlen = sum(lengths)
    state_len = width - 1
    # production layout: x is (dim, T_total), channel-last, stride(0)==1 (dim contiguous).
    # torch.randn(dim, T) would give stride (T, 1) (token-major), which scatters the
    # per-feature loads over T cache lines; keep the spec layout instead.
    x = torch.randn(cu_seqlen, dim, dtype=x_dtype).t()
    weight = torch.randn(dim, width, dtype=weight_dtype)
    bias = torch.randn(dim, dtype=x_dtype)
    query_start_loc = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    pool = torch.randn(num_slots, state_len, 3 * dim, dtype=x_dtype)
    conv_states = pool.transpose(-1, -2)[:, :dim]
    cache_indices = torch.randperm(num_slots)[:num_seqs].to(torch.int32)
    has_initial_state = torch.randint(0, 2, (num_seqs,), dtype=torch.bool)
    if padding:
        cache_indices = torch.cat(
            [cache_indices, torch.full((padding,), PAD_SLOT_ID, dtype=torch.int32)]
        )
        has_initial_state = torch.cat(
            [has_initial_state, torch.zeros(padding, dtype=torch.bool)]
        )
    return (
        tuple(
            t.to(DEVICE)
            for t in (
                x,
                weight,
                bias,
                conv_states,
                query_start_loc,
                cache_indices,
                has_initial_state,
            )
        ),
        lengths,
    )


def _make_update_inputs(
    dim: int = DIM_DECODE,
    width: int = WIDTH,
    x_dtype: torch.dtype = torch.bfloat16,
    weight_dtype: torch.dtype = torch.float32,
    batch: int = 8,
    num_slots: int = NUM_SLOTS,
    seqlen: int = 1,
    with_padding: bool = True,
):
    torch.manual_seed(42)
    padding = 2 if with_padding else 0
    total = batch + padding
    state_len = width - 1
    if seqlen == 1:
        x = torch.randn(total, dim, dtype=x_dtype)
    else:
        x = torch.randn(total, dim, seqlen, dtype=x_dtype)
    weight = torch.randn(dim, width, dtype=weight_dtype)
    bias = torch.randn(dim, dtype=x_dtype)
    pool = torch.randn(num_slots, state_len, dim, dtype=x_dtype)
    conv_state = pool.transpose(-1, -2)
    indices = torch.randperm(num_slots)[:batch].to(torch.int32)
    padded_indices = torch.cat(
        [indices, torch.full((padding,), PAD_SLOT_ID, dtype=torch.int32)]
    )
    return tuple(t.to(DEVICE) for t in (x, conv_state, weight, bias, padded_indices))


def _active_fwd_region(query_start_loc, cache_indices, cu_seqlen):
    region = torch.zeros(cu_seqlen, dtype=torch.bool)
    offsets = query_start_loc.cpu().tolist()
    for i in range(len(offsets) - 1):
        if cache_indices[i].item() == PAD_SLOT_ID:
            continue
        region[offsets[i] : offsets[i + 1]] = True
    return region


def _update_varlen_torch(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    conv_state_indices: torch.Tensor,
    activation: str | None = "silu",
) -> torch.Tensor:
    """Pure-PyTorch reference for the vLLM varlen decode path.

    ``x`` is ``[num_tokens, dim]`` with sequences concatenated; each sequence
    is processed like a standard decode update and its rows are written back
    at ``[query_start_loc[i]:query_start_loc[i+1]]``.
    """
    num_tokens, dim = x.shape
    width = weight.shape[1]
    state_len = width - 1
    compute = torch.float32 if weight.dtype == torch.float32 else weight.dtype
    out = torch.zeros_like(x)
    offsets = query_start_loc.tolist()
    w = weight.unsqueeze(1).to(compute)
    for i in range(len(offsets) - 1):
        bos, eos = offsets[i], offsets[i + 1]
        if bos == eos:
            continue
        line = conv_state_indices[i].item()
        state = conv_state[line][:, :state_len]
        x_new = torch.cat([state, x[bos:eos].t()], dim=-1).unsqueeze(0)
        conv_state[line][:, :state_len].copy_(x_new[0, :, -state_len:])
        out[bos:eos] = silu_if(
            F.conv1d(x_new.to(compute), w, None, groups=dim)[0], activation
        ).t()
    return out


# ---------------------------------------------------------------------------
# Cross-backend correctness through the unified entry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            dict(dim=DIM_PER_RANK, width=4, x_dtype=torch.bfloat16,
                 weight_dtype=torch.float32, has_bias=False, activation="silu"),
            id="k3-tp8-bf16x-fp32w",
        ),
        pytest.param(
            dict(dim=300, width=2, x_dtype=torch.float16,
                 weight_dtype=torch.float16, has_bias=True, activation="silu"),
            id="fp16-w2",
        ),
        # ---- scale sweep: uniform varlen prefill, T_total = B * L ----
        # small    : B=8, L=128  -> T=1024
        # medium   : B=8, L=512  -> T=4096
        # large    : B=8, L=2048 -> T=16384
        pytest.param(
            dict(dim=DIM_PER_RANK, width=4, x_dtype=torch.bfloat16,
                 weight_dtype=torch.float32, has_bias=False, activation="silu",
                 num_seqs=8, seqlen=128, with_padding=False),
            id="k3-tp8-bf16x-fp32w-t1024",
        ),
        pytest.param(
            dict(dim=DIM_PER_RANK, width=4, x_dtype=torch.bfloat16,
                 weight_dtype=torch.float32, has_bias=False, activation="silu",
                 num_seqs=8, seqlen=512, with_padding=False),
            id="k3-tp8-bf16x-fp32w-t4096",
        ),
        pytest.param(
            dict(dim=DIM_PER_RANK, width=4, x_dtype=torch.bfloat16,
                 weight_dtype=torch.float32, has_bias=False, activation="silu",
                 num_seqs=8, seqlen=2048, with_padding=False),
            id="k3-tp8-bf16x-fp32w-t16384",
        ),
        pytest.param(
            dict(dim=128, width=4, x_dtype=torch.float32,
                 weight_dtype=torch.float32, has_bias=True, activation="swish"),
            id="fp32-w4",
        ),
        pytest.param(
            dict(dim=259, width=3, x_dtype=torch.bfloat16,
                 weight_dtype=torch.bfloat16, has_bias=False, activation=None),
            id="masked-dim-tail",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@GPU_REQUIRED
def test_unified_fwd_matches_torch(case, backend):
    (x, weight, bias, conv_states, query_start_loc, cache_indices, has_init), lengths = (
        _make_fwd_inputs(
            **{k: v for k, v in case.items() if k not in ("has_bias", "activation")}
        )
    )
    bias_in = bias if case["has_bias"] else None
    conv_states_ref = conv_states.clone()
    kwargs = dict(
        cache_indices=cache_indices,
        has_initial_state=has_init,
        activation=case["activation"],
        pad_slot_id=PAD_SLOT_ID,
        backend=backend,
    )
    if backend == "sglang":
        kwargs["seq_lens_cpu"] = list(lengths)
    actual = causal_conv1d_fn(x, weight, bias_in, conv_states, query_start_loc, **kwargs)
    expected = causal_conv1d_fn_torch(
        x, weight, bias_in, conv_states_ref, query_start_loc,
        cache_indices, has_init, case["activation"],
    )
    region = _active_fwd_region(query_start_loc, cache_indices, x.shape[1])
    assert_close(actual[:, region], expected[:, region], case["x_dtype"])
    active = cache_indices[cache_indices != PAD_SLOT_ID]
    assert_close(conv_states[active], conv_states_ref[active], case["x_dtype"])


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            dict(dim=DIM_DECODE, width=4, x_dtype=torch.bfloat16,
                 weight_dtype=torch.float32, has_bias=False, activation="silu",
                 batch=MAX_DECODE_BATCH, seqlen=1),
            id="k3-tp8-max-batch-2d",
        ),
        pytest.param(
            dict(dim=DIM_DECODE, width=4, x_dtype=torch.bfloat16,
                 weight_dtype=torch.float32, has_bias=False, activation="silu",
                 batch=8, seqlen=5),
            id="k3-tp8-multi-token",
        ),
        pytest.param(
            dict(dim=300, width=2, x_dtype=torch.float16,
                 weight_dtype=torch.float16, has_bias=True, activation="silu",
                 batch=4, seqlen=5),
            id="fp16-w2",
        ),
        pytest.param(
            dict(dim=128, width=4, x_dtype=torch.float32,
                 weight_dtype=torch.float32, has_bias=True, activation="swish",
                 batch=3, seqlen=1),
            id="fp32-w4",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@GPU_REQUIRED
def test_unified_update_matches_torch(case, backend):
    keys = {k: v for k, v in case.items() if k not in ("has_bias", "activation")}
    x, conv_state, weight, bias, padded_indices = _make_update_inputs(**keys)
    conv_state_ref = conv_state.clone()
    actual = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias if case["has_bias"] else None,
        activation=case["activation"],
        conv_state_indices=padded_indices,
        pad_slot_id=PAD_SLOT_ID,
        backend=backend,
    )
    x3 = x.unsqueeze(-1) if x.dim() == 2 else x
    expected = causal_conv1d_update_torch(
        x3, conv_state_ref, weight, bias if case["has_bias"] else None,
        case["activation"], padded_indices,
    )
    if x.dim() == 2:
        expected = expected.squeeze(-1)
    batch = case["batch"]
    assert_close(actual[:batch], expected[:batch], case["x_dtype"])
    active = padded_indices[padded_indices != PAD_SLOT_ID]
    assert_close(conv_state[active], conv_state_ref[active], case["x_dtype"])


@GPU_REQUIRED
def test_unified_fwd_identity_default_fill():
    """No cache_indices / has_initial_state: both ports fall back to identity."""
    dim, width = DIM_PER_RANK, WIDTH
    for backend in ("sglang", "vllm"):
        (x, weight, _bias, conv_states, query_start_loc, _ci, _hi), lengths = (
            _make_fwd_inputs(num_seqs=3, with_padding=False)
        )
        ref = conv_states.clone()
        kwargs = dict(
            activation="silu",
            backend=backend,
            cache_indices=None,
            has_initial_state=None,
        )
        if backend == "sglang":
            kwargs["seq_lens_cpu"] = list(lengths)
        actual = causal_conv1d_fn(
            x, weight, None, conv_states, query_start_loc, **kwargs
        )
        expected = causal_conv1d_fn_torch(
            x, weight, None, ref, query_start_loc, None, None, "silu"
        )
        assert_close(actual, expected, torch.bfloat16)
        assert_close(conv_states, ref, torch.bfloat16)


@GPU_REQUIRED
def test_unified_update_identity_default_fill():
    """No conv_state_indices: both ports fall back to identity cache lines."""
    for backend in ("sglang", "vllm"):
        x, conv_state, weight, _bias, _idx = _make_update_inputs(
            dim=DIM_DECODE, batch=8, seqlen=1, with_padding=False
        )
        ref = conv_state.clone()
        actual = causal_conv1d_update(
            x, conv_state, weight, None, activation="silu", backend=backend
        )
        expected = causal_conv1d_update_torch(
            x.unsqueeze(-1), ref, weight, None, "silu", None
        ).squeeze(-1)
        assert_close(actual, expected, torch.bfloat16)
        assert_close(conv_state, ref, torch.bfloat16)


def test_torch_reference_known_values_cpu():
    """Hand-computed check for the PyTorch fwd/update references (no GPU)."""
    x = torch.tensor([[1.0, 2.0]])
    weight = torch.tensor([[0.5, -0.25, 1.0]])
    bias = torch.tensor([0.1])
    conv_states = torch.tensor([[[0.3, -0.7]]])
    query_start_loc = torch.tensor([0, 2])
    out = causal_conv1d_fn_torch(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        None,
        torch.tensor([True]),
        None,
    )
    # out[0] = 0.5*0.3 + (-0.25)*(-0.7) + 1.0*1.0 + 0.1 = 1.425
    # out[1] = 0.5*(-0.7) + (-0.25)*1.0 + 1.0*2.0 + 0.1 = 1.5
    torch.testing.assert_close(out.flatten(), torch.tensor([1.425, 1.5]))
    # Rolling state becomes the last two input tokens.
    torch.testing.assert_close(conv_states[0], torch.tensor([[1.0, 2.0]]))

    x_upd = torch.tensor([[[5.0]]])
    conv_state = torch.tensor([[[0.3, -0.7]]])
    out_upd = causal_conv1d_update_torch(x_upd, conv_state, weight, bias, None)
    # out = 0.5*0.3 + (-0.25)*(-0.7) + 1.0*5.0 + 0.1 = 5.425
    torch.testing.assert_close(out_upd.flatten(), torch.tensor([5.425]))
    torch.testing.assert_close(conv_state[0], torch.tensor([[-0.7, 5.0]]))


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@GPU_REQUIRED
def test_unified_fwd_measured_debug_shape(backend):
    """Reproduce the SGLANG_DEBUG_CONV1D_SHAPES capture (TP8, single seq).

    x [1536,46] bf16 stride (1,6144); weight [1536,4] fp32 stride (4,1);
    conv_states [33,1536,3] view stride (13824,1,4608); query_start_loc [2]
    int32; cache_indices [1] int32; has_initial_state [1] bool.
    """
    dim, width, num_slots = DIM_PER_RANK, WIDTH, NUM_SLOTS
    torch.manual_seed(42)
    lengths = [46]
    x_base = torch.randn(lengths[0], 6144, dtype=torch.bfloat16, device=DEVICE)
    x = x_base.t()[:dim]  # [1536, 46], strides (1, 6144): channel-last view
    weight = torch.randn(dim, width, dtype=torch.float32, device=DEVICE)
    pool = torch.randn(
        num_slots, width - 1, 3 * dim, dtype=torch.bfloat16, device=DEVICE
    )
    conv_states = pool.transpose(-1, -2)[:, :dim]
    query_start_loc = torch.tensor([0, 46], dtype=torch.int32, device=DEVICE)
    cache_indices = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    has_initial_state = torch.tensor([True], dtype=torch.bool, device=DEVICE)
    assert x.stride() == (1, 6144)
    assert weight.stride() == (width, 1)
    assert conv_states.stride() == MEASURED_STATE_STRIDES

    conv_states_ref = conv_states.clone()
    kwargs = dict(
        activation="silu",
        pad_slot_id=PAD_SLOT_ID,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        backend=backend,
    )
    if backend == "sglang":
        kwargs["seq_lens_cpu"] = list(lengths)
    actual = causal_conv1d_fn(
        x, weight, None, conv_states, query_start_loc, **kwargs
    )
    expected = causal_conv1d_fn_torch(
        x, weight, None, conv_states_ref, query_start_loc,
        cache_indices, has_initial_state, "silu",
    )
    assert_close(actual, expected, torch.bfloat16)
    assert_close(conv_states, conv_states_ref, torch.bfloat16)


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@GPU_REQUIRED
def test_unified_fwd_noncontiguous_input(backend):
    """The Triton path accepts arbitrary strides without a .contiguous() copy."""
    dim, width, dtype = 259, 4, torch.bfloat16
    base = torch.randn(dim + 5, 300, dtype=dtype, device=DEVICE)
    x = base[:dim]  # non-contiguous view
    weight = torch.randn(dim, width, dtype=torch.float32, device=DEVICE)
    conv_states = torch.randn(1, dim, width - 1, dtype=dtype, device=DEVICE)
    conv_states_ref = conv_states.clone()
    query_start_loc = torch.tensor([0, 300], dtype=torch.int32, device=DEVICE)
    kwargs = dict(activation="silu", pad_slot_id=PAD_SLOT_ID, backend=backend)
    if backend == "sglang":
        kwargs["seq_lens_cpu"] = [300]
    actual = causal_conv1d_fn(
        x, weight, None, conv_states, query_start_loc, **kwargs
    )
    expected = causal_conv1d_fn_torch(
        x, weight, None, conv_states_ref, query_start_loc, None, None, "silu"
    )
    assert_close(actual, expected, dtype)
    assert_close(conv_states, conv_states_ref, dtype)


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@GPU_REQUIRED
def test_unified_update_measured_debug_shape(backend):
    """Reproduce the exact decode capture (TP8, single request).

    x [1,4608] bf16 stride (6144,1); weight [4608,4] fp32 stride (4,1);
    conv_state [33,4608,3] view stride (13824,1,4608); conv_state_indices [1]
    int32; activation silu; bias None.
    """
    dim, width, batch = DIM_DECODE, WIDTH, 1
    x = torch.randn(batch + 2, 6144, dtype=torch.bfloat16, device=DEVICE)[:, :dim]
    weight = torch.randn(dim, width, dtype=torch.float32, device=DEVICE)
    pool = torch.randn(
        NUM_SLOTS, width - 1, dim, dtype=torch.bfloat16, device=DEVICE
    )
    conv_state = pool.transpose(-1, -2)
    padded_indices = torch.tensor(
        [1, PAD_SLOT_ID, PAD_SLOT_ID], dtype=torch.int32, device=DEVICE
    )
    assert x.stride() == (6144, 1)
    assert weight.stride() == (width, 1)
    assert conv_state.stride() == MEASURED_STATE_STRIDES

    conv_state_ref = conv_state.clone()
    actual = causal_conv1d_update(
        x,
        conv_state,
        weight,
        None,
        activation="silu",
        conv_state_indices=padded_indices,
        pad_slot_id=PAD_SLOT_ID,
        backend=backend,
    )
    expected = causal_conv1d_update_torch(
        x.unsqueeze(-1), conv_state_ref, weight, None, "silu", padded_indices
    ).squeeze(-1)
    # Padded rows are uninitialized in both ports; compare active rows only.
    assert_close(actual[:batch], expected[:batch], torch.bfloat16)
    active = padded_indices[padded_indices != PAD_SLOT_ID]
    assert_close(conv_state[active], conv_state_ref[active], torch.bfloat16)


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@GPU_REQUIRED
def test_unified_update_large_state_cache(backend):
    """Only the first ``width - 1`` slots of a larger state cache are used."""
    dim, width, dtype, batch = 64, 4, torch.bfloat16, 2
    cache_state_len = 8
    x = torch.randn(batch, dim, 3, dtype=dtype, device=DEVICE)
    weight = torch.randn(dim, width, dtype=torch.float32, device=DEVICE)
    conv_state = torch.randn(batch, dim, cache_state_len, dtype=dtype, device=DEVICE)
    conv_state_ref = conv_state.clone()
    actual = causal_conv1d_update(
        x, conv_state, weight, activation=None, backend=backend
    )
    expected = causal_conv1d_update_torch(x, conv_state_ref, weight, None, None)
    assert_close(actual, expected, dtype)
    assert_close(conv_state, conv_state_ref, dtype)


@GPU_REQUIRED
def test_unified_update_varlen_decode():
    """vLLM-specific path: continuous-batching decode via query_start_loc."""
    dim, width, batch, seqlen = 256, WIDTH, 4, 4  # seqlen >= state_len
    x_dtype, weight_dtype = torch.bfloat16, torch.float32
    state_len = width - 1
    torch.manual_seed(42)
    x = torch.randn(batch * seqlen, dim, dtype=x_dtype)
    weight = torch.randn(dim, width, dtype=weight_dtype)
    pool = torch.randn(batch, state_len, dim, dtype=x_dtype)
    conv_state = pool.transpose(-1, -2)
    query_start_loc = torch.arange(batch + 1, dtype=torch.int32) * seqlen
    conv_state_indices = torch.arange(batch, dtype=torch.int32)
    x, conv_state, weight, query_start_loc, conv_state_indices = tuple(
        t.to(DEVICE)
        for t in (x, conv_state, weight, query_start_loc, conv_state_indices)
    )
    conv_state_ref = conv_state.clone()
    x_ref = x.clone()
    actual = causal_conv1d_update(
        x,
        conv_state,
        weight,
        None,
        activation="silu",
        conv_state_indices=conv_state_indices,
        query_start_loc=query_start_loc,
        max_query_len=seqlen,
        backend="vllm",
    )
    expected = _update_varlen_torch(
        x_ref, conv_state_ref, weight, query_start_loc, conv_state_indices, "silu"
    )
    assert_close(actual, expected, x_dtype)
    assert_close(conv_state, conv_state_ref, x_dtype)


# ---------------------------------------------------------------------------
# Bake-off: both backends on the same real-model shapes (pytest --bench)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BenchShape:
    name: str
    dim: int
    width: int = WIDTH
    x_dtype: torch.dtype = torch.bfloat16
    weight_dtype: torch.dtype = torch.float32
    lengths: tuple[int, ...] | None = None
    batch: int = 1
    seqlen: int = 1


def _varlen_lengths(batch: int, seed: int = 7) -> tuple[int, ...]:
    rng = torch.Generator().manual_seed(seed)
    return tuple(int(v) for v in torch.randint(40, 250, (batch,), generator=rng))


BENCH_FWD_SHAPES = {
    "measured-single": _BenchShape("measured-single", DIM_PER_RANK, lengths=(46,)),
    "varlen-b8": _BenchShape(
        "varlen-b8", DIM_PER_RANK, lengths=(46, 91, 137, 182, 57, 203, 118, 76)
    ),
    "varlen-max-batch": _BenchShape(
        "varlen-max-batch", DIM_PER_RANK, lengths=_varlen_lengths(MAX_DECODE_BATCH)
    ),
    "long-prefill": _BenchShape("long-prefill", DIM_PER_RANK, lengths=(4096,)),
    "full-model": _BenchShape("full-model", 12288, lengths=(4156,)),
}
BENCH_UPDATE_SHAPES = {
    "decode-b1-measured": _BenchShape("decode-b1-measured", DIM_DECODE, batch=1, seqlen=1),
    "decode-b8": _BenchShape("decode-b8", DIM_DECODE, batch=8, seqlen=1),
    "decode-b32-max": _BenchShape("decode-b32-max", DIM_DECODE, batch=MAX_DECODE_BATCH, seqlen=1),
    "decode-multi-token": _BenchShape("decode-multi-token", DIM_DECODE, batch=8, seqlen=5),
}


def _bench_fwd_inputs(case: _BenchShape):
    torch.manual_seed(42)
    lengths = case.lengths
    cu_seqlen = sum(lengths)
    state_len = case.width - 1
    # production layout: channel-last, x.stride(0) == 1 (feature contiguous);
    # row-major x would defeat the vec8 dwordx4 path and skew bench numbers.
    x = torch.randn(cu_seqlen, case.dim, dtype=case.x_dtype).t()
    weight = torch.randn(case.dim, case.width, dtype=case.weight_dtype)
    pool = torch.randn(2 * len(lengths), state_len, 3 * case.dim, dtype=case.x_dtype)
    conv_states = pool.transpose(-1, -2)[:, : case.dim]
    query_start_loc = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    batch = len(lengths)
    cache_indices = torch.arange(batch, dtype=torch.int32)
    has_initial_state = torch.ones(batch, dtype=torch.bool)
    return (
        tuple(
            t.to(DEVICE)
            for t in (x, weight, conv_states, query_start_loc, cache_indices, has_initial_state)
        ),
        list(lengths),
    )


def _bench_update_inputs(case: _BenchShape):
    torch.manual_seed(42)
    state_len = case.width - 1
    x = torch.randn(case.batch, case.dim, case.seqlen, dtype=case.x_dtype)
    weight = torch.randn(case.dim, case.width, dtype=case.weight_dtype)
    pool = torch.randn(case.batch, state_len, case.dim, dtype=case.x_dtype)
    conv_state = pool.transpose(-1, -2)
    return tuple(t.to(DEVICE) for t in (x, conv_state, weight))


def _useful_flops(case: _BenchShape, fwd: bool) -> int:
    if fwd:
        return 2 * case.dim * sum(case.lengths) * case.width
    return 2 * case.batch * case.dim * case.seqlen * case.width


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("shape_name", list(BENCH_FWD_SHAPES))
def test_bench_causal_conv1d_fn_compare(
    shape_name, time_ms, bench_warmup, bench_repeat
):
    """Time fwd on both ports; the ratio is the kernel-merge decision data."""
    case = BENCH_FWD_SHAPES[shape_name]
    results: dict[str, float] = {}
    for backend in ("sglang", "vllm"):
        (x, weight, conv_states, query_start_loc, cache_indices, has_init), lengths = (
            _bench_fwd_inputs(case)
        )
        ref = conv_states.clone()
        kwargs = dict(
            cache_indices=cache_indices,
            has_initial_state=has_init,
            activation="silu",
            pad_slot_id=PAD_SLOT_ID,
            backend=backend,
        )
        if backend == "sglang":
            kwargs["seq_lens_cpu"] = list(lengths)
        triton_fn = lambda: causal_conv1d_fn(  # noqa: E731
            x, weight, None, conv_states, query_start_loc, **kwargs
        )
        actual = triton_fn()
        expected = causal_conv1d_fn_torch(
            x, weight, None, ref, query_start_loc, cache_indices, has_init, "silu"
        )
        assert_close(actual, expected, case.x_dtype)
        assert_close(conv_states, ref, case.x_dtype)
        results[backend] = time_ms(triton_fn, bench_warmup, bench_repeat)
    _report("fwd", case.name, results, _useful_flops(case, fwd=True))


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize("shape_name", list(BENCH_UPDATE_SHAPES))
def test_bench_causal_conv1d_update_compare(
    shape_name, time_ms, bench_warmup, bench_repeat
):
    """Time update on both ports; the ratio is the kernel-merge decision data."""
    case = BENCH_UPDATE_SHAPES[shape_name]
    results: dict[str, float] = {}
    for backend in ("sglang", "vllm"):
        x, conv_state, weight = _bench_update_inputs(case)
        ref = conv_state.clone()
        triton_fn = lambda: causal_conv1d_update(  # noqa: E731
            x, conv_state, weight, None, activation="silu", backend=backend
        )
        actual = triton_fn()
        expected = causal_conv1d_update_torch(x, ref, weight, None, "silu")
        assert_close(actual, expected, case.x_dtype)
        assert_close(conv_state, ref, case.x_dtype)
        results[backend] = time_ms(triton_fn, bench_warmup, bench_repeat)
    _report("update", case.name, results, _useful_flops(case, fwd=False))


def _report(kind: str, name: str, results: dict[str, float], flops: int) -> None:
    ms_sgl, ms_vllm = results["sglang"], results["vllm"]
    print(
        f"[bench] causal_conv1d {kind} {name}: "
        f"sglang {ms_sgl:.4f} ms ({flops / (ms_sgl * 1e9):.3f} TFLOPS) | "
        f"vllm {ms_vllm:.4f} ms ({flops / (ms_vllm * 1e9):.3f} TFLOPS) | "
        f"vllm/sglang {ms_vllm / ms_sgl:.2f}x"
    )
