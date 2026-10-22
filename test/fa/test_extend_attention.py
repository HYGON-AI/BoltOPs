"""Correctness tests for the SGLang-compatible ExtendAttention API."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from boltops.fa.extend_attention import (
    build_unified_kv_indices,
    dense_prefill_attention_fwd,
    dense_prefill_attention_fwd_4d,
    dense_prefill_attention_fwd_segmented,
    extend_attention_fwd,
    extend_attention_fwd_unified,
)
from boltops.tools.extend_attention_triton_tune.benchmark_cases import (
    AUTOTUNE_CASES,
    BENCHMARK_CASES,
    BENCHMARK_TUNING_CASES,
    HISTORICAL_LONG_TUNING_CASES,
    HISTORICAL_SHORT_TUNING_CASES,
    TUNING_CASES,
    make_benchmark_inputs,
)


triton = pytest.importorskip("triton")
import triton.language as tl

from ._fa_bench_utils import (
    case_parameters,
    prepared_case,
    strict_fp32_matmul,
    useful_flops,
)

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)
HIP_ATOL_RTOL = 2e-2 if torch.version.hip is not None else 2e-6


def test_public_api_is_sglang_ordered_and_scales_are_required():
    """Validate the ABI without requiring a device or compiling Triton."""
    from boltops.fa import extend_attention_fwd as fa_extend

    params = list(inspect.signature(fa_extend).parameters.values())
    assert [p.name for p in params[:15]] == [
        "q_extend", "k_extend", "v_extend", "o_extend", "k_buffer", "v_buffer",
        "qo_indptr", "kv_indptr", "kv_indices", "custom_mask", "is_causal",
        "mask_indptr", "max_len_extend", "k_scale", "v_scale",
    ]
    assert params[13].default is inspect.Parameter.empty
    assert params[14].default is inspect.Parameter.empty
    with pytest.raises(ValueError, match="k_scale and v_scale"):
        fa_extend(
            None, None, None, None, None, None, None, None, None, None, True,
            None, 1, None, None,
        )


@GPU_REQUIRED
def test_optimized_kernel_routing_boundary_and_fallbacks():
    from boltops.fa.extend_attention.triton import extend_attention as backend

    common = dict(
        page_size=1,
        store_lse=False,
        skip_prefix=False,
        skip_extend=False,
        score_mod=None,
        aux_tensors=None,
        identity_kv_indices=False,
        is_hcu_platform=True,
    )
    assert backend._select_optimized_kernel(31, **common) == backend.SHORT_QUERY
    assert backend._select_optimized_kernel(32, **common) == backend.LONG_QUERY
    assert backend._select_optimized_kernel(
        31, **{**common, "page_size": 16}
    ) is None
    assert backend._select_optimized_kernel(
        31, **{**common, "store_lse": True}
    ) is None


@GPU_REQUIRED
def test_config_lookup_uses_exact_then_closest_larger_grid(tmp_path, monkeypatch):
    from boltops.fa.extend_attention.triton import config

    directory = tmp_path / "triton_3_6"
    directory.mkdir(parents=True)
    path = directory / (
        "long_query-device=gfx938_cu64-"
        "Q_Extend=torch.bfloat16-K_Buffer=torch.float8_e4m3fn.json"
    )
    shape = (8, 192, 128, False, True)
    smaller = (4, 8, *shape)
    larger = (8, 8, *shape)
    path.write_text(
        json.dumps(
            {
                "config": {
                    str(smaller): {"BLOCK_M": 32, "BLOCK_N": 32},
                    str(larger): {"BLOCK_M": 64, "BLOCK_N": 32},
                }
            }
        ),
        encoding="utf-8",
    )
    fallback_directory = tmp_path / "triton_3_2"
    fallback_directory.mkdir(parents=True)
    fallback_path = fallback_directory / path.name
    fallback_path.write_text(
        json.dumps(
            {
                "config": {
                    str(smaller): {"BLOCK_M": 16, "BLOCK_N": 32},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_ROOT", tmp_path)
    monkeypatch.setattr(
        config,
        "get_compatible_triton_config_dirs",
        lambda root, capabilities: [
            root / "triton_3_6",
            root / "triton_3_2",
        ],
    )
    monkeypatch.setattr(
        config,
        "get_device_info",
        lambda device=None: SimpleNamespace(arch="gfx938", cu_count=64),
    )
    config._load_configs.cache_clear()
    config.get_kernel_config.cache_clear()
    file_key = (
        "Q_Extend=torch.bfloat16",
        "K_Buffer=torch.float8_e4m3fn",
    )

    exact = config.get_kernel_config(
        config.LONG_QUERY, file_key, smaller, device="cuda"
    )
    nearest = config.get_kernel_config(
        config.LONG_QUERY, file_key, (6, 8, *shape), device="cuda"
    )

    assert exact["BLOCK_M"] == 32
    assert nearest["BLOCK_M"] == 64

    path.unlink()
    config.get_kernel_config.cache_clear()
    config._load_configs.cache_clear()
    fallback = config.get_kernel_config(
        config.LONG_QUERY, file_key, smaller, device="cuda"
    )
    assert fallback["BLOCK_M"] == 16
    config.get_kernel_config.cache_clear()
    config._load_configs.cache_clear()


def test_tune_search_space_is_split_by_kv_cache_dtype():
    from boltops.fa.extend_attention.triton import config
    from boltops.tools.extend_attention_triton_tune.tune_extend_attention import (
        _search_candidates,
    )

    bf16 = _search_candidates(config.LONG_QUERY, torch.bfloat16)
    fp8 = _search_candidates(config.LONG_QUERY, torch.float8_e4m3fn)

    assert len(bf16) == 54
    assert {candidate["kpack"] for candidate in bf16} == {1}
    assert {candidate["BLOCK_N"] for candidate in bf16} == {16, 32, 64}
    assert len(fp8) == 72
    assert {candidate["kpack"] for candidate in fp8} == {1, 2}
    assert {candidate["BLOCK_N"] for candidate in fp8} == {32, 64}


def test_fp8_arch_policy_matches_supported_architectures(monkeypatch):
    from boltops.utility import arch_info
    from boltops.tools.extend_attention_triton_tune.tune_extend_attention import (
        _fp8_kv_cache_tuning_available,
        _requested_kv_cache_dtypes,
    )

    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx946")
    assert arch_info.is_fp8_avail()
    assert all(
        _fp8_kv_cache_tuning_available(arch)
        for arch in ("gfx938", "gfx92a")
    )
    assert all(
        not _fp8_kv_cache_tuning_available(arch)
        for arch in ("gfx936", "gfx946")
    )

    assert _requested_kv_cache_dtypes("both", fp8_available=True) == (
        "bf16",
        "fp8",
    )
    assert _requested_kv_cache_dtypes("both", fp8_available=False) == ("bf16",)
    with pytest.raises(ValueError, match="gfx938 or gfx92a"):
        _requested_kv_cache_dtypes("fp8", fp8_available=False)


def test_packaged_fp8_configs_only_target_supported_architectures():
    from boltops.fa.extend_attention.triton import config

    config_root = Path(config.__file__).resolve().parent / "configs"
    fp8_configs = tuple(config_root.rglob("*K_Buffer=torch.float8_e4m3fn.json"))
    assert fp8_configs
    assert all("device=gfx938_" in path.name for path in fp8_configs)


def test_tuning_manifest_is_the_known_case_superset():
    assert len(AUTOTUNE_CASES) == 8
    assert len(BENCHMARK_CASES) == 8
    assert len(BENCHMARK_TUNING_CASES) == 8
    assert len(HISTORICAL_SHORT_TUNING_CASES) == 27
    assert len(HISTORICAL_LONG_TUNING_CASES) == 4
    assert len(TUNING_CASES) == 47
    assert len({case.name for case in TUNING_CASES}) == len(TUNING_CASES)

    applicable_jobs = [
        (case.name, dtype)
        for case in TUNING_CASES
        for dtype in case.kv_cache_dtypes
    ]
    assert len(applicable_jobs) == 55
    assert {case.kv_cache_dtypes for case in AUTOTUNE_CASES} == {
        ("bf16", "fp8")
    }
    assert {case.kv_cache_dtypes for case in BENCHMARK_TUNING_CASES} == {
        ("bf16",)
    }
    historical = (
        HISTORICAL_SHORT_TUNING_CASES + HISTORICAL_LONG_TUNING_CASES
    )
    assert {case.kv_cache_dtypes for case in historical} == {("fp8",)}
    assert {
        (case.batch_size, case.extend_length)
        for case in HISTORICAL_LONG_TUNING_CASES
    } == {
        (1, 16429),
        (1, 16507),
        (1, 20003),
    }


def test_historical_tuning_manifest_is_covered_by_packaged_fp8_configs():
    from boltops.fa.extend_attention.triton import config
    from boltops.tools.extend_attention_triton_tune.tune_extend_attention import (
        _config_key,
    )

    def tensor(dtype):
        return SimpleNamespace(dtype=dtype)

    manifest = set()
    historical = (
        HISTORICAL_SHORT_TUNING_CASES + HISTORICAL_LONG_TUNING_CASES
    )
    for case in historical:
        with_window_offsets = case.tune_with_window_offsets
        if with_window_offsets is None:
            with_window_offsets = (
                case.use_custom_mask and case.sliding_window_size > 0
            )
        data = {
            "q": tensor(torch.bfloat16),
            "k": tensor(torch.bfloat16),
            "v": tensor(torch.bfloat16),
            "output": tensor(torch.bfloat16),
            "k_buffer": tensor(torch.float8_e4m3fn),
            "v_buffer": tensor(torch.float8_e4m3fn),
            "qo_indptr": tensor(case.tune_qo_indptr_dtype),
            "kv_indptr": tensor(case.tune_kv_indptr_dtype),
            "kv_indices": tensor(case.tune_indices_dtype),
            "custom_mask": tensor(torch.bool) if case.use_custom_mask else None,
            "mask_indptr": (
                tensor(case.tune_mask_indptr_dtype)
                if case.use_custom_mask
                else None
            ),
            "sinks": tensor(case.tune_sinks_dtype) if case.with_sinks else None,
            "window_kv_offsets": (
                tensor(case.tune_window_offsets_dtype)
                if with_window_offsets
                else None
            ),
        }
        manifest.add((case.kind, str(_config_key(case, data))))

    reference_root = Path(config.__file__).resolve().parent / "configs" / "triton_3_6"
    reference = set()
    for path in reference_root.glob("*cu72*float8*.json"):
        kind = path.name.split("-", 1)[0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        reference.update((kind, key) for key in payload["config"])

    assert len(manifest) == 31
    assert manifest <= reference


def _parameters(fn):
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


def _cache_rows(cache: torch.Tensor, indices: torch.Tensor, page_size: int) -> torch.Tensor:
    if cache.ndim == 3:
        return cache[indices.long()]
    return cache[indices.long() // page_size, indices.long() % page_size]


def _reference(
    q, k, v, k_cache, v_cache, qo, kv, indices, *, causal, k_scale, v_scale,
    custom_mask=None, mask_indptr=None, skip_prefix_custom_mask=True,
    window=-1, sinks=None, offsets=None, logit_cap=0.0, xai_temperature_len=-1,
    skip_prefix=False, skip_extend=False, page_size=1, relative_bias=None,
    unified_xai_temperature=False,
):
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], device=q.device, dtype=torch.float32)
    lse_out = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    groups = q.shape[1] // k.shape[1]
    for batch in range(qo.numel() - 1):
        qb, qe = int(qo[batch]), int(qo[batch + 1])
        kb, ke = int(kv[batch]), int(kv[batch + 1])
        q_len, p_len = qe - qb, ke - kb
        q_rows = q[qb:qe].float().transpose(0, 1)
        k_ext = k[qb:qe].float().transpose(0, 1).repeat_interleave(groups, 0)
        v_ext = v[qb:qe].float().transpose(0, 1).repeat_interleave(groups, 0)
        k_pre = _cache_rows(k_cache, indices[kb:ke], page_size).float().transpose(0, 1).repeat_interleave(groups, 0)
        v_pre = _cache_rows(v_cache, indices[kb:ke], page_size).float().transpose(0, 1).repeat_interleave(groups, 0)
        # Match the SGLang kernel order exactly: apply sm/K scale before
        # soft-capping, rather than scaling the capped scores afterwards.
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
        prefix_scores = q_rows @ k_pre.transpose(-1, -2) * (sm_scale * k_scale)
        extend_scores = q_rows @ k_ext.transpose(-1, -2) * sm_scale
        if logit_cap:
            prefix_scores = logit_cap * torch.tanh(prefix_scores / logit_cap)
            extend_scores = logit_cap * torch.tanh(extend_scores / logit_cap)
        if xai_temperature_len > 0:
            positions = p_len + torch.arange(q_len, device=q.device, dtype=torch.float32)
            if unified_xai_temperature:
                # SGLang's Unified kernel uses a distinct, inverse
                # temperature schedule from the regular two-stage kernel.
                reg = torch.where(
                    positions < xai_temperature_len,
                    torch.ones_like(positions),
                    xai_temperature_len / (positions + 1.0),
                )
            else:
                reg = torch.where(
                    positions > xai_temperature_len,
                    torch.log2(positions) / math.log2(xai_temperature_len),
                    torch.ones_like(positions),
                )
            prefix_scores *= reg[None, :, None]
            extend_scores *= reg[None, :, None]
        if relative_bias is not None:
            q_positions = p_len + torch.arange(q_len, device=q.device)
            q_indices = torch.arange(qb, qe, device=q.device)

            def _bias_for(key_positions: torch.Tensor) -> torch.Tensor:
                rel = q_positions[:, None] - key_positions[None, :]
                valid = (rel >= 0) & (rel < relative_bias.shape[-1])
                rel = rel.clamp(0, relative_bias.shape[-1] - 1)
                per_query = relative_bias[q_indices].float()
                values = torch.stack(
                    [per_query[i, :, rel[i]] for i in range(q_len)], dim=1
                )
                return torch.where(valid[None, :, :], values, 0.0)

            prefix_scores += _bias_for(torch.arange(p_len, device=q.device))
            extend_scores += _bias_for(
                p_len + torch.arange(q_len, device=q.device)
            )
        prefix_ok = torch.ones(q_len, p_len, device=q.device, dtype=torch.bool)
        extend_ok = torch.ones(q_len, q_len, device=q.device, dtype=torch.bool)
        if custom_mask is not None:
            offset = 0 if offsets is None else int(offsets[batch])
            cm = custom_mask[int(mask_indptr[batch]):int(mask_indptr[batch + 1])].reshape(q_len, p_len + q_len + offset)[:, offset:]
            if not skip_prefix_custom_mask:
                prefix_ok &= cm[:, :p_len]
            extend_ok &= cm[:, p_len:]
        elif causal:
            extend_ok &= torch.ones(q_len, q_len, device=q.device, dtype=torch.bool).tril()
        if window > 0:
            # The regular SGLang kernel uses prefix_len + local_q for the
            # prefix stage, and local positions for the extend stage.
            prefix_ok &= (p_len + torch.arange(q_len, device=q.device))[:, None] <= (
                torch.arange(p_len, device=q.device)[None, :] + window
            )
            extend_ok &= torch.arange(q_len, device=q.device)[:, None] <= (
                torch.arange(q_len, device=q.device)[None, :] + window
            )
        if skip_prefix:
            prefix_ok.zero_()
        if skip_extend:
            extend_ok.zero_()
        scores = torch.cat((prefix_scores, extend_scores), -1)
        scores.masked_fill_(~torch.cat((prefix_ok, extend_ok), -1)[None], -torch.inf)
        values = torch.cat((v_pre * v_scale, v_ext), 1)
        if sinks is not None:
            scores = torch.cat((scores, sinks.float()[:, None, None].expand(-1, q_len, 1)), -1)
            values = torch.cat((values, torch.zeros_like(values[:, :1])), 1)
        lse = torch.logsumexp(scores, -1)
        out[qb:qe] = (torch.softmax(scores, -1) @ values).transpose(0, 1)
        lse_out[qb:qe] = lse.transpose(0, 1)
    return out, lse_out


def _inputs(*, page_size=1, d=32, dv=32, dtype=torch.float32):
    torch.manual_seed(123)
    q_lens, p_lens, hq, hkv = [2, 3], [3, 2], 4, 2
    total_q, slots = sum(q_lens), max(8, page_size)
    q = torch.randn(total_q, hq, d, device=DEVICE, dtype=dtype)
    k = torch.randn(total_q, hkv, d, device=DEVICE, dtype=dtype)
    v = torch.randn(total_q, hkv, dv, device=DEVICE, dtype=dtype)
    k_flat = torch.randn(slots, hkv, d, device=DEVICE, dtype=dtype)
    v_flat = torch.randn(slots, hkv, dv, device=DEVICE, dtype=dtype)
    if page_size == 1:
        k_cache, v_cache = k_flat, v_flat
    else:
        k_cache = k_flat.reshape(slots // page_size, page_size, hkv, d)
        v_cache = v_flat.reshape(slots // page_size, page_size, hkv, dv)
    qo = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE)
    kv = torch.tensor([0, *torch.tensor(p_lens).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE)
    indices = torch.tensor([slots - 1, 2, 5, 1, 4], dtype=torch.int32, device=DEVICE)
    return q, k, v, k_cache, v_cache, qo, kv, indices


def _ragged_inputs(*, q_lens, p_lens, d=32, dv=32, dtype=torch.float32):
    """Build an irregular batch large enough to cross regular HIP tile bounds."""
    assert len(q_lens) == len(p_lens)
    torch.manual_seed(456)
    hq, hkv = 4, 2
    total_q, total_prefix = sum(q_lens), sum(p_lens)
    slots = total_prefix + 17
    q = torch.randn(total_q, hq, d, device=DEVICE, dtype=dtype)
    k = torch.randn(total_q, hkv, d, device=DEVICE, dtype=dtype)
    v = torch.randn(total_q, hkv, dv, device=DEVICE, dtype=dtype)
    k_cache = torch.randn(slots, hkv, d, device=DEVICE, dtype=dtype)
    v_cache = torch.randn(slots, hkv, dv, device=DEVICE, dtype=dtype)
    qo = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=DEVICE,
    )
    kv = torch.tensor(
        [0, *torch.tensor(p_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=DEVICE,
    )
    indices = torch.randperm(slots, device=DEVICE)[:total_prefix].to(torch.int32)
    return q, k, v, k_cache, v_cache, qo, kv, indices


@triton.jit
def _relative_bias_score_mod(
    qk,
    q_pos,
    kv_pos,
    q_idx,
    head,
    mask,
    aux0,
    aux0_stride_t,
    aux0_stride_h,
    aux0_len,
):
    """SGLang score_mod contract: add a causal relative-position bias."""
    rel_dist = q_pos - kv_pos
    rel_idx = tl.minimum(tl.maximum(rel_dist, 0), aux0_len - 1)
    bias = tl.load(
        aux0 + q_idx * aux0_stride_t + head * aux0_stride_h + rel_idx,
        mask=mask & (rel_dist >= 0) & (rel_dist < aux0_len),
        other=0.0,
    )
    return qk + bias


@GPU_REQUIRED
def test_public_signature_matches_backend_and_sglang_contract() -> None:
    from boltops.fa.extend_attention.triton.extend_attention import (
        build_unified_kv_indices as build_backend,
        extend_attention_fwd as backend,
        extend_attention_fwd_unified as unified_backend,
    )

    assert _parameters(extend_attention_fwd) == _parameters(backend)
    assert not hasattr(backend, "redundant_attention")
    assert [name for name, _, _ in _parameters(extend_attention_fwd)] == [
        "q_extend", "k_extend", "v_extend", "o_extend", "k_buffer", "v_buffer",
        "qo_indptr", "kv_indptr", "kv_indices", "custom_mask", "is_causal",
        "mask_indptr", "max_len_extend", "k_scale", "v_scale", "sm_scale",
        "logit_cap", "skip_prefix_custom_mask", "sliding_window_size", "sinks",
        "window_kv_offsets", "xai_temperature_len", "lse_extend", "skip_prefix",
        "skip_extend", "page_size", "score_mod", "aux_tensors", "extend_seq_lens_cpu",
        "identity_kv_indices",
    ]
    assert _parameters(build_unified_kv_indices) == _parameters(build_backend)
    assert [name for name, _, _ in _parameters(build_unified_kv_indices)] == [
        "prefix_kv_indptr", "prefix_kv_indices", "extend_start_loc",
        "extend_seq_lens", "extend_kv_indices", "bs",
    ]
    assert _parameters(extend_attention_fwd_unified) == _parameters(unified_backend)
    assert [name for name, _, _ in _parameters(extend_attention_fwd_unified)] == [
        "q", "o", "k_buffer", "v_buffer", "k_scale", "v_scale", "qo_indptr",
        "kv_indptr", "kv_indices", "prefix_lens", "max_len_extend",
        "custom_mask", "mask_indptr", "sm_scale", "logit_cap", "is_causal",
        "sliding_window_size", "sinks", "window_start_pos",
        "xai_temperature_len", "page_size", "score_mod", "aux_tensors",
    ]


@GPU_REQUIRED
def test_build_unified_kv_indices_matches_sglang_layout() -> None:
    """Copy each SGLang prefix segment followed by its packed current segment."""
    batch_size = 4
    prefix_lens = torch.tensor([10, 20, 15, 25], dtype=torch.int32, device=DEVICE)
    extend_lens = torch.tensor([5, 3, 7, 4], dtype=torch.int32, device=DEVICE)
    prefix_indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
            torch.cumsum(prefix_lens, dim=0),
        )
    )
    prefix_indices = torch.arange(
        int(prefix_lens.sum()), dtype=torch.int64, device=DEVICE
    )
    extend_start_loc = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
            torch.cumsum(extend_lens[:-1], dim=0),
        )
    )
    extend_indices = torch.arange(
        int(prefix_lens.sum()),
        int(prefix_lens.sum() + extend_lens.sum()),
        dtype=torch.int64,
        device=DEVICE,
    )

    indptr, indices, returned_prefix_lens = build_unified_kv_indices(
        prefix_indptr,
        prefix_indices,
        extend_start_loc,
        extend_lens,
        extend_indices,
        batch_size,
    )

    expected_lens = prefix_lens + extend_lens
    expected_indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
            torch.cumsum(expected_lens, dim=0),
        )
    )
    expected_indices = torch.cat(
        [
            torch.cat(
                (
                    prefix_indices[
                        int(prefix_indptr[i]) : int(prefix_indptr[i + 1])
                    ],
                    extend_indices[
                        int(extend_start_loc[i]) : int(
                            extend_start_loc[i] + extend_lens[i]
                        )
                    ],
                )
            )
            for i in range(batch_size)
        ]
    )
    torch.testing.assert_close(indptr, expected_indptr, rtol=0, atol=0)
    # SGLang consumes these values as lengths; its source test does not require
    # a particular integer dtype. On HCU, advanced indexing may promote this
    # intermediate to int64.
    torch.testing.assert_close(
        returned_prefix_lens,
        prefix_lens.to(returned_prefix_lens.dtype),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)


def _unified_inputs(
    *,
    page_size: int,
    dtype: torch.dtype = torch.float32,
    regular_inputs=None,
):
    """Place the regular path's current K/V in a larger paged cache."""
    if regular_inputs is None:
        regular_inputs = _inputs(page_size=page_size, dtype=dtype)
    q, k, v, k_cache, v_cache, qo, kv, prefix_indices = regular_inputs
    prefix_slots = (
        k_cache.shape[0] if k_cache.ndim == 3 else k_cache.shape[0] * page_size
    )
    total_q = q.shape[0]
    total_slots = (
        (prefix_slots + total_q + page_size - 1) // page_size
    ) * page_size
    k_flat = torch.empty(
        total_slots, k.shape[1], k.shape[-1], dtype=k.dtype, device=DEVICE
    )
    v_flat = torch.empty(
        total_slots, v.shape[1], v.shape[-1], dtype=v.dtype, device=DEVICE
    )
    k_flat[:prefix_slots] = _cache_rows(
        k_cache, torch.arange(prefix_slots, device=DEVICE), page_size
    )
    v_flat[:prefix_slots] = _cache_rows(
        v_cache, torch.arange(prefix_slots, device=DEVICE), page_size
    )
    extend_indices = torch.arange(
        prefix_slots, prefix_slots + total_q, dtype=torch.int64, device=DEVICE
    )
    k_flat[extend_indices] = k
    v_flat[extend_indices] = v
    if page_size == 1:
        unified_k_cache, unified_v_cache = k_flat, v_flat
    else:
        unified_k_cache = k_flat.reshape(
            total_slots // page_size, page_size, k.shape[1], k.shape[-1]
        )
        unified_v_cache = v_flat.reshape(
            total_slots // page_size, page_size, v.shape[1], v.shape[-1]
        )
    extend_lens = qo[1:] - qo[:-1]
    extend_start_loc = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
            torch.cumsum(extend_lens[:-1], dim=0),
        )
    )
    unified_kv_indptr, unified_kv_indices, prefix_lens = (
        build_unified_kv_indices(
            kv,
            prefix_indices.to(torch.int64),
            extend_start_loc,
            extend_lens,
            extend_indices,
            len(extend_lens),
        )
    )
    return (
        q,
        k,
        v,
        k_cache,
        v_cache,
        qo,
        kv,
        prefix_indices,
        unified_k_cache,
        unified_v_cache,
        unified_kv_indptr,
        unified_kv_indices,
        prefix_lens,
        int(extend_lens.max()),
    )


@pytest.mark.parametrize("page_size", [1, 2, 16, 32, 64])
@GPU_REQUIRED
def test_extend_attention_unified_matches_regular_path(page_size: int) -> None:
    """Check SGLang's one-stage combined-KV path against its regular path."""
    (
        q,
        k,
        v,
        k_cache,
        v_cache,
        qo,
        kv,
        prefix_indices,
        unified_k_cache,
        unified_v_cache,
        unified_kv_indptr,
        unified_kv_indices,
        prefix_lens,
        max_len_extend,
    ) = _unified_inputs(page_size=page_size)
    regular_out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], device=DEVICE)
    unified_out = torch.empty_like(regular_out)
    extend_attention_fwd(
        q,
        k,
        v,
        regular_out,
        k_cache,
        v_cache,
        qo,
        kv,
        prefix_indices,
        None,
        True,
        None,
        max_len_extend,
        1.0,
        1.0,
        page_size=page_size,
    )
    extend_attention_fwd_unified(
        q,
        unified_out,
        unified_k_cache,
        unified_v_cache,
        1.0,
        1.0,
        qo,
        unified_kv_indptr,
        unified_kv_indices,
        prefix_lens,
        max_len_extend,
        is_causal=True,
        page_size=page_size,
    )
    torch.testing.assert_close(
        unified_out, regular_out, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL
    )


@GPU_REQUIRED
def test_extend_attention_unified_complex_controls_match_reference() -> None:
    """Cover Unified mask, window, sinks, temperature, and score-mod controls."""
    (
        q,
        k,
        v,
        k_cache,
        v_cache,
        qo,
        kv,
        prefix_indices,
        unified_k_cache,
        unified_v_cache,
        unified_kv_indptr,
        unified_kv_indices,
        prefix_lens,
        max_len_extend,
    ) = _unified_inputs(page_size=1)
    masks = []
    for batch in range(qo.numel() - 1):
        query_len = int(qo[batch + 1] - qo[batch])
        prefix_len = int(kv[batch + 1] - kv[batch])
        mask = torch.ones(
            query_len, prefix_len + query_len, dtype=torch.bool, device=DEVICE
        )
        mask[:, 0] = False
        mask[:, prefix_len:] = torch.ones(
            query_len, query_len, dtype=torch.bool, device=DEVICE
        ).tril()
        masks.append(mask.flatten())
    custom_mask = torch.cat(masks)
    mask_indptr = torch.tensor(
        [0, masks[0].numel(), custom_mask.numel()],
        dtype=torch.int32,
        device=DEVICE,
    )
    # The regular SGLang kernel requires this pointer when custom masking and
    # sliding-window attention are combined. These masks start at KV position
    # zero, so every per-sequence offset is zero.
    window_kv_offsets = torch.zeros(qo.numel() - 1, dtype=torch.int32, device=DEVICE)
    sinks = torch.tensor([0.2, -0.3, 0.1, 0.4], device=DEVICE)
    aux0 = torch.randn(q.shape[0], q.shape[1], 7, device=DEVICE)
    regular_out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], device=DEVICE)
    unified_out = torch.empty_like(regular_out)
    common_kwargs = {
        "logit_cap": 1.3,
        "sliding_window_size": 3,
        "sinks": sinks,
        "xai_temperature_len": 3,
        "score_mod": _relative_bias_score_mod,
        "aux_tensors": [aux0],
    }
    extend_attention_fwd(
        q,
        k,
        v,
        regular_out,
        k_cache,
        v_cache,
        qo,
        kv,
        prefix_indices,
        custom_mask,
        False,
        mask_indptr,
        max_len_extend,
        1.0,
        1.0,
        skip_prefix_custom_mask=False,
        window_kv_offsets=window_kv_offsets,
        **common_kwargs,
    )
    extend_attention_fwd_unified(
        q,
        unified_out,
        unified_k_cache,
        unified_v_cache,
        1.0,
        1.0,
        qo,
        unified_kv_indptr,
        unified_kv_indices,
        prefix_lens,
        max_len_extend,
        custom_mask=custom_mask,
        mask_indptr=mask_indptr,
        is_causal=False,
        window_start_pos=torch.tensor([11, 29], dtype=torch.int32, device=DEVICE),
        **common_kwargs,
    )
    regular_expected, _ = _reference(
        q,
        k,
        v,
        k_cache,
        v_cache,
        qo,
        kv,
        prefix_indices,
        causal=False,
        k_scale=1.0,
        v_scale=1.0,
        custom_mask=custom_mask,
        mask_indptr=mask_indptr,
        skip_prefix_custom_mask=False,
        window=3,
        sinks=sinks,
        logit_cap=1.3,
        xai_temperature_len=3,
        relative_bias=aux0,
    )
    unified_expected, _ = _reference(
        q,
        k,
        v,
        k_cache,
        v_cache,
        qo,
        kv,
        prefix_indices,
        causal=False,
        k_scale=1.0,
        v_scale=1.0,
        custom_mask=custom_mask,
        mask_indptr=mask_indptr,
        skip_prefix_custom_mask=False,
        window=3,
        sinks=sinks,
        logit_cap=1.3,
        xai_temperature_len=3,
        relative_bias=aux0,
        unified_xai_temperature=True,
    )
    torch.testing.assert_close(
        regular_out, regular_expected, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL
    )
    torch.testing.assert_close(
        unified_out, unified_expected, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@GPU_REQUIRED
def test_extend_attention_unified_page16_low_precision_matches_regular(
    dtype: torch.dtype,
) -> None:
    """Exercise page-shaped Unified KV caches in the inference dtypes."""
    (
        q,
        k,
        v,
        k_cache,
        v_cache,
        qo,
        kv,
        prefix_indices,
        unified_k_cache,
        unified_v_cache,
        unified_kv_indptr,
        unified_kv_indices,
        prefix_lens,
        max_len_extend,
    ) = _unified_inputs(page_size=16, dtype=dtype)
    regular_out = torch.empty(
        q.shape[0], q.shape[1], v.shape[-1], dtype=dtype, device=DEVICE
    )
    unified_out = torch.empty_like(regular_out)
    extend_attention_fwd(
        q,
        k,
        v,
        regular_out,
        k_cache,
        v_cache,
        qo,
        kv,
        prefix_indices,
        None,
        True,
        None,
        max_len_extend,
        1.0,
        1.0,
        page_size=16,
    )
    extend_attention_fwd_unified(
        q,
        unified_out,
        unified_k_cache,
        unified_v_cache,
        1.0,
        1.0,
        qo,
        unified_kv_indptr,
        unified_kv_indices,
        prefix_lens,
        max_len_extend,
        is_causal=True,
        page_size=16,
    )
    torch.testing.assert_close(unified_out, regular_out, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("page_size", [1, 2])
@GPU_REQUIRED
def test_extend_attention_matches_independent_reference(page_size: int) -> None:
    q, k, v, k_cache, v_cache, qo, kv, indices = _inputs(page_size=page_size)
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], device=DEVICE)
    lse = torch.empty(q.shape[:2], device=DEVICE)
    sinks = torch.tensor([0.2, -0.3, 0.1, 0.4], device=DEVICE)
    assert extend_attention_fwd(
        q, k, v, out, k_cache, v_cache, qo, kv, indices, None, True, None, 3,
        1.25, 0.8, logit_cap=1.7, sliding_window_size=3, sinks=sinks,
        xai_temperature_len=3, lse_extend=lse, page_size=page_size,
    ) is None
    expected, expected_lse = _reference(
        q, k, v, k_cache, v_cache, qo, kv, indices, causal=True, k_scale=1.25,
        v_scale=0.8, window=3, sinks=sinks, logit_cap=1.7,
        xai_temperature_len=3, page_size=page_size,
    )
    torch.testing.assert_close(out, expected, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL)
    torch.testing.assert_close(lse, expected_lse, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL)


@GPU_REQUIRED
def test_identity_kv_indices_matches_indexed_path() -> None:
    q, k, v, k_cache, v_cache, qo, kv, _ = _inputs(page_size=1)
    identity = torch.arange(kv[-1], dtype=torch.int32, device=DEVICE)
    indexed = torch.empty(q.shape[0], q.shape[1], v.shape[-1], device=DEVICE)
    direct = torch.empty_like(indexed)
    extend_attention_fwd(
        q, k, v, indexed, k_cache, v_cache, qo, kv, identity, None, True,
        None, 3, 1.0, 1.0,
    )
    extend_attention_fwd(
        q, k, v, direct, k_cache, v_cache, qo, kv, identity, None, True,
        None, 3, 1.0, 1.0, identity_kv_indices=True,
    )
    torch.testing.assert_close(direct, indexed, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL)


@GPU_REQUIRED
def test_fully_masked_rows_are_zero_with_negative_infinite_lse() -> None:
    q, k, v, k_cache, v_cache, qo, kv, indices = _inputs()
    q_lens, prefix_lens = [2, 3], [3, 2]
    masks = [
        torch.zeros(q_len * (prefix_len + q_len), dtype=torch.bool, device=DEVICE)
        for q_len, prefix_len in zip(q_lens, prefix_lens)
    ]
    custom_mask = torch.cat(masks)
    mask_indptr = torch.tensor(
        [0, masks[0].numel(), custom_mask.numel()],
        dtype=torch.int32,
        device=DEVICE,
    )
    out = torch.full(
        (q.shape[0], q.shape[1], v.shape[-1]), float("nan"), device=DEVICE
    )
    lse = torch.full(q.shape[:2], float("nan"), device=DEVICE)
    extend_attention_fwd(
        q, k, v, out, k_cache, v_cache, qo, kv, indices, custom_mask, False,
        mask_indptr, 3, 1.0, 1.0, skip_prefix_custom_mask=False,
        lse_extend=lse,
    )
    torch.testing.assert_close(out, torch.zeros_like(out))
    assert torch.isneginf(lse).all()


@GPU_REQUIRED
def test_short_query_kernel_fully_masked_rows_are_zero() -> None:
    """Keep BoltOPs' empty-row behavior when no LSE output is requested."""
    q, k, v, k_cache, v_cache, qo, kv, indices = _inputs()
    q_lens, prefix_lens = [2, 3], [3, 2]
    masks = [
        torch.zeros(q_len * (prefix_len + q_len), dtype=torch.bool, device=DEVICE)
        for q_len, prefix_len in zip(q_lens, prefix_lens)
    ]
    custom_mask = torch.cat(masks)
    mask_indptr = torch.tensor(
        [0, masks[0].numel(), custom_mask.numel()],
        dtype=torch.int32,
        device=DEVICE,
    )
    out = torch.full(
        (q.shape[0], q.shape[1], v.shape[-1]), float("nan"), device=DEVICE
    )
    extend_attention_fwd(
        q, k, v, out, k_cache, v_cache, qo, kv, indices, custom_mask, False,
        mask_indptr, 3, 1.0, 1.0, skip_prefix_custom_mask=False,
    )
    torch.testing.assert_close(out, torch.zeros_like(out))


@GPU_REQUIRED
def test_dense_prefill_matches_independent_reference() -> None:
    torch.manual_seed(987)
    q_lens, kv_lens, hq, hkv, d, dv = [2, 3], [4, 5], 4, 2, 32, 24
    q = torch.randn(sum(q_lens), hq, d, device=DEVICE)
    k = torch.randn(sum(kv_lens), hkv, d, device=DEVICE)
    v = torch.randn(sum(kv_lens), hkv, dv, device=DEVICE)
    qo = torch.tensor([0, 2, 5], dtype=torch.int32, device=DEVICE)
    kv = torch.tensor([0, 4, 9], dtype=torch.int32, device=DEVICE)
    out = torch.empty(sum(q_lens), hq, dv, device=DEVICE)
    lse = torch.empty(sum(q_lens), hq, device=DEVICE)
    k_scale, v_scale = 1.25, 0.8
    dense_prefill_attention_fwd(
        q, k, v, out, qo, kv, max(q_lens), k_scale=k_scale,
        v_scale=v_scale, lse=lse,
    )

    expected = torch.empty_like(out)
    expected_lse = torch.empty_like(lse)
    groups = hq // hkv
    for batch, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        qb, kb = int(qo[batch]), int(kv[batch])
        q_rows = q[qb : qb + q_len].transpose(0, 1)
        k_rows = k[kb : kb + kv_len].transpose(0, 1).repeat_interleave(groups, 0)
        v_rows = v[kb : kb + kv_len].transpose(0, 1).repeat_interleave(groups, 0)
        scores = q_rows @ k_rows.transpose(-1, -2) * (k_scale / math.sqrt(d))
        prefix_len = kv_len - q_len
        causal = torch.arange(kv_len, device=DEVICE)[None, :] <= (
            prefix_len + torch.arange(q_len, device=DEVICE)[:, None]
        )
        scores.masked_fill_(~causal[None], -torch.inf)
        expected[qb : qb + q_len] = (
            torch.softmax(scores, -1) @ (v_rows * v_scale)
        ).transpose(0, 1)
        expected_lse[qb : qb + q_len] = torch.logsumexp(scores, -1).transpose(0, 1)

    torch.testing.assert_close(out, expected, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL)
    torch.testing.assert_close(lse, expected_lse, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL)


# Four-dimensional dense prefill.


def _dense_4d_inputs(batch=2, heads=3, q_len=7, kv_len=23):
    shape = (batch, heads)
    q = torch.randn(*shape, q_len, 192, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(*shape, kv_len, 192, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(*shape, kv_len, 128, device=DEVICE, dtype=torch.bfloat16)
    return q, k, v


def _causal_positions(q, k):
    """Global positions of the last ``Sq`` of ``Sk`` tokens: bottom-right causal."""
    q_len, kv_len = q.shape[-2], k.shape[-2]
    return torch.arange(kv_len - q_len, kv_len, device=q.device)


def _attention_reference(q, k, v, sm_scale, positions=None):
    """FP32 attention over ``[..., S, D]``; query i sees keys ``0 .. positions[i]``.

    ``positions=None`` attends every key (non-causal).
    """
    scores = q.float() @ k.float().transpose(-1, -2) * sm_scale
    if positions is not None:
        mask = torch.arange(k.shape[-2], device=q.device)[None, :] > positions[:, None]
        scores.masked_fill_(mask, -torch.inf)
    return torch.softmax(scores, -1) @ v.float()


@pytest.fixture(params=["packed", "per_request"])
def dense_4d_path(request, monkeypatch):
    """Force the 4-D entry onto one of its two launch strategies."""
    from boltops.fa.extend_attention.triton import extend_attention as backend

    monkeypatch.setattr(
        backend, "_use_packed_dense_prefill_4d", lambda *args: request.param == "packed"
    )


@pytest.mark.parametrize(
    ("batch", "heads", "q_len", "kv_len", "packed"),
    [
        (1, 8, 16, 16, False),  # a single request is already one launch
        (32, 8, 16, 16, True),  # host-bound: one launch instead of 32
        (4, 8, 64, 4096, True),  # 8 * cdiv(64, 64) * 4096 == 2**15 key rows
        (4, 8, 64, 4097, False),  # one key row past the threshold
        (4, 64, 1, 4096, False),  # decode-like, long KV: per request is faster
        (4, 64, 6656, 13312, False),  # Kimi mha_slice1
        (100_000, 1, 1, 128, False),  # small, but packed K offsets pass int32
    ],
)
def test_dense_prefill_4d_packs_only_small_batched_requests(
    batch, heads, q_len, kv_len, packed
) -> None:
    from boltops.fa.extend_attention.triton import extend_attention as backend

    assert backend._use_packed_dense_prefill_4d(batch, heads, q_len, kv_len, 192, 128) is packed


@GPU_REQUIRED
def test_dense_prefill_4d_public_signature_and_exports() -> None:
    import boltops.fa
    from boltops.fa.extend_attention.triton.extend_attention import (
        dense_prefill_attention_fwd_4d as backend,
    )

    assert boltops.fa.dense_prefill_attention_fwd_4d is dense_prefill_attention_fwd_4d
    assert _parameters(dense_prefill_attention_fwd_4d) == _parameters(backend)
    assert _parameters(dense_prefill_attention_fwd_4d) == [
        ("q", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("k", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("v", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("sm_scale", inspect.Parameter.KEYWORD_ONLY, None),
        ("is_causal", inspect.Parameter.KEYWORD_ONLY, True),
    ]


@pytest.mark.parametrize(
    ("batch", "heads", "q_len", "kv_len", "is_causal", "sm_scale"),
    [
        (1, 1, 1, 1, True, None),
        (2, 3, 17, 17, True, None),  # no prefix
        (2, 3, 5, 77, True, 0.3),  # prefix, ragged KV tile
        (3, 2, 1, 130, True, 0.3),  # single query over several KV tiles
        (2, 2, 130, 200, True, 0.3),  # several query and KV tiles
        (2, 2, 70, 150, False, 0.3),
    ],
)
@pytest.mark.usefixtures("dense_4d_path")
@GPU_REQUIRED
def test_dense_prefill_4d_matches_independent_reference(
    batch, heads, q_len, kv_len, is_causal, sm_scale
) -> None:
    torch.manual_seed(2468)
    q, k, v = _dense_4d_inputs(batch, heads, q_len, kv_len)

    out = dense_prefill_attention_fwd_4d(
        q, k, v, sm_scale=sm_scale, is_causal=is_causal
    )

    assert out.shape == (batch, heads, q_len, 128)
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    expected = _attention_reference(
        q, k, v, 192**-0.5 if sm_scale is None else sm_scale,
        _causal_positions(q, k) if is_causal else None,
    )
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.usefixtures("dense_4d_path")
@GPU_REQUIRED
def test_dense_prefill_4d_matches_packed_api_and_is_deterministic() -> None:
    torch.manual_seed(1357)
    batch, heads, q_len, kv_len = 2, 2, 33, 100
    q, k, v = _dense_4d_inputs(batch, heads, q_len, kv_len)

    def pack(t):
        return t.transpose(1, 2).reshape(-1, heads, t.shape[-1])

    packed_out = torch.empty(batch * q_len, heads, 128, device=DEVICE, dtype=torch.bfloat16)
    dense_prefill_attention_fwd(
        pack(q), pack(k), pack(v), packed_out,
        torch.arange(0, (batch + 1) * q_len, q_len, device=DEVICE, dtype=torch.int32),
        torch.arange(0, (batch + 1) * kv_len, kv_len, device=DEVICE, dtype=torch.int32),
        q_len,
    )
    out = dense_prefill_attention_fwd_4d(q, k, v)

    expected = packed_out.view(batch, q_len, heads, 128).transpose(1, 2)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert torch.equal(dense_prefill_attention_fwd_4d(q, k, v), out)


@pytest.mark.usefixtures("dense_4d_path")
@GPU_REQUIRED
def test_dense_prefill_4d_accepts_strided_inputs() -> None:
    torch.manual_seed(97531)
    batch, heads, q_len, kv_len = 2, 3, 40, 90
    # Token-major projections, with K and V sliced out of one buffer.
    q = torch.randn(batch, q_len, heads, 192, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(batch, kv_len, heads, 320, device=DEVICE, dtype=torch.bfloat16)
    q, k, v = q.transpose(1, 2), kv[..., :192].transpose(1, 2), kv[..., 192:].transpose(1, 2)
    # Head dim not unit-stride: the only layout that has to be copied.
    q_strided_dim = q.transpose(-1, -2).contiguous().transpose(-1, -2)

    expected = dense_prefill_attention_fwd_4d(q.contiguous(), k.contiguous(), v.contiguous())

    torch.testing.assert_close(dense_prefill_attention_fwd_4d(q, k, v), expected, atol=0, rtol=0)
    torch.testing.assert_close(
        dense_prefill_attention_fwd_4d(q_strided_dim, k, v), expected, atol=0, rtol=0
    )


@pytest.mark.parametrize("padded_operand", ["q", "k", "v"])
def test_dense_prefill_4d_packed_checks_actual_strides(monkeypatch, padded_operand) -> None:
    """A small CPU view reproduces the padded packed-span overflow without a GPU."""
    from boltops.fa.extend_attention.triton import extend_attention as backend

    batch, heads, q_len, kv_len = 4, 8, 1, 7
    limit = 60000
    monkeypatch.setattr(backend, "_DENSE_PREFILL_MAX_SPAN", limit)

    def projection(length, width, storage_width):
        storage = torch.empty(batch, length, heads, storage_width, dtype=torch.bfloat16)
        return storage[..., :width].transpose(1, 2)

    q = projection(q_len, 192, 2048 if padded_operand == "q" else 192)
    k = projection(kv_len, 192, 320 if padded_operand == "k" else 192)
    v = projection(kv_len, 128, 320 if padded_operand == "v" else 128)
    assert all(backend._request_span(t) < limit for t in (q, k, v))
    assert backend._use_packed_dense_prefill_4d(batch, heads, q_len, kv_len, 192, 128)

    sentinel = object()
    calls = []

    def fallback(*args):
        calls.append(args)
        return sentinel

    def unsafe_launch(*args, **kwargs):
        pytest.fail("packed launcher must not receive an overflowing padded view")

    monkeypatch.setattr(backend, "_dense_prefill_4d_per_request", fallback)
    monkeypatch.setattr(backend, "dense_prefill_attention_fwd", unsafe_launch)
    assert backend._dense_prefill_4d_packed(q, k, v, 0.3, True) is sentinel
    assert len(calls) == 1
    assert all(actual is expected for actual, expected in zip(calls[0][:3], (q, k, v)))
    assert calls[0][3:] == (0.3, True)


@GPU_REQUIRED
def test_dense_prefill_4d_padded_packed_overflow_falls_back_correctly(monkeypatch) -> None:
    from boltops.fa.extend_attention.triton import extend_attention as backend

    torch.manual_seed(31337)
    batch, heads, q_len, kv_len = 4, 8, 1, 7
    q = torch.randn(batch, heads, q_len, 192, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(batch, kv_len, heads, 320, device=DEVICE, dtype=torch.bfloat16)
    k, v = kv[..., :192].transpose(1, 2), kv[..., 192:].transpose(1, 2)
    monkeypatch.setattr(backend, "_DENSE_PREFILL_MAX_SPAN", 60000)
    assert backend._use_packed_dense_prefill_4d(batch, heads, q_len, kv_len, 192, 128)

    original = backend._dense_prefill_4d_per_request
    fallback_calls = []

    def fallback(*args):
        fallback_calls.append(True)
        return original(*args)

    monkeypatch.setattr(backend, "_dense_prefill_4d_per_request", fallback)
    out = dense_prefill_attention_fwd_4d(q, k, v, sm_scale=0.3)
    assert fallback_calls == [True]
    expected = _attention_reference(q, k, v, 0.3, _causal_positions(q, k))
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("make_args", "error", "match"),
    [
        pytest.param(lambda q, k, v: ((q[0], k, v), {}), ValueError, "q must be 4-D", id="ndim"),
        pytest.param(
            lambda q, k, v: ((q.float(), k, v), {}), TypeError, "q must be torch.bfloat16",
            id="dtype",
        ),
        pytest.param(
            lambda q, k, v: ((q, k.cpu(), v), {}), ValueError, "k must be a CUDA/HIP tensor",
            id="cpu",
        ),
        pytest.param(
            lambda q, k, v: ((q[:, :, :0], k, v), {}), ValueError, "q must not have empty dims",
            id="empty",
        ),
        pytest.param(
            lambda q, k, v: ((q, k[:1], v[:1]), {}), ValueError, "batch sizes must match",
            id="batch",
        ),
        pytest.param(
            lambda q, k, v: ((q, k, v[:, :, :4]), {}), ValueError, "k/v heads and lengths",
            id="kv_len",
        ),
        pytest.param(
            lambda q, k, v: ((q, k[..., :128], v), {}), ValueError, "q/k head dims must match",
            id="qk_dim",
        ),
        pytest.param(lambda q, k, v: ((q[:, :1], k, v), {}), ValueError, "only MHA", id="gqa"),
        pytest.param(
            lambda q, k, v: ((q[..., :128], k[..., :128], v), {}),
            ValueError, r"unsupported \(Dqk, Dv\)=\(128, 128\)", id="head_dims",
        ),
        pytest.param(
            lambda q, k, v: ((q, k[:, :, :2], v[:, :, :2]), {}), ValueError, "Sk must be >= Sq",
            id="short_kv",
        ),
        pytest.param(
            lambda q, k, v: ((q, k, v), {"is_causal": 1}), TypeError, "is_causal must be bool",
            id="is_causal",
        ),
        pytest.param(
            lambda q, k, v: ((q.clone().requires_grad_(), k, v), {}),
            RuntimeError, "does not support autograd", id="requires_grad",
        ),
    ],
)
@GPU_REQUIRED
def test_dense_prefill_4d_rejects_invalid_inputs(make_args, error, match) -> None:
    args, kwargs = make_args(*_dense_4d_inputs(2, 2, 3, 5))
    with pytest.raises(error, match=match):
        dense_prefill_attention_fwd_4d(*args, **kwargs)


@pytest.mark.parametrize(
    ("sm_scale", "error"),
    [
        (True, TypeError),
        ("0.1", TypeError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (math.inf, ValueError),
        (math.nan, ValueError),
    ],
)
@GPU_REQUIRED
def test_dense_prefill_4d_rejects_invalid_sm_scale(sm_scale, error) -> None:
    with pytest.raises(error, match="sm_scale must be"):
        dense_prefill_attention_fwd_4d(*_dense_4d_inputs(1, 1, 3, 5), sm_scale=sm_scale)


@GPU_REQUIRED
def test_dense_prefill_4d_rejects_per_request_span_beyond_int32(monkeypatch) -> None:
    from boltops.fa.extend_attention.triton import extend_attention as backend

    q, k, v = _dense_4d_inputs(2, 2, 3, 5)
    # k[b] is the widest operand here: 2 heads * 5 tokens * 192 elements.
    monkeypatch.setattr(backend, "_DENSE_PREFILL_MAX_SPAN", 2 * 5 * 192)
    with pytest.raises(ValueError, match="k\\[b\\] spans 1920 elements"):
        dense_prefill_attention_fwd_4d(q, k, v)


@pytest.mark.usefixtures("dense_4d_path")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA/HIP devices")
def test_dense_prefill_4d_runs_on_non_current_device() -> None:
    torch.manual_seed(8642)
    with torch.cuda.device(1):
        q, k, v = _dense_4d_inputs(2, 2, 9, 30)
    with torch.cuda.device(0):
        out = dense_prefill_attention_fwd_4d(q, k, v)

    assert out.device == q.device
    expected = _attention_reference(q, k, v, 192**-0.5, _causal_positions(q, k))
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


@GPU_REQUIRED
def test_dense_prefill_4d_long_context_past_packed_int32_offsets() -> None:
    """Packed, these two requests would put K offsets past 2**31; per request they fit."""
    torch.manual_seed(1928)
    q, k, v = _dense_4d_inputs(2, 64, 64, 131072)

    # A peaked softmax keeps outputs O(1); at the default scale, averaging
    # 131072 random values leaves outputs within the tolerance of zero.
    out = dense_prefill_attention_fwd_4d(q, k, v, sm_scale=1.0)

    # The last heads of the last request are where a wrapped offset would land.
    expected = _attention_reference(
        q[1:, -4:], k[1:, -4:], v[1:, -4:], 1.0, _causal_positions(q, k)
    )
    torch.testing.assert_close(out[1:, -4:].float(), expected, atol=2e-2, rtol=2e-2)


# Four-dimensional prefill isolation and independent-B32 coverage.

@pytest.fixture
def strict_fp32():
    """Run one test without TF32, restoring global matmul settings afterwards."""
    with strict_fp32_matmul():
        yield


@GPU_REQUIRED
@pytest.mark.usefixtures("strict_fp32")
@pytest.mark.parametrize("q_len,kv_len,strided,shared", [
    (5, 19, False, False),
    (17, 513, False, False),  # B32, per-request route and ragged KV tile
    (17, 513, True, False),
    (17, 513, False, True),
])
def test_dense_prefill_4d_b32_matches_full_independent_fp32(q_len, kv_len, strided, shared):
    torch.manual_seed(732)
    batch, heads = 32, 64
    storage_d = 320 if strided else 192
    q = torch.randn(batch, heads, q_len, storage_d, device=DEVICE, dtype=torch.bfloat16)[..., :192]
    k = torch.randn(1 if shared else batch, heads, kv_len, storage_d,
                    device=DEVICE, dtype=torch.bfloat16)[..., :192]
    v = torch.randn(1 if shared else batch, heads, kv_len, 256 if strided else 128,
                    device=DEVICE, dtype=torch.bfloat16)[..., :128]
    if shared:
        k, v = k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1)
    out = dense_prefill_attention_fwd_4d(q, k, v, sm_scale=0.11, is_causal=True)
    assert out.shape == (batch, heads, q_len, 128)
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    assert bool(torch.isfinite(out).all())
    positions = _causal_positions(q, k)
    # Keep the independent reference bounded; every output element is checked.
    for b in range(batch):
        expected = _attention_reference(q[b], k[b], v[b], 0.11, positions)
        torch.testing.assert_close(out[b].float(), expected, atol=0.02, rtol=0.02)


@GPU_REQUIRED
def test_dense_prefill_4d_shared_kv_is_readonly_and_batch_isolated():
    torch.manual_seed(733)
    q = torch.randn(32, 64, 5, 192, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(1, 64, 19, 192, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(1, 64, 19, 128, device=DEVICE, dtype=torch.bfloat16)
    saved_k, saved_v = k.clone(), v.clone()
    first = dense_prefill_attention_fwd_4d(q, k.expand(32, -1, -1, -1),
                                          v.expand(32, -1, -1, -1), sm_scale=0.11)
    q[17].neg_()
    second = dense_prefill_attention_fwd_4d(q, k.expand(32, -1, -1, -1),
                                           v.expand(32, -1, -1, -1), sm_scale=0.11)
    torch.testing.assert_close(k, saved_k, atol=0, rtol=0)
    torch.testing.assert_close(v, saved_v, atol=0, rtol=0)
    torch.testing.assert_close(first[:17], second[:17], atol=0, rtol=0)
    torch.testing.assert_close(first[18:], second[18:], atol=0, rtol=0)
    assert not torch.equal(first[17], second[17])


@pytest.mark.parametrize("packed", [False, True])
@GPU_REQUIRED
def test_dense_prefill_4d_causal_future_and_batch_isolation_preserves_inputs(monkeypatch, packed):
    from boltops.fa.extend_attention.triton import extend_attention as backend
    monkeypatch.setattr(backend, "_use_packed_dense_prefill_4d", lambda *args: packed)
    torch.manual_seed(834)
    q, k, v = _dense_4d_inputs(2, 3, 5, 9)
    snapshots = [t.clone() for t in (q, k, v)]
    original = dense_prefill_attention_fwd_4d(q, k, v, sm_scale=0.17)
    for t, saved in zip((q, k, v), snapshots):
        assert torch.equal(t, saved)
    # Query 0 sees keys 0..4; changes to future keys 5..8 cannot affect it.
    k2, v2 = k.clone(), v.clone()
    k2[0, :, 5:] = 20
    v2[0, :, 5:] = -20
    changed = dense_prefill_attention_fwd_4d(q, k2, v2, sm_scale=0.17)
    torch.testing.assert_close(changed[0, :, :1], original[0, :, :1], atol=0, rtol=0)
    torch.testing.assert_close(changed[1], original[1], atol=0, rtol=0)
    # Noncausal attention must see the same changed tokens.
    a = dense_prefill_attention_fwd_4d(q, k, v, sm_scale=0.17, is_causal=False)
    b = dense_prefill_attention_fwd_4d(q, k2, v2, sm_scale=0.17, is_causal=False)
    assert not torch.equal(a[0, :, :1], b[0, :, :1])


# Segmented prefill: explicit global positions and metadata contracts.

@pytest.fixture
def segmented_adapter():
    """Load the Triton host adapter only when a test needs an active target."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP target required to load the Triton host adapter")
    from boltops.fa.extend_attention.triton import extend_attention as backend

    return backend


def test_segmented_exports_and_signature():
    from boltops.fa.extend_attention import dense_prefill_attention_fwd_segmented as entry
    assert entry is dense_prefill_attention_fwd_segmented
    assert list(inspect.signature(entry).parameters) == [
        "q", "k", "v", "q_segment_lens", "kv_prefix_lens", "sm_scale"]
    for p in list(inspect.signature(entry).parameters.values())[3:]:
        assert p.kind == p.KEYWORD_ONLY and p.default is p.empty


@pytest.mark.parametrize("value,error", [
    ([[1]], ValueError), ([[1, 2], [3, 4]], ValueError),
    ([[True, 2]], TypeError), ([[1.0, 2]], TypeError), ([[-1, 2]], ValueError),
    ("12", ValueError), (torch.ones(1, 2), TypeError),
    (torch.ones(1, 2, dtype=torch.bool), TypeError),
])
def test_segmented_host_metadata_rejects_ambiguous_values(segmented_adapter, value, error):
    with pytest.raises(error):
        segmented_adapter._host_lengths(value, 1, "lengths")


def test_segmented_host_metadata_snapshots_integer_lists_and_cpu_tensor(segmented_adapter):
    values = [[0, 7], [3, 4]]
    assert segmented_adapter._host_lengths(values, 2, "lengths") == ((0, 7), (3, 4))
    assert segmented_adapter._host_lengths(torch.tensor(values), 2, "lengths") == ((0, 7), (3, 4))


@GPU_REQUIRED
@pytest.mark.usefixtures("strict_fp32")
@pytest.mark.parametrize("batch,heads,total,prefix", [(32, 64, 33, 0), (2, 4, 257, 19), (2, 3, 7, 3)])
def test_segmented_all_eight_ranks_restore_full_global_attention(batch, heads, total, prefix):
    torch.manual_seed(743)
    q, k, v = _dense_4d_inputs(batch, heads, total, prefix + total)
    positions = torch.arange(prefix, prefix + total, device=DEVICE)
    expected = _attention_reference(q, k, v, 0.11, positions)
    restored = torch.empty_like(expected, dtype=q.dtype)
    blocks = torch.tensor_split(torch.arange(total, device=DEVICE), 16)
    visited = torch.zeros(total, dtype=torch.int64, device=DEVICE)
    for rank in range(8):
        first, second = blocks[rank], blocks[15-rank]
        ids = torch.cat([first, second])
        # Physical padding also lets ranks with no tokens exercise zero output.
        local = torch.full((batch, heads, len(ids)+3, 192), torch.nan, device=DEVICE, dtype=q.dtype)
        local[:, :, :len(ids)] = q[:, :, ids]
        lens = [[len(first), len(second)]] * batch
        ends = [[int(chunk[-1])+prefix+1 if len(chunk) else 0 for chunk in (first, second)]] * batch
        out = dense_prefill_attention_fwd_segmented(
            local, k, v, q_segment_lens=lens, kv_prefix_lens=ends, sm_scale=0.11)
        assert out.is_contiguous() and out.dtype == q.dtype
        assert not bool(out[:, :, len(ids):].any())
        if len(ids):
            torch.testing.assert_close(out[:, :, :len(ids)].float(), expected[:, :, ids], atol=0.02, rtol=0.02)
            restored[:, :, ids] = out[:, :, :len(ids)]
            visited[ids] += 1
    assert bool((visited == 1).all())
    torch.testing.assert_close(restored.float(), expected, atol=0.02, rtol=0.02)


@GPU_REQUIRED
@pytest.mark.parametrize("layout", ["ordinary", "padded_features", "nonunit_features", "shared_kv"])
def test_segmented_ragged_batch_padding_empty_segments_strides_and_readonly_kv(layout):
    torch.manual_seed(744)
    q, k, v = _dense_4d_inputs(4, 3, 7, 23)
    lens = [[2, 3], [0, 4], [3, 0], [0, 0]]
    ends = [[5, 19], [0, 11], [9, 0], [0, 0]]
    if layout in ("padded_features", "nonunit_features"):
        step = 2 if layout == "nonunit_features" else 1
        def strided(t):
            storage = torch.empty(*t.shape[:-1], t.shape[-1]*2+8, device=t.device, dtype=t.dtype)
            view = storage[..., :t.shape[-1]*step:step]
            view.copy_(t)
            return view
        q, k, v = map(strided, (q, k, v))
    if layout == "shared_kv":
        k, v = k[:1].expand(4, -1, -1, -1), v[:1].expand(4, -1, -1, -1)
    saved = [t.clone() for t in (q, k, v)]
    out = dense_prefill_attention_fwd_segmented(
        q, k, v, q_segment_lens=torch.tensor(lens), kv_prefix_lens=ends, sm_scale=0.11)
    for b, (lengths, boundaries) in enumerate(zip(lens, ends)):
        pos = torch.tensor([p for length, end in zip(lengths, boundaries)
                            for p in range(end-length, end)], device=DEVICE)
        if len(pos):
            expected = _attention_reference(q[b:b+1, :, :len(pos)], k[b:b+1], v[b:b+1], 0.11, pos)
            torch.testing.assert_close(out[b:b+1, :, :len(pos)].float(), expected, atol=0.02, rtol=0.02)
        assert not bool(out[b, :, len(pos):].any())
    for original, snapshot in zip((q, k, v), saved):
        torch.testing.assert_close(original, snapshot, atol=0, rtol=0)
    # Altering one request's query must not change another request's output.
    q2 = q.clone()
    q2[1].neg_()
    out2 = dense_prefill_attention_fwd_segmented(
        q2, k, v, q_segment_lens=lens, kv_prefix_lens=ends, sm_scale=0.11)
    torch.testing.assert_close(out2[[0, 2, 3]], out[[0, 2, 3]], atol=0, rtol=0)
    assert not torch.equal(out2[1], out[1])


@GPU_REQUIRED
def test_segmented_wrong_whole_query_tail_assumption_is_detected():
    q = torch.zeros(1, 2, 4, 192, device=DEVICE, dtype=torch.bfloat16)
    k = torch.zeros(1, 2, 32, 192, device=DEVICE, dtype=torch.bfloat16)
    v = torch.arange(32, device=DEVICE, dtype=torch.bfloat16)[None, None, :, None].expand(1, 2, 32, 128)
    correct = dense_prefill_attention_fwd_segmented(
        q, k, v, q_segment_lens=[[2, 2]], kv_prefix_lens=[[2, 32]], sm_scale=0.11)
    expected = _attention_reference(q, k, v, 0.11, torch.tensor([0, 1, 30, 31], device=DEVICE))
    wrong = dense_prefill_attention_fwd_4d(q, k, v, sm_scale=0.11)
    torch.testing.assert_close(correct.float(), expected, atol=0.02, rtol=0.02)
    assert (wrong[:, :, :2].float() - expected[:, :, :2]).abs().min().item() > 10


@GPU_REQUIRED
@pytest.mark.parametrize("lens,ends,error", [
    ([[4, 4], [2, 3]], [[5, 19], [5, 19]], ValueError),
    ([[2, 3], [2, 3]], [[1, 19], [5, 19]], ValueError),
    ([[2, 3], [2, 3]], [[5, 24], [5, 19]], ValueError),
    ([[2, 3], [2, 3]], [[5, 19], [9, 10]], ValueError),
    ([[2, 3]], [[5, 19]], ValueError),
    ([[2, 3], [True, 3]], [[5, 19], [5, 19]], TypeError),
])
def test_segmented_invalid_metadata_is_rejected_before_any_segment_launch(segmented_adapter, monkeypatch, lens, ends, error):
    q, k, v = _dense_4d_inputs()
    def unexpected(*args, **kwargs):
        pytest.fail("invalid batch must be rejected before the first launch")
    monkeypatch.setattr(segmented_adapter, "dense_prefill_attention_fwd_4d", unexpected)
    with pytest.raises(error):
        dense_prefill_attention_fwd_segmented(q, k, v, q_segment_lens=lens, kv_prefix_lens=ends, sm_scale=0.11)


@GPU_REQUIRED
@pytest.mark.parametrize("scale,error", [(True, TypeError), (None, TypeError), (0, ValueError),
                                         (-1, ValueError), (float("nan"), ValueError), (float("inf"), ValueError)])
def test_segmented_invalid_scale(scale, error):
    with pytest.raises(error):
        dense_prefill_attention_fwd_segmented(*_dense_4d_inputs(), q_segment_lens=[[2, 3]]*2,
                                               kv_prefix_lens=[[5, 19]]*2, sm_scale=scale)


@GPU_REQUIRED
def test_segmented_device_metadata_and_invalid_tensors_are_rejected():
    q, k, v = _dense_4d_inputs()
    kwargs = dict(q_segment_lens=[[2, 3]]*2, kv_prefix_lens=[[5, 19]]*2, sm_scale=0.11)
    with pytest.raises(ValueError, match="host metadata"):
        dense_prefill_attention_fwd_segmented(q, k, v, **dict(kwargs, q_segment_lens=torch.tensor([[2, 3]]*2, device=DEVICE)))
    for bad, error in [((q.float(), k, v), TypeError), ((q, k.cpu(), v), ValueError),
                       ((q, k[:1], v), ValueError), ((q[..., :128], k, v), ValueError),
                       ((q[:, :, :0], k, v), ValueError), ((None, k, v), TypeError)]:
        with pytest.raises(error):
            dense_prefill_attention_fwd_segmented(*bad, **kwargs)
    q.requires_grad_()
    with pytest.raises(RuntimeError, match="autograd"):
        dense_prefill_attention_fwd_segmented(q, k, v, **kwargs)
    with torch.no_grad():
        assert torch.isfinite(dense_prefill_attention_fwd_segmented(q, k, v, **kwargs)).all()


@GPU_REQUIRED
def test_segmented_actual_batch_pointer_offset_beyond_int32():
    # Two tiny requests separated by >2^31 elements in the underlying K/V
    # storages. This checks a real 64-bit batch pointer, not a mocked limit.
    q, _, _ = _dense_4d_inputs(2, 1, 3, 5)
    offset = 2**31 + 256
    caches = []
    for dim in (192, 128):
        storage = torch.empty(offset + 5*dim, dtype=torch.bfloat16, device=DEVICE)
        cache = storage.as_strided((2, 1, 5, dim), (offset, 5*dim, dim, 1))
        cache.normal_()
        caches.append(cache)
    k, v = caches
    out = dense_prefill_attention_fwd_segmented(q, k, v, q_segment_lens=[[1, 2]]*2,
                                               kv_prefix_lens=[[1, 5]]*2, sm_scale=0.11)
    for b in range(2):
        expected = _attention_reference(
            q[b:b+1], k[b:b+1], v[b:b+1], 0.11, torch.tensor([0, 3, 4], device=DEVICE)
        )
        torch.testing.assert_close(out[b:b+1].float(), expected, atol=0.02, rtol=0.02)


@GPU_REQUIRED
def test_long_ragged_batch_crosses_query_and_kv_tiles(monkeypatch) -> None:
    """Validate multi-tile Q/KV reduction and masked tail tiles on the legacy grid."""
    q_lens, p_lens = [1, 65], [65, 129]
    q, k, v, k_cache, v_cache, qo, kv, indices = _ragged_inputs(
        q_lens=q_lens, p_lens=p_lens
    )
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], device=DEVICE)
    lse = torch.empty(q.shape[:2], device=DEVICE)
    # Keep this test on the rectangular grid: the second sequence crosses
    # HCU's regular 64-token Q/KV tile boundary and has a one-token tail.
    monkeypatch.setenv("BOLT_OPS_TRITON_COMPACT_EXTEND_ATTENTION", "0")
    extend_attention_fwd(
        q, k, v, out, k_cache, v_cache, qo, kv, indices, None, True, None,
        max(q_lens), 1.0, 1.0, lse_extend=lse,
    )
    expected, expected_lse = _reference(
        q, k, v, k_cache, v_cache, qo, kv, indices, causal=True, k_scale=1.0,
        v_scale=1.0,
    )
    torch.testing.assert_close(out, expected, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL)
    torch.testing.assert_close(lse, expected_lse, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL)


@GPU_REQUIRED
def test_short_query_kernel_matches_long_query_kernel(monkeypatch) -> None:
    """Compare both optimized layouts on the same short-query input."""
    from boltops.fa.extend_attention.triton import extend_attention as backend

    if not backend._is_hcu:
        pytest.skip("optimized Extend Attention kernels are HCU-only")

    q, k, v, k_cache, v_cache, qo, kv, indices = _inputs(dtype=torch.bfloat16)
    short_query_out = torch.empty(
        q.shape[0], q.shape[1], v.shape[-1], device=DEVICE, dtype=q.dtype
    )
    long_query_out = torch.empty_like(short_query_out)
    max_len_extend = 3
    extend_attention_fwd(
        q, k, v, short_query_out, k_cache, v_cache, qo, kv, indices, None, True,
        None, max_len_extend, 1.0, 1.0,
    )

    original_selector = backend._select_optimized_kernel
    monkeypatch.setattr(
        backend,
        "_select_optimized_kernel",
        lambda *args, **kwargs: backend.LONG_QUERY,
    )
    extend_attention_fwd(
        q, k, v, long_query_out, k_cache, v_cache, qo, kv, indices, None, True,
        None, max_len_extend, 1.0, 1.0,
    )
    monkeypatch.setattr(backend, "_select_optimized_kernel", original_selector)

    torch.testing.assert_close(
        short_query_out, long_query_out, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL
    )


@GPU_REQUIRED
@pytest.mark.parametrize("query_len", [4, 32])
def test_optimized_kernels_use_shipped_gfx938_fp8_config(query_len: int) -> None:
    """Compile and validate both optimized kernels with shipped configs."""
    from boltops.fa.extend_attention.triton import extend_attention as backend

    target = triton.runtime.driver.active.get_current_target()
    if not backend._is_hcu or not target.arch.startswith("gfx938"):
        pytest.skip("shipped tuning data targets gfx938")

    torch.manual_seed(321)
    batch, prefix_len = 1, 32
    hq, hkv, dqk, dv = 8, 1, 192, 128
    q = torch.randn(query_len, hq, dqk, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(query_len, hkv, dqk, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(query_len, hkv, dv, device=DEVICE, dtype=torch.bfloat16)
    k_cache = torch.randn(
        prefix_len, hkv, dqk, device=DEVICE, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    v_cache = torch.randn(
        prefix_len, hkv, dv, device=DEVICE, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    qo = torch.tensor([0, query_len], device=DEVICE, dtype=torch.int64)
    kv = torch.tensor([0, prefix_len], device=DEVICE, dtype=torch.int32)
    indices = torch.arange(prefix_len, device=DEVICE, dtype=torch.int64)
    sinks = torch.randn(hq, device=DEVICE, dtype=torch.bfloat16)
    window_offsets = torch.zeros(batch, device=DEVICE, dtype=torch.int32)
    out = torch.empty(query_len, hq, dv, device=DEVICE, dtype=torch.bfloat16)

    extend_attention_fwd(
        q, k, v, out, k_cache, v_cache, qo, kv, indices, None, True, None,
        query_len, 1.0, 1.0, sliding_window_size=128, sinks=sinks,
        window_kv_offsets=window_offsets,
    )
    expected, _ = _reference(
        q, k, v, k_cache, v_cache, qo, kv, indices, causal=True, k_scale=1.0,
        v_scale=1.0, window=128, sinks=sinks,
    )
    torch.testing.assert_close(out, expected.to(out.dtype), atol=5e-2, rtol=5e-2)


@GPU_REQUIRED
def test_compact_grid_with_extend_seq_lens_cpu_matches_legacy_grid(monkeypatch) -> None:
    """Exercise the HIP compact-grid branch through the public CPU-length argument."""
    from boltops.fa.extend_attention.triton import extend_attention as backend

    if not backend._is_hip:
        pytest.skip("compact extend-attention grid is HIP-only")

    q_lens, p_lens = [1, 65], [65, 129]
    q, k, v, k_cache, v_cache, qo, kv, indices = _ragged_inputs(
        q_lens=q_lens, p_lens=p_lens
    )
    _, _, _, block_m, _, _ = backend._get_block_sizes_for_extend_attention(
        q.shape[-1], v.shape[-1]
    )
    compact_tiles = backend._compact_extend_q_tiles_per_head(
        batch_size=len(q_lens),
        max_len_extend=max(q_lens),
        total_extend_tokens=q.shape[0],
        block_m=block_m,
        extend_seq_lens_cpu=torch.tensor(q_lens, dtype=torch.int32),
    )
    assert compact_tiles is not None

    legacy_out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], device=DEVICE)
    legacy_lse = torch.empty(q.shape[:2], device=DEVICE)
    monkeypatch.setenv("BOLT_OPS_TRITON_COMPACT_EXTEND_ATTENTION", "0")
    extend_attention_fwd(
        q, k, v, legacy_out, k_cache, v_cache, qo, kv, indices, None, True,
        None, max(q_lens), 1.0, 1.0, lse_extend=legacy_lse,
    )

    compact_out = torch.empty_like(legacy_out)
    compact_lse = torch.empty_like(legacy_lse)
    monkeypatch.setenv("BOLT_OPS_TRITON_COMPACT_EXTEND_ATTENTION", "1")
    extend_attention_fwd(
        q, k, v, compact_out, k_cache, v_cache, qo, kv, indices, None, True,
        None, max(q_lens), 1.0, 1.0, lse_extend=compact_lse,
        extend_seq_lens_cpu=torch.tensor(q_lens, dtype=torch.int32),
    )

    torch.testing.assert_close(
        compact_out, legacy_out, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL
    )
    torch.testing.assert_close(
        compact_lse, legacy_lse, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL
    )


@GPU_REQUIRED
def test_score_mod_and_aux_tensors_match_reference() -> None:
    """Exercise SGLang's Triton score_mod ABI with one relative-bias tensor."""
    q, k, v, k_cache, v_cache, qo, kv, indices = _inputs()
    aux0 = torch.randn(q.shape[0], q.shape[1], 7, device=DEVICE)
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], device=DEVICE)
    lse = torch.empty(q.shape[:2], device=DEVICE)
    extend_attention_fwd(
        q, k, v, out, k_cache, v_cache, qo, kv, indices, None, True, None, 3,
        1.0, 1.0, lse_extend=lse, score_mod=_relative_bias_score_mod,
        aux_tensors=[aux0],
    )
    expected, expected_lse = _reference(
        q, k, v, k_cache, v_cache, qo, kv, indices, causal=True, k_scale=1.0,
        v_scale=1.0, relative_bias=aux0,
    )
    torch.testing.assert_close(out, expected, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL)
    torch.testing.assert_close(lse, expected_lse, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL)


@GPU_REQUIRED
def test_windowed_kv_with_nonzero_window_offsets_matches_reference() -> None:
    """Validate the real windowed-KV custom-mask layout used by SGLang.

    ``kv_indptr``/``kv_indices`` describe only the visible window.  The custom
    mask retains the full-sequence columns, and ``window_kv_offsets`` tells the
    kernel where the visible window begins in each mask row.
    """
    q, k, v, k_cache, v_cache, qo, kv, indices = _inputs()
    visible_prefix_lens, q_lens = [3, 2], [2, 3]
    offsets = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
    masks = []
    for prefix, query, offset in zip(
        visible_prefix_lens, q_lens, offsets.tolist()
    ):
        mask = torch.zeros(
            query, offset + prefix + query, dtype=torch.bool, device=DEVICE
        )
        mask[:, offset : offset + prefix] = True
        mask[:, offset + prefix :] = torch.ones(
            query, query, dtype=torch.bool, device=DEVICE
        ).tril()
        masks.append(mask.flatten())
    custom_mask = torch.cat(masks)
    mask_indptr = torch.tensor(
        [0, masks[0].numel(), custom_mask.numel()],
        dtype=torch.int32,
        device=DEVICE,
    )
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], device=DEVICE)
    lse = torch.empty(q.shape[:2], device=DEVICE)
    extend_attention_fwd(
        q, k, v, out, k_cache, v_cache, qo, kv, indices, custom_mask, False,
        mask_indptr, 3, 1.0, 1.0, skip_prefix_custom_mask=False,
        sliding_window_size=2, window_kv_offsets=offsets, lse_extend=lse,
    )
    expected, expected_lse = _reference(
        q, k, v, k_cache, v_cache, qo, kv, indices, causal=False, k_scale=1.0,
        v_scale=1.0, custom_mask=custom_mask, mask_indptr=mask_indptr,
        skip_prefix_custom_mask=False, offsets=offsets, window=2,
    )
    torch.testing.assert_close(out, expected, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL)
    torch.testing.assert_close(lse, expected_lse, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL)


@pytest.mark.parametrize(
    "qk_dim,v_dim,page_size",
    [
        (32, 32, 1),
        (64, 64, 1),
        (128, 128, 16),
        (192, 128, 1),
        (288, 128, 1),
        (576, 512, 1),
    ],
)
@GPU_REQUIRED
def test_extend_attention_head_dim_matrix(
    qk_dim: int, v_dim: int, page_size: int
) -> None:
    """Cover generic and SGLang-specialized head-dimension branches.

    Dims 192, 288, and 576 exercise SGLang's dedicated DPE/block-size paths;
    page size 16 covers the 4-D paged-KV layout beyond the small page-size
    smoke test.
    """
    q, k, v, k_cache, v_cache, qo, kv, indices = _inputs(
        page_size=page_size, d=qk_dim, dv=v_dim
    )
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], device=DEVICE)
    lse = torch.empty(q.shape[:2], device=DEVICE)
    extend_attention_fwd(
        q, k, v, out, k_cache, v_cache, qo, kv, indices, None, True, None, 3,
        1.0, 1.0, lse_extend=lse, page_size=page_size,
    )
    expected, expected_lse = _reference(
        q, k, v, k_cache, v_cache, qo, kv, indices, causal=True, k_scale=1.0,
        v_scale=1.0, page_size=page_size,
    )
    # Different Triton backends can use different MMA reduction orders. Keep
    # the existing strict small-shape test above; this branch-coverage matrix
    # uses the HCU tolerance appropriate for larger production head dimensions.
    tolerance = 2e-2 if torch.version.hip is not None else 2e-6
    torch.testing.assert_close(out, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(lse, expected_lse, atol=tolerance, rtol=tolerance)


@GPU_REQUIRED
def test_hcu_lq576_lv512_multitile_regular_and_unified_match_reference(
    monkeypatch,
) -> None:
    """Validate BoltOps' HCU-only 32x32 tile beyond both tile boundaries."""
    from boltops.fa.extend_attention.triton import extend_attention as backend

    if not backend._is_hcu:
        pytest.skip("HCU-specific 576/512 tile validation")

    assert backend._get_block_sizes_for_extend_attention(576, 512)[3:] == (
        32,
        32,
        4,
    )

    q_lens, prefix_lens = [33, 65], [64, 97]
    regular_inputs = _ragged_inputs(
        q_lens=q_lens,
        p_lens=prefix_lens,
        d=576,
        dv=512,
        dtype=torch.bfloat16,
    )
    (
        q,
        k,
        v,
        k_cache,
        v_cache,
        qo,
        kv,
        prefix_indices,
        unified_k_cache,
        unified_v_cache,
        unified_kv_indptr,
        unified_kv_indices,
        returned_prefix_lens,
        max_len_extend,
    ) = _unified_inputs(page_size=1, regular_inputs=regular_inputs)

    # Isolate the HCU tile adaptation from the separate compact-grid path.
    monkeypatch.setenv("BOLT_OPS_TRITON_COMPACT_EXTEND_ATTENTION", "0")
    regular_out = torch.empty(
        q.shape[0], q.shape[1], v.shape[-1], dtype=q.dtype, device=DEVICE
    )
    regular_lse = torch.empty(q.shape[:2], dtype=torch.float32, device=DEVICE)
    extend_attention_fwd(
        q,
        k,
        v,
        regular_out,
        k_cache,
        v_cache,
        qo,
        kv,
        prefix_indices,
        None,
        True,
        None,
        max_len_extend,
        1.0,
        1.0,
        lse_extend=regular_lse,
    )

    unified_out = torch.empty_like(regular_out)
    extend_attention_fwd_unified(
        q,
        unified_out,
        unified_k_cache,
        unified_v_cache,
        1.0,
        1.0,
        qo,
        unified_kv_indptr,
        unified_kv_indices,
        returned_prefix_lens,
        max_len_extend,
        is_causal=True,
    )

    expected, expected_lse = _reference(
        q,
        k,
        v,
        k_cache,
        v_cache,
        qo,
        kv,
        prefix_indices,
        causal=True,
        k_scale=1.0,
        v_scale=1.0,
    )
    torch.testing.assert_close(
        regular_out, expected.to(regular_out.dtype), atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        regular_lse, expected_lse, atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        unified_out, expected.to(unified_out.dtype), atol=3e-2, rtol=3e-2
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@GPU_REQUIRED
def test_extend_attention_low_precision(dtype: torch.dtype) -> None:
    """Exercise the regular SGLang API in the inference dtypes used on HIP."""
    q, k, v, k_cache, v_cache, qo, kv, indices = _inputs(
        page_size=16, d=128, dv=128, dtype=dtype
    )
    out = torch.empty(
        q.shape[0], q.shape[1], v.shape[-1], device=DEVICE, dtype=dtype
    )
    lse = torch.empty(q.shape[:2], device=DEVICE)
    extend_attention_fwd(
        q, k, v, out, k_cache, v_cache, qo, kv, indices, None, True, None, 3,
        1.0, 1.0, lse_extend=lse, page_size=16,
    )
    expected, expected_lse = _reference(
        q, k, v, k_cache, v_cache, qo, kv, indices, causal=True, k_scale=1.0,
        v_scale=1.0, page_size=16,
    )
    torch.testing.assert_close(out, expected.to(out.dtype), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, expected_lse, atol=3e-2, rtol=3e-2)


@GPU_REQUIRED
def test_custom_mask_and_dcp_partial_calls_match_reference() -> None:
    q, k, v, k_cache, v_cache, qo, kv, indices = _inputs()
    p_lens, q_lens = [3, 2], [2, 3]
    masks = []
    for prefix, query in zip(p_lens, q_lens):
        current = torch.ones(query, prefix + query, dtype=torch.bool, device=DEVICE)
        current[:, prefix:] = torch.ones(
            query, query, dtype=torch.bool, device=DEVICE
        ).tril()
        masks.append(current.flatten())
    custom = torch.cat(masks)
    mask_indptr = torch.tensor([0, masks[0].numel(), custom.numel()], dtype=torch.int32, device=DEVICE)
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], device=DEVICE)
    lse = torch.empty(q.shape[:2], device=DEVICE)
    extend_attention_fwd(
        q, k, v, out, k_cache, v_cache, qo, kv, indices, custom, False, mask_indptr, 3,
        1.0, 1.0, skip_prefix_custom_mask=False,
        lse_extend=lse,
    )
    expected, expected_lse = _reference(
        q, k, v, k_cache, v_cache, qo, kv, indices, causal=False, k_scale=1.0,
        v_scale=1.0, custom_mask=custom, mask_indptr=mask_indptr,
        skip_prefix_custom_mask=False,
    )
    torch.testing.assert_close(out, expected, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL)
    torch.testing.assert_close(lse, expected_lse, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL)

    full = torch.empty_like(out)
    full_lse = torch.empty_like(lse)
    prefix_only = torch.empty_like(out)
    prefix_lse = torch.empty_like(lse)
    extend_only = torch.empty_like(out)
    extend_lse = torch.empty_like(lse)
    extend_attention_fwd(q, k, v, full, k_cache, v_cache, qo, kv, indices, None, True, None, 3, 1.0, 1.0, lse_extend=full_lse)
    extend_attention_fwd(q, k, v, prefix_only, k_cache, v_cache, qo, kv, indices, None, True, None, 3, 1.0, 1.0, lse_extend=prefix_lse, skip_extend=True)
    extend_attention_fwd(q, k, v, extend_only, k_cache, v_cache, qo, kv, indices, None, True, None, 3, 1.0, 1.0, lse_extend=extend_lse, skip_prefix=True)
    merged = (
        prefix_only * torch.exp(prefix_lse - full_lse).unsqueeze(-1)
        + extend_only * torch.exp(extend_lse - full_lse).unsqueeze(-1)
    )
    torch.testing.assert_close(merged, full, atol=HIP_ATOL_RTOL, rtol=HIP_ATOL_RTOL)


@pytest.mark.bench
@GPU_REQUIRED
@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case.name) for case in BENCHMARK_CASES],
)
@torch.inference_mode()
def test_extend_attention_benchmark(case) -> None:
    """Measure the active upstream production shapes on the optimized path."""
    from boltops.fa.extend_attention.triton import extend_attention as backend

    if not backend._is_hcu:
        pytest.skip("optimized Extend Attention benchmark requires HCU")

    data = make_benchmark_inputs(case)

    def run() -> torch.Tensor:
        output = torch.empty_like(data["output"])
        extend_attention_fwd(
            data["q"],
            data["k"],
            data["v"],
            output,
            data["k_buffer"],
            data["v_buffer"],
            data["qo_indptr"],
            data["kv_indptr"],
            data["kv_indices"],
            None,
            True,
            None,
            case.extend_length,
            1.0,
            1.0,
            sm_scale=1.0,
            sliding_window_size=case.sliding_window_size,
            sinks=data["sinks"],
            window_kv_offsets=data["window_kv_offsets"],
        )
        return output

    assert torch.isfinite(run()).all()
    elapsed_ms = float(triton.testing.do_bench_cudagraph(run))
    kernel = "short-query" if case.extend_length < 32 else "long-query"
    print(
        f"Extend Attention {kernel}: B={case.batch_size}, Hq={case.query_heads}, "
        f"Hkv={case.kv_heads}, prefix={case.prefix_length}, "
        f"extend={case.extend_length}, window={case.sliding_window_size}, "
        f"sinks={case.with_sinks}: "
        f"{elapsed_ms:.4f} ms"
    )


# Original-shape correctness uses the same independent reference, without timing.
@GPU_REQUIRED
@pytest.mark.parametrize("prepared_case", case_parameters({"dense", "segmented"}, full=True),
                         indirect=True, scope="module", ids=lambda c: c["case_id"])
def test_original_shape_prefill(prepared_case):
    case, _ = prepared_case  # Input preparation executes and checks the reference.
    print(f"[correctness] {case['case_id']}: passed")


# Original-shape benchmarks, using the shared time_ms fixture.

@pytest.mark.bench
@pytest.mark.parametrize("prepared_case", case_parameters({"dense", "segmented"}),
                         indirect=True, scope="module", ids=lambda c: c["case_id"])
def test_bench_fa_prefill(prepared_case, time_ms, bench_warmup, bench_repeat):
    case, run = prepared_case
    ms = time_ms(run, bench_warmup, bench_repeat,
                 flops=useful_flops(case), dtype=torch.bfloat16)
    assert math.isfinite(ms) and ms > 0
    print(f"[bench] {case['case_id']}: {ms:.6f} ms")
