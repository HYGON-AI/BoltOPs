# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Self-contained coverage for a value head that differs from the qk head.

Kimi ``mha_slice*`` dense attention uses a 192-dim qk head and a 128-dim value
head. Every shape below is written inline (no external shape manifest), so the
feature is verified without depending on generated shape data. The reference is
computed on CPU copies because GPU-side rocBLAS has no gfx946 TensileLibrary on
the PMD build; kernel inputs stay on device.

The design mirrors vLLM's triton DiffKV backend
(``triton_unified_attention_diffkv``): K is loaded at the qk width, V at the
value width, and the accumulator + output are sized to the value head.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from boltops.fa.unified_attention import unified_attention


def _reference_paged_attention_diffkv(q, k_cache, v_cache, q_lens, kv_lens, table, scale, causal):
    """CPU reference for a value head that differs from the qk head.

    ``k_cache`` carries the qk head on its last dim; ``v_cache`` carries the
    value head. Scores run over the qk dim (``einsum("qhd,khd->hqk")``) while the
    output runs over the value dim (``einsum("hqk,khd->qhd")``), so the returned
    tensor's last dimension equals the value head.
    """
    qk_head = q.shape[-1]
    v_head = v_cache.shape[-1]
    assert k_cache.shape[-1] == qk_head, "k cache must carry the qk head"
    out = torch.empty(q.shape[0], q.shape[1], v_head, dtype=q.dtype)
    cursor = 0
    block_size = k_cache.shape[1]
    for seq, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        blocks = table[seq, : math.ceil(kv_len / block_size)].long()
        k = k_cache[blocks].reshape(-1, k_cache.shape[2], k_cache.shape[3])[:kv_len]
        v = v_cache[blocks].reshape(-1, v_cache.shape[2], v_cache.shape[3])[:kv_len]
        q_seq = q[cursor : cursor + q_len]
        group = q_seq.shape[1] // k.shape[1]
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q_seq.float(), k.float()) * scale
        seq_causal = bool(causal[seq]) if isinstance(causal, (list, tuple, torch.Tensor)) else causal
        if seq_causal:
            positions = torch.arange(q_len, device=q.device) + kv_len - q_len
            scores.masked_fill_(
                torch.arange(kv_len, device=q.device)[None, None, :] > positions[None, :, None],
                float("-inf"),
            )
        probs = torch.softmax(scores, dim=-1)
        out[cursor : cursor + q_len] = torch.einsum("hqk,khd->qhd", probs, v.float()).to(q.dtype)
        cursor += q_len
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an HCU/CUDA Triton device")
@pytest.mark.parametrize("q_len, kv_len", [(12, 24), (20, 48)])
def test_qk_value_head_diffkv_matches_reference(q_len: int, kv_len: int) -> None:
    """qk head 192 / value head 128 causal prefill (MHA, q_per_kv = 1)."""
    torch.manual_seed(11)
    device, page, hq, hkv, qk_dim, v_dim = "cuda", 16, 4, 4, 192, 128
    pages = (kv_len + page - 1) // page
    q_cpu = torch.randn(q_len, hq, qk_dim, dtype=torch.bfloat16)
    k_cpu = torch.randn(pages, page, hkv, qk_dim, dtype=torch.bfloat16)
    v_cpu = torch.randn(pages, page, hkv, v_dim, dtype=torch.bfloat16)
    q, k, v = q_cpu.to(device), k_cpu.to(device), v_cpu.to(device)
    out = torch.empty(q_len, hq, v_dim, device=device, dtype=torch.bfloat16)
    table_cpu = torch.arange(pages, dtype=torch.int32).reshape(1, -1)
    table = table_cpu.to(device)
    unified_attention(
        q, k, v, out,
        torch.tensor([0, q_len], device=device, dtype=torch.int32), q_len,
        torch.tensor([kv_len], device=device, dtype=torch.int64), kv_len,
        qk_dim ** -0.5, True, (-1, -1), table, 0.0, None, None, None,
    )
    reference = _reference_paged_attention_diffkv(
        q_cpu, k_cpu, v_cpu, [q_len], [kv_len], table_cpu, qk_dim ** -0.5, True
    )
    assert out.shape[-1] == v_dim
    torch.testing.assert_close(out.cpu(), reference, atol=3e-2, rtol=3e-2)


def _load_extractor() -> object:
    path = Path(__file__).resolve().parents[1].parent / "scripts" / "extract_gfx946_shapes.py"
    spec = importlib.util.spec_from_file_location("extract_gfx946_shapes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The 23 Kimi mha_slice (batch, query_length, kv_length) combinations: q/k head
# 192, value head 128, bf16, H64:Hkv64 MHA, causal prefill.
_KIMI_QKV = [
    (4, 4608, 131072), (4, 6656, 6656), (4, 6656, 13312), (4, 6656, 19968),
    (4, 6656, 26624), (4, 6656, 33280), (4, 6656, 39936), (4, 6656, 46592),
    (4, 6656, 53248), (4, 6656, 59904), (4, 6656, 66560), (4, 6656, 73216),
    (4, 6656, 79872), (4, 6656, 86528), (4, 6656, 93184), (4, 6656, 99840),
    (4, 6656, 106496), (4, 6656, 113152), (4, 6656, 119808), (4, 6656, 126464),
    (32, 3072, 131072), (32, 6656, 53248), (32, 6656, 106496),
]


def test_kimi_qk192_v128_shapes_classify_ready() -> None:
    """The extractor must classify all 23 Kimi qk192/v128 shapes as ``ready``."""
    extractor = _load_extractor()
    assert len(_KIMI_QKV) == 23
    for batch, query_length, kv_length in _KIMI_QKV:
        case = {
            "operator": "unified_attention",
            "input_shapes": [
                [batch, 64, query_length, 192],
                [batch, 64, kv_length, 192],
                [batch, 64, kv_length, 128],
            ],
            "input_dtypes": ["BF16", "BF16", "BF16"],
            "sources": [{"variant": "mha_slice0"}],
        }
        extractor.normalize_case(case)
        assert case["validation"] == "ready", (
            batch, query_length, kv_length, case["validation"])
        assert case["parameters"]["head_size"] == 192
        assert case["parameters"]["value_head_size"] == 128
