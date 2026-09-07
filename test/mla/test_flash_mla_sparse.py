"""Correctness and API tests for FlashMLA sparse forward."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import pytest
import torch

pytest.importorskip("triton")

from boltops.mla import flash_mla_sparse_fwd


GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    d_v: int,
    topk_length: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_q, h_q, _ = q.shape
    h_kv = kv.shape[1]
    group_size = h_q // h_kv
    out = torch.zeros(s_q, h_q, d_v, device=q.device, dtype=torch.float32)
    max_logits = torch.full(
        (s_q, h_q), -torch.inf, device=q.device, dtype=torch.float32
    )
    lse = torch.full_like(max_logits, -torch.inf)
    for row in range(s_q):
        length = indices.shape[-1] if topk_length is None else int(topk_length[row])
        for head in range(h_q):
            kv_head = head // group_size
            selected = indices[row, kv_head, : max(0, length)].long()
            selected = selected[(selected >= 0) & (selected < kv.shape[0])]
            if selected.numel() == 0:
                continue
            keys = kv[selected, kv_head].float()
            scores = (q[row, head].float() @ keys.T) * scale
            max_logits[row, head] = scores.max()
            lse[row, head] = torch.logsumexp(scores, dim=0)
            out[row, head] = torch.softmax(scores, dim=0) @ keys[:, :d_v]
    return out, max_logits, lse


@GPU_REQUIRED
@pytest.mark.parametrize("use_lengths", [False, True])
def test_flash_mla_sparse_fwd_matches_reference(use_lengths: bool):
    torch.manual_seed(0)
    s_q, s_kv, h_q, h_kv, d_qk, d_v, topk = 4, 80, 16, 1, 576, 512, 33
    q = torch.randn(s_q, h_q, d_qk, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_kv, h_kv, d_qk, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(
        0, s_kv, (s_q, h_kv, topk), device="cuda", dtype=torch.int32
    )
    indices[0, 0, 0] = -1
    indices[1, 0, 1] = s_kv + 1
    lengths = (
        torch.tensor([0, 7, topk, 19], device="cuda", dtype=torch.int32)
        if use_lengths
        else None
    )
    scale = 1.0 / math.sqrt(d_qk)

    out, max_logits, lse = flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        softmax_scale=scale,
        d_v=d_v,
        topk_length=lengths,
    )
    ref_out, ref_max, ref_lse = _reference(q, kv, indices, scale, d_v, lengths)

    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(max_logits, ref_max, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-2, rtol=2e-2)


def test_flash_mla_sparse_fwd_validates_shapes_before_launch():
    q = torch.empty(2, 16, 576, dtype=torch.bfloat16)
    kv = torch.empty(8, 1, 576, dtype=torch.bfloat16)
    indices = torch.empty(3, 1, 4, dtype=torch.int32)
    with pytest.raises(ValueError, match="indices must have shape"):
        flash_mla_sparse_fwd(q, kv, indices, 1.0)


def test_flash_mla_sparse_fwd_is_public():
    from boltops.mla import __all__

    assert "flash_mla_sparse_fwd" in __all__


def test_flash_mla_sparse_tuned_config_schema():
    config_root = (
        Path(__file__).parents[2]
        / "boltops"
        / "mla"
        / "triton"
        / "configs"
    )
    paths = list(config_root.glob("triton_*_*/flash_mla_sparse_*.json"))
    assert paths
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        metadata = data["metadata"]
        assert path.parent.name == metadata["version_tag"]
        assert metadata["arch"] in path.name
        assert metadata["device_slug"] in path.name
        matches = [tuple(sorted(entry["match"].items())) for entry in data["configs"]]
        assert len(matches) == len(set(matches))
        for entry in data["configs"]:
            assert entry["match"]["dtype"] == metadata["dtype"]
            assert entry["latency_ms"] > 0
            assert entry["speedup"] > 0
