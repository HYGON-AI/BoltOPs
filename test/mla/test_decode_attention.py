"""Correctness and optional performance tests for MLA decode attention."""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("triton")

from boltops.mla import (  # noqa: E402
    compute_num_kv_splits,
    decode_attention_fwd,
)

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def _reference_mla_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense gather + attention reference for one decode step (MLA layout)."""
    B, H, Lk = q.shape
    Lv = v_cache.shape[-1]
    out = torch.empty(B, H, Lv, dtype=torch.float32, device=q.device)
    lse = torch.empty(B, H, dtype=torch.float32, device=q.device)

    for b in range(B):
        seq_len = int(seq_lens[b].item())
        pages = (seq_len + page_size - 1) // page_size
        tokens = []
        for p in range(pages):
            page_idx = int(block_table[b, p].item())
            start = p * page_size
            end = min(start + page_size, seq_len)
            n = end - start
            tokens.append(k_cache[page_idx, :n, 0, :])
        k = torch.cat(tokens, dim=0)  # [S, Lk]
        v = k[:, :Lv]
        # q[b]: [H, Lk]
        logits = torch.matmul(q[b].float(), k.float().T) * sm_scale  # [H, S]
        lse_b = torch.logsumexp(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        out[b] = torch.matmul(probs, v.float())
        lse[b] = lse_b
    return out, lse


def _make_mla_inputs(
    batch: int,
    num_heads: int,
    seq_len: int,
    page_size: int,
    kv_lora_rank: int = 512,
    qk_rope_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    num_kv_splits: int = 1,
) -> dict:
    Lk = kv_lora_rank + qk_rope_dim
    Lv = kv_lora_rank
    num_pages_needed = (seq_len + page_size - 1) // page_size
    # Allocate contiguous pool large enough for the batch.
    num_pages = batch * num_pages_needed
    torch.manual_seed(0)
    k_cache = torch.randn(
        num_pages, page_size, 1, Lk, device=DEVICE, dtype=dtype
    )
    v_cache = k_cache[..., :Lv]
    q = torch.randn(batch, num_heads, Lk, device=DEVICE, dtype=dtype)

    block_table = torch.zeros(batch, num_pages_needed, device=DEVICE, dtype=torch.int32)
    for b in range(batch):
        for p in range(num_pages_needed):
            block_table[b, p] = b * num_pages_needed + p
    seq_lens = torch.full((batch,), seq_len, device=DEVICE, dtype=torch.int32)

    o = torch.empty(batch, num_heads, Lv, device=DEVICE, dtype=dtype)
    lse = torch.empty(batch, num_heads, device=DEVICE, dtype=dtype)
    attn_logits = torch.empty(
        batch, num_heads, num_kv_splits, Lv + 1, device=DEVICE, dtype=torch.float32
    )
    sm_scale = 1.0 / math.sqrt(Lk)
    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_table": block_table,
        "seq_lens": seq_lens,
        "o": o,
        "lse": lse,
        "attn_logits": attn_logits,
        "num_kv_splits": num_kv_splits,
        "sm_scale": sm_scale,
        "page_size": page_size,
        "Lk": Lk,
        "Lv": Lv,
    }


@GPU_REQUIRED
@pytest.mark.parametrize(
    ("batch", "num_heads", "seq_len", "page_size", "num_kv_splits"),
    [
        pytest.param(1, 16, 64, 16, 1, id="decode-short"),
        pytest.param(2, 16, 128, 16, 1, id="decode-batch"),
        pytest.param(1, 16, 256, 64, 2, id="split-kv"),
    ],
)
def test_mla_decode_attention_correctness(
    batch: int,
    num_heads: int,
    seq_len: int,
    page_size: int,
    num_kv_splits: int,
):
    """MLA absorb-decode vs dense gather reference (DeepSeek Lk=576 / Lv=512)."""
    inp = _make_mla_inputs(
        batch=batch,
        num_heads=num_heads,
        seq_len=seq_len,
        page_size=page_size,
        num_kv_splits=num_kv_splits,
    )
    decode_attention_fwd(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        inp["o"],
        inp["lse"],
        inp["block_table"],
        inp["seq_lens"],
        inp["attn_logits"],
        inp["num_kv_splits"],
        inp["sm_scale"],
        page_size=inp["page_size"],
        is_mla=True,
    )
    ref_o, ref_lse = _reference_mla_decode(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        inp["block_table"],
        inp["seq_lens"],
        inp["sm_scale"],
        inp["page_size"],
    )
    torch.testing.assert_close(
        inp["o"].float().cpu(), ref_o.cpu(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        inp["lse"].float().cpu(), ref_lse.cpu(), atol=2e-2, rtol=2e-2
    )


def test_compute_num_kv_splits_formula():
    assert compute_num_kv_splits(100, sm_count=64) == 1
    assert compute_num_kv_splits(1024, sm_count=64) == 2
    assert compute_num_kv_splits(8192, sm_count=4) == 8  # capped by sm*2


@GPU_REQUIRED
@pytest.mark.bench
@pytest.mark.parametrize(
    ("batch", "num_heads", "seq_len", "page_size"),
    [
        pytest.param(8, 16, 4096, 64, id="dsv3-like"),
    ],
)
def test_bench_mla_decode_attention(
    batch: int,
    num_heads: int,
    seq_len: int,
    page_size: int,
    time_ms,
    bench_warmup: int,
    bench_repeat: int,
):
    num_kv_splits = compute_num_kv_splits(seq_len)
    inp = _make_mla_inputs(
        batch=batch,
        num_heads=num_heads,
        seq_len=seq_len,
        page_size=page_size,
        num_kv_splits=num_kv_splits,
    )

    def _run():
        decode_attention_fwd(
            inp["q"],
            inp["k_cache"],
            inp["v_cache"],
            inp["o"],
            inp["lse"],
            inp["block_table"],
            inp["seq_lens"],
            inp["attn_logits"],
            inp["num_kv_splits"],
            inp["sm_scale"],
            page_size=inp["page_size"],
            is_mla=True,
        )

    _run()
    torch.cuda.synchronize()
    ms = time_ms(_run, bench_warmup, bench_repeat)
    print(
        f"[bench] mla decode_attention B={batch} H={num_heads} "
        f"S={seq_len} splits={num_kv_splits}: {ms:.3f} ms"
    )
