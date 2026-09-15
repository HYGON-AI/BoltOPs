"""Correctness and input-contract tests for generic QSA operators."""

from __future__ import annotations

import math
import pytest
import torch
import triton

from boltops.generic import (
    expand_qsa_block_indices,
    qsa_mqa_paged,
    qsa_select_paged_tokens,
    qsa_sparse_paged_attention,
)
from boltops.generic.triton.qsa import (
    _qsa_mqa_paged_kernel,
    TUNED_TARGET_SEGMENTS,
    _select_qsa_mqa_config,
    _select_qsa_sparse_block_m,
    _select_qsa_sparse_config,
    _select_sparse_launch_config,
)


DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def test_qsa_config_selector_covers_required_range():
    """Every supported exact row count yields the static dispatch schema."""
    for rows in (1, 8, 9, 31, 32, 256, 512, 10240):
        config = _select_qsa_sparse_config(rows, 1, 2051, 256)
        assert set(config) == {
            "block_n",
            "target_splits",
            "num_warps",
            "num_stages",
            "merge_warps",
            "merge_dim_tile",
        }


def test_qsa_decode_merge_selector_uses_measured_launch():
    """Small decode gets extra dim programs; larger decode stays full-head."""
    for rows, expected_dim_tile in ((1, 64), (8, 64), (9, 256), (512, 256)):
        config = _select_qsa_sparse_config(rows, 1, 2051, 256)
        assert config["merge_warps"] == 1
        assert config["merge_dim_tile"] == expected_dim_tile


def test_public_sparse_launch_config_matches_selector():
    """The public API must not silently rebuild a different merge policy."""
    cases = [
        (4, 16, 64, 4, 64),
        (40, 64, 8, 2, 256),
        (1024, 16, 1, 2, 256),
    ]
    for rows, block_n, target_splits, warps, dim_tile in cases:
        config = _select_sparse_launch_config(rows, 1, 2051, 256, 6)
        assert config["block_n"] == block_n
        assert config["target_splits"] == target_splits
        assert config["num_warps"] == warps
        assert config["merge_dim_tile"] == dim_tile
        # Rows >512 do not launch merge under the runtime split policy.
        if rows <= 512:
            assert config["merge_warps"] == 1


def test_tuned_long_prefill_segment_is_authoritative():
    """The measured table must not be silently overridden by public API."""
    _, _, values = TUNED_TARGET_SEGMENTS[0]
    assert values[:3] == (16, 1, 2)


def test_qsa_sparse_hcu_gqa_uses_mma_compatible_block_m():
    expected = 16 if torch.version.hip is not None else 8
    assert _select_qsa_sparse_block_m(6, 256) == expected
    assert _select_qsa_sparse_block_m(4, 128) == 4


def test_qsa_configs_are_merged_into_kernel_module():
    """Dispatch policy and measured segments live beside the QSA kernels."""
    from boltops.generic.triton import qsa

    assert qsa.TUNED_TARGET_SEGMENTS
    assert qsa._select_qsa_mqa_config(512, 65536, 128, 4)["use_grouped_mma"]
    assert hasattr(qsa, "_qsa_mqa_paged_mma_kernel")

def test_qsa_tuned_segments_are_valid_when_present():
    """Measured segments must be ordered, non-overlapping, and in-bounds."""
    if not TUNED_TARGET_SEGMENTS:
        return
    starts = [segment[0] for segment in TUNED_TARGET_SEGMENTS]
    assert starts == sorted(starts)
    for (start, end, values), (next_start, _, _) in zip(
        TUNED_TARGET_SEGMENTS, TUNED_TARGET_SEGMENTS[1:]
    ):
        assert start <= end
        assert end + 1 == next_start
        assert len(values) == 6
        block_n, target_splits, warps, stages, merge_warps, dim_tile = values
        assert block_n in (16, 32, 64)
        assert warps in (1, 2, 4, 8)
        assert stages in (1, 2)
        assert merge_warps in (1, 2, 4)
        assert dim_tile in (32, 64, 128, 256)
    for rows in (1, 2, 512, 513, 10240, 10241):
        config = _select_qsa_sparse_config(rows, 1, 2051, 256)
        assert set(config) == {
            "block_n",
            "target_splits",
            "num_warps",
            "num_stages",
            "merge_warps",
            "merge_dim_tile",
        }


def _reference_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    num_columns: int | None = None,
    score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent Torch implementation of the compressed paged-score contract."""
    rows, _, head_dim = q.shape
    columns = (
        page_table.shape[1] * k_cache.shape[1]
        if num_columns is None
        else num_columns
    )
    scores = torch.full(
        (rows, columns), -float("inf"), dtype=torch.float32, device=q.device
    )
    visible = torch.zeros(rows, dtype=torch.int32, device=q.device)
    divisor = head_dim**0.5 if score_scale is None else score_scale
    for row in range(rows):
        request = int(token_to_req[row])
        if not 0 <= request < page_table.shape[0]:
            continue
        visible_blocks = min(
            (int(query_positions[row]) + 1) // compress_ratio,
            int(sequence_lengths[request]) // compress_ratio,
        )
        visible[row] = visible_blocks
        for column in range(min(columns, visible_blocks)):
            logical_page, page_offset = divmod(column, k_cache.shape[1])
            if logical_page >= page_table.shape[1]:
                continue
            physical_page = int(page_table[request, logical_page])
            if not 0 <= physical_page < k_cache.shape[0]:
                continue
            dot = (q[row].float() * k_cache[physical_page, page_offset, 0].float()).sum(
                dim=-1
            )
            scores[row, column] = torch.relu(dot).sum() / divisor
    return scores, visible


def _reference_expand_qsa_blocks(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Independent Torch implementation of QSA block expansion and causal tail."""
    rows = block_indices.shape[0]
    output_width = token_topk + compress_ratio - 1
    output = torch.full(
        (rows, output_width), -1, dtype=torch.int32, device=block_indices.device
    )
    block_topk = token_topk // compress_ratio
    for row in range(rows):
        request = int(token_to_req[row])
        if not 0 <= request < sequence_lengths.numel():
            continue
        query_position = int(query_positions[row])
        sequence_length = int(sequence_lengths[request])
        complete_blocks = min(
            (query_position + 1) // compress_ratio,
            sequence_length // compress_ratio,
            block_topk,
        )
        for rank in range(complete_blocks):
            block = int(block_indices[row, rank])
            for offset in range(compress_ratio):
                token = block * compress_ratio + offset
                if 0 <= token < sequence_length:
                    output[row, rank * compress_ratio + offset] = token
        tail_start = ((query_position + 1) // compress_ratio) * compress_ratio
        tail_count = min(
            (query_position + 1) - tail_start,
            compress_ratio - 1,
        )
        for offset in range(tail_count):
            token = tail_start + offset
            if token < sequence_length:
                output[row, complete_blocks * compress_ratio + offset] = token
    return output


def _reference_select_paged_tokens(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Independently select visible compressed blocks then expand their ids."""
    scores, visible_blocks = _reference_mqa_paged(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
    )
    columns = torch.arange(scores.shape[1], device=scores.device)
    scores.masked_fill_(
        columns.unsqueeze(0) >= visible_blocks.unsqueeze(1), -float("inf")
    )
    block_indices = torch.topk(
        scores,
        k=token_topk // compress_ratio,
        dim=1,
        largest=True,
        sorted=False,
    ).indices.to(torch.int32)
    return _reference_expand_qsa_blocks(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio,
        token_topk,
    )


def _reference_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
) -> torch.Tensor:
    """Independent Torch implementation of sparse paged grouped attention."""
    rows, num_query_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    group_size = num_query_heads // num_kv_heads
    output = torch.zeros_like(q)
    for row in range(rows):
        request = int(token_to_req[row])
        if not 0 <= request < block_table.shape[0]:
            continue
        for kv_head in range(num_kv_heads):
            keys = []
            values = []
            for logical_token_tensor in logical_indices[row]:
                logical_token = int(logical_token_tensor)
                if logical_token < 0:
                    continue
                logical_page, page_offset = divmod(
                    logical_token, k_cache.shape[1]
                )
                if logical_page >= block_table.shape[1]:
                    continue
                physical_page = int(block_table[request, logical_page])
                if not 0 <= physical_page < k_cache.shape[0]:
                    continue
                keys.append(k_cache[physical_page, page_offset, kv_head].float())
                values.append(v_cache[physical_page, page_offset, kv_head].float())
            if not keys:
                continue
            key = torch.stack(keys)
            value = torch.stack(values)
            first_head = kv_head * group_size
            query = q[row, first_head : first_head + group_size].float()
            probabilities = torch.softmax(
                query @ key.transpose(0, 1) / head_dim**0.5, dim=-1
            )
            output[row, first_head : first_head + group_size] = (
                probabilities @ value
            ).to(q.dtype)
    return output


def _sparse_paged_fixture(rows: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(5678)
    q = torch.randn((rows, 4, 16), device=DEVICE, dtype=torch.bfloat16)
    k_cache = torch.randn((6, 4, 2, 16), device=DEVICE, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    logical_indices = torch.arange(32, device=DEVICE, dtype=torch.int32).repeat(
        rows, 1
    )
    logical_indices[:, 5::7] = -1
    block_table = torch.tensor(
        [[2, 0, -1, 5, 1, 8, 4, 3], [-1, 8, -1, 8, -1, 8, -1, 8]],
        device=DEVICE,
        dtype=torch.int32,
    )
    token_to_req = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    if rows > 1:
        token_to_req[-1] = 1
    return q, k_cache, v_cache, logical_indices, block_table, token_to_req


def _paged_mqa_fixture() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1234)
    q = torch.randn((3, 2, 8), device=DEVICE, dtype=torch.bfloat16)
    k_cache = torch.randn((4, 2, 1, 8), device=DEVICE, dtype=torch.bfloat16)
    page_table = torch.tensor([[2, 0], [3, -1]], device=DEVICE, dtype=torch.int32)
    token_to_req = torch.tensor([0, 1, 0], device=DEVICE, dtype=torch.int32)
    positions = torch.tensor([15, 11, 3], device=DEVICE, dtype=torch.int32)
    lengths = torch.tensor([16, 12], device=DEVICE, dtype=torch.int32)
    return q, k_cache, page_table, token_to_req, positions, lengths


@GPU_REQUIRED
def test_qsa_mqa_paged_matches_reference():
    q, k_cache, page_table, token_to_req, positions, lengths = _paged_mqa_fixture()

    actual, visible = qsa_mqa_paged(
        q, k_cache, page_table, token_to_req, positions, lengths, compress_ratio=4
    )
    expected, expected_visible = _reference_mqa_paged(
        q, k_cache, page_table, token_to_req, positions, lengths, 4
    )

    torch.testing.assert_close(visible, expected_visible, rtol=0, atol=0)
    torch.testing.assert_close(
        actual, expected, rtol=2e-3, atol=2e-3, equal_nan=True
    )


@GPU_REQUIRED
def test_qsa_grouped_mma_uses_each_token_visible_range():
    """Grouped H=4 MMA must score each same-request token through its own tail."""
    torch.manual_seed(4128)
    rows, query_heads, head_dim = 4, 4, 128
    page_size, width = 16, 8
    columns = page_size * width
    q = torch.randn(
        (rows, query_heads, head_dim), device=DEVICE, dtype=torch.bfloat16
    ) * 0.25
    k_cache = torch.randn(
        (width, page_size, 1, head_dim), device=DEVICE, dtype=torch.bfloat16
    ) * 0.25
    page_table = torch.randperm(width, device=DEVICE).to(torch.int32).unsqueeze(0)
    token_to_req = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    positions = torch.tensor([3, 7, 15, 31], device=DEVICE, dtype=torch.int32)
    lengths = torch.full((1,), columns * 4, device=DEVICE, dtype=torch.int32)

    actual, visible = qsa_mqa_paged(
        q, k_cache, page_table, token_to_req, positions, lengths,
        compress_ratio=4,
    )
    expected, expected_visible = _reference_mqa_paged(
        q, k_cache, page_table, token_to_req, positions, lengths, 4,
    )

    torch.testing.assert_close(visible, expected_visible, rtol=0, atol=0)
    torch.testing.assert_close(
        actual, expected, rtol=2e-3, atol=2e-3, equal_nan=True
    )


@GPU_REQUIRED
def test_qsa_grouped_mma_falls_back_for_mixed_requests():
    """A mixed-request group must retain the scalar-per-token score contract."""
    torch.manual_seed(4129)
    rows, query_heads, head_dim = 4, 4, 128
    page_size, width = 16, 8
    columns = page_size * width
    q = torch.randn(
        (rows, query_heads, head_dim), device=DEVICE, dtype=torch.bfloat16
    ) * 0.25
    k_cache = torch.randn(
        (width, page_size, 1, head_dim), device=DEVICE, dtype=torch.bfloat16
    ) * 0.25
    page_table = torch.stack(
        (torch.randperm(width, device=DEVICE), torch.randperm(width, device=DEVICE))
    ).to(torch.int32)
    token_to_req = torch.tensor([0, 1, 0, 1], device=DEVICE, dtype=torch.int32)
    positions = torch.tensor([3, 7, 15, 31], device=DEVICE, dtype=torch.int32)
    lengths = torch.full((2,), columns * 4, device=DEVICE, dtype=torch.int32)

    actual, visible = qsa_mqa_paged(
        q, k_cache, page_table, token_to_req, positions, lengths,
        compress_ratio=4,
    )
    expected, expected_visible = _reference_mqa_paged(
        q, k_cache, page_table, token_to_req, positions, lengths, 4,
    )

    torch.testing.assert_close(visible, expected_visible, rtol=0, atol=0)
    torch.testing.assert_close(
        actual, expected, rtol=2e-3, atol=2e-3, equal_nan=True
    )


@GPU_REQUIRED
@pytest.mark.parametrize(
    ("query_heads", "head_dim"),
    [(4, 128), (8, 64), (16, 128), (32, 32)],
)
def test_qsa_mma_mqa_matches_scalar(query_heads, head_dim):
    """Generic MMA routing must preserve the scalar kernel's score contract."""
    torch.manual_seed(query_heads * 1000 + head_dim)
    rows, page_size, width = 17, 16, 8
    columns = page_size * width
    q = torch.randn(
        (rows, query_heads, head_dim), device=DEVICE, dtype=torch.bfloat16
    ) * 0.25
    k_cache = torch.randn(
        (width, page_size, 1, head_dim), device=DEVICE, dtype=torch.bfloat16
    ) * 0.25
    page_table = torch.randperm(width, device=DEVICE).to(torch.int32).unsqueeze(0)
    token_to_req = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    positions = torch.linspace(
        0, columns * 4 - 1, rows, device=DEVICE
    ).to(torch.int32)
    lengths = torch.full((1,), columns * 4, device=DEVICE, dtype=torch.int32)
    config = _select_qsa_mqa_config(rows, columns, head_dim, query_heads)
    assert config["use_mma"]

    actual, actual_visible = qsa_mqa_paged(
        q, k_cache, page_table, token_to_req, positions, lengths,
        compress_ratio=4,
    )
    expected = torch.empty_like(actual)
    expected_visible = torch.empty_like(actual_visible)
    _qsa_mqa_paged_kernel[(rows, triton.cdiv(columns, config["block_n"]))](
        q, k_cache, page_table, token_to_req, positions, lengths,
        expected_visible, expected, q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(3),
        page_table.stride(0), page_table.stride(1), expected.stride(0), rows,
        columns, width, 1, math.sqrt(head_dim), PAGE_SIZE=page_size,
        PAGE_TABLE_WIDTH=width, NUM_HEADS=query_heads, HEAD_DIM=head_dim,
        BLOCK_N=config["block_n"], BLOCK_D=triton.next_power_of_2(head_dim),
        COMPRESS_RATIO=4, num_warps=2,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual_visible, expected_visible, rtol=0, atol=0)
    torch.testing.assert_close(
        actual, expected, rtol=2e-3, atol=2e-3, equal_nan=True
    )


@GPU_REQUIRED
def test_qsa_select_matches_masked_torch_topk():
    q, k_cache, page_table, token_to_req, positions, lengths = _paged_mqa_fixture()

    actual = qsa_select_paged_tokens(
        q,
        k_cache,
        page_table,
        token_to_req,
        positions,
        lengths,
        token_topk=8,
        compress_ratio=4,
    )
    expected = _reference_select_paged_tokens(
        q,
        k_cache,
        page_table,
        token_to_req,
        positions,
        lengths,
        8,
        4,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@GPU_REQUIRED
def test_expand_qsa_blocks_handles_invalid_blocks_and_causal_tails():
    blocks = torch.tensor([[0, -1], [1, 0], [0, 1]], device=DEVICE, dtype=torch.int32)
    positions = torch.tensor([9, 6, 10], device=DEVICE, dtype=torch.int32)
    lengths = torch.tensor([10, 10], device=DEVICE, dtype=torch.int32)
    token_to_req = torch.tensor([0, 0, -1], device=DEVICE, dtype=torch.int32)

    actual = expand_qsa_block_indices(
        blocks, positions, lengths, token_to_req, compress_ratio=4, token_topk=8
    )
    expected = _reference_expand_qsa_blocks(
        blocks, positions, lengths, token_to_req, compress_ratio=4, token_topk=8
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@GPU_REQUIRED
def test_expand_qsa_blocks_uses_caller_owned_output_buffer():
    blocks = torch.tensor([[1, 0]], device=DEVICE, dtype=torch.int32)
    positions = torch.tensor([6], device=DEVICE, dtype=torch.int32)
    lengths = torch.tensor([8], device=DEVICE, dtype=torch.int32)
    token_to_req = torch.tensor([0], device=DEVICE, dtype=torch.int32)
    out = torch.full((1, 11), -99, device=DEVICE, dtype=torch.int32)

    actual = expand_qsa_block_indices(
        blocks,
        positions,
        lengths,
        token_to_req,
        compress_ratio=4,
        token_topk=8,
        out=out,
    )
    expected = _reference_expand_qsa_blocks(
        blocks, positions, lengths, token_to_req, compress_ratio=4, token_topk=8
    )

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@GPU_REQUIRED
def test_expand_qsa_blocks_empty_rows_preserve_output_contract():
    blocks = torch.empty((0, 2), device=DEVICE, dtype=torch.int32)
    positions = torch.empty(0, device=DEVICE, dtype=torch.int32)
    lengths = torch.tensor([8], device=DEVICE, dtype=torch.int32)
    token_to_req = torch.empty(0, device=DEVICE, dtype=torch.int32)
    out = torch.empty((0, 11), device=DEVICE, dtype=torch.int32)

    actual = expand_qsa_block_indices(
        blocks,
        positions,
        lengths,
        token_to_req,
        compress_ratio=4,
        token_topk=8,
        out=out,
    )

    assert actual.data_ptr() == out.data_ptr()
    assert actual.shape == (0, 11)
    assert actual.dtype == torch.int32


@GPU_REQUIRED
@pytest.mark.parametrize("rows", [1, 33])
def test_sparse_paged_attention_matches_reference(rows):
    inputs = _sparse_paged_fixture(rows)

    actual = qsa_sparse_paged_attention(*inputs)
    expected = _reference_sparse_paged_attention(*inputs)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=8e-2)


@GPU_REQUIRED
def test_sparse_paged_attention_page16_fast_decode_preserves_invalid_tokens():
    """The PAGE_SIZE=16 address fast path must retain invalid-token semantics."""
    torch.manual_seed(9016)
    rows, heads, dim, page_size, width = 3, 4, 16, 16, 2
    q = torch.randn((rows, heads, dim), device=DEVICE, dtype=torch.bfloat16)
    k_cache = torch.randn((width, page_size, 1, dim), device=DEVICE, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    indices = torch.arange(page_size * width, device=DEVICE, dtype=torch.int32).repeat(rows, 1)
    indices[0, 3] = -1
    indices[1, 17] = -1
    block_table = torch.tensor([[1, 0], [0, -1]], device=DEVICE, dtype=torch.int32)
    token_to_req = torch.tensor([0, 1, -1], device=DEVICE, dtype=torch.int32)
    inputs = (q, k_cache, v_cache, indices, block_table, token_to_req)
    actual = qsa_sparse_paged_attention(*inputs)
    expected = _reference_sparse_paged_attention(*inputs)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=8e-2)



def test_qsa_rejects_nonempty_cpu_input():
    q = torch.empty((1, 2, 16), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA/HIP"):
        qsa_mqa_paged(
            q,
            q,
            torch.empty((1, 1), dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int64),
            torch.ones(1, dtype=torch.int32),
            4,
        )


def test_qsa_rejects_empty_cpu_input():
    q = torch.empty((0, 2, 16), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA/HIP"):
        qsa_mqa_paged(
            q,
            q,
            torch.empty((0, 1), dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int32),
            4,
        )
