# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Matrix-axis layouts must not be confused with packing or batch ordering."""

import pytest
import torch
import triton

from boltops.gemm import (
    gemm_a8w8,
    gemm_a8w8_blockscale,
    gemm_a16w4,
    gemm_w4a4,
    batched_gemm_a16w4,
)


def require_hcu():
    if (
        not torch.cuda.is_available()
        or triton.runtime.driver.active.get_current_target().arch != "gfx946"
    ):
        pytest.skip("requires gfx946")


def matrix_view(value, column_major):
    value = value.cuda()
    return (
        value.transpose(-1, -2).contiguous().transpose(-1, -2)
        if column_major
        else value
    )


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize(
    "blockscale,scale_column_major", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize(
    "a_column_major,b_column_major",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_a8w8_matrix_layouts(
    dtype, blockscale, a_column_major, b_column_major, scale_column_major
):
    require_hcu()
    torch.manual_seed(812)
    m, n, k = 5, 17, 64
    a_cpu = torch.randint(-2, 3, (m, k)).to(dtype)
    b_cpu = torch.randint(-2, 3, (n, k)).to(dtype)
    a = matrix_view(a_cpu, a_column_major)
    b = matrix_view(b_cpu, b_column_major)
    if blockscale:
        sa = (torch.arange(m * 2).reshape(m, 2) % 4 + 1).float() / 8
        sb = torch.tensor([[0.25, 0.5], [0.5, 0.25]])
        golden = torch.zeros((m, n))
        for group in range(2):
            golden += (
                (
                    a_cpu[:, group * 32 : (group + 1) * 32].float()
                    @ b_cpu[:, group * 32 : (group + 1) * 32].float().T
                )
                * sa[:, group, None]
                * sb[torch.arange(n) // 16, group][None, :]
            )
        result = gemm_a8w8_blockscale(
            a,
            b,
            matrix_view(sa, scale_column_major),
            matrix_view(sb, scale_column_major),
            dtype=torch.float32,
            backend="triton",
        )
    else:
        sa = torch.arange(1, m + 1).float() / 8
        sb = torch.arange(1, n + 1).float() / 16
        golden = (a_cpu.float() @ b_cpu.float().T) * sa[:, None] * sb[None, :] + 0.5
        result = gemm_a8w8(
            a,
            b,
            sa.cuda(),
            sb.cuda(),
            bias=torch.full((n,), 0.5, device="cuda"),
            dtype=torch.float32,
            backend="triton",
        )
    torch.testing.assert_close(result.cpu(), golden, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("a_column_major", [False, True])
@pytest.mark.parametrize("scale_column_major", [False, True])
def test_awq_unpacked_matrix_layouts(dtype, a_column_major, scale_column_major):
    require_hcu()
    torch.manual_seed(813)
    m, n, k = 3, 64, 64
    a_cpu = (torch.randint(-2, 3, (m, k)).float() / 4).to(dtype)
    scales_cpu = ((torch.arange(2 * n).reshape(2, n) % 4 + 1).float() / 8).to(dtype)
    a = matrix_view(a_cpu, a_column_major)
    scales = matrix_view(scales_cpu, scale_column_major)
    w = torch.full((k, n // 8), 0x33333333, dtype=torch.int32, device="cuda")
    zeros = torch.full((2, n // 8), 0x11111111, dtype=torch.int32, device="cuda")
    result = gemm_a16w4(a, w, scales, zeros)
    golden = a_cpu.float() @ (2 * scales_cpu.float().repeat_interleave(32, dim=0))
    torch.testing.assert_close(result.cpu(), golden.to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "a_column_major,b_column_major", [(False, True), (True, False), (True, True)]
)
def test_fp4_unpacked_scale_layouts(dtype, a_column_major, b_column_major):
    require_hcu()
    m, n, k = 3, 5, 128
    sa = (126 + torch.arange(m * 4).reshape(m, 4) % 3).to(torch.uint8)
    sb = (126 + torch.arange(n * 4).reshape(n, 4) % 3).to(torch.uint8)
    a = torch.full((m, k // 2), 0x22, dtype=torch.uint8, device="cuda")
    b = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda")
    result = gemm_w4a4(
        a,
        b,
        matrix_view(sa, a_column_major),
        matrix_view(sb, b_column_major),
        dtype=dtype,
    )
    golden = 32 * (torch.exp2(sa.float() - 127) @ torch.exp2(sb.float() - 127).T)
    torch.testing.assert_close(result.cpu(), golden.to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "a_column_major,scale_column_major", [(False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("split_k", [1, 2])
@pytest.mark.parametrize("transpose_bm", [False, True])
def test_batched_fp4_unpacked_matrix_layouts(
    dtype, a_column_major, scale_column_major, split_k, transpose_bm, request
):
    require_hcu()
    if not a_column_major and split_k == 1:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Known gfx946 PMD/compiler mismatch for K-contiguous online FP4 A and varying K32 scales; target-main kernel also reproduces",
                raises=AssertionError,
                strict=True,
            )
        )
    batch, m, n, k = 2, 3, 5, 128
    rows = torch.tensor([[1, 2, 4], [2, 4, 8]], dtype=dtype)
    a_cpu = (rows[:, :, None] * (torch.arange(k) // 32 + 1)[None, None, :] / 4).to(
        dtype
    )
    packed = torch.tensor([0x22, 0x44, 0x66, 0x22, 0x44], dtype=torch.uint8)
    b = packed[None, :, None].expand(batch, n, k // 2).contiguous().cuda()
    scales_cpu = (126 + torch.arange(batch * n * 4).reshape(batch, n, 4) % 3).to(
        torch.uint8
    )
    a = matrix_view(a_cpu, a_column_major)
    scales = matrix_view(scales_cpu, scale_column_major)
    config = dict(
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=1,
        NUM_KSPLIT=split_k,
        num_warps=4,
        num_stages=1,
        cache_modifier=".cg",
    )
    result = batched_gemm_a16w4(
        a, b, scales, dtype=dtype, config=config, transpose_bm=transpose_bm
    )
    weights = torch.tensor([1, 2, 4, 1, 2])[None, :, None] * torch.exp2(
        scales_cpu.float() - 127
    ).repeat_interleave(32, dim=-1)
    golden = torch.bmm(a_cpu.float(), weights.transpose(1, 2))
    if transpose_bm:
        golden = golden.transpose(0, 1).contiguous()
    torch.testing.assert_close(result.cpu(), golden.to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize("scale", [False, True])
def test_batch_only_contiguous_is_rejected(scale):
    require_hcu()
    a = torch.ones((2, 3, 64), dtype=torch.bfloat16, device="cuda")
    b = torch.full((2, 5, 32), 0x22, dtype=torch.uint8, device="cuda")
    s = torch.full((2, 5, 2), 127, dtype=torch.uint8, device="cuda")
    if scale:
        s = s.permute(1, 2, 0).contiguous().permute(2, 0, 1)
        assert s.stride(0) == 1 and all(v != 1 for v in s.stride()[-2:])
    else:
        a = a.permute(1, 2, 0).contiguous().permute(2, 0, 1)
        assert a.stride(0) == 1 and all(v != 1 for v in a.stride()[-2:])
    with pytest.raises(ValueError, match="contiguous matrix axis"):
        batched_gemm_a16w4(a, b, s)


@pytest.mark.parametrize("operation", [gemm_a8w8, gemm_a8w8_blockscale])
def test_positive_strided_matrices(operation):
    require_hcu()
    a = torch.ones((10, 128), dtype=torch.int8, device="cuda")[::2, ::2]
    b = torch.ones((34, 128), dtype=torch.int8, device="cuda")[::2, ::2]
    sa = (
        torch.ones((5, 2), device="cuda")
        if operation is gemm_a8w8_blockscale
        else torch.ones(5, device="cuda")
    )
    sb = (
        torch.ones((1, 2), device="cuda")
        if operation is gemm_a8w8_blockscale
        else torch.ones(17, device="cuda")
    )
    result = operation(a, b, sa, sb, dtype=torch.float32)
    torch.testing.assert_close(result.cpu(), torch.full((5, 17), 64.0), atol=0, rtol=0)


def test_explicit_gluon_rejects_noncontiguous_operands():
    require_hcu()
    a = torch.empty_strided(
        (128, 256), (1, 128), dtype=torch.float8_e4m3fn, device="cuda"
    )
    b = torch.empty((256, 256), dtype=a.dtype, device=a.device)
    sa = torch.ones((128, 2), device=a.device)
    sb = torch.ones((2, 2), device=a.device)
    with pytest.raises(NotImplementedError, match="requires contiguous"):
        gemm_a8w8_blockscale(a, b, sa, sb, backend="gluon")


@pytest.mark.parametrize("blockscale", [False, True])
def test_padded_matrix_axes(blockscale):
    require_hcu()
    m, n, k = 5, 17, 64
    a_storage = torch.ones((m, k + 7), dtype=torch.int8, device="cuda")
    b_storage = torch.ones((k, n + 3), dtype=torch.int8, device="cuda")
    a, b = a_storage[:, :k], b_storage[:, :n].T
    assert a.stride() == (k + 7, 1) and b.stride() == (1, n + 3)
    if blockscale:
        sa = torch.full((m, 4), float("nan"), device="cuda")[:, :2]
        sb = torch.full((2, 4), float("nan"), device="cuda")[:, :2]
        sa.fill_(0.5)
        sb.fill_(0.25)
        result = gemm_a8w8_blockscale(a, b, sa, sb, dtype=torch.float32)
    else:
        result = gemm_a8w8(
            a,
            b,
            torch.tensor(0.5, device="cuda"),
            torch.tensor(0.25, device="cuda"),
            dtype=torch.float32,
        )
    torch.testing.assert_close(result.cpu(), torch.full((m, n), 8.0), atol=0, rtol=0)


@pytest.mark.parametrize("weight", [False, True])
def test_batched_bf16_batch_only_contiguous_is_rejected(weight):
    require_hcu()
    from boltops.gemm import batched_gemm_bf16

    a = torch.ones((2, 3, 64), dtype=torch.float16, device="cuda")
    b = torch.ones((2, 5, 64), dtype=a.dtype, device=a.device)
    if weight:
        b = b.permute(1, 2, 0).contiguous().permute(2, 0, 1)
    else:
        a = a.permute(1, 2, 0).contiguous().permute(2, 0, 1)
    with pytest.raises(ValueError, match="batch-only contiguous"):
        batched_gemm_bf16(a, b)


@pytest.mark.parametrize("operation", ["awq", "fp4", "batched_fp4"])
def test_packed_operand_layouts(operation):
    require_hcu()
    if operation == "awq":
        m, n, k, group = 3, 64, 64, 32
        q = (torch.arange(k)[:, None] + torch.arange(n)[None, :]) % 4 + 1
        z = (torch.arange(2)[:, None] + torch.arange(n)[None, :]) % 2
        shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28])
        packed = (q.reshape(k, n // 8, 8).long() << shifts).sum(-1).int()
        zeros = (z.reshape(2, n // 8, 8).long() << shifts).sum(-1).int()
        scale_cpu = (
            (torch.arange(2)[:, None] + torch.arange(n)[None, :]) % 4 + 1
        ).half() / 8
        a_cpu = torch.arange(m * k).reshape(m, k).remainder(5).half() / 4
        a = matrix_view(a_cpu, True)
        result = gemm_a16w4(
            a,
            matrix_view(packed, True),
            matrix_view(scale_cpu, True),
            matrix_view(zeros, True),
        )
        golden = a_cpu.float() @ (
            (q.float() - z.float().repeat_interleave(group, dim=0))
            * scale_cpu.float().repeat_interleave(group, dim=0)
        )
    elif operation == "fp4":
        a = matrix_view(torch.full((3, 32), 0x22, dtype=torch.uint8), True)
        b = matrix_view(torch.full((5, 32), 0x44, dtype=torch.uint8), True)
        sa = matrix_view(torch.full((3, 2), 126, dtype=torch.uint8), True)
        sb = matrix_view(torch.full((5, 2), 128, dtype=torch.uint8), True)
        result = gemm_w4a4(a, b, sa, sb)
        golden = torch.full((3, 5), 128.0)
    else:
        a = matrix_view(torch.ones((2, 3, 64), dtype=torch.bfloat16), True)
        b = matrix_view(torch.full((2, 5, 32), 0x44, dtype=torch.uint8), True)
        scales = matrix_view(torch.full((2, 5, 2), 126, dtype=torch.uint8), True)
        result = batched_gemm_a16w4(a, b, scales)
        golden = torch.full((2, 3, 5), 64.0)
    torch.testing.assert_close(result.cpu(), golden.to(result.dtype), atol=0, rtol=0)


def test_whole_k_strided_scale_vectors():
    require_hcu()
    a = torch.ones((5, 64), dtype=torch.int8, device="cuda")
    b = torch.ones((17, 64), dtype=torch.int8, device="cuda")
    sa_cpu = torch.arange(1, 6).float() / 8
    sb_cpu = torch.arange(1, 18).float() / 16
    sa = torch.full((10,), float("nan"), device="cuda")[::2]
    sb = torch.full((34,), float("nan"), device="cuda")[::2]
    sa.copy_(sa_cpu.cuda())
    sb.copy_(sb_cpu.cuda())
    result = gemm_a8w8(a, b, sa, sb, dtype=torch.float32)
    golden = 64 * sa_cpu[:, None] * sb_cpu[None, :]
    torch.testing.assert_close(result.cpu(), golden, atol=0, rtol=0)
