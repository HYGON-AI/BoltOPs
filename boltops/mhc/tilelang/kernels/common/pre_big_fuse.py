import functools
import math

import tilelang
import torch
from tilelang import language as T

from boltops.mhc.tilelang.configs.pre_big_fuse import resolve_get_config
from boltops.utility.arch_info import get_device_info

from .pre_norm import _mhc_pre_norm_fn_fwd_mul
from .pre_norm_splitk import mhc_pre_gemm_sqrsum_splitk_kernel
from .pre_big_fuse_kernel import _mhc_pre_big_fuse


@functools.lru_cache(maxsize=128)
def _round_to_tf32_kernel(n_elem: int) -> tilelang.JITKernel:
    return _compile_round_to_tf32(n_elem)


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True},
)  # inp, out both passed in; out_idx would mean only inp is passed and out is allocated inside the adapter
def _compile_round_to_tf32(n_elem: int) -> tilelang.JITKernel:
    """Bitcast float32 -> int32, add 0x1000, bitcast back (1D linear scan for coalescing)."""
    _TF32_ROUND_BITS = 0x1000
    _ROUND_TO_TF32_BLK_MAX = 2048
    n_blk = math.gcd(_ROUND_TO_TF32_BLK_MAX, n_elem)

    @T.prim_func
    def _round_to_tf32_prim(
        inp: T.Tensor[(n_elem,), T.float32],
        out: T.Tensor[(n_elem,), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(n_elem, n_blk)) as pid:
            input_frag = T.alloc_fragment((n_blk,), T.float32)
            output_frag = T.alloc_fragment((n_blk,), T.float32)
            T.copy(inp[pid * n_blk], input_frag)
            input_int = T.view(input_frag, (n_blk,), T.int32)

            for t in T.Parallel(n_blk):
                input_int[t] += T.int32(_TF32_ROUND_BITS)
                output_frag[t] = T.reinterpret(input_int[t], T.float32)
            T.copy(output_frag, out[pid * n_blk])

    return _round_to_tf32_prim


def round_to_tf32(fn: torch.Tensor) -> torch.Tensor:
    """TF32 grid rounding via TileLang (flat numel; preserves original shape)."""
    ne = int(fn.numel())
    out = torch.empty_like(fn)
    _round_to_tf32_kernel(ne)(fn.reshape(ne), out.reshape(ne))
    return out


def mhc_pre_big_fuse(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    _splitk_kernel_factory=mhc_pre_gemm_sqrsum_splitk_kernel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    mhc_hidden_size = mhc_mult * hidden_size
    assert fn.shape[0] == mhc_mult3
    assert fn.shape[1] == mhc_hidden_size
    assert mhc_scale.shape == (3,)
    assert mhc_base.shape == (mhc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    post_mix = torch.empty(
        num_tokens, mhc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    # Bucket by 32 so get_block_info cache keys align with common launch granularity; real buffers still use num_tokens.
    num_tokens_align = (int(num_tokens) + 31) // 32 * 32
    device_info = get_device_info(residual.device)
    get_config = resolve_get_config(device_info.arch)
    block_info = get_config(num_tokens_align, mhc_hidden_size, device_info.cu_count)
    token_block = block_info.token_block
    hidden_block = block_info.hidden_block
    MHC_PRE_BIG_FUSE_N_SPLITS_PRE = block_info.n_splits_pre
    use_small_token_splitk = block_info.use_small_token_splitk

    if use_small_token_splitk:
        kernel_0, _ = _splitk_kernel_factory(
            mhc_mult3,
            mhc_hidden_size,
            split_k=MHC_PRE_BIG_FUSE_N_SPLITS_PRE,
            token_block=token_block,
            hidden_block=hidden_block,
            round_fn_to_tf32=True,
            separate_x_fn_smem=block_info.separate_x_fn_smem,
        )
        partial_out = torch.empty(
            MHC_PRE_BIG_FUSE_N_SPLITS_PRE,
            num_tokens,
            mhc_mult3,
            dtype=torch.float32,
            device=residual.device,
        )
        partial_sqrsum = torch.empty(
            MHC_PRE_BIG_FUSE_N_SPLITS_PRE,
            num_tokens,
            dtype=torch.float32,
            device=residual.device,
        )
        # gemm_out_mul = torch.empty(
        #     1, num_tokens, mhc_mult3, dtype=torch.float32, device=residual.device
        # )
        # gemm_out_sqrsum = torch.empty(1, num_tokens, dtype=torch.float32, device=residual.device)
        kernel_0(
            residual_flat.view(-1, mhc_hidden_size),
            fn,
            partial_out,
            partial_sqrsum,
        )
        gemm_out_mul = partial_out
        gemm_out_sqrsum = partial_sqrsum
        # kernel_1(
        #     partial_out,
        #     partial_sqrsum,
        #     gemm_out_mul.squeeze(0),
        #     gemm_out_sqrsum.squeeze(0),
        # )
        n_splits = MHC_PRE_BIG_FUSE_N_SPLITS_PRE
    else:
        gemm_out_mul = torch.empty(
            1, num_tokens, mhc_mult3, dtype=torch.float32, device=residual.device
        )
        gemm_out_sqrsum = torch.empty(
            1, num_tokens, dtype=torch.float32, device=residual.device
        )
        n_splits = 1
        fwd_mul_kernel = _mhc_pre_norm_fn_fwd_mul(
            mhc_mult3,
            1,
            mhc_hidden_size,
            token_block=token_block,
            hidden_block=hidden_block,
            round_fn_to_tf32=True,
        )
        fwd_mul_kernel(
            residual_flat.view(-1, mhc_hidden_size),
            fn,
            gemm_out_mul.view(-1, 1, mhc_mult3),
            gemm_out_sqrsum.view(-1, 1),
        )
    # END of TileLang implementation of pre-norm-fn forward matmul

    _mhc_pre_big_fuse(
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        mhc_mult=mhc_mult,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
    )

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input
