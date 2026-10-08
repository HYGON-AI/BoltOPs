# Copyright (c) 2026 The Qwen team, Alibaba Group.
# SPDX-License-Identifier: MIT
"""HCU TileLang fused GDN chunk state/output kernel.

The algebra is adapted from FlashQLA's fused forward kernel.  This HCU
specialization deliberately uses only full-workgroup barriers and explicit
global-to-LDS async copies.  GEMM operands and linearly accessed staging data
use distinct LDS buffers; storage is reused only after the previous value's
lifetime has ended.
"""

import tilelang
import tilelang.language as T


_LOG2E = 1.4426950408889634
_THREADS = 768
_ROLE_THREADS = 256
_STATE_BASE = _ROLE_THREADS
_VALUE_BASE = 2 * _ROLE_THREADS


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
    },
)
def _build_fused_chunk_delta_rule_fwd(
    batch: int,
    seq_len: int,
    q_heads: int,
    value_heads: int,
    dim_k: int,
    dim_v: int,
    qkv_dtype: str,
    gate_dtype: str,
    beta_dtype: str,
    chunk_size: int,
    scale: float,
    use_initial_state: bool,
    store_final_state: bool,
    block_v: int = 32,
):
    """Build the scalar-gate GDN specialization of the shared fused skeleton."""
    num_chunks = seq_len // chunk_size

    @T.prim_func
    def fused_chunk_gdn_fwd_kernel(
        q: T.Tensor(  # type: ignore[valid-type]
            [batch, seq_len, q_heads * dim_k], qkv_dtype
        ),
        k: T.Tensor(  # type: ignore[valid-type]
            [batch, seq_len, q_heads * dim_k], qkv_dtype
        ),
        v: T.Tensor(  # type: ignore[valid-type]
            [batch, seq_len, value_heads, dim_v], qkv_dtype
        ),
        a: T.Tensor(  # type: ignore[valid-type]
            [batch, seq_len, value_heads, chunk_size], qkv_dtype
        ),
        g: T.Tensor(  # type: ignore[valid-type]
            [batch, seq_len, value_heads], gate_dtype
        ),
        beta: T.Tensor(  # type: ignore[valid-type]
            [batch, seq_len, value_heads], beta_dtype
        ),
        initial_state: T.Tensor(  # type: ignore[valid-type]
            [batch, value_heads, dim_v, dim_k], "float32"
        ),
        output: T.Tensor(  # type: ignore[valid-type]
            [batch, seq_len, value_heads, dim_v], qkv_dtype
        ),
        final_state: T.Tensor(  # type: ignore[valid-type]
            [batch, value_heads, dim_v, dim_k], "float32"
        ),
    ):
        with T.Kernel(
            T.ceildiv(dim_v, block_v), batch * value_heads, threads=_THREADS
        ) as (vid, bidh):
            bid, hid = bidh // value_heads, bidh % value_heads
            qhid = hid // (value_heads // q_heads)
            v_start = vid * block_v
            tx = T.get_thread_binding()

            # 60.5 KiB at BT=64, K=128, BV=32 and bf16 matrix storage.
            # Keep Q and K in distinct buffers so their LDS layouts are
            # inferred from their own consumer graphs.  Each buffer is
            # overwritten only after the current chunk's operand is dead.
            # Q is only consumed as GEMM A/AT by Q @ H^T and Q @ K^T.
            q_shared = T.alloc_shared((chunk_size, dim_k), qkv_dtype)
            # K has mixed consumers: A/AT in K @ H^T, B/BN in Q @ K^T,
            # and B/BT in Vn^T @ K.  AT/BN share one LDS strategy; BT is
            # the conflicting requirement that still needs separate handling.
            k_shared = T.alloc_shared((chunk_size, dim_k), qkv_dtype)
            raw_v_shared = T.alloc_shared((chunk_size, block_v), qkv_dtype)
            vd_shared = T.alloc_shared((chunk_size, block_v), qkv_dtype)
            # GEMM-layout W is overwritten by Vn only after W's lifetime ends.
            w_vn_shared = T.alloc_shared((chunk_size, block_v), qkv_dtype)
            a_p_shared = T.alloc_shared((chunk_size, chunk_size), qkv_dtype)
            state_shared = T.alloc_shared((block_v, dim_k), qkv_dtype)
            g_shared = T.alloc_shared((chunk_size,), gate_dtype)
            beta_shared = T.alloc_shared((chunk_size,), beta_dtype)

            # Keep each role's loop in a separate warp-uniform branch.  Every
            # role executes two bootstrap barriers and six barriers per
            # chunk; equal barrier generations provide the cross-role protocol
            # even though the instructions reside at different program counters.
            # On HCU, T.sync_warp lowers to the workgroup-wide s_barrier used
            # by this protocol.
            if tx < _STATE_BASE:
                a_fragment = T.alloc_fragment(
                    (chunk_size, chunk_size), "float32"
                )

                # Bootstrap chunk 0.
                T.async_copy(
                    q[
                        bid,
                        0:chunk_size,
                        qhid * dim_k : (qhid + 1) * dim_k,
                    ],
                    q_shared,
                )
                T.async_copy(
                    k[
                        bid,
                        0:chunk_size,
                        qhid * dim_k : (qhid + 1) * dim_k,
                    ],
                    k_shared,
                )
                T.async_copy(
                    v[bid, 0:chunk_size, hid, v_start : v_start + block_v],
                    raw_v_shared,
                )
                T.async_copy(g[bid, 0:chunk_size, hid], g_shared)
                T.async_copy(beta[bid, 0:chunk_size, hid], beta_shared)
                T.ptx_wait_group(0)
                # BI (G0): publish initial Q/K/V/g/beta.
                T.sync_warp()
                T.copy(
                    a[bid, 0:chunk_size, hid, 0:chunk_size],
                    a_fragment,
                )
                for it, jt in T.Parallel(chunk_size, chunk_size):
                    a_fragment[it, jt] = (
                        a_fragment[it, jt]
                        * T.cast(beta_shared[jt], "float32")
                        * T.cast(it >= jt, "float32")
                    )
                T.copy(a_fragment, a_p_shared)
                # B0 (G0): publish A-beta[0].
                T.sync_warp()

                for cid in T.serial(num_chunks):
                    base = cid * chunk_size
                    next_base = base + chunk_size
                    # S0: prefetch beta[n+1].
                    if cid + 1 < num_chunks:
                        T.async_copy(
                            beta[
                                bid,
                                next_base : next_base + chunk_size,
                                hid,
                            ],
                            beta_shared,
                        )
                    # BX (G0): G1 computes while G2 preloads fragments.
                    T.sync_warp()
                    # B1 (G0): beta[n+1] prefetch has been issued.
                    T.sync_warp()

                    # S1: Q[n] is dead; prefetch Q[n+1].
                    if cid + 1 < num_chunks:
                        T.async_copy(
                            q[
                                bid,
                                next_base : next_base + chunk_size,
                                qhid * dim_k : (qhid + 1) * dim_k,
                            ],
                            q_shared,
                        )
                    # B2 (G0): Q[n+1] prefetch has been issued.
                    T.sync_warp()

                    # S2: raw V[n] is dead; prefetch V[n+1].
                    if cid + 1 < num_chunks:
                        T.async_copy(
                            v[
                                bid,
                                next_base : next_base + chunk_size,
                                hid,
                                v_start : v_start + block_v,
                            ],
                            raw_v_shared,
                        )
                    # B3 (G0): V[n+1] prefetch has been issued.
                    T.sync_warp()

                    # S3: K[n] and g[n] are dead.  Prefetch K/g[n+1]
                    # consecutively before the partial async wait.  Keeping
                    # g last lets wait_group(1) publish K before G1 reads it.
                    if cid + 1 < num_chunks:
                        T.async_copy(
                            k[
                                bid,
                                next_base : next_base + chunk_size,
                                qhid * dim_k : (qhid + 1) * dim_k,
                            ],
                            k_shared,
                        )
                        T.async_copy(
                            g[
                                bid,
                                next_base : next_base + chunk_size,
                                hid,
                            ],
                            g_shared,
                        )

                    # S3a: wait for beta/Q/V/K while retaining the g group.
                    if cid + 1 < num_chunks:
                        T.ptx_wait_group(1)
                    # B5a (G0): publish beta/Q/V/K[n+1].
                    T.sync_warp()

                    # S5b: prepare A while g finishes, then publish all inputs.
                    if cid + 1 < num_chunks:
                        T.copy(
                            a[
                                bid,
                                next_base : next_base + chunk_size,
                                hid,
                                0:chunk_size,
                            ],
                            a_fragment,
                        )
                        for it, jt in T.Parallel(chunk_size, chunk_size):
                            a_fragment[it, jt] = (
                                a_fragment[it, jt]
                                * T.cast(beta_shared[jt], "float32")
                                * T.cast(it >= jt, "float32")
                            )
                        T.copy(a_fragment, a_p_shared)
                        T.ptx_wait_group(0)
                    # B6 (G0): publish all inputs and A-beta[n+1].
                    T.sync_warp()

            elif tx < _VALUE_BASE:
                state_fragment = T.alloc_fragment(
                    (block_v, dim_k), "float32"
                )
                u_fragment = T.alloc_fragment(
                    (chunk_size, block_v), "float32"
                )
                vd_fragment = T.alloc_fragment(
                    (chunk_size, block_v), "float32"
                )
                k_u_fragment = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )
                state_u_fragment = T.alloc_fragment(
                    (block_v, dim_k), qkv_dtype
                )
                vn_state_fragment = T.alloc_fragment(
                    (chunk_size, block_v), qkv_dtype
                )
                k_state_fragment = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )

                if use_initial_state:
                    T.copy(
                        initial_state[
                            bid,
                            hid,
                            v_start : v_start + block_v,
                            0:dim_k,
                        ],
                        state_fragment,
                    )
                else:
                    T.clear(state_fragment)
                T.copy(state_fragment, state_shared)
                # BI (G1): publish H[0] and wait for initial inputs.
                T.sync_warp()

                # Preload the first K/H operands.  Their LDS reads can remain
                # outstanding across B0 and are consumed after the barrier.
                T.copy(k_shared, k_u_fragment)
                T.copy(state_shared, state_u_fragment)
                # B0 (G1): K/H[0] preload has been issued.
                T.sync_warp()

                for cid in T.serial(num_chunks):
                    T.s_waitcnt(0, "lgkmcnt")
                    T.sched_barrier()
                    # U[n] = K[n] @ H[n]^T: K is GEMM A/AT.
                    T.gemm(
                        k_u_fragment,
                        state_u_fragment,
                        u_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.sched_barrier()
                    # BX (G1): K@H[n] is complete.
                    T.sync_warp()

                    decay = T.exp2(
                        T.cast(g_shared[chunk_size - 1], "float32") * _LOG2E
                    )
                    for iv, ik in T.Parallel(block_v, dim_k):
                        state_fragment[iv, ik] *= decay
                    for it, iv in T.Parallel(chunk_size, block_v):
                        u_fragment[it, iv] = (
                            T.cast(raw_v_shared[it, iv], "float32")
                            - u_fragment[it, iv]
                            * T.exp2(
                                T.cast(g_shared[it], "float32") * _LOG2E
                            )
                        )
                    T.copy(u_fragment, w_vn_shared)
                    # B1 (G1): publish W[n].
                    T.sync_warp()

                    T.sched_barrier()
                    T.gemm(
                        a_p_shared,
                        w_vn_shared,
                        vd_fragment,
                        clear_accum=True,
                    )
                    T.sched_barrier()
                    for it, iv in T.Parallel(chunk_size, block_v):
                        value = vd_fragment[it, iv]
                        vd_shared[it, iv] = value
                        diff = T.cast(
                            g_shared[chunk_size - 1], "float32"
                        ) - T.cast(g_shared[it], "float32")
                        safe_diff = T.min(diff, T.float32(0.0))
                        w_vn_shared[it, iv] = T.cast(
                            value
                            * T.exp2(safe_diff * _LOG2E)
                            * T.cast(diff <= 0, "float32"),
                            qkv_dtype,
                        )
                    # B2 (G1): publish Vd/Vn[n].
                    T.sync_warp()

                    T.copy(w_vn_shared, vn_state_fragment)
                    T.copy(
                        k_shared,
                        k_state_fragment,
                    )
                    T.s_waitcnt(0, "lgkmcnt")
                    T.sched_barrier()
                    # H[n+1] += Vn[n]^T @ K[n]: K is GEMM B/BT.
                    T.gemm(
                        vn_state_fragment,
                        k_state_fragment,
                        state_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.sched_barrier()
                    T.copy(state_fragment, state_shared)
                    # B3 (G1): publish H[n+1].
                    T.sync_warp()

                    # Preload H[n+1] while G0 fetches K/g[n+1].
                    if cid + 1 < num_chunks:
                        T.copy(state_shared, state_u_fragment)
                    # B5a (G1): wait for beta/Q/V/K[n+1].
                    T.sync_warp()

                    # Preload K[n+1] while G0 prepares A-beta[n+1].
                    if cid + 1 < num_chunks:
                        T.copy(
                            k_shared,
                            k_u_fragment,
                        )
                    # B6 (G1): K/H[n+1] preload has been issued.
                    T.sync_warp()

                if store_final_state:
                    T.copy(
                        state_fragment,
                        final_state[
                            bid,
                            hid,
                            v_start : v_start + block_v,
                            0:dim_k,
                        ],
                    )

            else:
                p_fragment = T.alloc_fragment(
                    (chunk_size, chunk_size), "float32"
                )
                o_fragment = T.alloc_fragment(
                    (chunk_size, block_v), "float32"
                )
                state_o_fragment = T.alloc_fragment(
                    (block_v, dim_k), qkv_dtype
                )
                q_p_fragment = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )
                k_p_fragment = T.alloc_fragment(
                    (chunk_size, dim_k), qkv_dtype
                )

                # BI (G2): wait for initial inputs and H[0].
                T.sync_warp()
                # B0 (G2): G1 preloads K/H[0]; G2 starts after the barrier.
                T.sync_warp()

                for cid in T.serial(num_chunks):
                    base = cid * chunk_size
                    # Temporarily prioritize G2 so its LDS-to-VGPR preload is
                    # not starved by G1's concurrent MMAC issue.
                    T.call_extern("tl::promote_prio", dtype="void")
                    T.copy(state_shared, state_o_fragment)
                    T.copy(
                        q_shared,
                        q_p_fragment,
                    )
                    T.copy(
                        k_shared,
                        k_p_fragment,
                    )
                    T.call_extern("tl::restore_prio", dtype="void")
                    T.sched_barrier()
                    # BX (G2): Q/K/H[n] fragment prefetch has been issued.
                    T.sync_warp()

                    T.s_waitcnt(0, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(
                        q_p_fragment,
                        state_o_fragment,
                        o_fragment,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.Square,
                        clear_accum=True,
                    )
                    # P[n] = Q[n] @ K[n]^T: Q is A/AT; K is B/BN.
                    T.gemm(
                        q_p_fragment,
                        k_p_fragment,
                        p_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.sched_barrier()
                    # B1 (G2): O-state/P-raw[n] are ready in VGPRs.
                    T.sync_warp()

                    for it, jt in T.Parallel(chunk_size, chunk_size):
                        diff = T.cast(g_shared[it], "float32") - T.cast(
                            g_shared[jt], "float32"
                        )
                        safe_diff = T.min(diff, T.float32(0.0))
                        p_fragment[it, jt] = (
                            p_fragment[it, jt]
                            * scale
                            * T.exp2(safe_diff * _LOG2E)
                            * T.cast(it >= jt, "float32")
                            * T.cast(diff <= 0, "float32")
                        )
                    # B2 (G2): keep gated P[n] in VGPR until A[n] is dead.
                    T.sync_warp()

                    for it, iv in T.Parallel(chunk_size, block_v):
                        o_fragment[it, iv] *= scale * T.exp2(
                            T.cast(g_shared[it], "float32") * _LOG2E
                        )
                    T.copy(p_fragment, a_p_shared)
                    # B3 (G2): publish gated P[n].
                    T.sync_warp()

                    T.sched_barrier()
                    T.gemm(
                        a_p_shared,
                        vd_shared,
                        o_fragment,
                        policy=T.GemmWarpPolicy.Square,
                        clear_accum=False,
                    )
                    T.sched_barrier()
                    T.copy(
                        o_fragment,
                        output[
                            bid,
                            base : base + chunk_size,
                            hid,
                            v_start : v_start + block_v,
                        ],
                    )
                    # B5a (G2): O[n] has been stored.
                    T.sync_warp()

                    # B6 (G2): wait for Q/A-beta[n+1], then advance.
                    T.sync_warp()

    return fused_chunk_gdn_fwd_kernel


__all__ = ["_build_fused_chunk_delta_rule_fwd"]
