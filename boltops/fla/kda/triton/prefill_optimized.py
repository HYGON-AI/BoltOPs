"""Fixed-schedule Kimi-K3 KDA prefill implementation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...common.triton.math import exp2


RCP_LN2 = 1.4426950408889634
K1_CHUNK = 16
# The recurrence carries a [BV, 64] state tile across chunks.  Keep the time
# loop software-pipelined so loads for the next K1 record can overlap the
# current record's matrix work without unrolling multiple state transitions.
K2_PIPE_STAGES = 2
D = 128


def can_run_prefill_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    lower_bound: float | None,
    cu_seqlens: torch.Tensor | None = None,
) -> bool:
    """Whether inputs fit the fixed K1/K2 optimized prefill schedule."""
    if cu_seqlens is not None or lower_bound is None or q.ndim != 4:
        return False
    B, T, H, K = q.shape
    if B != 1 or K != D or not (K1_CHUNK <= T <= 16384) or T % K1_CHUNK:
        return False
    if H <= 0 or k.shape != q.shape or v.shape != q.shape or g.shape != q.shape:
        return False
    if beta.shape != (B, T, H):
        return False
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if g.dtype != q.dtype or beta.dtype != q.dtype:
        return False
    return all(x.is_contiguous() for x in (q, k, v, g, beta))


def _get_prepare_launch_config(H: int, T: int) -> dict[str, int]:
    # The kernel is launched as grid ``(H, nt)`` so that concurrently scheduled
    # workgroups walk contiguous [K1_CHUNK, H*DIM] rows of q/k/g.  With that
    # order the kernel is purely bandwidth-bound and a single wavefront per
    # program wins at every measured (T, H): widening splits the 16-row cumsum
    # and l2norm reductions across wavefronts and costs 25-30%.
    config = {"num_warps": 1, "num_stages": 1}
    if torch.version.hip is not None:
        config["waves_per_eu"] = 1
    return config


def _get_recurrence_launch_config(
    H: int, T: int
) -> dict[str, int | bool | str]:
    # BV picks how many programs split the V axis: 128/BV per head.  It has
    # to be ranked end to end, not on an isolated kernel sweep: replayed on
    # its own the recurrence reads a workspace that is already warm, but in
    # the real pipeline it starts against an L2 full of the lines K1 just
    # wrote, and BV=16 re-reads that workspace 8x where BV=32 re-reads it 4x.
    # The isolated sweep therefore inverts the ranking at H=32 (it preferred
    # BV=16 at 3.06 vs 3.78 ms; in situ BV=16 is the slower one, 4.13 vs
    # 3.79).  Measured in situ at T=16384, BV=16 vs BV=32 in ms: H=8
    # 2.97/3.64, H=16 3.00/3.70, H=32 4.13/3.79, H=64 7.52/5.63 -- so the
    # crossover sits between H=16 and H=32, where 128/BV*H first covers the
    # 72 CUs well enough that halving the workspace traffic beats adding
    # programs.  BW1100 prefers one warp for this dot-heavy recurrence; two
    # warps increase VGPR pressure without adding useful work.  The time loop
    # keeps a two-record software pipeline at every T and every BV in this
    # policy -- dropping to one stage cost 20-25% (H=64/T=1024: 0.311 vs
    # 0.260 ms) because the stall it removes is the loop-carried state
    # dependency, not launch buffer pressure -- while the HCU launch buffer
    # stays at one stage to avoid duplicating the large state tile in VGPRs.
    bv = 16 if H <= 24 else 32
    config: dict[str, int | bool | str] = {
        "BV": bv,
        "PIPE_STAGES": K2_PIPE_STAGES,
        "num_warps": 1,
        "num_stages": 1,
    }
    if torch.version.hip is not None:
        config["amdgpu_enable_max_ilp_scheduling_strategy"] = True
    return config


def _get_operator_bc(H: int, bv: int) -> int:
    """Column tile for the segment-operator prepass.

    The prepass has twice the main pass's programs (M and Delta halves) but
    re-reads the same kd/kr/inv workspace once per column tile, so it can
    trade some of that surplus parallelism for fewer redundant reads.  A tile
    of 32 wins everywhere measured and does not track ``BV``: at T=16384/H=8
    the totals over BC=16/32/64/128 are 2.09/1.92/2.38/5.02 ms, and the same
    ordering holds at H=16 and H=32.  Wider than 32 collapses -- BC=128 leaves
    one program per half per segment-head and runs 2.5x slower than BC=32.
    """
    return 32


def _resolve_segments(segments: int | None, H: int, T: int, nt: int) -> int:
    """How many time segments the context-parallel path should cut ``T`` into.

    Segmenting replaces one sequential sweep with a prepass that summarizes
    each segment as an affine map, a short scan over those maps, and a main
    pass over all segments at once.  That is roughly three times the work for
    a critical path SEG times shorter, so it only pays where the unsegmented
    grid leaves the machine idle.  The grid is ``128/BV * H`` programs, so the
    question is entirely about ``H``: BW1100 has 72 CUs, and H=8 at BV=16
    offers 64 programs -- under one wave, with the whole machine waiting on a
    single dependency chain.

    Measured end to end on BW1100 (best segment count against SEG=1, ms):

        H=4    T=8192  1.51 -> 0.61 (2.5x)   T=16384  3.00 -> 1.08 (2.8x)
        H=8    T=8192  1.56 -> 1.01 (1.5x)   T=16384  3.09 -> 1.85 (1.7x)
        H=12   T=8192  1.67 -> 1.41 (1.2x)   T=16384  3.31 -> 2.63 (1.3x)
        H=16   T=8192  1.70 -> 1.82 (loss)   T=16384  3.38 -> 3.45 (loss)
        H=64   T=8192  3.59 -> 5.74 (loss)   T=16384  7.16 -> 11.7 (loss)

    By H=16 the plain grid already covers the CUs and the prepass is pure
    added work, so the policy stops at H=12.  Within the winning range the
    optimum sits at about 512 tokens per segment at every H and T measured,
    and SEG=2 always loses -- it buys too little parallelism to cover the
    prepass -- so the path needs at least T=2048 to engage.
    """
    if segments is not None:
        if segments < 1:
            raise ValueError("segments must be >= 1")
        if nt % segments:
            raise ValueError(
                f"segments={segments} must divide the {nt} K1 chunks of T={T}"
            )
        return segments
    if H > 12 or T < 2048:
        return 1
    seg = min(T // 512, 32)
    return seg if seg > 1 and nt % seg == 0 else 1


@triton.jit
def _prefill_optimized_prepare_kernel(
    q,
    k,
    g,
    beta,
    beta_out,
    A_log,
    g_bias,
    kd,
    qd,
    kr,
    gt,
    inv,
    mqk,
    scale: tl.constexpr,
    lower_bound: tl.constexpr,
    LOG2E: tl.constexpr,
    K1_CHUNK: tl.constexpr,
    DIM: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    FUSE_GATE: tl.constexpr,
    HAS_G_BIAS: tl.constexpr,
    NORMALIZE_QK: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    STORE_BETA: tl.constexpr,
):
    # Head is the fast-varying grid axis: co-scheduled workgroups then cover
    # one contiguous [K1_CHUNK, H*DIM] slab of q/k/g instead of striding
    # H*DIM*2 bytes apart, which is worth 5x at H=64.
    i_h, i_t = tl.program_id(0), tl.program_id(1)
    tile = i_t * H + i_h

    p_q = tl.make_block_ptr(
        q + i_h * DIM, (T, DIM), (H * DIM, 1),
        (i_t * K1_CHUNK, 0), (K1_CHUNK, DIM), (1, 0),
    )
    p_k = tl.make_block_ptr(
        k + i_h * DIM, (T, DIM), (H * DIM, 1),
        (i_t * K1_CHUNK, 0), (K1_CHUNK, DIM), (1, 0),
    )
    p_g = tl.make_block_ptr(
        g + i_h * DIM, (T, DIM), (H * DIM, 1),
        (i_t * K1_CHUNK, 0), (K1_CHUNK, DIM), (1, 0),
    )
    b_q = tl.load(p_q)
    b_k = tl.load(p_k)
    b_g = tl.load(p_g).to(tl.float32)

    if NORMALIZE_QK:
        b_qf = b_q.to(tl.float32)
        b_kf = b_k.to(tl.float32)
        b_q = (
            b_qf * tl.rsqrt(tl.sum(b_qf * b_qf, axis=1)[:, None] + 1e-6)
        ).to(tl.bfloat16)
        b_k = (
            b_kf * tl.rsqrt(tl.sum(b_kf * b_kf, axis=1)[:, None] + 1e-6)
        ).to(tl.bfloat16)

    if FUSE_GATE:
        o_d = tl.arange(0, DIM)
        if HAS_G_BIAS:
            b_g += tl.load(g_bias + i_h * DIM + o_d)[None, :]
        b_a = tl.exp(tl.load(A_log + i_h).to(tl.float32))
        b_g = lower_bound * tl.sigmoid(b_a * b_g)

    b_gc = tl.cumsum(b_g * LOG2E, axis=0)
    b_decay = exp2(b_gc)
    chunk_rows = tl.arange(0, K1_CHUNK)
    b_total = exp2(
        tl.sum(tl.where(chunk_rows[:, None] == K1_CHUNK - 1, b_gc, 0.0), axis=0)
    )
    b_qd = (b_q * b_decay * scale).to(tl.bfloat16)
    b_kd = (b_k * b_decay).to(tl.bfloat16)
    b_ki = (b_k * exp2(-b_gc)).to(tl.bfloat16)
    b_kr = (b_ki * b_total[None, :]).to(tl.bfloat16)

    o_t = tl.arange(0, K1_CHUNK)
    b_beta = tl.load(beta + (i_t * K1_CHUNK + o_t) * H + i_h).to(tl.float32)
    if APPLY_BETA_SIGMOID:
        b_beta = tl.sigmoid(b_beta)
    if STORE_BETA:
        tl.store(beta_out + (i_t * K1_CHUNK + o_t) * H + i_h, b_beta)

    rows = chunk_rows
    strict_lower = rows[:, None] > rows[None, :]
    lower_equal = rows[:, None] >= rows[None, :]
    b_l = tl.dot(b_kd, tl.trans(b_ki), input_precision="ieee")
    b_a = tl.where(strict_lower, b_l * b_beta[:, None], 0.0)
    identity = (rows[:, None] == rows[None, :]).to(tl.float32)
    b_a2 = tl.dot(b_a, b_a, input_precision="ieee", out_dtype=tl.float32)
    b_x = tl.dot(identity + b_a, identity + b_a2,
              input_precision="ieee", out_dtype=tl.float32)
    b_a4 = tl.dot(b_a2, b_a2, input_precision="ieee", out_dtype=tl.float32)
    b_x = tl.dot(b_x, identity + b_a4,
                input_precision="ieee", out_dtype=tl.float32)
    b_a8 = tl.dot(b_a4, b_a4, input_precision="ieee", out_dtype=tl.float32)
    b_x = tl.dot(b_x, identity + b_a8,
                input_precision="ieee", out_dtype=tl.float32)
    # ``b_a`` is strictly lower triangular with K1_CHUNK=16, so b_a**16 is
    # exactly zero.  The Neumann series is complete after the A^8 step.
    b_mqk = tl.where(
        lower_equal,
        tl.dot(b_qd, tl.trans(b_ki), input_precision="ieee"),
        0.0,
    )

    # Folding ``b_x`` into ``mqk``/``kr`` here is algebraically exact and does
    # shorten K2's loop-carried chain by one dot, but it was measured slower on
    # BW1100 at every H<=32 (16384/H=32: 3.06 -> 3.52 ms best-config to
    # best-config): producing ``b_u`` from VALU ops instead of an MFMA forces a
    # real transpose before it can be an MFMA A-operand.  Keep the solve in K2.

    tile_kd = tile * K1_CHUNK * DIM
    tile_lm = tile * K1_CHUNK * K1_CHUNK
    p_kd = tl.make_block_ptr(
        kd + tile_kd, (K1_CHUNK, DIM), (DIM, 1), (0, 0),
        (K1_CHUNK, DIM), (1, 0),
    )
    p_qd = tl.make_block_ptr(
        qd + tile_kd, (K1_CHUNK, DIM), (DIM, 1), (0, 0),
        (K1_CHUNK, DIM), (1, 0),
    )
    p_kr = tl.make_block_ptr(
        kr + tile_kd, (K1_CHUNK, DIM), (DIM, 1), (0, 0),
        (K1_CHUNK, DIM), (1, 0),
    )
    p_inv = tl.make_block_ptr(
        inv + tile_lm, (K1_CHUNK, K1_CHUNK), (K1_CHUNK, 1), (0, 0),
        (K1_CHUNK, K1_CHUNK), (1, 0),
    )
    p_mqk = tl.make_block_ptr(
        mqk + tile_lm, (K1_CHUNK, K1_CHUNK), (K1_CHUNK, 1), (0, 0),
        (K1_CHUNK, K1_CHUNK), (1, 0),
    )
    tl.store(p_kd, b_kd)
    tl.store(p_qd, b_qd)
    tl.store(p_kr, b_kr)
    tl.store(gt + tile * DIM + tl.arange(0, DIM), b_total)
    tl.store(p_inv, b_x.to(tl.bfloat16))
    tl.store(p_mqk, b_mqk.to(tl.bfloat16))


@triton.jit
def _prefill_recurrence_step(
    v,
    beta_active,
    kd,
    qd,
    kr,
    gt,
    inv,
    mqk,
    out,
    b_s,
    i_t,
    i_h,
    v0,
    K1_CHUNK: tl.constexpr,
    DIM: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    BV: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
):
    tile = i_t * H + i_h
    tile_kd = tile * K1_CHUNK * DIM
    tile_lm = tile * K1_CHUNK * K1_CHUNK

    p_kd = tl.make_block_ptr(
        kd + tile_kd, (K1_CHUNK, DIM), (DIM, 1), (0, 0),
        (K1_CHUNK, DIM), (1, 0),
    )
    p_qd = tl.make_block_ptr(
        qd + tile_kd, (K1_CHUNK, DIM), (DIM, 1), (0, 0),
        (K1_CHUNK, DIM), (1, 0),
    )
    b_kd = tl.load(p_kd, eviction_policy="evict_last")
    b_qd = tl.load(p_qd, eviction_policy="evict_last")

    b_s_t = tl.trans(b_s.to(tl.bfloat16))
    # The state is carried as one [BV, DIM] tile rather than two [BV, 64] halves.
    # Same FLOPs and the same fp32 accumulator, but one K=DIM dot replaces each
    # pair of K=64 dots, and the gate, the workspace loads and the state
    # load/store all halve in count: 17.6% at 8192/H=16, 13.0% at 1024/H=8,
    # 7.8% at 16384/H=64.  Merging the two dots also shortens the accumulation
    # chain, so the relative error against the fp32 oracle is unchanged or
    # slightly better -- never worse.
    #
    # The chunk gate is not needed until the state update at the bottom, but it
    # is issued here on purpose.  Left next to its use it lands after the whole
    # b_u chain (the dot, the beta scaling and the UT solve) and the loop carry
    # then waits on the load rather than on arithmetic.  Hoisting the load is
    # worth 4.8% at 8192/H=16 and 8.4% at 1024/H=8; hoisting the multiply with
    # it scores the same, which is how we know it is the load that stalls.
    o_k = tl.arange(0, DIM)
    b_gt = tl.load(gt + tile * DIM + o_k, eviction_policy="evict_last")
    b_u = tl.dot(b_kd, b_s_t)
    b_o = tl.dot(b_qd, b_s_t)

    p_v = tl.make_block_ptr(
        v + i_h * DIM, (T, DIM), (H * DIM, 1), (i_t * K1_CHUNK, v0),
        (K1_CHUNK, BV), (1, 0),
    )
    b_v = tl.load(p_v, eviction_policy="evict_first")
    rows = tl.arange(0, K1_CHUNK)
    b_beta = tl.load(
        beta_active + (i_t * K1_CHUNK + rows) * H + i_h,
        eviction_policy="evict_first",
    ).to(tl.float32)
    if APPLY_BETA_SIGMOID:
        b_beta = tl.sigmoid(b_beta)
    # Deliberately NOT symmetric with the fp32 output accumulation below: doing
    # this subtraction in fp32 instead (``b_v.to(tl.float32) - b_u``) keeps the
    # solve's operand in registers a stage longer and was measured *slower* at
    # small head counts -- 8192/H=16 -3.1%, 1024/H=8 -5.2% -- while only helping
    # at H=64.  The bf16 round here is on the critical path into the UT solve.
    b_u = ((b_v - b_u.to(b_v.dtype)) * b_beta[:, None]).to(tl.bfloat16)

    p_inv = tl.make_block_ptr(
        inv + tile_lm, (K1_CHUNK, K1_CHUNK), (K1_CHUNK, 1), (0, 0),
        (K1_CHUNK, K1_CHUNK), (1, 0),
    )
    b_u = tl.dot(
        tl.load(p_inv, eviction_policy="evict_last"), b_u
    ).to(tl.bfloat16)

    p_mqk = tl.make_block_ptr(
        mqk + tile_lm, (K1_CHUNK, K1_CHUNK), (K1_CHUNK, 1), (0, 0),
        (K1_CHUNK, K1_CHUNK), (1, 0),
    )
    # Both terms of the output stay in the fp32 accumulator and round once, at
    # the store.  Rounding each addend to bf16 first (the earlier spelling) cost
    # two extra [K1_CHUNK, BV] converts per chunk for no accuracy: measured
    # 8192/H=16 1.694 -> 1.670 ms, 16384/H=64 6.710 -> 6.355 ms, with the
    # relative error against the fp32 oracle unchanged to four digits.
    b_o = b_o + tl.dot(
        tl.load(p_mqk, eviction_policy="evict_last"), b_u
    )

    p_out = tl.make_block_ptr(
        out + i_h * DIM, (T, DIM), (H * DIM, 1), (i_t * K1_CHUNK, v0),
        (K1_CHUNK, BV), (1, 0),
    )
    tl.store(p_out, b_o.to(out.dtype.element_ty), eviction_policy="evict_first")

    p_kr = tl.make_block_ptr(
        kr + tile_kd, (K1_CHUNK, DIM), (DIM, 1), (0, 0),
        (K1_CHUNK, DIM), (1, 0),
    )
    b_u_t = tl.trans(b_u)
    b_s = (
        b_s * b_gt[None, :]
        + tl.dot(b_u_t, tl.load(p_kr, eviction_policy="evict_last"))
    )
    return b_s


@triton.jit
def _prefill_segment_operator_kernel(
    v,
    beta_active,
    kd,
    kr,
    gt,
    inv,
    op,
    NT,
    SEG,
    K1_CHUNK: tl.constexpr,
    DIM: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    BC: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
):
    """Summarize one time segment as the affine map it applies to the state.

    A chunk sends the state ``S`` in R^[K,V] to ``M_c S + d_c`` with
    ``M_c = (I - kr^T inv diag(beta) kd) diag(g)``, so a whole segment is
    ``F_s(S) = M_s S + Delta_s``.  ``M`` is only [K,K] -- the map acts the same
    way on every V column -- which is what makes the time axis parallelizable.

    Both halves obey the identical per-chunk update on a right-hand block
    ``R`` in R^[K,C]::

        P = diag(g) R;  Y = kd P;  Z = inv (beta .* (Vterm - Y));  R' = P + kr^T Z

    so one kernel computes both: columns [0, DIM) carry ``M`` (``Vterm`` = 0,
    init identity) and columns [DIM, 2*DIM) carry ``Delta`` (``Vterm`` = v,
    init zero).  This is ``_prefill_recurrence_step`` with the two output dots
    dropped, which is why the prepass costs about a third more than the main
    pass rather than double.

    ``op`` is [2, SEG, H, DIM, DIM] fp32, part 0 holding ``M`` transposed as
    ``op[0][c, k] = M[k, c]`` and part 1 holding ``Delta`` in the same [v, k]
    layout the recurrence already uses for its state.
    """
    i_c, i_s, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_col: tl.constexpr = DIM // BC
    is_delta = i_c >= n_col
    c0 = (i_c % n_col) * BC

    o_k = tl.arange(0, 64)
    o_c = c0 + tl.arange(0, BC)
    if is_delta:
        b_r1 = tl.zeros((BC, 64), dtype=tl.bfloat16)
        b_r2 = tl.zeros((BC, 64), dtype=tl.bfloat16)
    else:
        b_r1 = (o_c[:, None] == o_k[None, :]).to(tl.bfloat16)
        b_r2 = (o_c[:, None] == (o_k + 64)[None, :]).to(tl.bfloat16)

    o_t = tl.arange(0, K1_CHUNK)
    for i_t in tl.range(i_s * NT, i_s * NT + NT, num_stages=PIPE_STAGES):
        tile = i_t * H + i_h
        tile_kd = tile * K1_CHUNK * DIM
        tile_lm = tile * K1_CHUNK * K1_CHUNK

        # ``kd`` already carries the within-chunk gate, so the solve reads the
        # *undecayed* block and the decay lands only on the carry below --
        # exactly the order ``_prefill_recurrence_step`` uses.
        p_kd1 = tl.make_block_ptr(
            kd + tile_kd, (K1_CHUNK, DIM), (DIM, 1), (0, 0),
            (K1_CHUNK, 64), (1, 0),
        )
        p_kd2 = tl.make_block_ptr(
            kd + tile_kd, (K1_CHUNK, DIM), (DIM, 1), (0, 64),
            (K1_CHUNK, 64), (1, 0),
        )
        b_y = tl.dot(
            tl.load(p_kd1, eviction_policy="evict_last"), tl.trans(b_r1)
        )
        b_y += tl.dot(
            tl.load(p_kd2, eviction_policy="evict_last"), tl.trans(b_r2)
        )

        b_beta = tl.load(
            beta_active + (i_t * K1_CHUNK + o_t) * H + i_h,
            eviction_policy="evict_first",
        ).to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
        # The two halves differ only in this term: Delta carries v through the
        # solve, M has no source and only propagates the identity it started
        # from.  Triton needs one dtype across the branch, hence ``b_z``.
        b_z = tl.zeros((K1_CHUNK, BC), dtype=tl.bfloat16)
        if is_delta:
            p_v = tl.make_block_ptr(
                v + i_h * DIM, (T, DIM), (H * DIM, 1), (i_t * K1_CHUNK, c0),
                (K1_CHUNK, BC), (1, 0),
            )
            b_v = tl.load(p_v, eviction_policy="evict_first")
            b_z = ((b_v - b_y.to(b_v.dtype)) * b_beta[:, None]).to(tl.bfloat16)
        else:
            b_z = (-b_y.to(tl.bfloat16) * b_beta[:, None]).to(tl.bfloat16)

        p_inv = tl.make_block_ptr(
            inv + tile_lm, (K1_CHUNK, K1_CHUNK), (K1_CHUNK, 1), (0, 0),
            (K1_CHUNK, K1_CHUNK), (1, 0),
        )
        b_z = tl.dot(
            tl.load(p_inv, eviction_policy="evict_last"), b_z
        ).to(tl.bfloat16)

        p_kr1 = tl.make_block_ptr(
            kr + tile_kd, (K1_CHUNK, DIM), (DIM, 1), (0, 0),
            (K1_CHUNK, 64), (1, 0),
        )
        p_kr2 = tl.make_block_ptr(
            kr + tile_kd, (K1_CHUNK, DIM), (DIM, 1), (0, 64),
            (K1_CHUNK, 64), (1, 0),
        )
        b_gt1 = tl.load(gt + tile * DIM + o_k, eviction_policy="evict_last")
        b_gt2 = tl.load(gt + tile * DIM + 64 + o_k, eviction_policy="evict_last")
        b_y_t = tl.trans(b_z)
        b_r1 = (
            b_r1.to(tl.float32) * b_gt1[None, :]
            + tl.dot(b_y_t, tl.load(p_kr1, eviction_policy="evict_last"))
        ).to(tl.bfloat16)
        b_r2 = (
            b_r2.to(tl.float32) * b_gt2[None, :]
            + tl.dot(b_y_t, tl.load(p_kr2, eviction_policy="evict_last"))
        ).to(tl.bfloat16)

    part = i_c // n_col
    base = ((part * SEG + i_s) * H + i_h) * DIM * DIM
    p_o1 = tl.make_block_ptr(
        op + base, (DIM, DIM), (DIM, 1), (c0, 0), (BC, 64), (1, 0)
    )
    p_o2 = tl.make_block_ptr(
        op + base, (DIM, DIM), (DIM, 1), (c0, 64), (BC, 64), (1, 0)
    )
    tl.store(p_o1, b_r1.to(p_o1.dtype.element_ty))
    tl.store(p_o2, b_r2.to(p_o2.dtype.element_ty))


@triton.heuristics(
    {"USE_INITIAL_STATE": lambda args: args["initial_state"] is not None}
)
@triton.jit
def _prefill_segment_scan_kernel(
    op,
    initial_state,
    pool,
    SEG,
    DIM: tl.constexpr,
    H: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    """Turn the per-segment affine maps into per-segment entry states.

    ``S_{s+1} = M_s S_s + Delta_s``, which in the stored [v, k] layout is just
    ``cur = cur @ op[0][s] + op[1][s]``.  This is the only sequential part left.

    The product runs in BF16 with an FP32 accumulator, not in IEEE FP32.  That
    is not a precision concession: ``M`` was produced by a BF16 carry in the
    prepass, and the recurrence rounds whatever entry state it is handed to
    BF16 on load, so an IEEE FP32 multiply here only pays for precision that
    is discarded on both sides.  It is expensive to pay for -- at T=16384/H=8
    with 32 segments the IEEE version of this loop cost about 0.8 ms, roughly
    as much as the prepass it follows.  The carry stays FP32 so ``Delta``
    accumulates at full width.
    """
    i_v, i_h = tl.program_id(0), tl.program_id(1)
    v0 = i_v * BV

    if USE_INITIAL_STATE:
        p_i = tl.make_block_ptr(
            initial_state + i_h * DIM * DIM, (DIM, DIM), (DIM, 1),
            (v0, 0), (BV, DIM), (1, 0),
        )
        b_cur = tl.load(p_i).to(tl.float32)
    else:
        b_cur = tl.zeros((BV, DIM), dtype=tl.float32)

    for i_s in range(SEG):
        p_pool = tl.make_block_ptr(
            pool + (i_s * H + i_h) * DIM * DIM, (DIM, DIM), (DIM, 1),
            (v0, 0), (BV, DIM), (1, 0),
        )
        tl.store(p_pool, b_cur.to(p_pool.dtype.element_ty))
        p_m = tl.make_block_ptr(
            op + (i_s * H + i_h) * DIM * DIM, (DIM, DIM), (DIM, 1),
            (0, 0), (DIM, DIM), (1, 0),
        )
        p_d = tl.make_block_ptr(
            op + ((SEG + i_s) * H + i_h) * DIM * DIM, (DIM, DIM), (DIM, 1),
            (v0, 0), (BV, DIM), (1, 0),
        )
        b_cur = tl.dot(
            b_cur.to(tl.bfloat16),
            tl.load(p_m).to(tl.bfloat16),
            out_dtype=tl.float32,
        ) + tl.load(p_d)


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "STORE_FINAL_STATE": lambda args: args["final_state"] is not None,
    }
)
@triton.jit
def _prefill_optimized_recurrence_kernel(
    v,
    beta_active,
    kd,
    qd,
    kr,
    gt,
    inv,
    mqk,
    initial_state,
    final_state,
    out,
    NT,
    K1_CHUNK: tl.constexpr,
    DIM: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    BV: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    SEGMENTS: tl.constexpr,
):
    # ``i_s`` selects one context-parallel segment of the time axis.  With
    # SEGMENTS=1 the grid is two-dimensional as before, ``i_s`` is zero and
    # every index below reduces to the single-segment form.
    i_v, i_h = tl.program_id(0), tl.program_id(1)
    i_s = tl.program_id(2)
    v0 = i_v * BV

    if USE_INITIAL_STATE:
        # The entry-state pool is [SEGMENTS, H, DIM, DIM]; at SEGMENTS=1 this
        # is the caller's [1, H, DIM, DIM] initial state unchanged.
        state_base = initial_state + (i_s * H + i_h) * DIM * DIM
        p_s = tl.make_block_ptr(
            state_base, (DIM, DIM), (DIM, 1), (v0, 0), (BV, DIM), (1, 0)
        )
        b_s = tl.load(p_s).to(tl.float32)
    else:
        b_s = tl.zeros((BV, DIM), dtype=tl.float32)

    for i_t in tl.range(i_s * NT, i_s * NT + NT, num_stages=PIPE_STAGES):
        b_s = _prefill_recurrence_step(
            v, beta_active, kd, qd, kr, gt, inv, mqk, out,
            b_s, i_t, i_h, v0,
            K1_CHUNK=K1_CHUNK, DIM=DIM, T=T, H=H, BV=BV,
            APPLY_BETA_SIGMOID=APPLY_BETA_SIGMOID,
        )

    if STORE_FINAL_STATE and i_s == SEGMENTS - 1:
        state_base = final_state + i_h * DIM * DIM
        p_f = tl.make_block_ptr(
            state_base, (DIM, DIM), (DIM, 1), (v0, 0), (BV, DIM), (1, 0)
        )
        tl.store(p_f, b_s.to(p_f.dtype.element_ty))


def prefill_optimized_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    final_state: torch.Tensor | None = None,
    *,
    A_log: torch.Tensor | None = None,
    g_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm: bool = False,
    apply_beta_sigmoid: bool = False,
    segments: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run fixed-K1 optimized Kimi-K3 prefill."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    if B != 1 or K != D or V != D or T % K1_CHUNK:
        raise ValueError(
            "optimized prefill requires B=1, K=V=128 and T divisible by 16"
        )
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("optimized prefill requires bf16 Q/K/V")
    if k.shape != q.shape or v.shape != q.shape or g.shape != q.shape:
        raise ValueError("optimized prefill requires matching Q/K/V/G shapes")
    if beta.shape != (B, T, H):
        raise ValueError("optimized prefill beta shape must be [1,T,H]")
    fuse_gate = A_log is not None
    if fuse_gate and lower_bound is None:
        raise ValueError("optimized fused gate requires lower_bound")
    if initial_state is not None and initial_state.shape != (B, H, D, D):
        raise ValueError("optimized prefill initial state shape must be [1,H,128,128]")
    if final_state is not None:
        if not output_final_state:
            raise ValueError("final_state requires output_final_state=True")
        if (
            final_state.shape != (B, H, D, D)
            or final_state.dtype != torch.float32
            or final_state.device != q.device
            or not final_state.is_contiguous()
        ):
            raise ValueError(
                "optimized prefill final_state must be contiguous fp32 [1,H,128,128]"
            )

    if output_final_state and final_state is None:
        final_state = torch.empty(
            (B, H, D, D), device=q.device, dtype=torch.float32
        )
    nt = T // K1_CHUNK
    kd_qd_kr = torch.empty(
        (3, nt, H, K1_CHUNK, D), device=q.device, dtype=q.dtype
    )
    kd = kd_qd_kr[0]
    qd = kd_qd_kr[1]
    kr = kd_qd_kr[2]
    # Gate totals are consumed as BF16 by the K2 recurrence.  Keeping this
    # workspace in FP32 doubles the state-loop traffic without improving the
    # FP32 accumulator, which already widens the value on load.
    gt = torch.empty((nt, H, D), device=q.device, dtype=q.dtype)
    inv_mqk = torch.empty(
        (2, nt, H, K1_CHUNK, K1_CHUNK), device=q.device, dtype=q.dtype
    )
    inv = inv_mqk[0]
    mqk = inv_mqk[1]
    # Keep raw beta in place.  The recurrence applies sigmoid on load when
    # requested; materializing a second [T,H] tensor only adds a short-path
    # global store/load pair.
    beta_active = beta

    prepare_config = _get_prepare_launch_config(H, T)
    _prefill_optimized_prepare_kernel[(H, nt)](
        q=q,
        k=k,
        g=g,
        beta=beta,
        beta_out=beta_active,
        A_log=A_log,
        g_bias=g_bias,
        kd=kd,
        qd=qd,
        kr=kr,
        gt=gt,
        inv=inv,
        mqk=mqk,
        scale=scale,
        lower_bound=lower_bound or 0.0,
        LOG2E=RCP_LN2,
        K1_CHUNK=K1_CHUNK,
        DIM=D,
        T=T,
        H=H,
        FUSE_GATE=fuse_gate,
        HAS_G_BIAS=g_bias is not None,
        NORMALIZE_QK=use_qk_l2norm,
        APPLY_BETA_SIGMOID=apply_beta_sigmoid,
        STORE_BETA=False,
        **prepare_config,
    )

    out = torch.empty_like(v)
    recurrence_config = _get_recurrence_launch_config(H, T)
    bv = recurrence_config.pop("BV")
    pipe_stages = recurrence_config.pop("PIPE_STAGES")

    seg = _resolve_segments(segments, H, T, nt)
    if seg > 1:
        # Context-parallel prefill.  A chunk maps the state affinely, and the
        # map is only [K, K] because it acts the same way on every V column,
        # so a segment collapses to (M_s, Delta_s).  Summarize every segment
        # in parallel, scan the S maps to recover the segment entry states,
        # then run the ordinary recurrence on all segments at once.
        nt = nt // seg
        bc = _get_operator_bc(H, bv)
        op = torch.empty((2, seg, H, D, D), device=q.device, dtype=torch.float32)
        _prefill_segment_operator_kernel[(2 * (D // bc), seg, H)](
            v=v,
            beta_active=beta_active,
            kd=kd,
            kr=kr,
            gt=gt,
            inv=inv,
            op=op,
            NT=nt,
            SEG=seg,
            K1_CHUNK=K1_CHUNK,
            DIM=D,
            T=T,
            H=H,
            BC=bc,
            PIPE_STAGES=pipe_stages,
            APPLY_BETA_SIGMOID=apply_beta_sigmoid,
            **recurrence_config,
        )
        pool = torch.empty((seg, H, D, D), device=q.device, dtype=torch.float32)
        _prefill_segment_scan_kernel[(D // 32, H)](
            op=op,
            initial_state=initial_state,
            pool=pool,
            SEG=seg,
            DIM=D,
            H=H,
            BV=32,
            num_warps=4,
            num_stages=1,
        )
        initial_state = pool

    _prefill_optimized_recurrence_kernel[(triton.cdiv(D, bv), H, seg)](
        v=v,
        beta_active=beta_active,
        kd=kd,
        qd=qd,
        kr=kr,
        gt=gt,
        inv=inv,
        mqk=mqk,
        initial_state=initial_state,
        final_state=final_state,
        out=out,
        NT=nt,
        K1_CHUNK=K1_CHUNK,
        DIM=D,
        T=T,
        H=H,
        BV=bv,
        PIPE_STAGES=pipe_stages,
        APPLY_BETA_SIGMOID=apply_beta_sigmoid,
        SEGMENTS=seg,
        **recurrence_config,
    )
    return out, final_state


__all__ = [
    "K1_CHUNK",
    "K2_PIPE_STAGES",
    "can_run_prefill_optimized",
    "prefill_optimized_fwd",
]
