# gemm_fp16 WASP/WDRA tune results (gfx946)

Internal `gemm_fp16` lab notes. Each **shape** section records the best
**nowasp** vs **wdra\*** launches found so far. Winner JSON:
`boltops/gemm/triton/configs/gemm_fp16/gfx946/M{M}_N{N}_K{K}_{TT|TN}_{mode}.json`.

Per-scenario WDRA VGPR policy (load / main / tail): [`VGPR.md`](VGPR.md).

## How to read

- **Shape** = problem size `(M, N, K)`.
- **AB layout** (`ab_layout` in winner JSON) — contiguous (“主序”) dim of A/B:
  - **TT**: A is **K-major** (`[M,K]` with `stride_ak=1`); B is **N-major** (`[K,N]` with `stride_bn=1`).
  - **TN**: A is **K-major**; B is **K-major** (`[K,N]` with `stride_bk=1`, e.g. transposed-B style).
- **num_stages**: most recorded datapoints use **`num_stages=2`**. Exceptions
  (e.g. TN 4096×4608×4096 with **`num_stages=4`**) are tagged in Launch / JSON.
- Default `tune_gemm_wasp` is **TT** (`A[M,K]`, `B[K,N]` contiguous). Use
  `--ab-layout TN` for B K-major (`B = randn(N,K).T`). Record the tag on every promote.
- **nowasp** = baseline (`wasp=False`); **wdra48** = OnePTwoC (load/mma = 4/8).
- **`enable_consumer_pingpong` (`pp`)**: offset the two WDRA MMA consumers
  (default true in current harness; no-op unless wdra48/wdra88). Tag `pp=1`
  when recording a pingpong datapoint.
- **`use_matrix_store` (`mstore`)**: C uses `tl.matrix_store` (requires `--mls`
  and `triton-mls-store`; prefer `mlf=3`). Tag `mstore=1` on those datapoints.
- TOPS from PMD `totalGcSimCycles` / `gcFrequency` (`2·M·N·K / (cycles/freq)`).
- **Peak utilization** = measured TOPS / **432 TOPS**, where 432 TOPS is the
  FP16 GEMM peak for Shaobo at 2.2 GHz with 48 CUs.
- Append a new `## Shape: …` block when promoting another MNK; keep newest
  winners in JSON, older notes under that shape’s “History” if needed.

## Index

| Shape (MNK) | AB | Best nowasp TOPS | Best wdra48 TOPS | Speedup | triton | Updated |
| --- | --- | ---: | ---: | ---: | --- | --- |
| [512³](#shape-5123-mnk--512-512-512) `MNK=512,512,512` | TT | **23.771 (5.50%)** | **26.820 (6.21%)** | **1.128×** | `c2defdef3`* | 2026-09-02 |
| [1536³](#shape-15363-mnk--1536-1536-1536) `MNK=1536,1536,1536` | TT | **154.194 (35.69%)** | **223.279 (51.68%)** | **1.448×** | `e4fee6ec5` | 2026-09-09 |
| [3072×3072×128](#shape-3072x3072x128) `MNK=3072,3072,128` | TT | **83.402 (19.31%)** | **100.976 (23.37%)** | **1.211×** | `e4fee6ec5` | 2026-09-02 |
| [3072×3072×256](#shape-3072x3072x256) `MNK=3072,3072,256` | TT | **114.910 (26.60%)** | **164.209 (38.01%)** | **1.429×** | `e4fee6ec5` | 2026-09-02 |
| [3072×3072×512](#shape-3072x3072x512) `MNK=3072,3072,512` | TT | **141.461 (32.75%)** | **229.833 (53.20%)** | **1.625×** | `e4fee6ec5` | 2026-09-02 |
| [3072×3072×1024](#shape-3072x3072x1024) `MNK=3072,3072,1024` | TT | **162.704 (37.66%)** | **295.429 (68.39%)** | **1.816×** | `e4fee6ec5` | 2026-09-02 |
| [3072×3072×2048](#shape-3072x3072x2048) `MNK=3072,3072,2048` | TT | **174.796 (40.46%)** | **344.613 (79.77%)** | **1.972×** | `e4fee6ec5` | 2026-09-02 |
| [4096×4096×4608](#shape-4096x4096x4608) `MNK=4096,4096,4608` | TT | — | 208.517 (48.27%) | — | `e4fee6ec5` | 2026-08-10 |
| [4096×4608×512](#shape-4096x4608x512) `MNK=4096,4608,512` | TT | **137.508 (31.83%)** | **244.311 (56.55%)** | **1.777×** | `45e8a4c1a` | 2026-09-09 |
| [4096×4608×4096](#shape-4096x4608x4096) `MNK=4096,4608,4096` | TT | **158.358 (36.66%)** | **381.346 (88.27%)** | **2.408×** | `f014fb048` | 2026-09-09 |
| [4096×4608×16384](#shape-4096x4608x16384) `MNK=4096,4608,16384` | TT | — | 392.254 (90.80%) | — | `45e8a4c1a` | 2026-08-20 |
| [4096×4608×4096 TN](#shape-4096x4608x4096-tn) `MNK=4096,4608,4096` | TN | — | 377.095 (87.29%) | — | `7225e1068` | 2026-08-14 |

† Speedup from SUMMARY only; nowasp row not kept.
‡ Older TN notes referenced Triton MR tip; current TN best uses consumer-pingpong `7225e1068` + `abarrier_seq` (replaces `s_waitcnt` so MLS can issue continuously).
\* The 512³ nowasp baseline was refreshed at `c2defdef3`; the wdra48 result in
that row remains the independently recorded `d0ccddcc3` result below.

---

## Known PMD pipeline regression: LLVM `043b33c45cfa`

For the `256x256x4096` TT `wdra48` case (`block=256x256x64`,
`num_stages=2`, `mlf=3`, consumer ping-pong, MLS matrix-store interleave 2),
LLVM commit `043b33c45cfa` (`[HCU] Fold s_cselect+cmp+branch when SCC is live
but unread`) causes a reproducible PMD performance regression:

| LLVM revision | cycles | TOPS | correctness |
| --- | ---: | ---: | --- |
| parent `197389b4c641` | 153781 | 7.672833 | GOLDEN PASS |
| `043b33c45cfa` | 157267 | 7.502756 | GOLDEN PASS |

This is **+3486 cycles / +2.27%**. In the PMD pipeline view, one particular
`s_ebarrier_sync` has a visibly longer idle interval immediately **before** it
after the commit, resembling a pipeline discontinuity ("断流"). This corrects
an earlier observation that placed the idle after all barriers and before
`s_setprio`.

The complete ISA diff only folds two role-selection sequences, removing four
SALU instructions in total; MMAC/MLS ordering, barriers, waits, register counts
and static block order are otherwise unchanged. Adding four `s_nop` slots near
the folded branches restores approximately the former performance (153776--
153804 cycles) without restoring the removed compare/select semantics. This
shows sensitivity to instruction-slot/warp-role timing, but does **not** yet
prove whether the longer pre-barrier idle is caused by consumer phase timing,
code layout, or scheduler arbitration. Treat the exact mechanism as open.

PMD artifacts:

- before: `/xukang/code/pmd_perf_before_043b33c4/0/0/stats_xcd0.perf`
- after: `/xukang/code/pmd_perf_after_043b33c4/0/0/stats_xcd0.perf`

---

## Known PMD matrix-store regression: removing address bit exchange

Recorded 2026-08-27 for the `256x256x4096` TT `wdra48` reproducer
(`block=256x256x64`, `num_stages=2`, `mlf=3`, consumer ping-pong, MLS,
`matrix_store_32x16_b16`, matrix-store interleave 2). Both variants pass the
golden check and produce the same non-contiguous N store ownership and final
matrix-store offsets (`0, 32, 128, 160`):

| matrix-store address lowering | cycles | TOPS | vs legacy | correctness |
| --- | ---: | ---: | ---: | --- |
| legacy repetition/warp-N bit exchange | 153779 | 7.672933 | baseline | GOLDEN PASS |
| direct `[group][warp][inner]` non-contiguous mapping (B) | 157167 | 7.507530 | **-2.2%** | GOLDEN PASS |

The direct-B implementation was also measured on the production
`4096x4608x4096` TT shape with the same `256x256x64` launch, once with forced
`TRITON_HCU_MATRIX_STORE_INSN=32x16` and once with 32x32:

| matrix-store address lowering | cycles | TOPS | vs legacy | correctness |
| --- | ---: | ---: | ---: | --- |
| legacy bit exchange | 872743 | 389.372 | baseline | GOLDEN PASS |
| direct B, `matrix_store_32x16_b16` | 889477 | 382.046 | **-1.9%** | GOLDEN PASS |
| direct B, `matrix_store_32x32_b16` | 891132 | 381.337 | **-2.1%** | GOLDEN PASS |

These are two different matrix-store instructions, not repeated runs of one
configuration. Direct-B 32x16 is about **0.19%** faster than direct-B 32x32 on
this shape. Both are about 2% below the legacy 32x16 bit-exchange result, but a
repeatability claim requires additional same-instruction runs.

The regression is therefore not caused by changing the global-memory store
partition: the legacy bit exchange already generated the same physical store
pattern. PMD attribution shows that the direct mapping leaves MMOP runtime and
static matrix-store counts essentially unchanged, but increases synchronization
and memory-side stalls:

| PMD counter | legacy | direct B | change |
| --- | ---: | ---: | ---: |
| `waitLgkm` | 933 | 1458 | +56% |
| barrier stall | 4822 | 6616 | +37% |
| flat stall | 1286 | 1979 | +54% |

Current interpretation: the removed bit-exchange SALU operations were on the
store-address dependency chain and incidentally paced/staggered matrix-store
issue across waves. Removing them makes descriptors/addresses ready earlier
and increases pending-cacheline/store contention. An independent `s_nop 6`
before the store did **not** recover performance (157290 cycles / 7.501659
TOPS), so this is dependency/scheduling sensitive rather than a fixed seven-cycle
delay. Re-enabling LLVM's `s_cselect+cmp+branch` fold also did not recover it
(157969 cycles / 7.469415 TOPS). The exact backend scheduling fix remains open;
do not restore address bit exchange solely on the assumption that the physical
store layout requires it.

PMD artifacts:

- legacy bit exchange: `/xukang/code/perf_bit_exchange_ab/legacy_bit_exchange/m5out/0/0/`
- rebuilt direct B: `/xukang/code/m5out_b_rebuilt_32x16_256/0/0/`
- cselect-fold experiment: `/xukang/code/m5out_b_cselect_reenabled_32x16/0/0/`

The measured loss is consistently about **2%** on both the small reproducer
and `4096x4608x4096`, but other shapes and matrix-store instruction widths
still require independent measurement.

---

## Shape: 512³ (MNK = 512, 512, 512)

### Meta

| Item | Value |
| --- | --- |
| MNK | **M=512, N=512, K=512** |
| AB layout | **TT** (A K-major, B N-major) |
| arch | gfx946 |
| dtype | FP16 in / FP32 accumulate |
| gcFrequency | 2.197802 GHz |
| triton | nowasp: `c2defdef3` + LLVM `10899c9c`; wdra48: `d0ccddcc3` + triton-llvm `0cb57b8b81cc` |
| bolt_ops | nowasp: `8092776`; wdra48: `c54f635` |
| tool | `boltops.tools.tune_gemm_wasp`; fixed-config no-WASP driver for the 8-warp baseline |

### Winners

| mode | TOPS | cycles | wall_s | vs nowasp | JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| nowasp, non-MLS | **24.450** | **24130** | — | 1.000× | cold-start tuning record below |
| wdra48, non-MLS (prior) | 17.516 | 33681 | 9.484 | 0.716× | [`M512_N512_K512_TT_wdra48.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M512_N512_K512_TT_wdra48.json) |
| wdra48 + **MLS + `matrix_store_32x32_b16`**, `cl=1`, **`pp=0`** | **26.819** | **21998** | 9.792 | **1.097×** | tuning CSV (below) |

### Launch — nowasp

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | **64 × 64 × 128** |
| empty_arrive_after_mmac | false |
| enable_v_mmac_cluster | **1** |
| enable_consumer_pingpong | false (no-op for nowasp) |
| mmac_layout_force | **0** |
| num_warps / num_stages | **8 / 2** |
| use_mls / use_matrix_store | false / false |
| buffer cache swizzle / buffer ops | true / true |
| grid / wg (PMD) | (32768,1,1) / (512,1,1) |
| Triton / LLVM | `c2defdef3` / `10899c9c790344e11fcb80c96f9092938baa3d9f` |
| code object resources | 96 VGPR / 29 SGPR / 32 KiB shared |
| golden | PASS (`max_abs=0.03125`, `max_rel=0.0120798`, atol/rtol=1e-3) |

Two independent cold-start measurements were **24.432 / 24.450 TOPS**
(24147 / 24130 cycles), averaging **24.441 TOPS / 24138.5 cycles**. The best
single cold-start result, **24.450 TOPS**, is used as the nowasp reference above.

### Launch — wdra48 best (MLS + `mstore`, `cl=1`, `pp=0`)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | **128 × 128 × 128** |
| empty_arrive_after_mmac | **false** |
| enable_v_mmac_cluster | **1** |
| enable_consumer_pingpong | **false** |
| mmac_layout_force | **3** (mmac transpose) |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | **true** (`tl.matrix_load`) |
| use_matrix_store | **true** (`matrix_store_32x32_b16`, interleave **2**) |
| output blocks | **4 × 4 = 16** |
| grid / wg (PMD) | (12288,1,1) / (768,1,1) |
| source CSV | `tuning_512_mls_20260827/post_rebase_verify.csv` |
| golden | PASS (`max_abs=0.03125`, `max_rel=0.0120798`, atol/rtol=1e-3) |

Three independent cold-start measurements of the promoted configuration were
**26.819 / 26.750 / 26.802 TOPS** (21998 / 22055 / 22012 cycles), averaging
**26.790 TOPS / 22022 cycles**. The final post-rebase full-pipeline verification
reproduced the best result exactly: **26.819 TOPS / 21998 cycles**, GOLDEN PASS.

### Launch — wdra48 prior (non-MLS)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | 128 × 128 × 128 |
| empty_arrive_after_mmac | false |
| enable_v_mmac_cluster | 0 |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| wdra regs (load / main / tail) | 160 / 160 / 160 ([VGPR.md](VGPR.md#scenario-512cube-wdra48-non-mls)) |
| grid / wg (PMD) | (16,1,1) / (768,1,1) |
| source CSV | `gemm_wasp_tune_wdra48_incl128.csv` |

### Notes

- VGPR **88/144/140** + small tiles (64³ …) left wdra48 ~0.96 TOPS. **160/160/160**
  + **128³** unlocked the win above.
- nowasp / wdra48 bests are from different CSV runs; same freq and MNK.
- The refreshed nowasp search covered `num_warps=2/4/8/16`,
  `num_stages=1/2/3/4`, and square/asymmetric block families. Eight warps with
  `64x64x128`, stage 2 was stable across two independent cold starts; 16 warps
  regressed to 9.404 TOPS, and stage 4 `64x64x256` exceeded the 128 KiB LDS
  limit. PMD emitted `VGPR read before writing` diagnostics for some 8-warp
  candidates even though random-input GOLDEN checks passed.
- For the MLS matrix-store launch, disabling consumer ping-pong was the main
  improvement; adding `cluster=1` raised the cold-start result to 26.75--26.82
  TOPS. Forced `matrix_store_32x16_b16` was slower than the automatic 32x32
  selection.
- Attempts to increase output parallelism toward 48 blocks did not produce a
  legal faster launch. `64x96`/`96x64` cannot be expressed by the current
  power-of-two `tl.arange` kernel, while the 32-block `64x128`/`128x64`
  variants fail MLS tile selection under WDRA48's eight consumer warps.

<!-- AUTOTUNE_REFRESH:512x512x512:TT:BEGIN -->
### Autotune refresh — 2026-09-02

| mode | best TOPS | best cycles | 3-run TOPS range | 3-run cycles range | fixed configuration |
| --- | ---: | ---: | ---: | ---: | --- |
| nowasp | **23.771** | **24819** | 23.689–23.771 | 24819–24905 | `64x64x128`, ns=2, cl=1, pp=0, mlf=3, mls=0, mstore=0 |
| wdra48 | **26.820** | **21997** | 26.718–26.820 | 21997–22081 | `128x128x128`, ns=2, cl=1, pp=0, mlf=3, mls=1, mstore=1, regs=88/208/208 |

WDRA48 / nowasp = **1.128×**. All promoted reruns passed GOLDEN. Artifacts: `tuning_results_refresh_20260831/512x512x512_TT`.
<!-- AUTOTUNE_REFRESH:512x512x512:TT:END -->

### History

| date | change |
| --- | --- |
| 2026-08-06 | First promote: nowasp + wdra48 @ 128³; regs 160/160/160. |
| 2026-08-27 | New best: wdra48 + MLS + direct `matrix_store_32x32_b16`, `128x128x128`, `ns=2`, `mlf=3`, `cl=1`, `pp=0`: **26.819 TOPS / 21998 cycles**, GOLDEN PASS. |
| 2026-08-29 | Refresh nowasp reference: `64x64x128`, `num_warps=8`, `ns=2`, `mlf=0`, `cl=1`: best cold start **24.450 TOPS / 24130 cycles**, two-run mean **24.441 TOPS**, GOLDEN PASS. |

---

## Shape: 1536³ (MNK = 1536, 1536, 1536)

### Meta

| Item | Value |
| --- | --- |
| MNK | **M=1536, N=1536, K=1536** |
| AB layout | **TT** (A K-major, B N-major) |
| arch | gfx946 |
| dtype | FP16 in / FP32 accumulate |
| gcFrequency | 2.197802 GHz |
| triton | `e4fee6ec5` |
| bolt_ops | `901eb99` |
| tool | `python -m boltops.tools.tune_gemm_wasp` |

### Winners

| mode | TOPS | cycles | wall_s | vs nowasp | JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| nowasp | 115.468 | 137953 | 168.291 | 1.000× | [`M1536_N1536_K1536_TT_nowasp.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M1536_N1536_K1536_TT_nowasp.json) |
| wdra48 | 133.050 | 119723 | 131.826 | **1.152×** | [`M1536_N1536_K1536_TT_wdra48.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M1536_N1536_K1536_TT_wdra48.json) |

### Launch — nowasp

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | 128 × 128 × 128 |
| empty_arrive_after_mmac | false |
| enable_v_mmac_cluster | 0 |
| num_warps / num_stages | 4 / 2 |
| grid / wg (PMD) | (36864,1,1) / (256,1,1) |
| source CSV | `gemm_wasp_tune.csv` |

### Launch — wdra48

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | 256 × 256 × 64 |
| empty_arrive_after_mmac | false |
| enable_v_mmac_cluster | 0 |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | false |
| wdra regs (load / main / tail) | 112 / 196 / 196 ([VGPR.md](VGPR.md#scenario-1536cube-wdra48-non-mls)) |
| grid / wg (PMD) | (27648,1,1) / (768,1,1) |
| source CSV | `gemm_wasp_tune.csv` |

### Notes

- nowasp best so far is **128³** (not 256×256×64).
- wdra48 best is **256×256×64** + VGPR **112/196/196**.
- nowasp / wdra48 from separate solo runs; same MNK and freq.

<!-- AUTOTUNE_REFRESH:1536x1536x1536:TT:BEGIN -->
### Autotune refresh — 2026-09-09

| mode | best TOPS | best cycles | 3-run TOPS range | 3-run cycles range | fixed configuration |
| --- | ---: | ---: | ---: | ---: | --- |
| nowasp | **154.194 (35.69%)** | **103306** | 153.679–154.194 | 103306–103652 | `128x128x32`, ns=2, cl=1, pp=0, mlf=3, mls=0, mstore=0 |
| wdra48 | **223.279 (51.68%)** | **71342** | 222.586–223.279 | 71342–71564 | `256x256x64`, ns=2, cl=1, pp=1, mlf=3, mls=1, mstore=1 |

WDRA48 / nowasp = **1.448×**. All promoted reruns passed GOLDEN. Artifacts:
`tuning_1536_nowasp_20260831` and `tuning_1536_wdra48_20260831`.
<!-- AUTOTUNE_REFRESH:1536x1536x1536:TT:END -->

### History

| date | change |
| --- | --- |
| 2026-08-10 | Record wdra48 @ 256×256×64, 133.050 TOPS; nowasp ~116 TBD. |
| 2026-08-10 | Promote nowasp @ 128³, 115.468 TOPS (best so far). |

---

## Shape: 3072×3072×128 (MNK = 3072, 3072, 128)

### Meta

| Item | Value |
| --- | --- |
| MNK | **M=3072, N=3072, K=128** |
| AB layout | **TT** (A K-major, B N-major) |
| arch | gfx946 |
| dtype | FP16 in / FP32 accumulate |
| gcFrequency | 2.197802 GHz |
| triton | `e4fee6ec5` |
| bolt_ops | `901eb99` |
| tool | `python -m boltops.tools.tune_gemm_wasp` |
| status | **first datapoint** (not claimed best; nowasp not kept) |

### Winners

| mode | TOPS | cycles | wall_s | vs nowasp | JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| wdra48 | 27.049 | ~196300 | — | 7.909×† | [`M3072_N3072_K128_TT_wdra48.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M3072_N3072_K128_TT_wdra48.json) |

† SUMMARY `SPEEDUP wdra48/nowasp = 7.909x`; nowasp TOPS/config not recorded.

### Launch — wdra48

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | 256 × 256 × 64 |
| empty_arrive_after_mmac | false |
| enable_v_mmac_cluster | 0 |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | false |
| wdra regs (load / main / tail) | 112 / 196 / 196 (assumed; same 256-tile family) |
| tiles (ceil) | 12 × 12 = 144 |
| source CSV | lost (overwritten); TOPS from terminal SUMMARY |
| cycles | back-computed from TOPS + gcFrequency |

### Notes

- Square MN with short K (K=128 = 2×BK); first datapoint only.
- **nowasp not recorded** (only speedup ratio kept from SUMMARY).

<!-- AUTOTUNE_REFRESH:3072x3072x128:TT:BEGIN -->
### Autotune refresh — 2026-09-02

| mode | best TOPS | best cycles | 3-run TOPS range | 3-run cycles range | fixed configuration |
| --- | ---: | ---: | ---: | ---: | --- |
| nowasp | **83.402** | **63664** | 82.215–83.402 | 63664–64583 | `128x128x32`, ns=2, cl=0, pp=0, mlf=3, mls=0, mstore=0 |
| wdra48 | **100.976** | **52584** | 99.522–100.976 | 52584–53352 | `256x256x64`, ns=2, cl=2, pp=0, mlf=3, mls=1, mstore=1, regs=16/244/244 |

WDRA48 / nowasp = **1.211×**. All promoted reruns passed GOLDEN. Artifacts: `tuning_results_refresh_20260831/3072x3072x128_TT`.
<!-- AUTOTUNE_REFRESH:3072x3072x128:TT:END -->

### History

| date | change |
| --- | --- |
| 2026-08-10 | First datapoint: wdra48 @ 256×256×64, 27.049 TOPS. |

---

## Shape: 3072×3072×256 (MNK = 3072, 3072, 256)

### Meta

| Item | Value |
| --- | --- |
| MNK | **M=3072, N=3072, K=256** |
| AB layout | **TT** (A K-major, B N-major) |
| arch | gfx946 |
| dtype | FP16 in / FP32 accumulate |
| gcFrequency | 2.197802 GHz |
| triton | `e4fee6ec5` |
| bolt_ops | `901eb99` |
| tool | `python -m boltops.tools.tune_gemm_wasp` |
| status | **first datapoint** (not claimed best; nowasp not recorded) |

### Winners

| mode | TOPS | cycles | wall_s | vs nowasp | JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| wdra48 | 52.461 | 202427 | 86.552 | — | [`M3072_N3072_K256_TT_wdra48.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M3072_N3072_K256_TT_wdra48.json) |

### Launch — wdra48

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | 256 × 256 × 64 |
| empty_arrive_after_mmac | false |
| enable_v_mmac_cluster | 0 |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | false |
| wdra regs (load / main / tail) | 112 / 196 / 196 |
| tiles (ceil) | 12 × 12 = 144 |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS |

### Notes

- Solo `--skip-nowasp` run; **nowasp not recorded**.
- Square MN; short-K family with K=256.

<!-- AUTOTUNE_REFRESH:3072x3072x256:TT:BEGIN -->
### Autotune refresh — 2026-09-02

| mode | best TOPS | best cycles | 3-run TOPS range | 3-run cycles range | fixed configuration |
| --- | ---: | ---: | ---: | ---: | --- |
| nowasp | **114.910** | **92415** | 112.345–114.910 | 92415–94525 | `128x128x32`, ns=2, cl=0, pp=0, mlf=3, mls=0, mstore=0 |
| wdra48 | **164.209** | **64670** | 162.200–164.209 | 64670–65471 | `256x256x64`, ns=2, cl=2, pp=0, mlf=3, mls=1, mstore=1, regs=16/244/244 |

WDRA48 / nowasp = **1.429×**. All promoted reruns passed GOLDEN. Artifacts: `tuning_results_refresh_20260831/3072x3072x256_TT`.
<!-- AUTOTUNE_REFRESH:3072x3072x256:TT:END -->

### History

| date | change |
| --- | --- |
| 2026-08-10 | First datapoint: wdra48 @ 256×256×64, 52.461 TOPS. |

---

## Shape: 3072×3072×512 (MNK = 3072, 3072, 512)

### Meta

| Item | Value |
| --- | --- |
| MNK | **M=3072, N=3072, K=512** |
| AB layout | **TT** (A K-major, B N-major) |
| arch | gfx946 |
| dtype | FP16 in / FP32 accumulate |
| gcFrequency | 2.197802 GHz |
| triton | `e4fee6ec5` |
| bolt_ops | `901eb99` |
| tool | `python -m boltops.tools.tune_gemm_wasp` |
| status | **first datapoint** (not claimed best; nowasp not recorded) |

### Winners

| mode | TOPS | cycles | wall_s | vs nowasp | JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| wdra48 | 104.280 | 203672 | 170.88 | — | [`M3072_N3072_K512_TT_wdra48.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M3072_N3072_K512_TT_wdra48.json) |

### Launch — wdra48

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | 256 × 256 × 64 |
| empty_arrive_after_mmac | false |
| enable_v_mmac_cluster | 0 |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | false |
| wdra regs (load / main / tail) | 112 / 196 / 196 |
| tiles (ceil) | 12 × 12 = 144 |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS |

### Notes

- Solo `--skip-nowasp` run; **nowasp not recorded**.
- Same tile/VGPR as 3072×3072×256; ~2× TOPS with K doubled and cycles nearly flat (~203k vs ~202k).

<!-- AUTOTUNE_REFRESH:3072x3072x512:TT:BEGIN -->
### Autotune refresh — 2026-09-02

| mode | best TOPS | best cycles | 3-run TOPS range | 3-run cycles range | fixed configuration |
| --- | ---: | ---: | ---: | ---: | --- |
| nowasp | **141.461** | **150139** | 140.701–141.461 | 150139–150950 | `128x128x32`, ns=2, cl=1, pp=0, mlf=3, mls=0, mstore=0 |
| wdra48 | **229.833** | **92410** | 228.434–229.833 | 92410–92976 | `256x256x64`, ns=2, cl=1, pp=1, mlf=3, mls=1, mstore=1, regs=16/244/244 |

WDRA48 / nowasp = **1.625×**. All promoted reruns passed GOLDEN. Artifacts: `tuning_results_refresh_20260831/3072x3072x512_TT`.
<!-- AUTOTUNE_REFRESH:3072x3072x512:TT:END -->

### History

| date | change |
| --- | --- |
| 2026-08-10 | First datapoint: wdra48 @ 256×256×64, 104.280 TOPS. |

---

## Shape: 3072×3072×1024 (MNK = 3072, 3072, 1024)

### Meta

| Item | Value |
| --- | --- |
| MNK | **M=3072, N=3072, K=1024** |
| AB layout | **TT** (A K-major, B N-major) |
| arch | gfx946 |
| dtype | FP16 in / FP32 accumulate |
| gcFrequency | 2.197802 GHz |
| triton | `e4fee6ec5` |
| bolt_ops | `901eb99` |
| tool | `python -m boltops.tools.tune_gemm_wasp` |
| status | **first datapoint** (not claimed best; nowasp not recorded) |

### Winners

| mode | TOPS | cycles | wall_s | vs nowasp | JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| wdra48 | 207.980 | 204239 | 342.048 | — | [`M3072_N3072_K1024_TT_wdra48.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M3072_N3072_K1024_TT_wdra48.json) |

### Launch — wdra48

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | 256 × 256 × 64 |
| empty_arrive_after_mmac | false |
| enable_v_mmac_cluster | 0 |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | false |
| wdra regs (load / main / tail) | 112 / 196 / 196 |
| tiles (ceil) | 12 × 12 = 144 |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS |

### Notes

- Solo `--skip-nowasp` run; **nowasp not recorded**.
- Continues the 3072² short-K series; ~2× TOPS vs K=512 with cycles still ~204k.

<!-- AUTOTUNE_REFRESH:3072x3072x1024:TT:BEGIN -->
### Autotune refresh — 2026-09-02

| mode | best TOPS | best cycles | 3-run TOPS range | 3-run cycles range | fixed configuration |
| --- | ---: | ---: | ---: | ---: | --- |
| nowasp | **162.704** | **261073** | 162.318–162.704 | 261073–261695 | `128x128x32`, ns=2, cl=1, pp=0, mlf=3, mls=0, mstore=0 |
| wdra48 | **295.429** | **143783** | 293.330–295.429 | 143783–144812 | `256x256x64`, ns=2, cl=0, pp=1, mlf=3, mls=1, mstore=1, regs=16/244/244 |

WDRA48 / nowasp = **1.816×**. All promoted reruns passed GOLDEN. Artifacts: `tuning_results_refresh_20260831/3072x3072x1024_TT`.
<!-- AUTOTUNE_REFRESH:3072x3072x1024:TT:END -->

### History

| date | change |
| --- | --- |
| 2026-08-10 | First datapoint: wdra48 @ 256×256×64, 207.980 TOPS. |

---

## Shape: 3072×3072×2048 (MNK = 3072, 3072, 2048)

### Meta

| Item | Value |
| --- | --- |
| MNK | **M=3072, N=3072, K=2048** |
| AB layout | **TT** (A K-major, B N-major) |
| arch | gfx946 |
| dtype | FP16 in / FP32 accumulate |
| gcFrequency | 2.197802 GHz |
| triton | `e4fee6ec5` |
| bolt_ops | `901eb99` |
| tool | `python -m boltops.tools.tune_gemm_wasp` |
| status | **first datapoint** (not claimed best; nowasp not recorded) |

### Winners

| mode | TOPS | cycles | wall_s | vs nowasp | JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| wdra48 + `TRITON_BUFFER_CACHE_SWIZZLE` | 216.531 | 392348 | 695.249 | — | [`M3072_N3072_K2048_TT_wdra48_bcs.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M3072_N3072_K2048_TT_wdra48_bcs.json) |
| wdra48 (no swizzle) | 216.051 | 393220 | 697.861 | — | [`M3072_N3072_K2048_TT_wdra48.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M3072_N3072_K2048_TT_wdra48.json) |

### Launch — wdra48 (+ buffer cache swizzle)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | 256 × 256 × 64 |
| empty_arrive_after_mmac | false |
| enable_v_mmac_cluster | 0 |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | false |
| wdra regs (load / main / tail) | 112 / 196 / 196 |
| `TRITON_BUFFER_CACHE_SWIZZLE` | **on** |
| tiles (ceil) | 12 × 12 = 144 |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS |

### Notes

- Solo `--skip-nowasp` runs; **nowasp not recorded**.
- vs K=1024 (207.980 TOPS / ~204k cycles): K×2 → cycles ~×1.92–1.93, TOPS only +4% (plateau).
- Swizzle vs no-swizzle on same tile: **216.531 vs 216.051** (+0.22%, cycles 392348 vs 393220).

<!-- AUTOTUNE_REFRESH:3072x3072x2048:TT:BEGIN -->
### Autotune refresh — 2026-09-02

| mode | best TOPS | best cycles | 3-run TOPS range | 3-run cycles range | fixed configuration |
| --- | ---: | ---: | ---: | ---: | --- |
| nowasp | **174.796** | **486027** | 174.475–174.796 | 486027–486921 | `128x128x32`, ns=2, cl=1, pp=0, mlf=3, mls=0, mstore=0 |
| wdra48 | **344.613** | **246524** | 343.867–344.613 | 246524–247059 | `256x256x64`, ns=2, cl=2, pp=1, mlf=3, mls=1, mstore=1, regs=16/244/244 |

WDRA48 / nowasp = **1.972×**. All promoted reruns passed GOLDEN. Artifacts: `tuning_results_refresh_20260831/3072x3072x2048_TT`.
<!-- AUTOTUNE_REFRESH:3072x3072x2048:TT:END -->

### History

| date | change |
| --- | --- |
| 2026-08-10 | First datapoint: wdra48 @ 256×256×64, 216.051 TOPS (no swizzle). |
| 2026-08-10 | +`TRITON_BUFFER_CACHE_SWIZZLE`: 216.531 TOPS (best so far on this shape). |

---

## Shape: 4096×4096×4608 (MNK = 4096, 4096, 4608)

### Meta

| Item | Value |
| --- | --- |
| MNK | **M=4096, N=4096, K=4608** |
| AB layout | **TT** (A K-major, B N-major) |
| arch | gfx946 |
| dtype | FP16 in / FP32 accumulate |
| gcFrequency | 2.197802 GHz |
| triton | `e4fee6ec5` |
| bolt_ops | `901eb99` |
| tool | `python -m boltops.tools.tune_gemm_wasp` |
| status | **first datapoint** (not claimed best; nowasp not recorded) |

### Winners

| mode | TOPS | cycles | wall_s | vs nowasp | JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| wdra48 + `TRITON_BUFFER_CACHE_SWIZZLE` | 208.517 | 1629708 | 2780.032 | — | [`M4096_N4096_K4608_TT_wdra48_bcs.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M4096_N4096_K4608_TT_wdra48_bcs.json) |

### Launch — wdra48 (+ buffer cache swizzle)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | 256 × 256 × 64 |
| empty_arrive_after_mmac | false |
| enable_v_mmac_cluster | 0 |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | false |
| wdra regs (load / main / tail) | 112 / 196 / 196 |
| `TRITON_BUFFER_CACHE_SWIZZLE` | **on** |
| tiles (ceil) | 16 × 16 = 256 |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS |

### Notes

- Solo `--skip-nowasp` run; **nowasp not recorded**.
- CLI: `--mnk 4096x4096x4608` with `TRITON_BUFFER_CACHE_SWIZZLE=1`.

### History

| date | change |
| --- | --- |
| 2026-08-10 | First datapoint: wdra48 + BCS @ 256×256×64, 208.517 TOPS. |
| 2026-08-11 | Corrected MNK: N=4096, K=4608 (was N/K swapped). |

---

## Shape: 4096×4608×512 (MNK = 4096, 4608, 512)

### Meta

| Item | Value |
| --- | --- |
| MNK | **M=4096, N=4608, K=512** |
| AB layout | **TT** (A K-major, B N-major) |
| arch | gfx946 |
| dtype | FP16 in / FP32 accumulate |
| gcFrequency | 2.197802 GHz |
| triton | `45e8a4c1a` (`45e8a4c1a5c4ffb7ca1ee867c6e61fe3db674161`, `triton-mls-store`) |
| bolt_ops | `d46c3b5` |
| tool | `python -m boltops.tools.tune_gemm_wasp --ab-layout TT --mls --matrix-store` |
| status | best so far @ **256×256×64** / `ns=2` / **`mlf=3`** / **`pp=1`** / MLS + **`mstore`**; short-K overhead dominates; nowasp not recorded |

### Winners

| mode | TOPS | cycles | wall_s | vs nowasp | JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| wdra48 + BCS + **MLS** + **`mstore`**, `ea=0`, `ns=2`, BK=64, **`mlf=3`**, **`pp=1`**, VGPR 16/244/244 | **246.803** | 172112 | 396.122 | — | [`M4096_N4608_K512_TT_wdra48_bcs_mls_mlf3_pp_mstore.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M4096_N4608_K512_TT_wdra48_bcs_mls_mlf3_pp_mstore.json) |
| prior (same launch) | 229.675 | 184947 | 413.178 | — | — (superseded) |

### Launch — wdra48 (MLS + BCS + `mstore`, `ns=2`, BK=64, `mlf=3`, `pp=1`)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | **256 × 256 × 64** |
| empty_arrive_after_mmac | **false** |
| enable_v_mmac_cluster | 0 |
| enable_consumer_pingpong | **true** |
| mmac_layout_force | **3** (mmac transpose) |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | **true** (`tl.matrix_load`) |
| use_matrix_store | **true** (`tl.matrix_store`, interleave **2**) |
| wdra regs (load / main / tail) | **16 / 244 / 244** (Clang used 2 / 225 / 225) |
| `TRITON_BUFFER_CACHE_SWIZZLE` | **on** |
| tiles (ceil) | 16 × 18 = 288 |
| K-tile iters | 8 (`K / BK`) |
| grid / wg (PMD) | (221184,1,1) / (768,1,1) |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS (`max_abs=0.0625`, `max_rel=1.5`, atol/rtol=1e-3) |

### Notes

- Solo `--skip-nowasp --ab-layout TT` run; **nowasp not recorded**.
- Same launch family as [4096×4608×4096 TT](#shape-4096x4608x4096) and [4096×4608×16384 TT](#shape-4096x4608x16384), but short-K (**K=512**, only **8** BK iters).
- **Why TOPS is much lower than K=4096/16384 (~389–392):** K is small → few K-iters per block → **WDRA prologue setup** and **block-epilogue store** take a large fraction of time; during both phases **MMAC is idle**. Compute amortization is poor vs long-K.
- Harness printed `RESULT ... TOPS=246.80263263964898 cycles=172112 wall_s=396.122 golden=PASS`.
- Prior same launch: **229.675** TOPS / 184947 cycles / 413.178 s.

<!-- AUTOTUNE_REFRESH:4096x4608x512:TT:BEGIN -->
### Autotune refresh — 2026-09-09

| mode | best TOPS | best cycles | 3-run TOPS range | 3-run cycles range | fixed configuration |
| --- | ---: | ---: | ---: | ---: | --- |
| nowasp | **137.508 (31.83%)** | **308910** | 135.943–137.508 | 308910–312467 | `128x128x32`, ns=2, cl=0, pp=0, mlf=3, mls=0, mstore=0 |
| wdra48 | **244.311 (56.55%)** | **173867** | 242.443–244.311 | 173867–175207 | `256x256x64`, ns=2, cl=1, pp=1, mlf=3, mls=1, mstore=1, regs=16/244/244 |

WDRA48 / nowasp = **1.777×**. All promoted reruns passed GOLDEN. Artifacts: `tuning_results_refresh_20260831/4096x4608x512_TT`.
<!-- AUTOTUNE_REFRESH:4096x4608x512:TT:END -->

### History

| date | change |
| --- | --- |
| 2026-08-20 | First datapoint: **229.675** TOPS / 184947 cycles. |
| 2026-08-20 | Re-run: **246.803** TOPS / 172112 cycles; note short-K WDRA setup + store overhead (MMAC idle). |

---

## Shape: 4096×4608×4096 (MNK = 4096, 4608, 4096)

### Meta

| Item | Value |
| --- | --- |
| MNK | **M=4096, N=4608, K=4096** |
| AB layout | **TT** (A K-major, B N-major) |
| arch | gfx946 |
| dtype | FP16 in / FP32 accumulate |
| gcFrequency | 2.197802 GHz |
| triton | `45e8a4c1a` (`45e8a4c1a5c4ffb7ca1ee867c6e61fe3db674161`, `triton-mls-store`) |
| bolt_ops | `77c99f2` |
| tool | `python -m boltops.tools.tune_gemm_wasp --ab-layout TT --mls --matrix-store` |
| status | best so far is **MLS + matrix_store** @ **256×256×64** / `ns=2` / **`mlf=3`** / **`pp=1`** / VGPR **16/244/244**; nowasp not recorded |

### Winners

| mode | TOPS | cycles | wall_s | vs nowasp | JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| wdra48 + BCS + **MLS** + **`mstore`**, `ea=0`, `ns=2`, BK=64, **`mlf=3`**, **`pp=1`**, VGPR 16/244/244 | **389.372** | 872743 | 2806.281 | — | [`M4096_N4608_K4096_TT_wdra48_bcs_mls_mlf3_pp_mstore.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M4096_N4608_K4096_TT_wdra48_bcs_mls_mlf3_pp_mstore.json) |
| wdra48 + `TRITON_BUFFER_CACHE_SWIZZLE` | 230.378 | 1475058 | 3217.491 | — | [`M4096_N4608_K4096_TT_wdra48_bcs.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M4096_N4608_K4096_TT_wdra48_bcs.json) |

### Launch — wdra48 best (MLS + BCS + `mstore`, `ns=2`, BK=64, `mlf=3`, `pp=1`)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | **256 × 256 × 64** |
| empty_arrive_after_mmac | **false** |
| enable_v_mmac_cluster | 0 |
| enable_consumer_pingpong | **true** |
| mmac_layout_force | **3** (mmac transpose) |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | **true** (`tl.matrix_load`) |
| use_matrix_store | **true** (`tl.matrix_store`, interleave **2**) |
| wdra regs (load / main / tail) | **16 / 244 / 244** (Clang used 2 / 225 / 225) |
| `TRITON_BUFFER_CACHE_SWIZZLE` | **on** |
| tiles (ceil) | 16 × 18 = 288 |
| grid / wg (PMD) | (221184,1,1) / (768,1,1) |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS (`max_abs=0.125`, `max_rel=25.875`, atol/rtol=1e-3) |

### Launch — wdra48 prior (+ buffer cache swizzle, non-MLS)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | 256 × 256 × 64 |
| empty_arrive_after_mmac | false |
| enable_v_mmac_cluster | 0 |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | false |
| wdra regs (load / main / tail) | 112 / 196 / 196 |
| `TRITON_BUFFER_CACHE_SWIZZLE` | **on** |
| tiles (ceil) | 16 × 18 = 288 |
| grid / wg (PMD) | (221184,1,1) / (768,1,1) |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS |

### Notes

- Solo `--skip-nowasp --ab-layout TT` runs; **nowasp not recorded**.
- Distinct from [4096×4096×4608](#shape-4096x4096x4608) (N/K swapped): same FLOPs, different tile grid (288 vs 256).
- **MLS + `tl.matrix_store`** (`mstore=1`, B interleave **2**, `mlf=3`, `pp=1`, VGPR 16/244/244) vs prior non-MLS BCS: **389.372** vs 230.378 TOPS (**~+69.0%**, cycles 872743 vs 1475058).
- Same launch vs TN best (MLS `pp` + `abarrier_seq`, no mstore): **389.372** vs 377.095 TOPS (**~+3.3%**).
- The **389.372** TOPS winner used the legacy matrix-store address bit
  exchange with `matrix_store_32x16_b16`. With direct non-contiguous B
  lowering, 32x16 reached **382.046** TOPS / 889477 cycles and 32x32 reached
  **381.337** TOPS / 891132 cycles. Direct-B 32x16 is about **0.19%** faster,
  while both are **~1.9--2.1%** below the legacy result. See
  [the PMD regression note](#known-pmd-matrix-store-regression-removing-address-bit-exchange).
- Harness printed `GOLDEN max_abs=0.125 max_rel=25.875` and still **PASS** at 1e-3 (`|Δ| <= atol + rtol*|ref|`; large rel is near-zero C).

<!-- AUTOTUNE_REFRESH:4096x4608x4096:TT:BEGIN -->
### Autotune refresh — 2026-09-09

| mode | best TOPS | best cycles | tested variants | fixed configuration |
| --- | ---: | ---: | --- | --- |
| nowasp | **158.358 (36.66%)** | **2145905** | cl=0: 158.358; cl=1: 153.676 | `128x128x32`, ns=2, cl=0, pp=0, mlf=3, mls=0, mstore=0 |
| wdra48 | **381.346 (88.27%)** | **891111** | cl=0: 381.346; cl=1: 380.561; cl=2: 380.001 | `256x256x64`, ns=2, cl=0, pp=1, mlf=3, mls=1, mstore=1, regs=16/244/244 |

WDRA48 / nowasp = **2.408×**. Every result above passed GOLDEN. The WDRA
winner has strict register headroom (`225 < 244`, `225 < 244`, `2 < 16`) and
zero reported VGPR spills. The planned three-run promotion was paused during
the first nowasp rerun, so only completed search results are promoted here.
Triton: `f014fb048`; triton-llvm: `10899c9c7903`. Artifacts:
`tuning_results_refresh_20260831/4096x4608x4096_TT`.
<!-- AUTOTUNE_REFRESH:4096x4608x4096:TT:END -->

### History

| date | change |
| --- | --- |
| 2026-08-11 | First datapoint: wdra48 + BCS @ 256×256×64, 230.378 TOPS. |
| 2026-08-19 | MLS + **`matrix_store`** (`mlf=3`, `pp=1`, interleave=2, Triton `45e8a4c1a`): **389.372** TOPS (new TT best; ~+69.0% vs non-MLS BCS). |
| 2026-08-27 | Direct non-contiguous B store lowering: `matrix_store_32x16_b16` **382.046** TOPS; `matrix_store_32x32_b16` **381.337** TOPS. 32x16 is ~0.19% faster; both are ~2% below the legacy 32x16 bit-exchange winner. |

---

## Shape: 4096×4608×16384 (MNK = 4096, 4608, 16384)

### Meta

| Item | Value |
| --- | --- |
| MNK | **M=4096, N=4608, K=16384** |
| AB layout | **TT** (A K-major, B N-major) |
| arch | gfx946 |
| dtype | FP16 in / FP32 accumulate |
| gcFrequency | 2.197802 GHz |
| triton | `45e8a4c1a` (`45e8a4c1a5c4ffb7ca1ee867c6e61fe3db674161`, `triton-mls-store`) |
| bolt_ops | `d0f8917` |
| tool | `python -m boltops.tools.tune_gemm_wasp --ab-layout TT --mls --matrix-store` |
| status | first **TT** datapoint @ **256×256×64** / `ns=2` / **`mlf=3`** / **`pp=1`** / MLS + **`mstore`**; nowasp not recorded |

### Winners

| mode | TOPS | cycles | gc_time | wall_s | vs nowasp | JSON |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| wdra48 + BCS + **MLS** + **`mstore`**, `ea=0`, `ns=2`, BK=64, **`mlf=3`**, **`pp=1`**, VGPR 16/244/244 | **392.254** | 3465320 | 1576.721 μs | 11256.32 | — | [`M4096_N4608_K16384_TT_wdra48_bcs_mls_mlf3_pp_mstore.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M4096_N4608_K16384_TT_wdra48_bcs_mls_mlf3_pp_mstore.json) |

### Launch — wdra48 (MLS + BCS + `mstore`, `ns=2`, BK=64, `mlf=3`, `pp=1`)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | **256 × 256 × 64** |
| empty_arrive_after_mmac | **false** |
| enable_v_mmac_cluster | 0 |
| enable_consumer_pingpong | **true** |
| mmac_layout_force | **3** (mmac transpose) |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | **true** (`tl.matrix_load`) |
| use_matrix_store | **true** (`tl.matrix_store`, interleave **2**) |
| wdra regs (load / main / tail) | **16 / 244 / 244** |
| `TRITON_BUFFER_CACHE_SWIZZLE` | **on** |
| tiles (ceil) | 16 × 18 = 288 |
| K-tile iters | 256 (`K / BK`) |
| source CSV | `gemm_wasp_tune.csv` |
| stats | `/xukang/code/m5out/0/0/stats.txt` |

Command:

```bash
TRITON_BUFFER_CACHE_SWIZZLE=1 TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR=triton_cache \
PYTHONPATH=/xukang/code/triton-mls-store/python:/xukang/code/bolt_ops \
python -m boltops.tools.tune_gemm_wasp \
  --mnk 4096x4608x16384 --mode wdra48 --block 256x256x64 \
  --empty-arrive 0 --cluster 0 --check --skip-nowasp --ab-layout TT \
  --num-stages 2 --mmac-layout-force 3 --consumer-pingpong true \
  --mls --matrix-store --matrix-store-interleave 2
```

### Notes

- Solo `--skip-nowasp --ab-layout TT` run; **nowasp not recorded**.
- Same launch as [4096×4608×4096 TT](#shape-4096x4608x4096) MLS `mstore` best; **K=16384** (4× inner dim).
- **392.254 TOPS** from PMD `totalGcSimCycles=3465320` @ `gcFrequency=2.197802 GHz` (`2·M·N·K / (cycles/freq)`).
- vs TT **K=4096** (**389.372** TOPS / 872743 cycles): **~+0.75%** TOPS at 4× K (cycles scale ~3.97×).
- `--check` enabled; harness CSV not updated for this run (stats taken from `m5out/0/0/stats.txt`).

### History

| date | change |
| --- | --- |
| 2026-08-20 | First datapoint: wdra48 + BCS + MLS + **`matrix_store`** @ 256×256×64, **392.254** TOPS. |

---

## Shape: 4096×4608×4096 TN (MNK = 4096, 4608, 4096)

### Meta

| Item | Value |
| --- | --- |
| MNK | **M=4096, N=4608, K=4096** |
| AB layout | **TN** (A K-major, B K-major) |
| arch | gfx946 |
| dtype | FP16 in / FP32 accumulate |
| gcFrequency | 2.197802 GHz |
| triton | `7225e1068` (`7225e1068dd006d2648f248d71a1ae4d704d7a5b`, consumer-pingpong) + **`abarrier_seq`** (replaces `s_waitcnt`) |
| bolt_ops | `4a60e0e` |
| tool | `python -m boltops.tools.tune_gemm_wasp --ab-layout TN` |
| status | best so far is **MLS** @ **256×256×64** / `ns=2` / **`mlf=3`** / **`pp=1`** / VGPR **16/244/244** + **`abarrier_seq`**; nowasp not recorded |

### Winners

| mode | TOPS | cycles | wall_s | vs nowasp | JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| wdra48 + BCS + **MLS**, `ea=0`, `ns=2`, BK=64, **`mlf=3`**, **`pp=1`**, VGPR 16/244/244, **`abarrier_seq`** | **377.095** | 901156 | 2883.342 | — | [`M4096_N4608_K4096_TN_wdra48_bcs_mls_mlf3_pp_rsrc.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M4096_N4608_K4096_TN_wdra48_bcs_mls_mlf3_pp_rsrc.json) |
| wdra48 + BCS + **MLS**, `ea=0`, `ns=2`, BK=64, **`mlf=3`**, **`pp=1`**, VGPR 16/244/244 | 327.693 | 1037013 | 2915.037 | — | [`M4096_N4608_K4096_TN_wdra48_bcs_mls_mlf3_pp.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M4096_N4608_K4096_TN_wdra48_bcs_mls_mlf3_pp.json) |
| wdra48 + BCS + **MLS**, `ea=0`, `ns=2`, BK=64, **`mlf=3`** | 293.201 | 1159006 | 2911.787 | — | [`M4096_N4608_K4096_TN_wdra48_bcs_mls_mlf3.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M4096_N4608_K4096_TN_wdra48_bcs_mls_mlf3.json) |
| wdra48 + BCS + **MLS**, `ea=0`, `ns=2`, BK=64, **`mlf=4`** | 293.172 | 1159119 | 2945.821 | — | [`M4096_N4608_K4096_TN_wdra48_bcs_mls_mlf4.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M4096_N4608_K4096_TN_wdra48_bcs_mls_mlf4.json) |
| wdra48 + BCS + **MLS**, `ea=0`, `ns=2`, BK=64 (`mlf` auto) | 281.854 | 1205665 | 2965.39 | — | [`M4096_N4608_K4096_TN_wdra48_bcs_mls.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M4096_N4608_K4096_TN_wdra48_bcs_mls.json) |
| wdra48 + BCS, `ea=0`, **`ns=4`**, BK=32 | 255.252 | 1331316 | 3138.631 | — | [`M4096_N4608_K4096_TN_wdra48_bcs_ns4.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M4096_N4608_K4096_TN_wdra48_bcs_ns4.json) |
| wdra48 + BCS, `ea=0`, `ns=4` (prior) | 250.732 | 1355319 | 3128.193 | — | — (superseded; same JSON path overwritten) |
| wdra48 + BCS, `ea=0`, `ns=2`, BK=32 | 245.557 | 1383878 | 3184.459 | — | [`M4096_N4608_K4096_TN_wdra48_bcs.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M4096_N4608_K4096_TN_wdra48_bcs.json) |
| wdra48 + BCS + **MLS**, `ea=0`, `ns=4`, BK=32 | 244.320 | 1390887 | 3134.372 | — | [`M4096_N4608_K4096_TN_wdra48_bcs_ns4_mls.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M4096_N4608_K4096_TN_wdra48_bcs_ns4_mls.json) |
| wdra48 + BCS, `ea=1`, `ns=2` (rejected) | 227.233 | 1495478 | 3230.792 | — | — (CSV only; worse than `ea=0`) |

### Launch — wdra48 best (MLS + BCS, `ns=2`, BK=64, `mlf=3`, `pp=1`, `abarrier_seq`)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | **256 × 256 × 64** |
| empty_arrive_after_mmac | **false** |
| enable_v_mmac_cluster | 0 |
| enable_consumer_pingpong | **true** |
| mmac_layout_force | **3** (mmac transpose) |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | **true** (`tl.matrix_load`) |
| wdra regs (load / main / tail) | **16 / 244 / 244** |
| `TRITON_BUFFER_CACHE_SWIZZLE` | **on** |
| wait | **`abarrier_seq`** replaces `s_waitcnt`, so MLS can issue continuously |
| tiles (ceil) | 16 × 18 = 288 |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS |

### Launch — wdra48 prior (MLS + BCS, `ns=2`, BK=64, `mlf=3`, `pp=1`)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | **256 × 256 × 64** |
| empty_arrive_after_mmac | **false** |
| enable_v_mmac_cluster | 0 |
| enable_consumer_pingpong | **true** |
| mmac_layout_force | **3** (mmac transpose) |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | **true** (`tl.matrix_load`) |
| wdra regs (load / main / tail) | **16 / 244 / 244** |
| `TRITON_BUFFER_CACHE_SWIZZLE` | **on** |
| tiles (ceil) | 16 × 18 = 288 |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS |

### Launch — wdra48 prior best (MLS + BCS, `ns=2`, BK=64, `mlf=3`, `pp` off)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | **256 × 256 × 64** |
| empty_arrive_after_mmac | **false** |
| enable_v_mmac_cluster | 0 |
| mmac_layout_force | **3** (mmac transpose) |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | **true** (`tl.matrix_load`) |
| wdra regs (load / main / tail) | 88 / 208 / 208 |
| `TRITON_BUFFER_CACHE_SWIZZLE` | **on** |
| tiles (ceil) | 16 × 18 = 288 |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS |

### Launch — wdra48 prior MLS (BCS, `ns=2`, BK=64, `mlf` auto)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | **256 × 256 × 64** |
| empty_arrive_after_mmac | **false** |
| enable_v_mmac_cluster | 0 |
| mmac_layout_force | **-1** (auto) |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | **true** (`tl.matrix_load`) |
| wdra regs (load / main / tail) | 88 / 208 / 208 |
| `TRITON_BUFFER_CACHE_SWIZZLE` | **on** |
| tiles (ceil) | 16 × 18 = 288 |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS |

### Launch — wdra48 prior best non-MLS (`num_stages=4` + BCS, BK=32)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | 256 × 256 × 32 |
| empty_arrive_after_mmac | **false** |
| enable_v_mmac_cluster | 0 |
| load / mma warps | 4 / 8 |
| num_stages | **4** |
| use_mls | false |
| wdra regs (load / main / tail) | 88 / 208 / 208 |
| Clang Available (print order) | `[208, 208, 88]` ≈ tail / main / load |
| `TRITON_BUFFER_CACHE_SWIZZLE` | **on** |
| tiles (ceil) | 16 × 18 = 288 |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS |

### Launch — wdra48 (`num_stages=2` + BCS, BK=32, prior)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | 256 × 256 × 32 |
| empty_arrive_after_mmac | **false** (best among ns=2; `true` regresses — see below) |
| enable_v_mmac_cluster | 0 |
| load / mma warps | 4 / 8 |
| num_stages | **2** |
| use_mls | false |
| wdra regs (load / main / tail) | 88 / 208 / 208 |
| Clang Available (print order) | `[208, 208, 88]` ≈ tail / main / load |
| `TRITON_BUFFER_CACHE_SWIZZLE` | **on** |
| tiles (ceil) | 16 × 18 = 288 |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS |

### Notes

- Solo `--skip-nowasp --ab-layout TN` runs; **nowasp not recorded**.
- Non-MLS TN previously needed **BK=32** (BK=64 did not fit VGPR); **MLS unlocks BK=64** at the same **88/208/208** budgets — see [`VGPR.md`](VGPR.md).
- **`mmac_layout_force=3`** (transpose) vs auto (`-1`) on same MLS BK=64 / ns=2: **293.201** vs 281.854 TOPS (**~+4.0%**, cycles 1159006 vs 1205665).
- **`mmac_layout_force=4`** (interleave+transpose) on the same launch: **293.172** TOPS (cycles 1159119) — essentially tied with `mlf=3` (~−0.01%).
- **Same launch + `abarrier_seq` replacing `s_waitcnt`** (MLS issues continuously) vs prior `pp=1`: **377.095** vs 327.693 TOPS (**~+15.1%**, cycles 901156 vs 1037013).
- **MLS BK=64 / ns=2 / mlf=3 / `pp=1` / VGPR 16/244/244** vs prior `mlf=3` (pp off, 88/208/208): **327.693** vs 293.201 TOPS (**~+11.8%**, cycles 1037013 vs 1159006). Gain is **pingpong + VGPR rebalance** together (Triton `7225e1068`).
- **MLS BK=64 / ns=2 / mlf=3 / pp / `abarrier_seq`** vs prior non-MLS best (BK=32 / ns=4): **377.095** vs 255.252 TOPS (**~+47.7%**).
- **MLS BK=64 / ns=2 / mlf=3 / pp** vs prior non-MLS best (BK=32 / ns=4): **327.693** vs 255.252 TOPS (**~+28.4%**).
- **MLS BK=64 / ns=2 / mlf=3** vs prior non-MLS best (BK=32 / ns=4): **293.201** vs 255.252 TOPS (**~+14.9%**).
- **`num_stages=4` vs `2`** on non-MLS BK=32: **255.252** vs 245.557 TOPS (**~+4.0%**).
- Prior MLS @ BK=32 / ns=4 was **244.320** TOPS (~−4.3% vs non-MLS ns4).

### Launch — wdra48 + MLS (`num_stages=4` + BCS, BK=32, prior)

| knob | value |
| --- | --- |
| BLOCK_SIZE_{M,N,K} | 256 × 256 × 32 |
| empty_arrive_after_mmac | **false** |
| enable_v_mmac_cluster | 0 |
| load / mma warps | 4 / 8 |
| num_stages | **4** |
| use_mls | **true** (`tl.matrix_load`) |
| wdra regs (load / main / tail) | 88 / 208 / 208 |
| `TRITON_BUFFER_CACHE_SWIZZLE` | **on** |
| tiles (ceil) | 16 × 18 = 288 |
| source CSV | `gemm_wasp_tune.csv` |
| golden | PASS |

### Perf / bottleneck (PMD `.perf`)

On this scenario (`wdra48`, TN, BCS on):

- With **`empty_arrive_after_mmac=false` (`ea=0`)**, non-MLS BK=32 / `ns=2`: MMAC CU **idle** is mainly waiting on **`ds_read` → `lgkmcnt(0)`**, then **arrive empty barrier** (245.557 TOPS).
- With **`empty_arrive_after_mmac=true` (`ea=1`)**, `ns=2` — empty barrier arrive **after mmac**: perf **drops** to **227.233** TOPS (≈−7.5%). Perf shows **odd gaps / 断流 between consumer `ds_read`s**. Keep **`ea=0`**.
- Non-MLS **`num_stages=4`**, BK=32, `ea=0`: **255.252** TOPS.
- MLS **`num_stages=4`**, BK=32, `ea=0`: **244.320** TOPS (~−4.3% vs non-MLS ns4).
- MLS **`num_stages=2`**, **BK=64**, `ea=0`, `mlf` auto: **281.854** TOPS.
- MLS **`num_stages=2`**, **BK=64**, `ea=0`, **`mlf=3`**: **293.201** TOPS (prior best, VGPR 88/208/208).
- MLS **`num_stages=2`**, **BK=64**, `ea=0`, **`mlf=4`**: **293.172** TOPS (tied with `mlf=3` within ~0.01%).
- MLS **`num_stages=2`**, **BK=64**, `ea=0`, **`mlf=3`**, **`pp=1`**, VGPR **16/244/244**: **327.693** TOPS (prior pingpong best).
- Same + **`abarrier_seq`** (replaces `s_waitcnt`, MLS issues continuously): **377.095** TOPS (current best).

### History

| date | change |
| --- | --- |
| 2026-08-11 | First TN datapoint: wdra48 + BCS @ 256×256×32, `ns=2`, 245.557 TOPS. |
| 2026-08-11 | Perf (`ea=0`, `ns=2`): MMAC idle dominated by `ds_read` `lgkmcnt(0)` then empty-barrier arrive. |
| 2026-08-11 | Tried `ea=1` (`ns=2`): 227.233 TOPS (−7.5%); consumer `ds_read` 断流. |
| 2026-08-11 | `num_stages=4` + BCS @ 256×256×32, `ea=0`: **250.732** TOPS. |
| 2026-08-12 | Same launch + epilogue reorder / OnePTwoC VGPR pad `%12`: **255.252** TOPS (Triton `5f67c818e` / MR #301). |
| 2026-08-12 | First **MLS** datapoint: BK=32 / `ns=4`, **244.320** TOPS (~−4.3% vs non-MLS ns4). |
| 2026-08-12 | **MLS** @ **256×256×64** / `ns=2`: **281.854** TOPS (~+10.4% vs non-MLS ns4). |
| 2026-08-12 | Same + **`mmac_layout_force=3`**: **293.201** TOPS (new best; ~+4.0% vs MLS auto). |
| 2026-08-13 | Same + **`mmac_layout_force=4`**: **293.172** TOPS (tied with `mlf=3` within ~0.01%). |
| 2026-08-13 | Same + **`enable_consumer_pingpong`** + VGPR **16/244/244** (Triton `7225e1068`): **327.693** TOPS (new best; ~+11.8% vs `mlf=3`). |
| 2026-08-14 | Same launch + **`abarrier_seq`** replacing **`s_waitcnt`** (MLS can issue continuously): **377.095** TOPS (new best; ~+15.1% vs pingpong). |

---

<!-- Template for the next shape (copy below the line, fill, add a row to Index):

## Shape: {M}×{N}×{K} (MNK = {M}, {N}, {K})

### Meta
| Item | Value |
| --- | --- |
| MNK | **M={M}, N={N}, K={K}** |
| AB layout | **TT** (A K-major, B N-major) or **TN** (A K-major, B K-major) |
| num_stages | **2** (or 4) |
| arch | gfx946 |
| gcFrequency | |
| triton | |
| bolt_ops | |

### Winners
| mode | TOPS | cycles | wall_s | vs nowasp | JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| nowasp | | | | 1.000× | `M{M}_N{N}_K{K}_{TT|TN}_nowasp.json` |
| wdra48 | | | | | `M{M}_N{N}_K{K}_{TT|TN}_wdra48.json` |

### Launch — nowasp
### Launch — wdra48
### Notes
### History
| date | change |
| --- | --- |
| YYYY-MM-DD | |

-->
