# gemm_fp16 WDRA VGPR allocation strategy

When a **scenario** is fixed (shape × mode × MLS × tile family), pin the
WDRA VGPR budgets here. Winner JSON under `boltops/gemm/triton/configs/gemm_fp16/gfx946/` remains the runnable
source of truth; this file is the **policy note** (why / what / constraints).

Triton must **not** rewrite these budgets. Callers pass
`wdra_num_load_regs` / `wdra_num_mma_regs_main` / `wdra_num_mma_regs_tail`
unchanged into `SetVgpr` / `WdraInit`.

## Clang constraints (must hold)

Equal-weight branches, VGPR granularity = 4:

| topology | modes | branches (budgets) | required |
| --- | --- | --- | --- |
| OnePTwoC | `wdra48` | load + main + tail | `sum % 12 == 0` |
| TwoPTwoC | `wdra88` | load + load + main + tail | `sum % 16 == 0` |

Also: each budget `% 4 == 0`, and typically `<= 256` with `-support-512-vgprs`.

Triton (`ConvertWarpSpecializeToLLVM`) must pad OnePTwoC live budgets with
**`sum % 12`**, not `% 16`. Example: **88+208+208=504** already satisfies
`504%12==0` and avg 168 (`%4==0`); padding to 512 makes `sum%3!=0` and Clang
rejects the kernel.

`BranchAvailableVGPRs[i]` in Clang dumps is **not** `[load, main, tail]` order;
for OnePTwoC it prints roughly **`[tail, main, load]`**. Match by value, not index.

## Tile-family note (TT + `ds_read_m32x16_fp16`)

For **TT** (A K-major, B N-major) when the LDS path uses **`ds_read_m32x16_fp16`**
and the block is **256×64×256 (M×K×N)** (= **256×256×64** in `BLOCK_SIZE_M×N×K`):

| Item | Value |
| --- | --- |
| AB layout | **TT** |
| num_stages | **2** |
| LDS read | `ds_read_m32x16_fp16` |
| tile (M×N×K) | 256 × 256 × 64 |
| **VGPR enough?** | **yes** — Clang print **196 / 196 / 112** (= load / main / tail **112 / 196 / 196**) |
| sum | 504 (`504 % 12 == 0`) |

So on this TT + matrix-load tile, do **not** bump load above 112 for headroom;
MMA main/tail keep 196. Matches current `gemm_fp16.py` `_DEFAULT_WDRA_*` and the
large-tile rows below (1536³ / 3072²×K / 4096…).

| date | note |
| --- | --- |
| 2026-08-11 | Confirmed: TT + `ds_read_m32x16_fp16` @ 256×64×256 (MKN) → 196/196/112 sufficient. |

## Tile-family note (TN + `ds_read2_b64`)

For **TN** (A K-major, B K-major) at the same block **256×64×256 (M×K×N)**
(= **256×256×64** M×N×K), LDS falls back to **`ds_read2_b64`** (not
`ds_read_m32x16_fp16`). MMA (mmac) consumer VGPR pressure is much higher:

| Item | Value |
| --- | --- |
| AB layout | **TN** |
| num_stages | **2** |
| LDS read | `ds_read2_b64` |
| tile (M×N×K) | 256 × 256 × 64 |
| tried budgets (Clang print ≈ tail/main/load) | **208 / 208 / 88** (= load / main / tail **88 / 208 / 208**, sum 504) |
| **VGPR enough?** | **no** — mmac consumer usage still overflows / cannot be held |

Do **not** treat TT’s 112/196/196 (or 196/196/112 print) as valid for this TN
tile: the `ds_read2_b64` path needs a different tile and/or VGPR strategy.

**Workaround that fits (non-MLS):** shrink **BK to 32** (tile **256×256×32** / M×K×N
**256×32×256**) with the same Clang budgets **208 / 208 / 88** (=
**88 / 208 / 208**). Best non-MLS: **255.252** TOPS (`ns=4`).

**MLS unlocks BK=64:** same **88 / 208 / 208** with `use_mls=true` @
**256×256×64** / `ns=2`. With **`mmac_layout_force=3`** → **293.201** TOPS
(prior best without pingpong); auto layout was **281.854**.

**Pingpong rebalance:** `enable_consumer_pingpong=true` + VGPR **16 / 244 / 244**
(still `sum=504`, `%12==0`; Clang `s_set_vgpr_size` 16/244/244) @ same MLS
BK=64 / ns=2 / `mlf=3` → **327.693** TOPS. Same budgets + **`abarrier_seq`**
replacing `s_waitcnt` (MLS issues continuously) → **377.095** TOPS (TN best).
Same budgets on **TT** + MLS + **`tl.matrix_store`** → **246.803** TOPS @ K=512
(short-K: WDRA setup + store dominate, MMAC idle), **389.372** TOPS @ K=4096,
and **392.254** TOPS @ K=16384.

| date | note |
| --- | --- |
| 2026-08-11 | TN @ 256×64×256 (MKN) uses `ds_read2_b64`; 208/208/88 still not enough for mmac consumers. |
| 2026-08-11 | TN @ 256×32×256 (MKN) / 256×256×32: 88/208/208 fits; promoted first datapoint (245.557 @ ns=2). |
| 2026-08-12 | Same budgets @ ns=4 + epilogue reorder / Triton pad `%12`: **255.252** TOPS (non-MLS). |
| 2026-08-12 | MLS @ 256×256×64 / ns=2 with 88/208/208: **281.854** TOPS. |
| 2026-08-12 | Same + `mmac_layout_force=3`: **293.201** TOPS (new best). |
| 2026-08-13 | Same + `mmac_layout_force=4`: **293.172** TOPS (tied with `mlf=3`). |
| 2026-08-13 | Pingpong + VGPR **16/244/244** @ MLS BK=64 / `mlf=3`: **327.693** TOPS (new best). |
| 2026-08-14 | Same VGPR + `abarrier_seq` replacing `s_waitcnt` (MLS issues continuously): **377.095** TOPS (TN best). |
| 2026-08-20 | Same VGPR on TT + MLS + `tl.matrix_store` @ K=512: **246.803** TOPS (short-K; WDRA setup + store, MMAC idle). |
| 2026-08-19 | Same VGPR on TT + MLS + `tl.matrix_store`: **389.372** TOPS (TT best @ K=4096; Clang used 2/225/225). |
| 2026-08-20 | Same VGPR on TT + MLS + `tl.matrix_store` @ K=16384: **392.254** TOPS. |

## Scenario table

Key: `(M,N,K)` + `mode` + `use_mls` + tile + **AB layout** (`TT`/`TN`) + **num_stages**
(see [`RESULTS.md`](RESULTS.md#how-to-read)). Add a row when a scenario is chosen;
do not overwrite older rows — move them to History under that section.

| scenario | mode | MLS | AB | tile (M×N×K) | ns | load | main | tail | sum | notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [512³ wdra48](#scenario-512cube-wdra48-non-mls) | wdra48 | no | TT | 128³ | 2 | 160 | 160 | 160 | 480 | promoted winner |
| [1536³ wdra48](#scenario-1536cube-wdra48-non-mls) | wdra48 | no | TT | 256×256×64 | 2 | 112 | 196 | 196 | 504 | large-tile win |
| 3072×3072×128 wdra48 | wdra48 | no | TT | 256×256×64 | 2 | 112 | 196 | 196 | 504 | first datapoint; same budgets as 1536³ |
| 3072×3072×256 wdra48 | wdra48 | no | TT | 256×256×64 | 2 | 112 | 196 | 196 | 504 | first datapoint; same budgets as 1536³ |
| 3072×3072×512 wdra48 | wdra48 | no | TT | 256×256×64 | 2 | 112 | 196 | 196 | 504 | first datapoint; same budgets as 1536³ |
| 3072×3072×1024 wdra48 | wdra48 | no | TT | 256×256×64 | 2 | 112 | 196 | 196 | 504 | first datapoint; same budgets as 1536³ |
| 3072×3072×2048 wdra48 | wdra48 | no | TT | 256×256×64 | 2 | 112 | 196 | 196 | 504 | first datapoint; same budgets as 1536³ |
| 3072×3072×2048 wdra48 + BCS | wdra48 | no | TT | 256×256×64 | 2 | 112 | 196 | 196 | 504 | `TRITON_BUFFER_CACHE_SWIZZLE`; best on this shape |
| 4096×4096×4608 wdra48 + BCS | wdra48 | no | TT | 256×256×64 | 2 | 112 | 196 | 196 | 504 | `TRITON_BUFFER_CACHE_SWIZZLE`; first datapoint |
| 4096×4608×4096 wdra48 + BCS | wdra48 | no | TT | 256×256×64 | 2 | 112 | 196 | 196 | 504 | `TRITON_BUFFER_CACHE_SWIZZLE`; prior TT; **230.378** TOPS |
| 4096×4608×512 TT wdra48 + BCS MLS mstore | wdra48 | **yes** | TT | **256×256×64** | 2 | **16** | **244** | **244** | 504 | TT @ K=512: **246.803** TOPS; short-K WDRA setup+store (MMAC idle) |
| 4096×4608×4096 TT wdra48 + BCS MLS mstore | wdra48 | **yes** | TT | **256×256×64** | 2 | **16** | **244** | **244** | 504 | TT @ K=4096: **389.372** TOPS; `mlf=3` + `pp=1` + `tl.matrix_store` |
| 4096×4608×16384 TT wdra48 + BCS MLS mstore | wdra48 | **yes** | TT | **256×256×64** | 2 | **16** | **244** | **244** | 504 | TT @ K=16384: **392.254** TOPS; same launch |
| 4096×4608×4096 TN wdra48 + BCS | wdra48 | no | TN | 256×256×32 | 2 | 88 | 208 | 208 | 504 | `TRITON_BUFFER_CACHE_SWIZZLE`; BK=32 fits (BK=64 does not) |
| 4096×4608×4096 TN wdra48 + BCS ns4 | wdra48 | no | TN | 256×256×32 | 4 | 88 | 208 | 208 | 504 | prior non-MLS best; **255.252** TOPS; needs Triton pad `%12` |
| 4096×4608×4096 TN wdra48 + BCS ns4 MLS | wdra48 | **yes** | TN | 256×256×32 | 4 | 88 | 208 | 208 | 504 | first MLS; **244.320** TOPS (~−4.3% vs non-MLS ns4) |
| 4096×4608×4096 TN wdra48 + BCS MLS BK64 | wdra48 | **yes** | TN | **256×256×64** | 2 | 88 | 208 | 208 | 504 | MLS BK=64 / `mlf` auto; **281.854** TOPS |
| 4096×4608×4096 TN wdra48 + BCS MLS BK64 mlf3 | wdra48 | **yes** | TN | **256×256×64** | 2 | 88 | 208 | 208 | 504 | prior best without pingpong; **293.201** TOPS; `mmac_layout_force=3` |
| 4096×4608×4096 TN wdra48 + BCS MLS BK64 mlf4 | wdra48 | **yes** | TN | **256×256×64** | 2 | 88 | 208 | 208 | 504 | **293.172** TOPS; `mmac_layout_force=4`; tied with mlf3 |
| 4096×4608×4096 TN wdra48 + BCS MLS BK64 mlf3 pp | wdra48 | **yes** | TN | **256×256×64** | 2 | **16** | **244** | **244** | 504 | prior pingpong; **327.693** TOPS; `mlf=3` + `pp=1` |
| 4096×4608×4096 TN wdra48 + BCS MLS BK64 mlf3 pp abarrier_seq | wdra48 | **yes** | TN | **256×256×64** | 2 | **16** | **244** | **244** | 504 | **current best** **377.095** TOPS; `abarrier_seq` replaces `s_waitcnt` so MLS can issue continuously |

---

<a id="scenario-512cube-wdra48-non-mls"></a>

## 512³ / wdra48 / non-MLS

| Item | Value |
| --- | --- |
| shape | M=N=K=512 |
| AB layout | **TT** (A K-major, B N-major) |
| mode | `wdra48` (OnePTwoC, load/mma warps = 4/8) |
| use_mls | **false** |
| tile | 128 × 128 × 128 |
| num_stages | **2** |
| empty_arrive_after_mmac | false |
| enable_v_mmac_cluster | 0 |
| **VGPR (load / main / tail)** | **160 / 160 / 160** |
| sum / avg | 480 / 160 (`480 % 12 == 0`, avg `% 4 == 0`) |
| winner JSON | [`gfx946/M512_N512_K512_TT_wdra48.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M512_N512_K512_TT_wdra48.json) |
| perf (PMD) | ~17.516 TOPS vs nowasp ~15.406 (~1.14×); see [`RESULTS.md`](RESULTS.md) |

### Strategy

- Prefer **balanced** budgets on this path: MMA main/tail need headroom for
  128³ accumulators; equal 160 keeps Clang avg clean and avoids starving load.
- Rejected for this scenario: **88 / 144 / 140** with small tiles (64³ …) —
  wdra48 stayed ~0.96 TOPS (register / LDS pressure mismatch).
- `gemm_fp16.py` module defaults (`_DEFAULT_WDRA_*`) may differ for other
  tiles; **this scenario must launch with 160/160/160** (JSON / explicit kwargs).

### History

| date | allocation | note |
| --- | --- | --- |
| 2026-08-06 | 160 / 160 / 160 | Promoted with 128³; first solid 512³ wdra48 win. |
| (earlier) | 88 / 144 / 140 | Too tight for 128³ / ineffective with small tiles. |

---

<a id="scenario-1536cube-wdra48-non-mls"></a>

## 1536³ / wdra48 / non-MLS

| Item | Value |
| --- | --- |
| shape | M=N=K=1536 |
| AB layout | **TT** (A K-major, B N-major) |
| mode | `wdra48` (OnePTwoC, load/mma warps = 4/8) |
| use_mls | **false** |
| tile | 256 × 256 × 64 |
| num_stages | **2** |
| empty_arrive_after_mmac | false |
| enable_v_mmac_cluster | 0 |
| **VGPR (load / main / tail)** | **112 / 196 / 196** |
| sum / avg | 504 / 168 (`504 % 12 == 0`, avg `% 4 == 0`) |
| Clang Available (print order) | `[196, 196, 112]` ≈ tail / main / load |
| winner JSON | [`gfx946/M1536_N1536_K1536_TT_wdra48.json`](../../boltops/gemm/triton/configs/gemm_fp16/gfx946/M1536_N1536_K1536_TT_wdra48.json) |
| perf (PMD) | ~133.050 TOPS vs nowasp **115.468** @ 128³ (~1.15×); see [`RESULTS.md`](RESULTS.md) |

### Strategy

- Same budgets as current `gemm_fp16.py` defaults (`_DEFAULT_WDRA_*`).
- Prefer **asymmetric** 112/196/196 on this large MN tile: MMA keeps 196;
  load stays at 112 (dump NumVGPR load ~88).
- Do not silently retune to 160/160/160 here — that set is the 512³/128³ policy.

### History

| date | allocation | note |
| --- | --- | --- |
| 2026-08-10 | 112 / 196 / 196 | Solo wdra48 @ 256×256×64 → 133.050 TOPS. |

---

## How to extend

1. Lock scenario knobs (shape, mode, MLS, tile, barrier cluster).
2. Pick budgets that satisfy the Clang row above; verify with
   `GPU_DFLAGS="['StatLog']"` that `BranchAvailable` matches intended values
   (remember reverse print order).
3. Add / update a row in the scenario table and a `## …` section.
4. Promote winner JSON; link it from [`RESULTS.md`](RESULTS.md).
