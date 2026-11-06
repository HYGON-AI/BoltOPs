# GEMM interfaces and independent backends

Each public operation has one entry point, with its own JSON configuration
under `boltops/gemm/configs/`. All providers are independent (`triton`,
`gluon`, `tilelang`). There are no public aliases named after backend kernels.

| Public interface | Computation / input contract | Previous public entry points |
| --- | --- | --- |
| `gemm_dense` | Native A[M,K] @ B[N,K].T, optional bias | `matmul_persistent`, `gemm_a16w16` |
| `gemm_scaled` | Native block-scale or whole-K tensor/row/channel scale; FP8, INT8, FP16/BF16 where registered | `gemm_a8w8_blockscale`, `w8a8_triton_block_scaled_mm`, `triton_scaled_mm`, `w8a8_block_int8_matmul` |
| `gemm_batched` | Exact native A[B,M,K] @ B[B,N,K].T, optional [B,1,N] bias | `batched_gemm_bf16` |
| `gemm_grouped` | Group-addressed GEMM; equal groups or explicitly registered addressing modes | Independent grouped contracts |
| `gemm_scaled(..., b_format="awq_int4")` | A16 + packed INT4, optional zero points (default zero), AWQ nibble order | `awq_gemm_triton` |
| `gemm_scaled(..., a_format="fp4_e2m1", b_format="fp4_e2m1")` | Prequantized packed E2M1 A/B and E8M0 K32 scales | `gemm_afp4wfp4` |
| `gemm_batched_a16wfp4_online_quant` | Existing online MXFP4 activation quantization + packed FP4 B | `batched_gemm_a16wfp4` |

AWQ and prequantized MXFP4 are formats/recipes of `gemm_scaled`, not separate
public operations. Online A16WFP4 remains separate because it performs lossy
activation quantization inside the GEMM K loop. Each row/K32 group derives
its dynamic E8M0 scale and FP4 values, then immediately calls native scaled
dot; it does not materialize a whole quantized A in global memory. This is not
exact unquantized A16 x FP4 multiplication. Missing native support still fails
without emulation or backend retries.

Old public wrappers are removed without compatibility aliases. Existing
implementation functions remain in `boltops.gemm.triton.*` for backend-specific
configuration, partial split-K results, specialized output layouts and fused
activation experiments. These implementation controls are not new public
GEMM operations. NN callers of `matmul_persistent(a,b)` migrate to
`gemm_dense(a,b.T)`. `triton_scaled_mm` callers also transpose B and supply an
explicit scale recipe. Plain batched does not broadcast batch dimensions.

Scaled format examples:

```python
# AWQ storage is explicitly [K,N/8], [K/G,N], [K/G,N/8], all contiguous.
# A has no scale; its recipe entry is unused. Missing zero points mean zero.
y = gemm_scaled(a16, packed_b, b_scale=scales, b_zero_points=zeros,
                b_format="awq_int4", scale_recipe=((1, 1), (1, G)))

# E2M1 inputs store two values per byte: A[M,K/2], B[N,K/2].
# Both scales are uint8 E8M0 with shapes [M,K/32] and [N,K/32].
y = gemm_scaled(a4, b4, a_scale, b_scale,
                a_format="fp4_e2m1", b_format="fp4_e2m1",
                scale_recipe=((1, 32), (1, 32)), scale_encodings=("e8m0", "e8m0"))
```

AWQ output dtype equals its scale dtype. MXFP4 output defaults to BF16.
`gemm_batched_a16wfp4_online_quant` takes A[B,M,K], packed B[B,N,K/2], E8M0 B scales
[B,N,K/32]; A scales are generated in the kernel. Both sides use recipe
(1,32), and packed paths require K divisible by 32.

All interfaces accept `backend="auto"`, `"triton"`, `"gluon"`, `"tilelang"`.
Automatic selection queries the selected operation's own configuration file
before defaults. Explicit backend selection never retries another backend on
compile/execute failure. Unsupported combinations raise NotImplementedError;
a registered backend missing its runtime raises ImportError. Gluon and TileLang
remain empty registration points in this standalone upstream PR.

The five files are `gemm_dense.json`, `gemm_scaled.json`, `gemm_batched.json`,
`gemm_grouped.json`, and `gemm_batched_a16wfp4_online_quant.json`. Each carries its interface identifier. Schema 3
separates `match` (arch, num_cus, MNK, dtype/scale/layout contracts) from
`config` (backend, implementation, tile and optional existing config file).
Tile is not a matching condition. No measurements, frequency, commit IDs or
evidence paths are stored. Current tuned entries reuse existing gfx946 48-CU
FP16 TT configurations. Source kernels and default configurations are unchanged.
