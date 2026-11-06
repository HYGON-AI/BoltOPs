# GEMM API docstrings

The following signatures and docstrings describe every public `boltops.gemm`
interface. Import these functions from `boltops` or `boltops.gemm`.

## `gemm_a16w16`

```python
def gemm_a16w16(x, w, bias=None, dtype=torch.bfloat16, out=None, config=None, activation=None, backend=None):
    """Compute dense FP16/BF16 GEMM, optional bias and fused activation.

    Args:
        x (torch.Tensor): GPU activation [M,K], float16 or bfloat16. The general Triton kernel
            accepts positive-stride views: row-major (K,1) makes K contiguous;
            column-major (1,M) makes M contiguous. Padded positive strides are
            also allowed.
        w (torch.Tensor): GPU weights [N,K], same dtype/device as x. Row-major (K,1)
            makes K contiguous; column-major (1,N) makes N contiguous. The general
            Triton kernel accepts positive-stride views. The recorded MLS/WDRA
            pipeline specifically requires x.stride()=(K,1) and w.stride()=(1,N).
            The logical transpose w.T is internal.
        bias (torch.Tensor | None): Optional [N] bias in x.dtype on the same device,
            with unit stride. Broadcast across M.
        dtype (torch.dtype | None): Output float16, bfloat16 or float32. Defaults to
            bfloat16; None uses x.dtype.
        out (torch.Tensor | None): Optional contiguous output [M,N], on the input
            device with the selected output dtype. Reused and returned directly;
            None allocates a new tensor. Inputs and out must not overlap.
        config (dict[str, object] | None): Complete backend kernel launch options, such
            as
            BLOCK_SIZE_M/N/K, num_warps and num_stages. Overrides recorded/default
            configuration lookup and is copied before use; the caller dict is unchanged.
            None selects cached configuration files for the device and shape. No tuning
            is performed during an ordinary call.
        activation (str | None): Optional "gelu", "gelu_tanh", "silu",
            "silu_exp2", "relu" or "relu6", applied to FP32 results after bias
            and the complete K reduction, before output conversion. None is
            identity. Explicit MLS/WDRA configurations do not support activation.
        backend (str | None): "auto", "triton", "gluon" or "tilelang"; None means
            "auto".
            Explicit selection never retries another backend. Currently only Triton
            is implemented for this operation; unimplemented backends raise
            NotImplementedError for nonempty computation.

    Returns:
        torch.Tensor: Contiguous [M,N] result in dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.
    """
```

## `gemm_a32w32`

```python
def gemm_a32w32(x, w, bias=None, dtype=None, out=None, config=None, backend=None):
    """Compute dense FP32 ``x @ w.T + bias`` using the persistent kernel.

    Args:
        x (torch.Tensor): float32 GPU activation [M,K]. Positive-stride views are allowed:
            (K,1) is K-contiguous, (1,M) is M-contiguous; padded positive strides
            are supported.
        w (torch.Tensor): float32 GPU weights [N,K] on x.device. Positive-stride views are
            allowed: (K,1) is K-contiguous, (1,N) is N-contiguous; padded positive
            strides are supported. The logical transpose w.T is internal.
        bias (torch.Tensor | None): Optional float32 [N] bias on x.device with unit
            stride, broadcast across M.
        dtype (torch.dtype | None): Only torch.float32 is supported; None selects it.
        out (torch.Tensor | None): Optional contiguous output [M,N], on the input
            device with the selected output dtype. Reused and returned directly;
            None allocates a new tensor. Inputs and out must not overlap.
        config (dict[str, object] | None): Complete backend kernel launch options, such
            as
            BLOCK_SIZE_M/N/K, num_warps and num_stages. Overrides recorded/default
            configuration lookup and is copied before use; the caller dict is unchanged.
            None selects cached configuration files for the device and shape. No tuning
            is performed during an ordinary call.
        backend (str | None): "auto", "triton", "gluon" or "tilelang"; None means
            "auto".
            Explicit selection never retries another backend. Currently only Triton
            is implemented for this operation; unimplemented backends raise
            NotImplementedError for nonempty computation.

    Returns:
        torch.Tensor: Contiguous [M,N] result in torch.float32.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.
    """
```

## `batched_gemm_bf16`

```python
def batched_gemm_bf16(x, w, bias=None, dtype=torch.bfloat16, out=None, config=None, backend=None):
    """Compute batched FP16/BF16 ``x[b] @ w[b].T + bias[b]``.

    FP16 and BF16 share this interface despite its historical name.
    Inputs have equal batch counts; batch broadcasting is not supported.

    Args:
        x (torch.Tensor): float16/bfloat16 GPU activation [B,M,K], with positive strides.
            K-contiguous, M-contiguous and padded/strided matrix views are
            supported. Standard contiguous strides are (M*K,K,1). Batch-only
            contiguous storage (only batch stride is 1) is rejected.
        w (torch.Tensor): GPU weights [B,N,K], same B, K, dtype/device as x, with
            positive strides. K-contiguous, N-contiguous and padded/strided
            matrix views are supported. Standard strides are (N*K,K,1).
            Batch-only contiguous storage is rejected; the logical transpose
            is internal and batch broadcasting is not supported.
        bias (torch.Tensor | None): Optional [B,1,N] bias in x.dtype on the
            same device, unit stride along N. Broadcast across M, not B.
        dtype (torch.dtype | None): Output float16 or bfloat16. Defaults to
            bfloat16; None uses x.dtype.
        out (torch.Tensor | None): Optional contiguous output [B,M,N], on the input
            device with the selected output dtype. Reused and returned directly;
            None allocates a new tensor. Inputs and out must not overlap.
        config (dict[str, object] | None): Complete backend kernel launch options, such
            as
            BLOCK_SIZE_M/N/K, num_warps and num_stages. Overrides recorded/default
            configuration lookup and is copied before use; the caller dict is unchanged.
            None selects cached configuration files for the device and shape. No tuning
            is performed during an ordinary call.
        backend (str | None): "auto", "triton", "gluon" or "tilelang"; None means
            "auto".
            Explicit selection never retries another backend. Currently only Triton
            is implemented for this operation; unimplemented backends raise
            NotImplementedError for nonempty computation.

    Returns:
        torch.Tensor: Contiguous [B,M,N] result in dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.

    Notes:
        Equal-length grouped GEMM with this same storage contract uses this
        entry. Ragged offsets, indexed tokens and group masks are not supported.
    """
```

## `gemm_a8w8_blockscale`

```python
def gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=torch.bfloat16, out=None, config=None, backend=None):
    """Compute FP8/INT8 GEMM with A row/K-group and B N/K-block scales.

    For each output element, sum products over K after applying the A/B
    scales for that K group. The A recipe is (1,BK_SCALE) and the B recipe
    is (BN_SCALE,BK_SCALE); these scale blocks are independent of the
    compute tile BLOCK_SIZE_M/N/K.

    Args:
        x (torch.Tensor): GPU activation [M,K], float8_e4m3fn, float8_e5m2 or int8.
            Triton accepts any positive matrix strides, including M-contiguous,
            K-contiguous, transposed and padded/sliced views.
        w (torch.Tensor): GPU weights [N,K], same dtype/device as x. Triton accepts
            any positive matrix strides independently of x; w.T is internal.
        x_scale (torch.Tensor): Numeric float16/bfloat16/float32 GPU scales
            [M,ceil(K/BK_SCALE)] on x.device. One value covers one A row and
            BK_SCALE logical K elements. Both matrix strides may be any positive
            values on the Triton path.
        w_scale (torch.Tensor): Numeric float16/bfloat16/float32 GPU scales
            [ceil(N/BN_SCALE),ceil(K/BK_SCALE)] on x.device, with positive matrix
            strides on Triton. One value covers BN_SCALE weight rows and
            BK_SCALE logical K elements. Effective BN_SCALE/BK_SCALE are inferred
            from the shape as powers of two at least 16; scale shapes must match.
        dtype (torch.dtype | None): Output float16, bfloat16 or float32. Both the
            default and None select bfloat16.
        out (torch.Tensor | None): Optional contiguous output [M,N], on the input
            device with the selected output dtype. Reused and returned directly;
            None allocates a new tensor. Inputs and out must not overlap.
        config (dict[str, object] | None): Complete backend kernel launch options, such
            as
            BLOCK_SIZE_M/N/K, num_warps and num_stages. Overrides recorded/default
            configuration lookup and is copied before use; the caller dict is unchanged.
            None selects cached configuration files for the device and shape. No tuning
            is performed during an ordinary call.
        backend (str | None): "auto", "triton", "gluon" or "tilelang"; None means
            "auto".
            Auto selects Gluon only for a recorded precision/recipe/M/N/K/device/CU
            match, otherwise Triton. Explicit selection never retries a backend.
            TileLang is not implemented; explicit Gluon also requires a supported
            configuration and legal kernel inputs.

    Returns:
        torch.Tensor: Contiguous [M,N] result in dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.

    Notes:
        Scale values are numeric, not encoded
        E8M0. The current tuned Gluon recipe requires contiguous x/w/scales and uses E4M3FN inputs, FP32 A/B scales
        with A(1,128)/B(128,128), BF16 output and complete 128x256 tiles; its K
        is 256..6144 divisible by 256. Auto uses Triton for noncontiguous matrix views; explicit Gluon reports
        unsupported storage rather than copying tensors. Other combinations use the native Triton
        path when supported. There is no software matrix-instruction fallback.
    """
```

## `gemm_a8w8`

```python
def gemm_a8w8(x, w, x_scale, w_scale, bias=None, dtype=torch.bfloat16, out=None, config=None, backend=None):
    """Compute whole-K tensor/row/channel-scaled FP8 or INT8 GEMM.

    The result is ``(x @ w.T) * x_scale * w_scale + bias``, with scales
    broadcast over the output. Scales are dequantization multipliers and
    are applied after the complete K reduction, not to each K block.

    Args:
        x (torch.Tensor): GPU activation [M,K], float8_e4m3fn, float8_e5m2 or int8.
            Any positive matrix strides are accepted, including M-contiguous,
            K-contiguous, transposed and padded/sliced views.
        w (torch.Tensor): GPU weights [N,K], same dtype/device as x. Any positive
            matrix strides are accepted independently of x; the logical transpose
            w.T is internal.
        x_scale (torch.Tensor): Numeric float16/bfloat16/float32 GPU scales on x.device.
            One value per tensor or M values, one per row and covering the full K.
            Scalar and flattenable vector views with positive element stride are
            supported; no implicit contiguous copy is made.
        w_scale (torch.Tensor): Numeric GPU scales on x.device, same dtype as x_scale.
            One value per tensor or N values, one per output channel covering
            full K. Scalar and flattenable positive-stride vector views are
            supported.
        bias (torch.Tensor | None): Optional floating-point [N] bias on x.device
            with unit stride, added after scale multiplication.
        dtype (torch.dtype | None): Output float16, bfloat16 or float32. Both the
            default and None select bfloat16.
        out (torch.Tensor | None): Optional contiguous output [M,N], on the input
            device with the selected output dtype. Reused and returned directly;
            None allocates a new tensor. Inputs and out must not overlap.
        config (dict[str, object] | None): Complete backend kernel launch options, such
            as
            BLOCK_SIZE_M/N/K, num_warps and num_stages. Overrides recorded/default
            configuration lookup and is copied before use; the caller dict is unchanged.
            None selects cached configuration files for the device and shape. No tuning
            is performed during an ordinary call.
        backend (str | None): "auto", "triton", "gluon" or "tilelang"; None means
            "auto".
            Explicit selection never retries another backend. Currently only Triton
            is implemented for this operation; unimplemented backends raise
            NotImplementedError for nonempty computation.

    Returns:
        torch.Tensor: Contiguous [M,N] result in dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.

    Notes:
        Scale values are ordinary numbers, not encoded E8M0 bytes. Input
        FP8/INT8 operands are passed to native matrix instructions without
        software upcasting; unsupported target combinations fail compilation.
    """
```

## `gemm_a16w4`

```python
def gemm_a16w4(x, w, w_scales, w_zeros=None, dtype=None, out=None, config=None, backend=None):
    """Compute A16 times AWQ INT4 weights with group scales and zero points.

    The logical weight is [K,N]; unpack its INT4 values and zero points
    using the existing AWQ nibble order, subtract zero points, multiply
    by the corresponding K-group scale, then compute ``x @ weight``.

    Args:
        x (torch.Tensor): float16/bfloat16 GPU activation [M,K]. Any positive matrix
            strides are supported, including transpose, padded and sliced views.
        w (torch.Tensor): int32 GPU weights [K,N/8] on x.device, with positive stored
            word strides. Each integer packs eight unsigned INT4 values in AWQ
            order along logical N. The carrier tensor may be transposed or
            strided physically; the logical packing remains AWQ [K,N/8]. This
            is distinct from uint8 E2M1 FP4 storage.
        w_scales (torch.Tensor): [K/G,N] GPU scales, same dtype/device as x, with positive
            matrix strides. G is inferred from the first dimension and must
            divide K, with G in {32,64,128,K}; K and N must be positive.
        w_zeros (torch.Tensor | None): int32 [K/G,N/8] GPU packed zero points on x.device,
            using the same AWQ order as w and positive stored word strides.
            None means integer zero, not an implicit symmetric zero of eight.
        dtype (torch.dtype | None): Must equal w_scales.dtype when supplied.
            None selects that dtype, which also equals x.dtype.
        out (torch.Tensor | None): Optional contiguous output [M,N], on the input
            device with the selected output dtype. Reused and returned directly;
            None allocates a new tensor. Inputs and out must not overlap.
        config (dict[str, object] | None): Complete backend kernel launch options, such
            as
            BLOCK_SIZE_M/N/K, num_warps and num_stages. Overrides recorded/default
            configuration lookup and is copied before use; the caller dict is unchanged.
            None selects cached configuration files for the device and shape. No tuning
            is performed during an ordinary call.
        backend (str | None): "auto", "triton", "gluon" or "tilelang"; None means
            "auto".
            Explicit selection never retries another backend. Currently only Triton
            is implemented for this operation; unimplemented backends raise
            NotImplementedError for nonempty computation.

    Returns:
        torch.Tensor: Contiguous [M,N] result in x.dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.

    Notes:
        This operation preserves the repository AWQ packing contract and does
        not infer or repack other INT4 formats.
    """
```

## `gemm_w4a4`

```python
def gemm_w4a4(x, w, x_scales, w_scales, dtype=torch.bfloat16, out=None, config=None, backend=None):
    """Compute native MXFP4 GEMM from prequantized E2M1 A/B operands.

    Both operands already include packed FP4 values and E8M0 K32
    scales. No activation quantization is performed by this interface.

    Args:
        x (torch.Tensor): uint8 GPU activation [M,K/2], with positive stored byte
            strides. Earlier/later logical K elements occupy the low/high nibble
            of each byte. The carrier may be transposed or strided physically;
            logical K must be positive and divisible by 32.
        w (torch.Tensor): uint8 GPU weights [N,K/2] on x.device with positive stored
            byte strides. Packing matches x; physical transpose/padded views
            are accepted and the logical weight transpose is internal.
        x_scales (torch.Tensor): uint8 [M,K/32] E8M0-encoded GPU scales on x.device,
            with positive matrix strides. Each scale covers 32 logical K elements
            in one activation row; scale bytes are not packed FP4 nibbles.
        w_scales (torch.Tensor): uint8 [N,K/32] E8M0-encoded GPU scales on x.device,
            with positive matrix strides. Each scale covers 32 logical K elements
            in one weight row.
        dtype (torch.dtype | None): Output float16 or bfloat16. Both the default
            and None select bfloat16.
        out (torch.Tensor | None): Optional contiguous output [M,N], on the input
            device with the selected output dtype. Reused and returned directly;
            None allocates a new tensor. Inputs and out must not overlap.
        config (dict[str, object] | None): Complete backend kernel launch options, such
            as
            BLOCK_SIZE_M/N/K, num_warps and num_stages. Overrides recorded/default
            configuration lookup and is copied before use; the caller dict is unchanged.
            None selects cached configuration files for the device and shape. No tuning
            is performed during an ordinary call.
        backend (str | None): "auto", "triton", "gluon" or "tilelang"; None means
            "auto".
            Explicit selection never retries another backend. Currently only Triton
            is implemented for this operation; unimplemented backends raise
            NotImplementedError for nonempty computation.

    Returns:
        torch.Tensor: Contiguous [M,N] result in dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.

    Notes:
        Currently requires gfx946 native FP4 support. Unsupported instructions
        are reported as compiler/runtime failures; no software FP4 GEMM
        replacement is used.
    """
```

## `batched_gemm_a16w4`

```python
def batched_gemm_a16w4(x, w, w_scales, dtype=torch.bfloat16, out=None, config=None, transpose_bm=False, out_scale=None, backend=None):
    """Compute batched A16 x FP4 GEMM with online MXFP4 activation quantization.

    Each K tile of x is quantized inside the GEMM kernel to packed E2M1
    values and E8M0 K32 scales, then multiplied by prequantized w. This
    changes activation values through quantization; it is not exact
    FP16/BF16 x dequantized-FP4 multiplication.

    Args:
        x (torch.Tensor): float16/bfloat16 GPU activation [B,M,K], with positive strides.
            M-contiguous, K-contiguous and padded/strided matrix views are
            supported. Batch-only contiguous storage (only the batch stride is 1)
            is rejected. Logical K must be positive and divisible by 32.
        w (torch.Tensor): uint8 GPU weights [B,N,K/2] on x.device, with positive
            stored byte strides. Earlier/later K elements occupy the low/high
            nibble of each E2M1 byte. Physical transpose/padded matrix views are
            supported; batch count must match x. Batch-only contiguous storage
            is rejected.
        w_scales (torch.Tensor): uint8 [B,N,K/32] E8M0-encoded GPU scales on x.device;
            one scale per weight row and 32 logical K elements. Positive matrix
            strides are supported; batch-only contiguous storage is rejected.
        dtype (torch.dtype | None): Output float16 or bfloat16. Defaults to
            bfloat16; None uses x.dtype.
        out (torch.Tensor | None): Optional contiguous output [B,M,N], or [M,B,N] when
            transpose_bm=True, on the input
            device with the selected output dtype. Reused and returned directly;
            None allocates a new tensor. Inputs and out must not overlap.
        config (dict[str, object] | None): Complete backend kernel launch options, such
            as
            BLOCK_SIZE_M/N/K, num_warps and num_stages. Overrides recorded/default
            configuration lookup and is copied before use; the caller dict is unchanged.
            None selects cached configuration files for the device and shape. No tuning
            is performed during an ordinary call.
        transpose_bm (bool): False writes contiguous [B,M,N]; True writes contiguous
            [M,B,N]. Both direct stores and split-K reduction use output strides
            to write the requested layout without a separate transpose kernel.
        out_scale (torch.Tensor | None): Optional single-element floating-point GPU
            tensor on x.device, shared by all batches and output elements.
            Divide FP32 accumulators by this scalar before output conversion.
            Split-K applies it to each FP32 partial before reduction. None
            leaves the result unscaled. It does not select output quantization
            or change dtype, and the caller tensor is not modified.
        backend (str | None): "auto", "triton", "gluon" or "tilelang"; None means
            "auto".
            Explicit selection never retries another backend. Currently only Triton
            is implemented for this operation; unimplemented backends raise
            NotImplementedError for nonempty computation.

    Returns:
        torch.Tensor: Contiguous [B,M,N], or [M,B,N] when transpose_bm=True result in
        dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.

    Notes:
        Currently requires gfx946 native FP4 support. Config may select split-K
        through NUM_KSPLIT; intermediate partials are FP32. No software FP4 GEMM
        replacement or separate activation-quantization launch is used.

        A known gfx946 PMD/compiler numerical mismatch affects K-contiguous
        FP16/BF16 x with varying K32 weight scales at BM16/BN16/BK64,
        num_warps=4, num_stages=1 and NUM_KSPLIT=1. It reproduces with the
        target-main device kernel; affected numerical tests are strict XFAILs,
        not correctness passes. No software instruction replacement is used.
    """
```
