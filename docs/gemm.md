# GEMM API docstrings

The following signatures and docstrings describe every public `boltops.gemm`
interface. Import these functions from `boltops` or `boltops.gemm`.

## `gemm_a16w16`

```python
def gemm_a16w16(x, w, bias=None, dtype=torch.bfloat16, out=None, activation=None, backend=None):
    """Compute dense FP16/BF16 GEMM, optional bias and fused activation.

    Args:
        x (torch.Tensor): GPU input matrix [M,K] in float16 or bfloat16.
        w (torch.Tensor): GPU weight matrix [N,K] on the same device and in the
            same dtype as x. The product is x @ w.T.
        bias (torch.Tensor | None): Optional [N] bias in x.dtype on the same
            device, with unit stride. Broadcast across M.
        dtype (torch.dtype | None): Dtype used when allocating the output.
            Defaults to bfloat16. None uses x.dtype.
        out (torch.Tensor | None): Optional preallocated [M,N] output on the
            input device. Returned directly when supplied. It must not overlap
            the inputs.
        activation (str | None): Activation applied after the matrix product and
            bias, before output conversion. Supports "gelu", "gelu_tanh",
            "silu", "silu_exp2", "relu", and "relu6". None applies no activation.
        backend (str | None): "triton" to request a backend, or
            "auto" or None to select a backend automatically.

    Returns:
        torch.Tensor: [M,N] result.
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
def gemm_a32w32(x, w, bias=None, dtype=None, out=None, backend=None):
    """Compute dense FP32 ``x @ w.T + bias``.

    Args:
        x (torch.Tensor): float32 GPU input matrix [M,K] with positive strides.
        w (torch.Tensor): float32 GPU weight matrix [N,K] on x.device, with
            positive strides.
        bias (torch.Tensor | None): Optional float32 [N] bias on x.device,
            with unit stride. Broadcast across M.
        dtype (torch.dtype | None): Dtype used to allocate the output. None selects float32, the
            supported output dtype.
        out (torch.Tensor | None): Optional [M,N] output on
            the input device. Returned directly when supplied. It must not
            overlap the inputs or contain overlapping elements.
        backend (str | None): "triton" to request a backend, or
            "auto" or None to select a backend automatically.

    Returns:
        torch.Tensor: [M,N] result in torch.float32.
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
def batched_gemm_bf16(x, w, bias=None, dtype=torch.bfloat16, out=None, backend=None):
    """Compute batched FP16/BF16 ``x[b] @ w[b].T + bias[b]``.

    FP16 and BF16 share this interface despite its historical name.
    Inputs have equal batch counts; batch broadcasting is not supported.

    Args:
        x (torch.Tensor): float16 or bfloat16 GPU input [B,M,K] with positive
            strides. If B > 1 and the batch stride is 1, a matrix dimension
            must also have unit stride.
        w (torch.Tensor): GPU weights [B,N,K] with the same dtype and device
            as x, with matching B and K. The same stride rules as x apply.
        bias (torch.Tensor | None): Optional [B,1,N] bias in x.dtype on
            x.device, with unit stride along N. Broadcast across M.
        dtype (torch.dtype | None): Allocation dtype (float16 or bfloat16). Defaults to
            bfloat16. None uses x.dtype.
        out (torch.Tensor | None): Optional [B,M,N] output on
            the input device. Returned directly when supplied. It must not
            overlap the inputs or contain overlapping elements.
        backend (str | None): "triton" to request a backend, or
            "auto" or None to select a backend automatically.

    Returns:
        torch.Tensor: [B,M,N] result in dtype.
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
def gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=torch.bfloat16, out=None, backend=None):
    """Compute FP8/INT8 GEMM with A row/K-group and B N/K-block scales.

    For each output element, sum products over K after applying the A/B
    scales for that K group. The A recipe is (1,BK_SCALE) and the B recipe
    is (BN_SCALE,BK_SCALE).

    Args:
        x (torch.Tensor): GPU input [M,K] in float8_e4m3fn, float8_e5m2, or int8.
        w (torch.Tensor): GPU weights [N,K] with the same dtype and device as x.
        x_scale (torch.Tensor): float16, bfloat16, or float32 scales
            [M,ceil(K/BK_SCALE)] on x.device. Each scale covers one input row
            and BK_SCALE consecutive K elements.
        w_scale (torch.Tensor): float16, bfloat16, or float32 scales
            [ceil(N/BN_SCALE),ceil(K/BK_SCALE)] on x.device. Each scale covers
            BN_SCALE weight rows and BK_SCALE consecutive K elements. Group
            sizes are inferred from scale shapes and must be powers of two
            of at least 16. The K grouping must match x_scale.
        dtype (torch.dtype | None): Allocation dtype (float16, bfloat16, or float32).
            Defaults to bfloat16. None also selects bfloat16.
        out (torch.Tensor | None): Optional [M,N] output on
            the input device. Returned directly when supplied. It must not
            overlap the inputs or contain overlapping elements.
        backend (str | None): "triton" or "gluon" to request a backend, or
            "auto" or None to select a backend automatically.

    Returns:
        torch.Tensor: [M,N] result in dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.
    """
```

## `gemm_a8w8`

```python
def gemm_a8w8(x, w, x_scale, w_scale, bias=None, dtype=torch.bfloat16, out=None, backend=None):
    """Compute whole-K tensor/row/channel-scaled FP8 or INT8 GEMM.

    The result is ``(x @ w.T) * x_scale * w_scale + bias``, with scales
    broadcast over the output. Scales are dequantization multipliers and
    are applied after the complete K reduction, not to each K block.

    Args:
        x (torch.Tensor): GPU input [M,K] in float8_e4m3fn, float8_e5m2, or int8,
            with positive strides.
        w (torch.Tensor): GPU weights [N,K] with the same dtype and device as x
            and positive strides.
        x_scale (torch.Tensor): float16, bfloat16, or float32 scales on x.device.
            Supply a scalar or a positive-stride view flattenable to M values,
            one per input row.
        w_scale (torch.Tensor): Scales with the same dtype and device as
            x_scale. Supply a scalar or a positive-stride view flattenable to
            N values, one per output channel.
        bias (torch.Tensor | None): Optional floating-point [N] bias on x.device
            with unit stride. Added after scaling.
        dtype (torch.dtype | None): Allocation dtype (float16, bfloat16, or float32).
            Defaults to bfloat16. None also selects bfloat16.
        out (torch.Tensor | None): Optional [M,N] output on
            the input device. Returned directly when supplied. It must not
            overlap the inputs or contain overlapping elements.
        backend (str | None): "triton" to request a backend, or
            "auto" or None to select a backend automatically.

    Returns:
        torch.Tensor: [M,N] result in dtype.
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
def gemm_a16w4(x, w, w_scales, w_zeros=None, dtype=None, out=None, backend=None):
    """Compute A16 times AWQ INT4 weights with group scales and zero points.

    The logical weight is [K,N]; unpack its INT4 values and zero points
    using the existing AWQ nibble order, subtract zero points, multiply
    by the corresponding K-group scale, then compute ``x @ weight``.

    Args:
        x (torch.Tensor): float16 or bfloat16 GPU input [M,K] with positive strides.
        w (torch.Tensor): int32 weights [K,N/8] on x.device with positive
            strides. Each word packs eight unsigned INT4 values along N in
            AWQ order.
        w_scales (torch.Tensor): Scales [K/G,N] in x.dtype on x.device, with
            positive strides. K and N must be positive. G is inferred from
            the scale shape and must divide K and be 32, 64, 128, or K.
        w_zeros (torch.Tensor | None): Optional int32 packed zero points
            [K/G,N/8] on x.device with positive strides and the same AWQ
            packing as w. None uses zero.
        dtype (torch.dtype | None): Dtype used to allocate the output. Uses w_scales.dtype when None.
            An explicit dtype must match w_scales.dtype.
        out (torch.Tensor | None): Optional contiguous [M,N] output on
            the input device. Returned directly when supplied. It must not
            overlap the inputs or contain overlapping elements.
        backend (str | None): "triton" to request a backend, or
            "auto" or None to select a backend automatically.

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
def gemm_w4a4(x, w, x_scales, w_scales, dtype=torch.bfloat16, out=None, backend=None):
    """Compute native MXFP4 GEMM from prequantized E2M1 A/B operands.

    Both operands already include packed FP4 values and E8M0 K32
    scales. No activation quantization is performed by this interface.

    Args:
        x (torch.Tensor): uint8 packed E2M1 input [M,K/2] on GPU, with positive
            byte strides. Earlier and later K elements occupy the low and
            high nibbles. K must be positive and divisible by 32.
        w (torch.Tensor): uint8 packed E2M1 weights [N,K/2] on x.device, with
            positive byte strides and the same packing as x.
        x_scales (torch.Tensor): uint8 E8M0 scales [M,K/32] on x.device with
            positive strides. One scale per input row and 32 K elements.
        w_scales (torch.Tensor): uint8 E8M0 scales [N,K/32] on x.device with
            positive strides. One scale per weight row and 32 K elements.
        dtype (torch.dtype | None): Allocation dtype (float16 or bfloat16). Defaults to
            bfloat16. None also selects bfloat16.
        out (torch.Tensor | None): Optional [M,N] output on
            the input device. Returned directly when supplied. It must not
            overlap the inputs or contain overlapping elements.
        backend (str | None): "triton" to request a backend, or
            "auto" or None to select a backend automatically.

    Returns:
        torch.Tensor: [M,N] result in dtype.
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
def batched_gemm_a16w4(x, w, w_scales, dtype=torch.bfloat16, out=None, transpose_bm=False, out_scale=None, backend=None):
    """Compute batched A16 x FP4 GEMM with online MXFP4 activation quantization.

    The input x is quantized to E2M1 with E8M0 scales per 32 K elements
    before multiplication by the prequantized weights.

    Args:
        x (torch.Tensor): float16 or bfloat16 GPU input [B,M,K] with positive
            strides. If B > 1 and the batch stride is 1, a matrix dimension
            must also have unit stride. K must be positive and divisible by 32.
        w (torch.Tensor): uint8 packed E2M1 weights [B,N,K/2] on x.device.
            B must match x. The same stride rules as x apply. Earlier and later
            K elements occupy the low and high nibbles.
        w_scales (torch.Tensor): uint8 E8M0 scales [B,N,K/32] on x.device.
            The same stride rules as x apply. One scale per weight row and
            32 K elements.
        dtype (torch.dtype | None): Allocation dtype (float16 or bfloat16). Defaults to
            bfloat16. None uses x.dtype.
        out (torch.Tensor | None): Optional output [B,M,N], or [M,B,N] when transpose_bm is True,
            on the input device. Returned directly when supplied. It must not
            overlap the inputs or contain overlapping elements.
        transpose_bm (bool): Select [M,B,N] output when True. Defaults to
            False for [B,M,N] output.
        out_scale (torch.Tensor | None): Optional single-element floating-point
            tensor on x.device. Divides the result before conversion to dtype.
            None leaves the result unscaled.
        backend (str | None): "triton" to request a backend, or
            "auto" or None to select a backend automatically.

    Returns:
        torch.Tensor: [B,M,N], or [M,B,N] when transpose_bm=True result in
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
