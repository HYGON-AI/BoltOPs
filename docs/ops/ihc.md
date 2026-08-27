# Independent Hyper-Connections (IHC)

IHC keeps `hc_mult` parallel residual channels and reduces them to a single
hidden state before each attention/MLP sub-block, then scatters the sub-block
output back with post gates. Unlike MHC, there is **no comb / Sinkhorn** path:
the projection width is `2 * hc_mult` (pre + post only).

TileLang kernels live in `boltops.ihc`. Architecture dispatch
follows the same exact → family → `common` fallback as other HCU ops.
Launch config is **owned by IHC** and is not shared with `boltops.mhc`.

The numeric reference is the Hy4-p eager modules in
`vllm/models/hy_v4/nvidia/hc.py` (`HYV4HCPreLayer`, `HYV4HCPostLayer`,
`HYV4HCHeadLayer`). Default model knobs: `hc_mult=4`, `hc_magnitude=2.0`.

## Public API (TileLang)

```python
from boltops.ihc import ihc_pre, ihc_post, ihc_head

layer_in, post = ihc_pre(
    residual,   # [T, hc, d], bfloat16
    fn,         # [2 * hc, hc * d], float32
    scale,      # [2], float32
    base,       # [2 * hc], float32
    rms_eps,
    hc_eps,
    magnitude,
)
# layer_in: [T, d] bfloat16; post: [T, hc] float32

residual = ihc_post(
    x,          # [T, d], bfloat16 (sub-block output)
    residual,   # [T, hc, d], bfloat16
    post,       # [T, hc], float32
)
# residual: [T, hc, d] bfloat16

hidden = ihc_head(
    residual,   # [T, hc, d], bfloat16
    head_fn,    # [hc, hc * d], float32
    head_scale, # [1] or scalar, float32
    head_base,  # [hc], float32
    rms_eps,
    hc_eps,
)
# hidden: [T, d] bfloat16
```

High-level entry points allocate outputs. Residual / activations are BF16;
gates and projection weights are FP32. Typical `hc_mult=4`.

## Eager API (Hy4-p)

The vLLM port is eager PyTorch only (no fused HPC kernels). Shapes below use
`T` = tokens, `hc` = `config.hc_mult`, `d` = `hidden_size`.

**Pre** (`HYV4HCPreLayer.forward`):

- Input `x`: `[T, hc, d]`
- Flatten to `[T, hc * d]`, RMS over the last dim, project with `hc_fn`
  (`ReplicatedLinear`, `in=hc*d`, `out=2*hc`, fp32, no bias)
- `pre = sigmoid(mixes[:hc] * hc_scale[0] + hc_base[:hc]) + hc_eps`
- `post = magnitude * sigmoid(mixes[hc:] * hc_scale[1] + hc_base[hc:]) + hc_eps`
- Output: reduced hidden `[T, d]`, post gates `[T, hc]`

**Post** (`HYV4HCPostLayer.forward`):

```text
y[n, i, d] = post[n, i] * x[n, d] + residual[n, i, d]
```

- `x`: `[T, d]`, `residual`: `[T, hc, d]`, `post`: `[T, hc]`
- Output: `[T, hc, d]`

**Head** (`HYV4HCHeadLayer.forward`):

- Same RMS + project pattern as pre, but `hc_head_fn` maps `hc*d → hc`
- `pre = sigmoid(mixes * hc_head_scale + hc_head_base) + hc_eps`
- Weighted reduce over channels → `[T, d]`

Decoder usage (`HYV4DecoderLayer._forward_ihc`): attn and MLP each call
`prepare_input` → `pre` → sub-block → `post`. The multi-channel residual stays
in `hidden_states`; `hc_head` runs once before the final RMSNorm.

## Tests

```bash
pytest test/ihc
pytest test/ihc --bench -s
```

Correctness compares TileLang outputs to the eager formulas above, not to MHC
references. Benchmark tests use the `bench` marker; `--bench` only controls
collection of timing cases.
