# Multi-Head Connection (MHC)

MHC is implemented with TileLang kernels. The current `common` implementation
family supports HCU `gfx936` and `gfx938`; architecture dispatch prefers an
exact implementation and then falls back to the architecture family. Future
`gfx94x` or exact-architecture kernels can be added without changing the public
API.

## Public API

```python
from boltops.mhc import (
    hc_split_sinkhorn,
    mhc_fused_tilelang,
    mhc_post_fwd,
    mhc_pre_big_fuse,
    pre_big_fuse_tilelang,
)
```

The public package lazy-loads TileLang and queries device properties on first
use. Device architecture and CU count are cached per device. `mhc_post_fwd`
and `mhc_pre_big_fuse` are allocation-owning high-level interfaces;
`pre_big_fuse_tilelang` and `mhc_fused_tilelang` expose preallocated-buffer
interfaces used by fused call sites.

The migrated kernels currently validate BF16 residual/input tensors, FP32 mix
and weight tensors, and `mhc_mult=4` model configurations.

## Tests

```bash
pytest test/mhc
pytest test/mhc --bench -s
```

Correctness and benchmark cases are separated by purpose rather than tensor
size. Default correctness tests use a small CI subset (decode-like `tokens=1`
plus a modest aligned batch, and production `hidden_size` 4096/7168). The
original AITER regression grids remain commented in the test files for local
full sweeps. Benchmark tests use the AITER benchmark shape sets, carry the
`bench` marker, and report kernel latency (and bandwidth where applicable).
`--bench` controls collection of those timing tests; it does not classify a
shape as correctness or performance based on its size.

`mhc_pre_big_fuse` follows the original AITER CUDA Graph replay timing because
its allocation-owning Python API launches multiple kernels. Eager end-to-end
timing also includes Python dispatch, output allocation, and host launch gaps,
so it must not be interpreted as GPU kernel latency.
