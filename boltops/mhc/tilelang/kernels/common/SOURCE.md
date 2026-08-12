# Source

The initial gfx936/gfx938 MHC TileLang kernels were migrated from the local
`aiter/aiter/ops/tilelang/mhc` implementation. Experimental backup files,
generated TileLang caches, and profiling artifacts are intentionally excluded.

BoltOps adds lazy public APIs, architecture-family dispatch, cached per-device
metadata, and an externalized pre-big-fuse configuration policy around the
kernel implementation.
