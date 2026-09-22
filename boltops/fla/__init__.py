"""Flash Linear Attention (FLA) family umbrella.

One directory per FLA sub-family (KDA / GDN / ...), plus ``common`` for code
shared by two or more sub-families. Family-specific kernels remain in their
own directory; FLA code does not depend on ``boltops.utility.triton``. Callers
import public APIs from the sub-family package, e.g. ``boltops.fla.kda``.
"""
