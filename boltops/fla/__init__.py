"""Flash Linear Attention (FLA) family umbrella.

One directory per FLA sub-family (KDA / GDN / ...), plus ``common`` for
helpers shared across families. Callers import public APIs from the
sub-family package, e.g. ``boltops.fla.kda``.
"""

