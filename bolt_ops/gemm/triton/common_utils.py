"""Small helpers shared by vendored aiter GEMM wrappers."""

from __future__ import annotations

import json
from typing import Any


def serialize_dict(d: dict | None) -> str | None:
    if d is None:
        return None
    return json.dumps(d)


def deserialize_str(s: str | dict | None) -> dict | None:
    if s is None:
        return None
    if isinstance(s, dict):
        return dict(s)
    return json.loads(s)
