"""Canonical JSON serialization used for every hash in the project.

Rules (also implemented in the report viewer's JavaScript verifier):
  * object keys sorted lexicographically (by Unicode code point),
  * no insignificant whitespace (separators ',' and ':'),
  * UTF-8 output, non-ASCII characters emitted as-is (not \\u-escaped),
  * no floating point numbers (they do not round-trip identically across languages);
    format them as strings first.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _check_no_floats(obj: Any, path: str = "$") -> None:
    if isinstance(obj, float):
        raise TypeError(f"floats are not allowed in canonical JSON (at {path}); format as a string")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_no_floats(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_no_floats(v, f"{path}[{i}]")


def canonical_bytes(obj: Any, allow_floats: bool = False) -> bytes:
    if not allow_floats:
        _check_no_floats(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(obj: Any, allow_floats: bool = False) -> str:
    return sha256_hex(canonical_bytes(obj, allow_floats=allow_floats))
