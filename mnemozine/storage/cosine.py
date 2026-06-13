"""Small, dependency-free vector helpers used by the storage layer.

Kept in its own module so the in-memory backend, the FalkorDB backend, and the
tests all share one definition of cosine similarity (and the contract test can
verify ranking without numpy). These are pure/CPU helpers, so they are sync.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors, in ``[-1.0, 1.0]``.

    Returns ``0.0`` when either vector is all-zeros (undefined direction) or the
    lengths differ — callers treat that as "no signal" rather than raising, so a
    degenerate embedding never crashes a write/query path.
    """

    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
