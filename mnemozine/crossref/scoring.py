"""Pure scoring + reason-generation helpers for the cross-reference engine.

These are deliberately side-effect-free so the FR-RET-6 ranking, threshold
gating and human-readable reason generation can be unit-tested in isolation,
without a storage backend or embedding provider. The engine
(:mod:`mnemozine.crossref.engine`) composes them with I/O.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from mnemozine.schema.models import Edge

# Maximum number of shared entities mentioned verbatim in a generated reason
# before it switches to a "+N more" summary, so reasons stay short and readable.
_MAX_ENTITIES_IN_REASON = 4


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors in ``[-1, 1]`` (0 if either is degenerate).

    Used by the FR-RET-6 vector-similarity fallback to score a candidate idea
    against the working-context text when no shared-entity path exists.
    """

    if not a or not b or len(a) != len(b):
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


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Jaccard overlap of two entity sets in ``[0, 1]`` (0 when both empty)."""

    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def edge_weight_factor(edges: Sequence[Edge]) -> float:
    """Collapse the connecting edges' weights into a ``(0, 1]`` confidence factor.

    Stronger / more numerous relations between the context entities and the
    candidate's entities raise confidence in the connection. We use the *max*
    active edge weight (saturating at 1.0) so a single strong relation suffices;
    absence of any edge yields a neutral 1.0 (entity overlap alone still scores).
    """

    weights = [e.weight for e in edges if e.is_active]
    if not weights:
        return 1.0
    return min(1.0, max(weights))


def graph_relevance(
    shared_entities: Sequence[str],
    context_entities: Sequence[str],
    candidate_entities: Sequence[str],
    connecting_edges: Sequence[Edge],
) -> float:
    """Relevance score in ``[0, 1]`` for a shared-entity (graph) connection.

    Combines:
    * how much the candidate's entities overlap the working context (Jaccard),
    * the strength of the connecting edges (FR-RET-6 weight-rank).

    A direct shared entity is the strongest, most explainable signal, so the
    overlap term dominates; the edge factor only modulates it. Returns 0 when
    there is no shared entity at all (nothing explainable to surface).
    """

    if not shared_entities:
        return 0.0
    overlap = jaccard(context_entities, candidate_entities)
    if overlap <= 0.0:
        # Shared entities were supplied but don't intersect the context sets;
        # fall back to a small direct-share signal rather than 0.
        overlap = len(set(shared_entities)) / (len(set(candidate_entities)) or 1)
        overlap = min(1.0, overlap)
    factor = edge_weight_factor(connecting_edges)
    # Blend: overlap is the backbone, edge strength a multiplicative modulation
    # bounded so a weak edge never zeroes a strong overlap (floor at 0.5).
    return overlap * (0.5 + 0.5 * factor)


def build_reason(
    shared_entities: Sequence[str],
    connecting_edges: Sequence[Edge] | None = None,
    *,
    via_vector: bool = False,
) -> str:
    """Build the mandatory human-readable :attr:`CrossReference.reason`.

    Graph path: ``"shares <e1>, <e2>"`` (optionally noting the relation when a
    single distinctive edge connects them). Vector fallback: a clearly-labelled
    semantic-similarity reason so the operator can tell why it surfaced without
    a shared entity. A connection with no expressible reason must not surface
    (FR-RET-6), so callers gate on a non-empty return.
    """

    ordered = _dedupe_preserve_order(shared_entities)

    if via_vector and not ordered:
        return "semantically similar to current work (no shared entities)"

    if not ordered:
        # No shared entities and not a vector hit -> nothing explainable.
        return ""

    shown = ordered[:_MAX_ENTITIES_IN_REASON]
    extra = len(ordered) - len(shown)
    entity_list = ", ".join(shown)
    if extra > 0:
        entity_list = f"{entity_list} (+{extra} more)"

    reason = f"shares {entity_list}"

    relation = _distinctive_relation(connecting_edges or [])
    if relation:
        reason = f"{reason}; related via {relation}"

    if via_vector:
        reason = f"{reason}; also semantically similar"

    return reason


def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _distinctive_relation(edges: Sequence[Edge]) -> str | None:
    """Return the single relation label if all active edges agree on one."""

    relations = {e.relation for e in edges if e.is_active and e.relation}
    if len(relations) == 1:
        return next(iter(relations))
    return None
