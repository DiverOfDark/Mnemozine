"""Unit tests for the pure cross-reference scoring/reason helpers (FR-RET-6).

These cover the *ranking signal* and the mandatory human-readable reason
generation in isolation — no storage, no embeddings — so the math and the
explanation text can be pinned down independently of I/O.
"""

from __future__ import annotations

import math

from mnemozine.crossref.scoring import (
    build_reason,
    cosine_similarity,
    edge_weight_factor,
    graph_relevance,
    jaccard,
)
from mnemozine.schema.models import Edge


def _edge(weight: float = 1.0, relation: str = "relates_to") -> Edge:
    return Edge(from_entity="a", to_entity="b", relation=relation, weight=weight)


# --- cosine_similarity ----------------------------------------------------


def test_cosine_identical_vectors_is_one() -> None:
    v = [0.1, 0.2, 0.3, 0.4]
    assert math.isclose(cosine_similarity(v, v), 1.0, rel_tol=1e-9)


def test_cosine_orthogonal_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_degenerate_inputs_are_zero() -> None:
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0


# --- jaccard --------------------------------------------------------------


def test_jaccard_basic() -> None:
    assert jaccard(["a", "b"], ["b", "c"]) == 1 / 3
    assert jaccard(["a"], ["a"]) == 1.0
    assert jaccard(["a"], ["b"]) == 0.0
    assert jaccard([], []) == 0.0


# --- edge_weight_factor ---------------------------------------------------


def test_edge_weight_factor_uses_max_and_saturates() -> None:
    assert edge_weight_factor([_edge(0.3), _edge(0.9)]) == 0.9
    assert edge_weight_factor([_edge(5.0)]) == 1.0  # saturates at 1.0


def test_edge_weight_factor_no_edges_is_neutral() -> None:
    assert edge_weight_factor([]) == 1.0


def test_edge_weight_factor_ignores_inactive_edges() -> None:
    active = _edge(0.4)
    inactive = _edge(0.95)
    inactive.valid_to = inactive.valid_from
    assert edge_weight_factor([active, inactive]) == 0.4


# --- graph_relevance ------------------------------------------------------


def test_graph_relevance_zero_without_shared_entities() -> None:
    assert (
        graph_relevance(
            shared_entities=[],
            context_entities=["rust"],
            candidate_entities=["python"],
            connecting_edges=[],
        )
        == 0.0
    )


def test_graph_relevance_higher_with_more_overlap() -> None:
    # Two shared of two context entities vs one shared of two.
    high = graph_relevance(
        shared_entities=["async", "cli"],
        context_entities=["async", "cli"],
        candidate_entities=["async", "cli"],
        connecting_edges=[],
    )
    low = graph_relevance(
        shared_entities=["async"],
        context_entities=["async", "cli"],
        candidate_entities=["async", "db", "http"],
        connecting_edges=[],
    )
    assert high > low


def test_graph_relevance_edge_strength_modulates() -> None:
    weak = graph_relevance(
        shared_entities=["async"],
        context_entities=["async"],
        candidate_entities=["async"],
        connecting_edges=[_edge(0.1)],
    )
    strong = graph_relevance(
        shared_entities=["async"],
        context_entities=["async"],
        candidate_entities=["async"],
        connecting_edges=[_edge(1.0)],
    )
    assert strong > weak
    # Edge factor is bounded so a weak edge never zeroes a perfect overlap.
    assert weak >= 0.5 * 1.0


def test_graph_relevance_in_unit_range() -> None:
    score = graph_relevance(
        shared_entities=["async", "cli"],
        context_entities=["async", "cli", "rust"],
        candidate_entities=["async", "cli", "tokio"],
        connecting_edges=[_edge(0.8)],
    )
    assert 0.0 <= score <= 1.0


# --- build_reason (the mandatory human-readable explanation) --------------


def test_build_reason_lists_shared_entities() -> None:
    reason = build_reason(["async-runtime", "cli-parsing"])
    assert reason == "shares async-runtime, cli-parsing"


def test_build_reason_dedupes_and_preserves_order() -> None:
    reason = build_reason(["cli", "async", "cli", "async"])
    assert reason == "shares cli, async"


def test_build_reason_truncates_long_entity_lists() -> None:
    reason = build_reason(["a", "b", "c", "d", "e", "f"])
    assert "shares a, b, c, d" in reason
    assert "(+2 more)" in reason


def test_build_reason_includes_distinctive_relation() -> None:
    edges = [Edge(from_entity="x", to_entity="y", relation="depends_on", weight=0.9)]
    reason = build_reason(["tokio"], edges)
    assert "shares tokio" in reason
    assert "related via depends_on" in reason


def test_build_reason_omits_relation_when_mixed() -> None:
    edges = [
        Edge(from_entity="x", to_entity="y", relation="depends_on", weight=0.9),
        Edge(from_entity="x", to_entity="z", relation="mentions", weight=0.5),
    ]
    reason = build_reason(["tokio"], edges)
    assert reason == "shares tokio"


def test_build_reason_vector_fallback_label() -> None:
    # No shared entities, vector path -> explicit semantic-similarity reason.
    reason = build_reason([], via_vector=True)
    assert "semantically similar" in reason
    assert reason != ""


def test_build_reason_vector_with_shared_entities_notes_both() -> None:
    reason = build_reason(["async"], via_vector=True)
    assert "shares async" in reason
    assert "semantically similar" in reason


def test_build_reason_empty_when_nothing_to_explain() -> None:
    # No shared entities and not a vector hit -> unexplainable -> empty reason,
    # which the engine treats as "do not surface" (FR-RET-6).
    assert build_reason([]) == ""
