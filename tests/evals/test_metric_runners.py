"""Unit tests for the async §9 metric runners (PRD §9).

Each runner is driven against the conftest ``InMemoryStorage`` fake + the
harness adapters, with both passing and failing gold sets, so the pass/fail
verdict and the per-case detail are both exercised. No live infra.
"""

from __future__ import annotations

import pytest

from mnemozine.evals import metrics
from mnemozine.evals.goldset import (
    ClassifierCase,
    CrossRefCase,
    GoldMemory,
    GoldSet,
    InjectionCase,
    NoLeakCase,
    PreferenceCase,
    load_gold_set,
)
from mnemozine.evals.harness_adapters import (
    GraphCrossReferencer,
    KeywordExtractor,
    StorageBackedRetriever,
)
from mnemozine.schema.models import ScopeDecision
from tests.conftest import InMemoryStorage


async def _seed(gold_set: GoldSet) -> InMemoryStorage:
    store = InMemoryStorage()
    for unit in gold_set.materialize_memories():
        await store.upsert_memory(unit)
    return store


@pytest.fixture
def gold() -> GoldSet:
    return load_gold_set()


# --- injection precision@k --------------------------------------------------


async def test_injection_precision_passes_on_fixture(gold: GoldSet) -> None:
    store = await _seed(gold)
    retriever = StorageBackedRetriever(store)
    result = await metrics.injection_precision(retriever, gold)
    assert result.passed
    assert result.score == pytest.approx(1.0)
    assert result.n == len(gold.injection_cases)


async def test_injection_precision_fails_when_distractor_marked_relevant() -> None:
    # A gold set whose injection case claims a non-retrievable id should surface
    # -> recall fails -> case fails.
    gs = GoldSet(
        memories=[
            GoldMemory(
                gold_id="a",
                category="preference",
                content="prefers tabs over spaces",
                scope="global",
                entities=["style"],
            )
        ],
        injection_cases=[
            InjectionCase(
                case_id="bad",
                query="tabs spaces",
                scopes=["global"],
                entities=["style"],
                should_surface=["a", "missing"],  # 'missing' can never surface
                top_k=3,
            )
        ],
    )
    store = await _seed(gs)
    result = await metrics.injection_precision(StorageBackedRetriever(store), gs)
    assert not result.passed  # recall of 'missing' fails


# --- changed-preference correctness ----------------------------------------


async def test_changed_preference_passes_when_stale_superseded(gold: GoldSet) -> None:
    store = await _seed(gold)
    result = await metrics.changed_preference_correctness(StorageBackedRetriever(store), gold)
    assert result.passed
    assert result.score == pytest.approx(1.0)


async def test_changed_preference_fails_when_stale_still_active() -> None:
    # Same content/entities but the stale unit is NOT superseded -> it surfaces
    # as active alongside the current one -> the metric must fail.
    gs = GoldSet(
        memories=[
            GoldMemory(
                gold_id="cur",
                category="preference",
                content="prefers thiserror error handling",
                scope="global",
                entities=["rust", "errors"],
                age_days=1,
            ),
            GoldMemory(
                gold_id="old",
                category="preference",
                content="prefers anyhow error handling",
                scope="global",
                entities=["rust", "errors"],
                age_days=50,
                superseded=False,  # bug: never closed
            ),
        ],
        preference_cases=[
            PreferenceCase(
                case_id="chg",
                query="error handling",
                scopes=["global"],
                entities=["rust"],
                current_gold_id="cur",
                stale_gold_id="old",
                top_k=5,
            )
        ],
    )
    store = await _seed(gs)
    result = await metrics.changed_preference_correctness(StorageBackedRetriever(store), gs)
    assert not result.passed
    assert result.cases[0].detail["stale_suppressed"] is False


# --- cross-reference precision ---------------------------------------------


async def test_crossref_precision_passes_on_fixture(gold: GoldSet) -> None:
    store = await _seed(gold)
    result = await metrics.crossref_precision(GraphCrossReferencer(store), gold)
    assert result.passed
    assert result.score == pytest.approx(1.0)
    # Every surfaced reference must carry a human-readable reason (FR-RET-6).
    assert all(c.detail["reasons"] for c in result.cases if c.detail["surfaced"])


async def test_crossref_precision_fails_on_irrelevant_surface() -> None:
    # An idea-seed shares an entity with the context but the gold case says no
    # connection is relevant -> any surfaced item is a false positive.
    gs = GoldSet(
        memories=[
            GoldMemory(
                gold_id="idea",
                category="idea",
                cross_ref_candidate=True,
                content="idea: a thing about widgets",
                scope="global",
                entities=["widgets"],
            )
        ],
        crossref_cases=[
            CrossRefCase(
                case_id="xref",
                project="p",
                entities=["widgets"],
                relevant_gold_ids=[],  # nothing is relevant
                max_suggestions=2,
            )
        ],
    )
    store = await _seed(gs)
    result = await metrics.crossref_precision(GraphCrossReferencer(store), gs)
    assert not result.passed
    assert result.cases[0].detail["precision"] == 0.0


async def test_crossref_suppression_excludes_dismissed() -> None:
    gs = GoldSet(
        memories=[
            GoldMemory(
                gold_id="idea",
                category="idea",
                cross_ref_candidate=True,
                content="idea: widget thing",
                scope="global",
                entities=["widgets"],
            )
        ],
        crossref_cases=[
            CrossRefCase(
                case_id="xref",
                project="p",
                entities=["widgets"],
                relevant_gold_ids=[],
                max_suggestions=2,
            )
        ],
    )
    store = await _seed(gs)
    xref = GraphCrossReferencer(store)
    # Dismiss the idea in context 'p'; now it must not resurface -> precision 1.0.
    await xref.suppress(gs.runtime_id("idea"), "p")
    result = await metrics.crossref_precision(xref, gs)
    assert result.passed
    assert result.cases[0].detail["surfaced"] == []


# --- classifier accuracy ----------------------------------------------------


async def test_classifier_accuracy_on_fixture(gold: GoldSet) -> None:
    result = await metrics.classifier_accuracy(KeywordExtractor(), gold)
    assert result.passed
    assert result.score >= 0.9


async def test_classifier_accuracy_fails_below_threshold() -> None:
    # A statement the heuristic mislabels, with a high threshold, must fail.
    gs = GoldSet(
        classifier_cases=[
            ClassifierCase(
                case_id="c1",
                statement="totally ambiguous sentence with no markers",
                project="proj",
                expected_scope_decision=ScopeDecision.PROJECT,
            )
        ]
    )
    result = await metrics.classifier_accuracy(KeywordExtractor(), gs, threshold=0.9)
    # The heuristic defaults ambiguous -> global, so this is wrong -> fail.
    assert not result.passed
    assert result.score == 0.0


# --- retrieval p95 latency --------------------------------------------------


async def test_retrieval_latency_under_generous_target(gold: GoldSet) -> None:
    store = await _seed(gold)
    result = await metrics.retrieval_latency(
        StorageBackedRetriever(store), gold, target_ms=10_000.0, repeats=3
    )
    assert result.passed
    assert result.name == "retrieval_p95_latency_ms"
    assert result.n == 3 * len(gold.injection_cases)


async def test_retrieval_latency_fails_under_zero_target(gold: GoldSet) -> None:
    store = await _seed(gold)
    result = await metrics.retrieval_latency(
        StorageBackedRetriever(store), gold, target_ms=0.0, repeats=2
    )
    # Any real work takes >0ms, so a 0ms SLA must fail.
    assert not result.passed


# --- no-leak check ----------------------------------------------------------


async def test_no_leak_passes_on_fixture(gold: GoldSet) -> None:
    store = await _seed(gold)
    result = await metrics.no_leak_check(store, gold)
    assert result.passed
    assert result.score == pytest.approx(1.0)


async def test_no_leak_fails_when_project_fact_is_global() -> None:
    # A project_fact mis-scoped to global leaks into every project -> fail.
    gs = GoldSet(
        memories=[
            GoldMemory(
                gold_id="leaky",
                category="fact",
                content="secret pins tokio 1.38",
                scope="global",  # bug: should be project:rust-cli
                entities=["tokio"],
            )
        ],
        no_leak_cases=[
            NoLeakCase(
                case_id="leak",
                fact_gold_id="leaky",
                unrelated_project="webapp",
                query="tokio",
                entities=["tokio"],
                top_k=10,
            )
        ],
    )
    store = await _seed(gs)
    result = await metrics.no_leak_check(store, gs)
    assert not result.passed
    assert result.cases[0].detail["leaked"] is True
