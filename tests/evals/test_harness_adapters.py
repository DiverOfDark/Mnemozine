"""Unit tests for the harness adapters (offline Retriever/CrossRef/Extractor)."""

from __future__ import annotations

import pytest

from mnemozine.evals.harness_adapters import (
    GraphCrossReferencer,
    KeywordExtractor,
    StorageBackedRetriever,
)
from mnemozine.interfaces import (
    CrossReferencer,
    Extractor,
    RetrievalContext,
    Retriever,
)
from mnemozine.schema.models import MemoryUnit, Provenance, Scope, ScopeDecision
from tests.conftest import InMemoryStorage


def test_adapters_satisfy_protocols() -> None:
    store = InMemoryStorage()
    assert isinstance(StorageBackedRetriever(store), Retriever)
    assert isinstance(GraphCrossReferencer(store), CrossReferencer)
    assert isinstance(KeywordExtractor(), Extractor)


async def test_retriever_records_access() -> None:
    store = InMemoryStorage()
    unit = MemoryUnit(
        category="preference",
        content="prefers thiserror error handling",
        scope=Scope.global_(),
        entities=["rust", "errors"],
        provenance=Provenance(source="eval", session_id="s"),
    )
    await store.upsert_memory(unit)
    retriever = StorageBackedRetriever(store)
    ctx = RetrievalContext(scopes=[Scope.global_()], entities=["rust"])
    hits = await retriever.scoped_retrieve("thiserror error handling", ctx, top_k=5)
    assert hits
    # FR-MNT-3: deliberate reads record access.
    assert store.memories[unit.id].access_count == 1


async def test_recall_records_access() -> None:
    store = InMemoryStorage()
    unit = MemoryUnit(
        category="preference",
        content="prefers ruff",
        scope=Scope.global_(),
        entities=["python"],
        provenance=Provenance(source="eval", session_id="s"),
    )
    await store.upsert_memory(unit)
    retriever = StorageBackedRetriever(store)
    hits = await retriever.recall("ruff", Scope.global_(), top_k=5)
    assert hits
    assert store.memories[unit.id].access_count == 1


@pytest.mark.parametrize(
    ("statement", "project", "expected"),
    [
        ("I prefer thiserror over anyhow.", "rust-cli", ScopeDecision.GLOBAL),
        ("I always format Python with ruff.", "py", ScopeDecision.GLOBAL),
        ("This project pins tokio 1.38.", "rust-cli", ScopeDecision.PROJECT),
        ("The webapp uses postgres 16 as its datastore.", "webapp", ScopeDecision.PROJECT),
    ],
)
async def test_keyword_extractor_classify(
    statement: str, project: str, expected: ScopeDecision
) -> None:
    ctx = RetrievalContext(project=project)
    cls = await KeywordExtractor().classify(statement, ctx)
    assert cls.scope_decision is expected
    if expected is ScopeDecision.PROJECT:
        assert cls.scope == Scope.project(project)
    else:
        assert cls.scope.is_global


async def test_keyword_extractor_extract_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        await KeywordExtractor().extract([])
