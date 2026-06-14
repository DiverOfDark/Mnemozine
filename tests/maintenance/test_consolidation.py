"""FR-MNT-2 tests — tiered consolidation merges related facts; archives sources."""

from __future__ import annotations

import pytest

from mnemozine.config import Settings
from mnemozine.maintenance.consolidation import ConsolidationJob
from mnemozine.schema.models import MemoryUnit, Provenance, Scope, Tier
from tests.conftest import FakeEmbeddingProvider, FakeLLMProvider, InMemoryStorage


def _pref(content: str, *, entities: list[str]) -> MemoryUnit:
    # Category split: a global-scope memory carrying the free-form "preference"
    # category (clustering is per-category, so all members share one).
    return MemoryUnit(
        content=content,
        scope=Scope.global_(),
        category="preference",
        entities=entities,
        confidence=0.9,
        provenance=Provenance(source="claude_code", session_id="s1"),
    )


@pytest.mark.asyncio
async def test_consolidates_related_cluster_and_archives_sources() -> None:
    settings = Settings()
    settings.maintenance.dedup_equivalence_threshold = 0.0  # cluster aggressively
    storage = InMemoryStorage()
    llm = FakeLLMProvider(
        text_responder=lambda p, s: "Prefers thiserror with structured Rust errors."
    )
    embeddings = FakeEmbeddingProvider()

    a = _pref("Uses thiserror for Rust errors.", entities=["rust", "error-handling"])
    b = _pref("Likes structured error types in Rust.", entities=["rust", "error-handling"])
    await storage.upsert_memory(a)
    await storage.upsert_memory(b)

    job = ConsolidationJob(storage, llm, embeddings, settings=settings)
    report = await job.run()

    assert report.consolidated == 1
    # Sources archived (retained, not deleted); a new active consolidated unit added.
    assert a.tier is Tier.ARCHIVE
    assert b.tier is Tier.ARCHIVE
    active_hot = [
        m for m in storage.memories.values() if m.is_active and m.tier is Tier.HOT
    ]
    assert len(active_hot) == 1
    consolidated = active_hot[0]
    assert "thiserror" in consolidated.content.lower()
    # Union of entities preserved.
    assert set(consolidated.entities) == {"rust", "error-handling"}


@pytest.mark.asyncio
async def test_consolidation_is_idempotent() -> None:
    settings = Settings()
    settings.maintenance.dedup_equivalence_threshold = 0.0
    storage = InMemoryStorage()
    llm = FakeLLMProvider(text_responder=lambda p, s: "Consolidated rust pref.")
    embeddings = FakeEmbeddingProvider()
    await storage.upsert_memory(_pref("a rust pref.", entities=["rust"]))
    await storage.upsert_memory(_pref("another rust pref.", entities=["rust"]))

    job = ConsolidationJob(storage, llm, embeddings, settings=settings)
    first = await job.run()
    second = await job.run()
    assert first.consolidated == 1
    # The single surviving consolidated unit is a 1-member cluster -> no-op.
    assert second.consolidated == 0


@pytest.mark.asyncio
async def test_singletons_and_cross_scope_not_merged() -> None:
    settings = Settings()
    settings.maintenance.dedup_equivalence_threshold = 0.0
    storage = InMemoryStorage()
    llm = FakeLLMProvider(text_responder=lambda p, s: "X")
    embeddings = FakeEmbeddingProvider()

    # Same entity but DIFFERENT scopes => must never consolidate together.
    g = _pref("global rust pref.", entities=["rust"])
    p = MemoryUnit(
        content="project rust fact.",
        scope=Scope.project("p1"),
        category="project_fact",
        entities=["rust"],
        confidence=0.9,
        provenance=Provenance(source="claude_code", session_id="s1"),
    )
    await storage.upsert_memory(g)
    await storage.upsert_memory(p)

    job = ConsolidationJob(storage, llm, embeddings, settings=settings)
    report = await job.run()

    assert report.consolidated == 0
    assert g.is_active and g.tier is Tier.HOT
    assert p.is_active and p.tier is Tier.HOT


@pytest.mark.asyncio
async def test_empty_llm_response_leaves_cluster_intact() -> None:
    settings = Settings()
    settings.maintenance.dedup_equivalence_threshold = 0.0
    storage = InMemoryStorage()
    llm = FakeLLMProvider(text_responder=lambda p, s: "   ")  # blank -> skip
    embeddings = FakeEmbeddingProvider()
    a = _pref("a rust pref.", entities=["rust"])
    b = _pref("b rust pref.", entities=["rust"])
    await storage.upsert_memory(a)
    await storage.upsert_memory(b)

    job = ConsolidationJob(storage, llm, embeddings, settings=settings)
    report = await job.run()

    assert report.consolidated == 0
    # No memory lost; both still hot+active.
    assert a.tier is Tier.HOT and b.tier is Tier.HOT
