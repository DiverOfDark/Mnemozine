"""Category-merge tests — the category analogue of entity resolution (FR-MNT-2/4).

Covers the merge DECISION (``propose_merges``: which near-duplicate categories
fold into which canonical one, and the threshold) and the applied ``run`` pass,
all offline against the conftest ``InMemoryStorage`` + ``FakeEmbeddingProvider``.
"""

from __future__ import annotations

import pytest

from mnemozine.config import Settings
from mnemozine.interfaces import CategoryMerger, MaintenanceJob
from mnemozine.maintenance.category_merge import (
    CategoryMergeJob,
    name_similarity,
    normalize_category,
)
from mnemozine.schema.models import MemoryUnit, Provenance, Scope
from tests.conftest import FakeEmbeddingProvider, InMemoryStorage


def _mem(content: str, *, category: str) -> MemoryUnit:
    return MemoryUnit(
        content=content,
        scope=Scope.global_(),
        category=category,
        entities=["rust"],
        confidence=0.9,
        provenance=Provenance(source="claude_code", session_id="s1"),
    )


async def _seed(storage: InMemoryStorage, category: str, n: int) -> None:
    for i in range(n):
        await storage.upsert_memory(_mem(f"{category} memory {i}", category=category))


# --- protocol + pure helpers ----------------------------------------------


def test_satisfies_protocols() -> None:
    job = CategoryMergeJob(InMemoryStorage(), embeddings=FakeEmbeddingProvider())
    assert isinstance(job, CategoryMerger)
    assert isinstance(job, MaintenanceJob)
    assert job.name == "category_merge"


def test_normalize_category_lowercases_trims_and_defaults() -> None:
    assert normalize_category("  Gotcha ") == "gotcha"
    assert normalize_category("DECISION") == "decision"
    # Empty falls back to DEFAULT_CATEGORY ("fact"), matching MemoryUnit.
    assert normalize_category("   ") == "fact"


def test_name_similarity_catches_plural_variants() -> None:
    # The string-similarity fallback clears the 0.85 default for plural/spelling
    # variants but not for distinct concepts.
    assert name_similarity("gotcha", "gotchas") >= 0.85
    assert name_similarity("decision", "decisions") >= 0.85
    assert name_similarity("preference", "gotcha") < 0.85


# --- the merge decision (propose_merges) ----------------------------------


@pytest.mark.asyncio
async def test_propose_merges_orients_to_highest_count_canonical() -> None:
    # 'gotcha' (3) and 'gotchas' (1) are near-duplicates -> fold the smaller into
    # the higher-count canonical, oriented source -> canonical.
    storage = InMemoryStorage()
    await _seed(storage, "gotcha", 3)
    await _seed(storage, "gotchas", 1)
    job = CategoryMergeJob(
        storage, embeddings=FakeEmbeddingProvider(), settings=Settings()
    )

    proposals = await job.propose_merges()

    assert proposals == [("gotchas", "gotcha")]


@pytest.mark.asyncio
async def test_propose_merges_below_threshold_keeps_distinct() -> None:
    # Distinct categories far below the similarity threshold are never proposed.
    storage = InMemoryStorage()
    await _seed(storage, "preference", 2)
    await _seed(storage, "architecture", 2)
    job = CategoryMergeJob(
        storage, embeddings=FakeEmbeddingProvider(), settings=Settings()
    )

    # FakeEmbeddingProvider gives near-orthogonal hash vectors and the names are
    # textually dissimilar, so nothing should merge.
    assert await job.propose_merges() == []


@pytest.mark.asyncio
async def test_threshold_is_config_driven() -> None:
    # Lowering the threshold makes otherwise-distinct names merge: proves the
    # cutoff is read from category.merge_similarity_threshold, not a constant.
    storage = InMemoryStorage()
    await _seed(storage, "gotcha", 2)
    await _seed(storage, "pitfall", 1)
    settings = Settings()
    # name_similarity('gotcha','pitfall') is low; drop the bar under it.
    settings.category.merge_similarity_threshold = 0.0
    job = CategoryMergeJob(storage, embeddings=None, settings=settings)

    proposals = await job.propose_merges()
    # Everything collapses into the highest-count canonical ('gotcha').
    assert ("pitfall", "gotcha") in proposals


@pytest.mark.asyncio
async def test_single_category_proposes_nothing() -> None:
    storage = InMemoryStorage()
    await _seed(storage, "fact", 4)
    job = CategoryMergeJob(storage, embeddings=FakeEmbeddingProvider())
    assert await job.propose_merges() == []


# --- the applied pass (run) -----------------------------------------------


@pytest.mark.asyncio
async def test_run_merges_and_relabels_memories() -> None:
    storage = InMemoryStorage()
    await _seed(storage, "gotcha", 3)
    await _seed(storage, "gotchas", 2)
    job = CategoryMergeJob(
        storage, embeddings=FakeEmbeddingProvider(), settings=Settings()
    )

    report = await job.run()

    # One category merged; the two 'gotchas' memories re-labelled to 'gotcha'.
    assert report.categories_merged == 1
    assert all(m.category != "gotchas" for m in storage.memories.values())
    assert sum(1 for m in storage.memories.values() if m.category == "gotcha") == 5
    # The registry now has a single canonical category.
    cats = dict(await storage.list_categories())
    assert cats == {"gotcha": 5}


@pytest.mark.asyncio
async def test_run_is_idempotent() -> None:
    storage = InMemoryStorage()
    await _seed(storage, "gotcha", 3)
    await _seed(storage, "gotchas", 2)
    job = CategoryMergeJob(
        storage, embeddings=FakeEmbeddingProvider(), settings=Settings()
    )

    first = await job.run()
    second = await job.run()

    assert first.categories_merged == 1
    # Nothing left to merge on the second pass (FR-MNT-5 idempotency).
    assert second.categories_merged == 0


@pytest.mark.asyncio
async def test_run_with_no_categories_is_noop() -> None:
    storage = InMemoryStorage()
    job = CategoryMergeJob(storage, embeddings=FakeEmbeddingProvider())
    report = await job.run()
    assert report.categories_merged == 0
    assert report.job_name == "category_merge"


@pytest.mark.asyncio
async def test_merge_works_without_embeddings_via_string_similarity() -> None:
    # No embedding provider at all: the string-similarity fallback still catches
    # the plural variant.
    storage = InMemoryStorage()
    await _seed(storage, "decision", 2)
    await _seed(storage, "decisions", 1)
    job = CategoryMergeJob(storage, embeddings=None, settings=Settings())

    report = await job.run()

    assert report.categories_merged == 1
    cats = dict(await storage.list_categories())
    assert cats == {"decision": 3}
