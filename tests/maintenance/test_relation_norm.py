"""Relation-normalization tests — the relation analogue of category merge.

Covers the controlled-vocabulary helpers (:func:`normalize_relation` +
``RELATION_SYNONYMS``), the read-only merge DECISION (``propose_merges``: which
fragmented labels collapse to which canonical relation), and the applied ``run``
pass + its idempotency — all offline against the conftest ``InMemoryStorage``.
"""

from __future__ import annotations

import pytest

from mnemozine.config import Settings
from mnemozine.interfaces import MaintenanceJob
from mnemozine.maintenance.relation_norm import (
    CONTROLLED_RELATIONS,
    RELATION_SYNONYMS,
    RelationNormJob,
    normalize_relation,
)
from mnemozine.schema.models import Edge, Entity
from tests.conftest import InMemoryStorage


async def _seed_edge(
    storage: InMemoryStorage,
    *,
    frm: str,
    to: str,
    relation: str,
    weight: float = 0.5,
) -> None:
    for eid, name in ((frm, frm), (to, to)):
        if eid not in storage.entities:
            await storage.upsert_entity(Entity(id=eid, canonical_name=name))
    await storage.upsert_edge(
        Edge(from_entity=frm, to_entity=to, relation=relation, weight=weight)
    )


# --- protocol + pure helpers ----------------------------------------------


def test_satisfies_protocol() -> None:
    job = RelationNormJob(InMemoryStorage())
    assert isinstance(job, MaintenanceJob)
    assert job.name == "relation_norm"


def test_normalize_relation_slugs_and_collapses_punctuation() -> None:
    # Lowercase + hyphen/space/underscore collapse to '_'.
    assert normalize_relation("Used-In") == "uses"
    assert normalize_relation("used_in") == "uses"
    assert normalize_relation("uses") == "uses"
    assert normalize_relation("Depends On") == "depends_on"
    assert normalize_relation("depends-on") == "depends_on"
    assert normalize_relation("requires") == "depends_on"
    assert normalize_relation("composites on") == "composites_on"


def test_normalize_relation_lemmatizes_trivial_inflections() -> None:
    # Trailing plural / gerund on the last token is folded before the lookup.
    assert normalize_relation("uses") == "uses"  # 'use' -> synonym -> 'uses'
    assert normalize_relation("using") == "uses"
    assert normalize_relation("includes") == "contains"
    assert normalize_relation("generates") == "produces"


def test_normalize_relation_empty_falls_back_to_relates() -> None:
    assert normalize_relation("   ") == "relates"
    assert normalize_relation("-") == "relates"


def test_normalize_relation_passthrough_for_unknown_canonical() -> None:
    # A label with no synonym maps to its own slug (already canonical).
    assert normalize_relation("supersedes") == "supersede"  # plural-strip only
    assert normalize_relation("mentions") == "mention"


def test_normalize_relation_is_idempotent() -> None:
    # Normalizing an already-canonical label is a fixed point (FR-MNT-5 substrate).
    for canonical in CONTROLLED_RELATIONS:
        assert normalize_relation(canonical) == canonical


def test_controlled_relations_covers_synonym_targets() -> None:
    assert set(RELATION_SYNONYMS.values()) <= CONTROLLED_RELATIONS


# --- the merge decision (propose_merges) ----------------------------------


@pytest.mark.asyncio
async def test_propose_merges_maps_variants_to_canonical() -> None:
    storage = InMemoryStorage()
    await _seed_edge(storage, frm="rust", to="tokio", relation="used_in")
    await _seed_edge(storage, frm="rust", to="serde", relation="depends-on")
    await _seed_edge(storage, frm="tokio", to="serde", relation="uses")  # canonical
    job = RelationNormJob(storage, settings=Settings())

    proposals = await job.propose_merges()

    # 'used_in' -> 'uses' and 'depends-on' -> 'depends_on'; the already-canonical
    # 'uses' proposes nothing. Sorted by source label.
    assert proposals == [("depends-on", "depends_on"), ("used_in", "uses")]


@pytest.mark.asyncio
async def test_propose_merges_empty_when_all_canonical() -> None:
    storage = InMemoryStorage()
    await _seed_edge(storage, frm="rust", to="tokio", relation="uses")
    await _seed_edge(storage, frm="rust", to="serde", relation="depends_on")
    job = RelationNormJob(storage, settings=Settings())

    assert await job.propose_merges() == []


# --- the applied pass (run) -----------------------------------------------


@pytest.mark.asyncio
async def test_run_relabels_edges_to_canonical() -> None:
    storage = InMemoryStorage()
    await _seed_edge(storage, frm="rust", to="tokio", relation="used_in")
    await _seed_edge(storage, frm="rust", to="serde", relation="used-in")
    job = RelationNormJob(storage, settings=Settings())

    report = await job.run()

    assert report.relations_merged == 2
    # Both fragmented labels collapse to the canonical 'uses'.
    assert dict(await storage.list_relations()) == {"uses": 2}


@pytest.mark.asyncio
async def test_run_combines_parallel_edges_no_duplicate() -> None:
    storage = InMemoryStorage()
    # A canonical edge AND a fragmented variant between the SAME pair: the merge
    # folds onto the single canonical edge (max weight), never duplicating.
    await _seed_edge(storage, frm="rust", to="tokio", relation="uses", weight=0.4)
    await _seed_edge(storage, frm="rust", to="tokio", relation="used_in", weight=0.9)
    job = RelationNormJob(storage, settings=Settings())

    report = await job.run()

    assert report.relations_merged == 1
    edges = [e for e in storage.edges.values() if e.is_active]
    assert len(edges) == 1
    assert edges[0].relation == "uses"
    assert edges[0].weight == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_run_is_idempotent() -> None:
    storage = InMemoryStorage()
    await _seed_edge(storage, frm="rust", to="tokio", relation="used_in")
    await _seed_edge(storage, frm="rust", to="serde", relation="requires")
    job = RelationNormJob(storage, settings=Settings())

    first = await job.run()
    second = await job.run()

    assert first.relations_merged == 2
    # Nothing left to normalize on the second pass (FR-MNT-5 idempotency).
    assert second.relations_merged == 0


@pytest.mark.asyncio
async def test_run_with_no_relations_is_noop() -> None:
    storage = InMemoryStorage()
    job = RelationNormJob(storage)
    report = await job.run()
    assert report.relations_merged == 0
    assert report.job_name == "relation_norm"
