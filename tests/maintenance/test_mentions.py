"""MentionsJob tests — persist memory->entity mention edges from m.entities.

Covers Protocol conformance, the applied ``run`` pass against the conftest
``InMemoryStorage`` fake (resolution by canonical name + alias, case-folding),
and the FR-MNT-5 idempotency property (a re-run asserts the same edges and adds
nothing new — the second run's edges_added equals the first's, with no growth in
the stored mention set).
"""

from __future__ import annotations

import pytest

from mnemozine.config import Settings
from mnemozine.interfaces import MaintenanceJob
from mnemozine.maintenance.mentions import MentionsJob
from mnemozine.schema.models import Entity, MemoryUnit, Provenance, Scope
from tests.conftest import InMemoryStorage


def _mem(content: str, *, entities: list[str], mid: str | None = None) -> MemoryUnit:
    kwargs = {
        "content": content,
        "scope": Scope.global_(),
        "category": "fact",
        "entities": entities,
        "confidence": 0.9,
        "provenance": Provenance(source="claude_code", session_id="s1"),
    }
    if mid is not None:
        kwargs["id"] = mid
    return MemoryUnit(**kwargs)


# --- protocol --------------------------------------------------------------


def test_satisfies_maintenance_job_protocol() -> None:
    job = MentionsJob(InMemoryStorage())
    assert isinstance(job, MaintenanceJob)
    assert job.name == "mentions"


# --- the applied pass (run) ------------------------------------------------


@pytest.mark.asyncio
async def test_run_persists_mention_edges_from_m_entities() -> None:
    storage = InMemoryStorage()
    await storage.upsert_entity(Entity(id="e-rust", canonical_name="rust"))
    await storage.upsert_entity(Entity(id="e-tokio", canonical_name="tokio"))
    await storage.upsert_memory(_mem("uses tokio in rust", entities=["rust", "tokio"], mid="m1"))
    await storage.upsert_memory(_mem("rust note", entities=["rust"], mid="m2"))

    job = MentionsJob(storage, settings=Settings())
    report = await job.run()

    # 3 mention edges: (m1->rust), (m1->tokio), (m2->rust).
    assert report.edges_added == 3
    assert storage.mentions == {
        ("m1", "e-rust"),
        ("m1", "e-tokio"),
        ("m2", "e-rust"),
    }


@pytest.mark.asyncio
async def test_resolution_is_case_folded_and_alias_aware() -> None:
    # Linkage drift: a memory mentions 'Rust' (cased) and 'rustc' (an alias),
    # both resolve to the same canonical entity by case-folded name/alias match.
    storage = InMemoryStorage()
    await storage.upsert_entity(
        Entity(id="e-rust", canonical_name="rust", aliases=["rustc", "rust-lang"])
    )
    await storage.upsert_memory(_mem("Rust note", entities=["Rust"], mid="m1"))
    await storage.upsert_memory(_mem("compiler note", entities=["rustc"], mid="m2"))

    report = await MentionsJob(storage).run()

    assert report.edges_added == 2
    assert storage.mentions == {("m1", "e-rust"), ("m2", "e-rust")}


@pytest.mark.asyncio
async def test_unresolvable_names_are_skipped() -> None:
    # A mention name with no matching entity produces no edge (no orphan edges).
    storage = InMemoryStorage()
    await storage.upsert_entity(Entity(id="e-rust", canonical_name="rust"))
    await storage.upsert_memory(
        _mem("rust + unknown", entities=["rust", "no-such-entity"], mid="m1")
    )

    report = await MentionsJob(storage).run()

    assert report.edges_added == 1
    assert storage.mentions == {("m1", "e-rust")}


@pytest.mark.asyncio
async def test_run_is_idempotent() -> None:
    # FR-MNT-5: a second pass asserts the SAME edges (MERGE, not CREATE) — the
    # mention set does not grow and edges_added is stable.
    storage = InMemoryStorage()
    await storage.upsert_entity(Entity(id="e-rust", canonical_name="rust"))
    await storage.upsert_entity(Entity(id="e-tokio", canonical_name="tokio"))
    await storage.upsert_memory(
        _mem("uses tokio in rust", entities=["rust", "tokio"], mid="m1")
    )
    job = MentionsJob(storage)

    first = await job.run()
    edges_after_first = set(storage.mentions)
    second = await job.run()

    assert first.edges_added == 2
    # Re-running re-asserts the same 2 edges (idempotent MERGE) and adds none.
    assert second.edges_added == 2
    assert storage.mentions == edges_after_first


@pytest.mark.asyncio
async def test_run_with_no_entities_is_noop() -> None:
    storage = InMemoryStorage()
    report = await MentionsJob(storage).run()
    assert report.edges_added == 0
    assert report.job_name == "mentions"


@pytest.mark.asyncio
async def test_merge_entities_repoints_mention_edges() -> None:
    # When a duplicate entity is merged, the mentions that pointed at it must
    # repoint to the survivor (forward-compat with the entity-dedup pass) and
    # collapse via set semantics (no duplicate edge for a memory that mentioned
    # both).
    storage = InMemoryStorage()
    await storage.upsert_entity(Entity(id="e-rust", canonical_name="rust"))
    await storage.upsert_entity(Entity(id="e-rustlang", canonical_name="rust-lang"))
    await storage.upsert_memory(
        _mem("rust + rust-lang", entities=["rust", "rust-lang"], mid="m1")
    )
    await MentionsJob(storage).run()
    assert storage.mentions == {("m1", "e-rust"), ("m1", "e-rustlang")}

    await storage.merge_entities("e-rustlang", "e-rust")

    # Both mentions now point at the survivor and collapse to a single edge.
    assert storage.mentions == {("m1", "e-rust")}
