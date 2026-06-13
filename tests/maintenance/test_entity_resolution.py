"""FR-MNT-4 tests — duplicate-entity merge, low-weight edge prune, degree cap."""

from __future__ import annotations

import pytest

from mnemozine.config import Settings
from mnemozine.maintenance.entity_resolution import (
    EntityResolutionJob,
    normalize_entity_key,
)
from mnemozine.schema.models import Edge, Entity
from tests.conftest import InMemoryStorage


def test_normalize_entity_key_collapses_variants() -> None:
    keys = {
        normalize_entity_key("Rust"),
        normalize_entity_key("rust-lang"),
        normalize_entity_key("the Rust"),
        normalize_entity_key("the rust work"),
    }
    assert keys == {"rust"}
    # Distinct concepts stay distinct.
    assert normalize_entity_key("async") != normalize_entity_key("cli")


@pytest.mark.asyncio
async def test_merge_duplicate_entities() -> None:
    storage = InMemoryStorage()
    a = Entity(canonical_name="rust")
    b = Entity(canonical_name="rust-lang", aliases=["rustc"])
    c = Entity(canonical_name="the Rust work")
    other = Entity(canonical_name="async")
    for e in (a, b, c, other):
        await storage.upsert_entity(e)

    job = EntityResolutionJob(storage, settings=Settings())
    report = await job.run()

    # Two of the three rust duplicates merged into one survivor; async untouched.
    assert report.entities_merged == 2
    rust_entities = [
        e for e in storage.entities.values()
        if normalize_entity_key(e.canonical_name) == "rust"
    ]
    assert len(rust_entities) == 1
    survivor = rust_entities[0]
    # The merge folds the others' names/aliases into the survivor's aliases.
    assert "rustc" in survivor.aliases or survivor.canonical_name == "rust-lang"
    assert "async" in [e.canonical_name for e in storage.entities.values()]


@pytest.mark.asyncio
async def test_merge_is_idempotent() -> None:
    storage = InMemoryStorage()
    for name in ("rust", "rust-lang"):
        await storage.upsert_entity(Entity(canonical_name=name))
    job = EntityResolutionJob(storage, settings=Settings())
    first = await job.run()
    second = await job.run()
    assert first.entities_merged == 1
    assert second.entities_merged == 0  # nothing left to merge


@pytest.mark.asyncio
async def test_prune_low_weight_edges() -> None:
    settings = Settings()
    settings.maintenance.edge_weight_floor = 0.1
    storage = InMemoryStorage()
    a = Entity(canonical_name="rust")
    b = Entity(canonical_name="async")
    await storage.upsert_entity(a)
    await storage.upsert_entity(b)
    strong = Edge(from_entity=a.id, to_entity=b.id, relation="relates_to", weight=0.9)
    weak = Edge(from_entity=a.id, to_entity=b.id, relation="vaguely", weight=0.02)
    await storage.upsert_edge(strong)
    await storage.upsert_edge(weak)

    job = EntityResolutionJob(storage, settings=settings)
    report = await job.run()

    assert report.edges_pruned == 1
    # Weak edge's validity window closed (pruned, not deleted); strong stays open.
    assert storage.edges[weak.id].valid_to is not None
    assert storage.edges[strong.id].is_active


@pytest.mark.asyncio
async def test_node_degree_cap_keeps_highest_weight() -> None:
    settings = Settings()
    settings.maintenance.edge_weight_floor = 0.0  # nothing pruned by floor
    settings.maintenance.max_node_degree = 2
    storage = InMemoryStorage()
    hub = Entity(canonical_name="rust")
    await storage.upsert_entity(hub)
    spokes = []
    for i in range(5):
        s = Entity(canonical_name=f"spoke{i}")
        await storage.upsert_entity(s)
        spokes.append(s)
        await storage.upsert_edge(
            Edge(
                from_entity=hub.id,
                to_entity=s.id,
                relation=f"r{i}",
                weight=float(i + 1),  # weights 1..5
            )
        )

    job = EntityResolutionJob(storage, settings=settings)
    report = await job.run()

    active_edges = [e for e in storage.edges.values() if e.is_active]
    # Capped to max_node_degree=2: only the two highest-weight edges survive.
    assert len(active_edges) == 2
    surviving_weights = sorted(e.weight for e in active_edges)
    assert surviving_weights == [4.0, 5.0]
    assert report.edges_pruned == 3
