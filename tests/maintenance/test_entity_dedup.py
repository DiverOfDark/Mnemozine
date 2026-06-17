"""Entity-dedup tests — merge true-duplicate entities, repointing ALL edge types.

Covers the per-mode grouping DECISION (exact / alias / embedding) and the applied
``run`` pass against the conftest ``InMemoryStorage`` (whose ``merge_entities``
repoints the mentions + co-mention layers in lock-step with the real backend), plus
the FR-MNT-5 idempotency assertion (a second run merges 0). All offline.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from mnemozine.config import Settings
from mnemozine.interfaces import MaintenanceJob
from mnemozine.maintenance.entity_dedup import DEDUP_MODES, EntityDedupJob
from mnemozine.schema.models import Entity, MemoryUnit, Provenance, Scope
from tests.conftest import InMemoryStorage


def _settings(mode: str = "exact", *, threshold: float = 0.92) -> Settings:
    s = Settings()
    s.graph.entity_dedup_mode = mode
    s.graph.entity_dedup_similarity_threshold = threshold
    return s


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


class _ControlledEmbeddingProvider:
    """Embeds by a name->vector table so cosine is deterministic in tests."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    @property
    def dimensions(self) -> int:
        return 2

    async def embed(self, text: str) -> list[float]:
        return self._table.get(text, [1.0, 0.0])

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


# --- protocol + config -----------------------------------------------------


def test_satisfies_protocol_and_name() -> None:
    job = EntityDedupJob(InMemoryStorage())
    assert isinstance(job, MaintenanceJob)
    assert job.name == "entity_dedup"


def test_dedup_modes_constant() -> None:
    assert DEDUP_MODES == ("exact", "alias", "embedding")


def test_mode_defaults_from_config_and_cli_override() -> None:
    storage = InMemoryStorage()
    # Default reads graph.entity_dedup_mode.
    assert EntityDedupJob(storage, settings=_settings("alias"))._mode == "alias"
    # CLI --mode overrides the config value.
    job = EntityDedupJob(storage, settings=_settings("alias"), mode="EXACT")
    assert job._mode == "exact"


# --- exact mode (default) --------------------------------------------------


@pytest.mark.asyncio
async def test_exact_merges_case_collision_deterministic_survivor() -> None:
    storage = InMemoryStorage()
    # Same name, different case -> a true duplicate group; the survivor with more
    # aliases wins deterministically (entity_resolution._pick_survivor).
    survivor = Entity(id="e-gh", canonical_name="GitHub", aliases=["gh", "github.com"])
    dup = Entity(id="e-gh2", canonical_name="github", aliases=["GH"])
    await storage.upsert_entity(survivor)
    await storage.upsert_entity(dup)
    job = EntityDedupJob(storage, settings=_settings("exact"))

    report = await job.run()

    assert report.entities_merged == 1
    # Survivor kept, duplicate folded away; alias union carried over.
    assert "e-gh" in storage.entities
    assert "e-gh2" not in storage.entities
    assert "github" in storage.entities["e-gh"].aliases


@pytest.mark.asyncio
async def test_exact_leaves_distinct_names_untouched() -> None:
    storage = InMemoryStorage()
    await storage.upsert_entity(Entity(id="e1", canonical_name="rust"))
    await storage.upsert_entity(Entity(id="e2", canonical_name="tokio"))
    job = EntityDedupJob(storage, settings=_settings("exact"))

    report = await job.run()

    assert report.entities_merged == 0
    assert set(storage.entities) == {"e1", "e2"}


@pytest.mark.asyncio
async def test_exact_repoints_mentions_and_co_mention_onto_survivor() -> None:
    storage = InMemoryStorage()
    await storage.upsert_entity(
        Entity(id="e-rust", canonical_name="rust", aliases=["rustc"])
    )
    await storage.upsert_entity(Entity(id="e-rust2", canonical_name="Rust"))
    await storage.upsert_entity(Entity(id="e-tokio", canonical_name="tokio"))
    # A memory mentions the duplicate; a co-mention edge touches it.
    await storage.upsert_memory(
        _mem("rust note", entities=["Rust"], mid="m1")
    )
    await storage.persist_mentions()
    await storage.upsert_co_mention("e-rust2", "e-tokio", weight=2.0, shared=3)
    assert ("m1", "e-rust2") in storage.mentions
    assert ("e-rust2", "e-tokio") in storage.co_mentions

    report = await job_run(storage, "exact")

    assert report.entities_merged == 1
    # Mentions + co-mention repointed onto the survivor, none left dangling.
    assert ("m1", "e-rust") in storage.mentions
    assert ("m1", "e-rust2") not in storage.mentions
    assert ("e-rust", "e-tokio") in storage.co_mentions
    assert ("e-rust2", "e-tokio") not in storage.co_mentions
    # No memory was deleted.
    assert "m1" in storage.memories


# --- alias mode ------------------------------------------------------------


@pytest.mark.asyncio
async def test_alias_mode_folds_alias_linked_entities() -> None:
    storage = InMemoryStorage()
    # 'rust-lang' is already an alias of 'rust' -> alias mode folds them; exact
    # mode (different lower(canonical_name)) would NOT.
    await storage.upsert_entity(
        Entity(id="e-rust", canonical_name="rust", aliases=["rust-lang"])
    )
    await storage.upsert_entity(Entity(id="e-rl", canonical_name="rust-lang"))

    # exact mode: distinct names, nothing merges.
    exact_report = await job_run(storage_copy(storage), "exact")
    assert exact_report.entities_merged == 0

    # alias mode: the alias linkage folds the pair.
    alias_report = await job_run(storage, "alias")
    assert alias_report.entities_merged == 1
    assert "e-rl" not in storage.entities or "e-rust" not in storage.entities


# --- embedding mode --------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_mode_folds_near_dup_names_above_threshold() -> None:
    storage = InMemoryStorage()
    await storage.upsert_entity(Entity(id="e-a", canonical_name="kubernetes"))
    await storage.upsert_entity(Entity(id="e-b", canonical_name="k8s"))
    await storage.upsert_entity(Entity(id="e-c", canonical_name="postgres"))
    # Make kubernetes/k8s near-identical vectors, postgres orthogonal.
    table = {
        "kubernetes": [1.0, 0.0],
        "k8s": [0.99, 0.01],
        "postgres": [0.0, 1.0],
    }
    embeddings = _ControlledEmbeddingProvider(table)
    job = EntityDedupJob(
        storage, embeddings=embeddings, settings=_settings("embedding", threshold=0.9)
    )

    report = await job.run()

    # kubernetes + k8s fold (cosine ~1.0 >= 0.9); postgres stays.
    assert report.entities_merged == 1
    assert "e-c" in storage.entities
    assert len([e for e in storage.entities]) == 2


@pytest.mark.asyncio
async def test_embedding_mode_without_provider_only_exact() -> None:
    storage = InMemoryStorage()
    await storage.upsert_entity(Entity(id="e-a", canonical_name="kubernetes"))
    await storage.upsert_entity(Entity(id="e-b", canonical_name="k8s"))
    # No embedding provider: the embedding widening is skipped, distinct names stay.
    job = EntityDedupJob(storage, embeddings=None, settings=_settings("embedding"))

    report = await job.run()

    assert report.entities_merged == 0
    assert set(storage.entities) == {"e-a", "e-b"}


# --- idempotency + edge cases ----------------------------------------------


@pytest.mark.asyncio
async def test_run_is_idempotent() -> None:
    storage = InMemoryStorage()
    await storage.upsert_entity(Entity(id="e1", canonical_name="GitHub"))
    await storage.upsert_entity(Entity(id="e2", canonical_name="github"))
    await storage.upsert_entity(Entity(id="e3", canonical_name="GITHUB"))
    job = EntityDedupJob(storage, settings=_settings("exact"))

    first = await job.run()
    second = await job.run()

    # First pass folds the two duplicates into one survivor; the second finds no
    # collision and merges 0 (FR-MNT-5).
    assert first.entities_merged == 2
    assert second.entities_merged == 0
    assert len(storage.entities) == 1


@pytest.mark.asyncio
async def test_unknown_mode_is_noop() -> None:
    storage = InMemoryStorage()
    await storage.upsert_entity(Entity(id="e1", canonical_name="GitHub"))
    await storage.upsert_entity(Entity(id="e2", canonical_name="github"))
    job = EntityDedupJob(storage, settings=_settings("bogus"))

    report = await job.run()

    assert report.entities_merged == 0
    assert len(storage.entities) == 2
    assert any("unknown entity_dedup mode" in n for n in report.notes)


@pytest.mark.asyncio
async def test_empty_store_is_noop() -> None:
    job = EntityDedupJob(InMemoryStorage(), settings=_settings("exact"))
    report = await job.run()
    assert report.entities_merged == 0
    assert report.job_name == "entity_dedup"


# --- helpers ---------------------------------------------------------------


async def job_run(storage: InMemoryStorage, mode: str):
    return await EntityDedupJob(storage, settings=_settings(mode)).run()


def storage_copy(storage: InMemoryStorage) -> InMemoryStorage:
    """Shallow clone of the entity table so the exact-mode dry check is isolated."""

    clone = InMemoryStorage()
    clone.entities = {
        eid: Entity(
            id=e.id,
            canonical_name=e.canonical_name,
            aliases=list(e.aliases),
            type=e.type,
        )
        for eid, e in storage.entities.items()
    }
    return clone
