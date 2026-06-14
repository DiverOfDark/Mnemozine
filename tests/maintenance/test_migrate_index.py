"""Unit tests for the OQ3 index/re-embed migration (migrate-index).

Two layers, both fully offline:

* :func:`needs_migration` — the pure decision logic (configured-vs-actual
  dimension, ``force``, absent index).
* :class:`MigrateIndexJob.run` — the end-to-end pass against the shared
  ``InMemoryStorage`` fake + a tiny fake vector-index admin, asserting the
  drop/recreate seam fires only when needed and the hot tier (and only the hot
  tier) is re-embedded.
"""

from __future__ import annotations

from mnemozine.config import EmbeddingSettings, Settings
from mnemozine.maintenance.migrate_index import MigrateIndexJob, needs_migration
from mnemozine.schema.events import Source
from mnemozine.schema.models import (
    MemoryUnit,
    Provenance,
    Scope,
    Tier,
)
from tests.conftest import FakeEmbeddingProvider, InMemoryStorage

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeIndexAdmin:
    """Minimal :class:`VectorIndexAdmin` over an in-memory "live" dimension.

    ``current_dim`` is what the live FalkorDB index would report;
    :meth:`recreate_vector_index` simulates the drop+recreate by adopting the
    configured ``embedding_dimensions`` and recording that it ran.
    """

    def __init__(self, *, current_dim: int | None, configured_dim: int) -> None:
        self._current_dim = current_dim
        self._configured_dim = configured_dim
        self.recreate_calls = 0
        self.dimension_reads = 0

    @property
    def embedding_dimensions(self) -> int:
        return self._configured_dim

    async def current_vector_index_dimension(self) -> int | None:
        self.dimension_reads += 1
        return self._current_dim

    async def recreate_vector_index(self) -> None:
        self.recreate_calls += 1
        self._current_dim = self._configured_dim


def _settings(dim: int) -> Settings:
    return Settings(embedding=EmbeddingSettings(dimensions=dim))


def _memory(content: str, *, tier: Tier = Tier.HOT) -> MemoryUnit:
    return MemoryUnit(
        content=content,
        scope=Scope.global_(),
        category="preference",
        entities=["rust"],
        confidence=0.9,
        provenance=Provenance(source=Source.CLAUDE_CODE.value, session_id="s"),
        tier=tier,
    )


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


def test_needs_migration_true_on_dimension_change() -> None:
    assert needs_migration(configured_dim=1024, actual_dim=768) is True


def test_needs_migration_false_when_dimensions_match() -> None:
    assert needs_migration(configured_dim=1024, actual_dim=1024) is False


def test_needs_migration_false_when_index_absent() -> None:
    # Fresh store: connect() builds the index at the right width; nothing to fix.
    assert needs_migration(configured_dim=1024, actual_dim=None) is False


def test_needs_migration_force_overrides_everything() -> None:
    assert needs_migration(configured_dim=1024, actual_dim=1024, force=True) is True
    assert needs_migration(configured_dim=1024, actual_dim=None, force=True) is True


# ---------------------------------------------------------------------------
# Full pass (re-embed loop against the fake backend)
# ---------------------------------------------------------------------------


async def _seed_hot_and_archive(storage: InMemoryStorage) -> tuple[list[str], str]:
    hot_ids = []
    for content in ("alpha", "beta", "gamma"):
        m = _memory(content)
        await storage.upsert_memory(m)
        hot_ids.append(m.id)
    archived = _memory("cold", tier=Tier.ARCHIVE)
    await storage.upsert_memory(archived)
    return hot_ids, archived.id


async def test_migrate_recreates_index_and_reembeds_hot_only() -> None:
    storage = InMemoryStorage()
    hot_ids, archived_id = await _seed_hot_and_archive(storage)

    admin = FakeIndexAdmin(current_dim=768, configured_dim=1024)
    job = MigrateIndexJob(
        storage, admin, FakeEmbeddingProvider(), settings=_settings(1024)
    )
    report = await job.run()

    # Dimension changed -> index recreated exactly once.
    assert admin.recreate_calls == 1
    # Every hot memory re-embedded once; the archived one is skipped (lazy on
    # promotion), proving the loop honours tier=HOT.
    for mid in hot_ids:
        assert storage.reembed_calls.get(mid) == 1
    assert archived_id not in storage.reembed_calls
    assert report.consolidated == len(hot_ids)
    assert "migrate=True" in report.notes


async def test_migrate_noop_when_dimension_matches() -> None:
    storage = InMemoryStorage()
    hot_ids, _ = await _seed_hot_and_archive(storage)

    admin = FakeIndexAdmin(current_dim=1024, configured_dim=1024)
    job = MigrateIndexJob(
        storage, admin, FakeEmbeddingProvider(), settings=_settings(1024)
    )
    report = await job.run()

    # No change -> no drop/recreate, no re-embed (cheap, idempotent re-run).
    assert admin.recreate_calls == 0
    assert storage.reembed_calls == {}
    assert report.consolidated == 0
    assert "migrate=False" in report.notes


async def test_migrate_noop_when_index_absent() -> None:
    storage = InMemoryStorage()
    await _seed_hot_and_archive(storage)

    admin = FakeIndexAdmin(current_dim=None, configured_dim=1024)
    job = MigrateIndexJob(
        storage, admin, FakeEmbeddingProvider(), settings=_settings(1024)
    )
    report = await job.run()

    assert admin.recreate_calls == 0
    assert storage.reembed_calls == {}
    assert report.consolidated == 0


async def test_migrate_force_reembeds_without_dimension_change() -> None:
    storage = InMemoryStorage()
    hot_ids, archived_id = await _seed_hot_and_archive(storage)

    admin = FakeIndexAdmin(current_dim=1024, configured_dim=1024)
    job = MigrateIndexJob(
        storage, admin, FakeEmbeddingProvider(), settings=_settings(1024), force=True
    )
    report = await job.run()

    # force=True re-embeds the hot tier even though the width is unchanged.
    assert admin.recreate_calls == 1
    for mid in hot_ids:
        assert storage.reembed_calls.get(mid) == 1
    assert archived_id not in storage.reembed_calls
    assert report.consolidated == len(hot_ids)


async def test_migrate_is_idempotent_on_rerun() -> None:
    storage = InMemoryStorage()
    hot_ids, _ = await _seed_hot_and_archive(storage)

    admin = FakeIndexAdmin(current_dim=512, configured_dim=1024)
    job = MigrateIndexJob(
        storage, admin, FakeEmbeddingProvider(), settings=_settings(1024)
    )
    first = await job.run()
    assert first.consolidated == len(hot_ids)
    assert admin.recreate_calls == 1

    # Second run: the admin now reports the new width -> the pass becomes a no-op.
    second = await job.run()
    assert second.consolidated == 0
    assert admin.recreate_calls == 1  # not recreated again
    # Re-embed counts did not grow on the no-op second pass.
    for mid in hot_ids:
        assert storage.reembed_calls.get(mid) == 1


async def test_migrate_reports_provider_dimension_mismatch_warning() -> None:
    storage = InMemoryStorage()
    await _seed_hot_and_archive(storage)

    # Provider is 8-d (FakeEmbeddingProvider default) but config says 1024.
    admin = FakeIndexAdmin(current_dim=768, configured_dim=1024)
    job = MigrateIndexJob(
        storage, admin, FakeEmbeddingProvider(dimensions=8), settings=_settings(1024)
    )
    report = await job.run()
    assert any("WARNING" in n and "provider" in n.lower() for n in report.notes)


def test_migrate_index_job_name() -> None:
    job = MigrateIndexJob(
        InMemoryStorage(),
        FakeIndexAdmin(current_dim=1024, configured_dim=1024),
        FakeEmbeddingProvider(),
        settings=_settings(1024),
    )
    assert job.name == "migrate-index"


def test_migrate_index_job_satisfies_maintenance_job_protocol() -> None:
    from mnemozine.interfaces import MaintenanceJob

    job = MigrateIndexJob(
        InMemoryStorage(),
        FakeIndexAdmin(current_dim=1024, configured_dim=1024),
        FakeEmbeddingProvider(),
        settings=_settings(1024),
    )
    assert isinstance(job, MaintenanceJob)
