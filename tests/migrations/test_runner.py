"""Unit tests for the in-place migration RUNNER (FR-MNT-5).

Exercises the runner's contract against the conftest ``InMemoryStorage`` fake and
tiny fake migrations, fully offline:

* pending SELECTION + ORDERING — only migrations above ``min_data_version`` run,
  ascending, capped at the target;
* IDEMPOTENCY / resumability — a re-run over a migrated store is a no-op, and a
  migration whose version the store already reached is skipped mid-pass;
* DRY-RUN — plans the steps without touching the store;
* the CHEAP/HEAVY split — ``include_reextract=False`` skips heavy re-extract
  migrations, and a heavy migration with no extractor fails loudly.

The runner resolves the module-level ``MIGRATIONS`` registry through
``pending_migrations``; tests install a controlled registry via ``monkeypatch`` so
they do not depend on the real baseline being the only registered step.
"""

from __future__ import annotations

import pytest

import mnemozine.migrations as migrations_pkg
from mnemozine.interfaces import MaintenanceReport
from mnemozine.migrations import Migration
from mnemozine.migrations.runner import (
    MigrationExtractorRequired,
    MigrationRunner,
)
from mnemozine.schema.models import MemoryUnit, Provenance, Scope
from tests.conftest import InMemoryStorage


class _FakeMigration:
    """A controllable :class:`Migration`: stamps every below-version memory to v.

    Records each ``run`` call so ordering / idempotency can be asserted, and
    advances BOTH tiers (memories + raw chunks) to its version so
    ``min_data_version`` actually reaches it (the contract the real runner relies
    on for resumability).
    """

    def __init__(
        self,
        version: int,
        *,
        requires_reextract: bool = False,
        record: list[int] | None = None,
    ) -> None:
        self._version = version
        self._requires_reextract = requires_reextract
        self.record = record if record is not None else []

    @property
    def version(self) -> int:
        return self._version

    @property
    def description(self) -> str:
        return f"fake migration to v{self._version}"

    @property
    def requires_reextract(self) -> bool:
        return self._requires_reextract

    async def run(self, backend, *, extractor=None):  # type: ignore[no-untyped-def]
        self.record.append(self._version)
        ids = [
            m.id
            async for m in backend.iter_memories_below_version(self._version)
        ]
        n = await backend.set_data_version(ids, self._version)
        hashes = [
            c.content_hash
            async for c in backend.iter_chunks_below_version(self._version)
        ]
        await backend.set_chunk_data_version(hashes, self._version)
        return MaintenanceReport(job_name=f"fake_v{self._version}", re_extracted=n)


def _install_registry(monkeypatch, migrations: list[Migration]) -> None:
    """Point the package-level MIGRATIONS at a controlled, validated registry.

    Also raises ``CURRENT_DATA_VERSION`` to the registry's max version so the
    contiguous-1..N registry invariant (enforced by ``validate_migrations`` inside
    ``pending_migrations``) is satisfied for the fake multi-version registries.
    """

    monkeypatch.setattr(migrations_pkg, "MIGRATIONS", list(migrations))
    if migrations:
        monkeypatch.setattr(
            migrations_pkg,
            "CURRENT_DATA_VERSION",
            max(m.version for m in migrations),
        )


def _seed_memory(storage: InMemoryStorage, *, version: int) -> MemoryUnit:
    m = MemoryUnit(
        content="seed memory",
        scope=Scope.global_(),
        provenance=Provenance(source="claude_code", session_id="s1"),
        data_version=version,
    )
    storage.memories[m.id] = m
    return m


def test_fake_migration_satisfies_protocol() -> None:
    assert isinstance(_FakeMigration(1), Migration)


async def test_pending_selects_only_above_min_version(monkeypatch) -> None:
    _install_registry(monkeypatch, [_FakeMigration(1), _FakeMigration(2)])
    storage = InMemoryStorage()
    _seed_memory(storage, version=1)  # min_data_version == 1
    runner = MigrationRunner(storage, target_version=2)
    pending = await runner.pending()
    assert [m.version for m in pending] == [2]


async def test_run_applies_pending_in_ascending_order(monkeypatch) -> None:
    order: list[int] = []
    _install_registry(
        monkeypatch,
        [
            _FakeMigration(2, record=order),
            _FakeMigration(1, record=order),
            _FakeMigration(3, record=order),
        ],
    )
    storage = InMemoryStorage()
    _seed_memory(storage, version=0)  # below all migrations
    runner = MigrationRunner(storage, target_version=3)
    report = await runner.run()
    # Applied ascending despite registry insertion order.
    assert order == [1, 2, 3]
    assert report.applied == 3
    assert report.from_version == 0
    assert report.to_version == 3
    # Both tiers reached the target: re-run is a no-op.
    assert await storage.min_data_version() == 3


async def test_run_is_idempotent_no_op_on_rerun(monkeypatch) -> None:
    order: list[int] = []
    _install_registry(monkeypatch, [_FakeMigration(1, record=order)])
    storage = InMemoryStorage()
    _seed_memory(storage, version=0)
    runner = MigrationRunner(storage, target_version=1)

    first = await runner.run()
    assert first.applied == 1
    assert order == [1]

    second = await runner.run()
    # Nothing left below v1: the migration's run() is never called again.
    assert second.applied == 0
    assert order == [1]
    assert second.from_version == 1
    assert second.to_version == 1


async def test_run_skips_a_version_the_store_already_reached(monkeypatch) -> None:
    order: list[int] = []
    _install_registry(
        monkeypatch,
        [_FakeMigration(1, record=order), _FakeMigration(2, record=order)],
    )
    storage = InMemoryStorage()
    # Store already at v1: only v2 should run.
    _seed_memory(storage, version=1)
    runner = MigrationRunner(storage, target_version=2)
    report = await runner.run()
    assert order == [2]
    assert report.applied == 1
    assert report.to_version == 2


async def test_dry_run_plans_without_writing(monkeypatch) -> None:
    order: list[int] = []
    _install_registry(
        monkeypatch,
        [_FakeMigration(1, record=order), _FakeMigration(2, record=order)],
    )
    storage = InMemoryStorage()
    _seed_memory(storage, version=0)
    runner = MigrationRunner(storage, target_version=2)

    plan = await runner.plan()
    assert plan.dry_run is True
    assert [p.version for p in plan.plan] == [1, 2]
    assert plan.to_version == 2
    # No migration ran, store untouched.
    assert order == []
    assert plan.applied == 0
    assert await storage.min_data_version() == 0


async def test_dry_run_empty_when_fully_migrated(monkeypatch) -> None:
    _install_registry(monkeypatch, [_FakeMigration(1)])
    storage = InMemoryStorage()
    _seed_memory(storage, version=1)
    runner = MigrationRunner(storage, target_version=1)
    plan = await runner.plan()
    assert plan.plan == []
    assert plan.from_version == 1
    assert plan.to_version == 1
    assert any("nothing pending" in n for n in plan.notes)


async def test_include_reextract_false_skips_heavy_migration(monkeypatch) -> None:
    order: list[int] = []
    _install_registry(
        monkeypatch,
        [
            _FakeMigration(1, record=order),
            _FakeMigration(2, requires_reextract=True, record=order),
        ],
    )
    storage = InMemoryStorage()
    _seed_memory(storage, version=0)
    runner = MigrationRunner(storage, target_version=2)

    report = await runner.run(include_reextract=False)
    # Only the cheap v1 ran; the heavy v2 was skipped and noted.
    assert order == [1]
    assert report.applied == 1
    assert any("heavy re-extract migration skipped" in n for n in report.notes)
    # The store is at v1 (the heavy step did not advance it).
    assert await storage.min_data_version() == 1


async def test_dry_run_marks_heavy_step_skipped(monkeypatch) -> None:
    _install_registry(
        monkeypatch,
        [
            _FakeMigration(1),
            _FakeMigration(2, requires_reextract=True),
        ],
    )
    storage = InMemoryStorage()
    _seed_memory(storage, version=0)
    runner = MigrationRunner(storage, target_version=2)
    plan = await runner.plan(include_reextract=False)
    by_version = {p.version: p for p in plan.plan}
    assert by_version[1].skipped is False
    assert by_version[2].skipped is True
    assert by_version[2].skip_reason is not None
    # to_version reflects only the applicable (cheap) steps.
    assert plan.to_version == 1


async def test_heavy_migration_without_extractor_raises(monkeypatch) -> None:
    _install_registry(
        monkeypatch, [_FakeMigration(1, requires_reextract=True)]
    )
    storage = InMemoryStorage()
    _seed_memory(storage, version=0)
    runner = MigrationRunner(storage, target_version=1)  # no extractor
    with pytest.raises(MigrationExtractorRequired):
        await runner.run(include_reextract=True)


async def test_run_no_op_on_empty_store(monkeypatch) -> None:
    _install_registry(monkeypatch, [_FakeMigration(1)])
    storage = InMemoryStorage()  # empty -> min_data_version == CURRENT
    runner = MigrationRunner(storage, target_version=1)
    report = await runner.run()
    assert report.applied == 0
    assert any("nothing to do" in n for n in report.notes)
