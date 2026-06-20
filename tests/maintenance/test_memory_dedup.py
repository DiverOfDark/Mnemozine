"""MemoryDedupJob tests — collapse exact-duplicate active memos to one survivor.

Covers the deterministic (no-LLM) offline pass that buckets active hot memos by
``(normalized content, scope)`` and, for each cluster of >=2, keeps one
deterministic survivor while superseding (never deleting) the redundant copies via
:meth:`StorageBackend.close_validity_window`. Pure/deterministic — no LLM, no
embeddings — so it runs offline against the conftest
:class:`~tests.conftest.InMemoryStorage` fake (which already implements
``iter_memories(active_only=, tier=)`` + ``close_validity_window``; no fake change
needed).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mnemozine.app import Container, maintenance_app
from mnemozine.config import Settings
from mnemozine.interfaces import MaintenanceJob
from mnemozine.maintenance import MemoryDedupJob as MemoryDedupJobExport
from mnemozine.maintenance.memory_dedup import MemoryDedupJob
from mnemozine.maintenance.runner import build_default_jobs
from mnemozine.schema.models import MemoryUnit, Scope, Tier
from tests.conftest import (
    FakeEmbeddingProvider,
    FakeLLMProvider,
    InMemoryStorage,
)

_cli = CliRunner()

_DUP = "If any build is failing - retry it"


def _mem(
    content: str,
    *,
    scope: Scope | None = None,
    confidence: float = 0.9,
    valid_from: datetime | None = None,
    memory_id: str | None = None,
    tier: Tier = Tier.HOT,
) -> MemoryUnit:
    kwargs: dict = {
        "content": content,
        "scope": scope or Scope.global_(),
        "category": "fact",
        "confidence": confidence,
        "tier": tier,
    }
    if valid_from is not None:
        kwargs["valid_from"] = valid_from
    if memory_id is not None:
        kwargs["id"] = memory_id
    return MemoryUnit(**kwargs)


async def _seed(storage: InMemoryStorage, *mems: MemoryUnit) -> None:
    for m in mems:
        await storage.upsert_memory(m)


def _active(storage: InMemoryStorage) -> list[MemoryUnit]:
    return [m for m in storage.memories.values() if m.is_active]


def test_dedup_satisfies_protocol() -> None:
    job = MemoryDedupJob(InMemoryStorage())
    assert isinstance(job, MaintenanceJob)
    assert job.name == "dedup_memories"


@pytest.mark.asyncio
async def test_three_identical_collapse_to_one_active_survivor() -> None:
    storage = InMemoryStorage()
    a = _mem(_DUP)
    b = _mem(_DUP)
    c = _mem(_DUP)
    await _seed(storage, a, b, c)

    report = await MemoryDedupJob(storage, settings=Settings()).run()

    # One survivor stays active; the two redundant copies are superseded.
    active = _active(storage)
    assert len(active) == 1
    superseded = [m for m in storage.memories.values() if not m.is_active]
    assert len(superseded) == 2
    # Superseded copies are RETAINED (still present, valid_to set) — never deleted.
    assert len(storage.memories) == 3
    for m in superseded:
        assert m.valid_to is not None
    # consolidated counts the redundant copies removed from the active set.
    assert report.consolidated == 2


@pytest.mark.asyncio
async def test_distinct_content_is_not_collapsed() -> None:
    storage = InMemoryStorage()
    a = _mem(_DUP)
    b = _mem("Always run ruff before committing")
    await _seed(storage, a, b)

    report = await MemoryDedupJob(storage).run()

    assert report.consolidated == 0
    assert len(_active(storage)) == 2


@pytest.mark.asyncio
async def test_same_content_different_scope_is_not_collapsed() -> None:
    # Same wording but different scope -> distinct keys, NOT merged (preserve the
    # scope split / no-leak boundary).
    storage = InMemoryStorage()
    g = _mem(_DUP, scope=Scope.global_())
    p = _mem(_DUP, scope=Scope.project("aipack"))
    await _seed(storage, g, p)

    report = await MemoryDedupJob(storage).run()

    assert report.consolidated == 0
    assert len(_active(storage)) == 2


@pytest.mark.asyncio
async def test_whitespace_and_case_only_differences_collapse() -> None:
    # The key normalizes content via strip().lower(), so whitespace/case-only
    # variants collapse together.
    storage = InMemoryStorage()
    a = _mem(_DUP)
    b = _mem(f"  {_DUP.upper()}  ")
    await _seed(storage, a, b)

    report = await MemoryDedupJob(storage).run()

    assert report.consolidated == 1
    assert len(_active(storage)) == 1


@pytest.mark.asyncio
async def test_survivor_selection_is_deterministic_by_confidence() -> None:
    # Highest confidence wins regardless of insertion order.
    storage = InMemoryStorage()
    low = _mem(_DUP, confidence=0.50, memory_id="id-low")
    high = _mem(_DUP, confidence=0.95, memory_id="id-high")
    mid = _mem(_DUP, confidence=0.70, memory_id="id-mid")
    await _seed(storage, low, high, mid)

    await MemoryDedupJob(storage).run()

    active = _active(storage)
    assert len(active) == 1
    assert active[0].id == "id-high"


@pytest.mark.asyncio
async def test_survivor_tiebreak_earliest_valid_from_then_smallest_id() -> None:
    # Equal confidence -> earliest valid_from wins; equal valid_from -> smallest id.
    storage = InMemoryStorage()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    early = _mem(
        _DUP, confidence=0.9, valid_from=base, memory_id="id-zzz"
    )
    late = _mem(
        _DUP, confidence=0.9, valid_from=base + timedelta(days=5), memory_id="id-aaa"
    )
    # Same valid_from as `early` but a larger id -> loses the id tiebreak.
    same_time_bigger_id = _mem(
        _DUP, confidence=0.9, valid_from=base, memory_id="id-zzzz"
    )
    await _seed(storage, late, early, same_time_bigger_id)

    await MemoryDedupJob(storage).run()

    active = _active(storage)
    assert len(active) == 1
    # earliest valid_from (base) AND smallest id among the two base-time copies.
    assert active[0].id == "id-zzz"


@pytest.mark.asyncio
async def test_archived_duplicate_is_not_collapsed() -> None:
    # The pass scans the HOT tier only; an archived copy is outside the active hot
    # set and must not be touched (or counted) by dedup.
    storage = InMemoryStorage()
    hot = _mem(_DUP, tier=Tier.HOT)
    archived = _mem(_DUP, tier=Tier.ARCHIVE)
    await _seed(storage, hot, archived)

    report = await MemoryDedupJob(storage).run()

    assert report.consolidated == 0
    assert hot.is_active
    assert archived.is_active  # untouched
    assert archived.tier is Tier.ARCHIVE


@pytest.mark.asyncio
async def test_rerun_is_idempotent_noop() -> None:
    storage = InMemoryStorage()
    await _seed(storage, _mem(_DUP), _mem(_DUP), _mem(_DUP))

    first = await MemoryDedupJob(storage).run()
    second = await MemoryDedupJob(storage).run()

    assert first.consolidated == 2
    # The superseded copies left the active set, so a re-run finds each key
    # single-membered and collapses nothing.
    assert second.consolidated == 0
    assert len(_active(storage)) == 1


@pytest.mark.asyncio
async def test_report_surfaces_counts_and_sample() -> None:
    storage = InMemoryStorage()
    await _seed(storage, _mem(_DUP), _mem(_DUP))
    await _seed(storage, _mem("unique fact"))

    report = await MemoryDedupJob(storage, settings=Settings()).run()

    blob = "\n".join(report.notes)
    assert "collapsed 1 exact-duplicate group(s)" in blob
    assert "superseded 1 redundant cop" in blob
    assert "not deleted" in blob
    assert "global" in blob


# ---------------------------------------------------------------------------
# Wiring: exports, default-job-set exclusion, CLI subcommand
# ---------------------------------------------------------------------------


def test_job_is_exported_from_maintenance_package() -> None:
    assert MemoryDedupJobExport is MemoryDedupJob


def test_dedup_is_not_in_default_scheduled_jobs() -> None:
    # Operator-triggered like reclassify/rescope — must NOT run on every cron tick.
    jobs = build_default_jobs(
        InMemoryStorage(),
        FakeLLMProvider(),
        FakeEmbeddingProvider(),
        settings=Settings(),
    )
    assert not any(isinstance(j, MemoryDedupJob) for j in jobs)


def test_dedup_memories_subcommand_is_registered() -> None:
    names = {c.name for c in maintenance_app.registered_commands}
    assert "dedup-memories" in names


def _offline_container(storage: InMemoryStorage) -> Container:
    settings = Settings()
    settings.web.static_dir = Path("/nonexistent-spa-dir-for-tests")
    c = Container(settings=settings)
    c._storage = storage
    c._embedding = FakeEmbeddingProvider()
    c._llm = FakeLLMProvider()
    return c


def test_cli_dedup_memories_applies_in_place(monkeypatch) -> None:
    storage = InMemoryStorage()
    a = _mem(_DUP)
    b = _mem(_DUP)
    storage.memories[a.id] = a
    storage.memories[b.id] = b
    container = _offline_container(storage)
    monkeypatch.setattr(Container, "from_env", classmethod(lambda cls: container))

    result = _cli.invoke(maintenance_app, ["dedup-memories"])

    assert result.exit_code == 0, result.output
    active = [m for m in storage.memories.values() if m.is_active]
    assert len(active) == 1


def test_cli_dedup_memories_dry_run_does_not_write(monkeypatch) -> None:
    storage = InMemoryStorage()
    a = _mem(_DUP)
    b = _mem(_DUP)
    storage.memories[a.id] = a
    storage.memories[b.id] = b
    container = _offline_container(storage)
    monkeypatch.setattr(Container, "from_env", classmethod(lambda cls: container))

    result = _cli.invoke(maintenance_app, ["dedup-memories", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert "would collapse" in result.output
    # Nothing written: both copies still active.
    active = [m for m in storage.memories.values() if m.is_active]
    assert len(active) == 2
