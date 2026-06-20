"""FR-MNT-3 decay tests — recency+access ranking and the archive sweep.

Pure ranking is tested deterministically with a fixed ``now``; the sweep is
tested against the conftest ``InMemoryStorage`` and asserts archive (never
hard-delete) + idempotent re-run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mnemozine.config import Settings
from mnemozine.maintenance.decay import DecayJob, decay_score, rank_by_decay
from mnemozine.schema.models import MemoryUnit, Provenance, Scope, Tier
from tests.conftest import InMemoryStorage

NOW = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)


def _mem(
    *,
    content: str,
    last_accessed: datetime | None,
    access_count: int = 0,
    valid_from: datetime | None = None,
    tier: Tier = Tier.HOT,
) -> MemoryUnit:
    return MemoryUnit(
        content=content,
        scope=Scope.global_(),
        category="preference",
        entities=["rust"],
        confidence=0.9,
        provenance=Provenance(source="claude_code", session_id="s1"),
        valid_from=valid_from or NOW,
        last_accessed=last_accessed,
        access_count=access_count,
        tier=tier,
    )


# --- pure ranking ---------------------------------------------------------


def test_recent_outranks_stale() -> None:
    recent = _mem(content="recent", last_accessed=NOW - timedelta(days=1))
    stale = _mem(content="stale", last_accessed=NOW - timedelta(days=200))
    assert decay_score(recent, now=NOW, half_life_days=30) > decay_score(
        stale, now=NOW, half_life_days=30
    )


def test_access_frequency_breaks_recency_tie() -> None:
    when = NOW - timedelta(days=10)
    hot = _mem(content="hot", last_accessed=when, access_count=50)
    cold = _mem(content="cold", last_accessed=when, access_count=0)
    assert decay_score(hot, now=NOW) > decay_score(cold, now=NOW)


def test_never_accessed_sinks_below_recently_accessed() -> None:
    # Never-accessed but recently created vs recently accessed: the accessed one
    # wins; both still decay off their anchor.
    never = _mem(
        content="never",
        last_accessed=None,
        valid_from=NOW - timedelta(days=120),
    )
    accessed = _mem(content="accessed", last_accessed=NOW - timedelta(days=2))
    assert decay_score(accessed, now=NOW, half_life_days=30) > decay_score(
        never, now=NOW, half_life_days=30
    )


def test_rank_by_decay_orders_keep_first_and_is_deterministic() -> None:
    a = _mem(content="a", last_accessed=NOW - timedelta(days=1))
    b = _mem(content="b", last_accessed=NOW - timedelta(days=100))
    c = _mem(content="c", last_accessed=NOW - timedelta(days=50))
    ranked = rank_by_decay([b, c, a], now=NOW, half_life_days=30)
    assert [m.content for m in ranked] == ["a", "c", "b"]
    # Deterministic across calls (stable tiebreak on id).
    assert rank_by_decay([a, b, c], now=NOW) == rank_by_decay([c, b, a], now=NOW)


# --- the archive sweep ----------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_archives_old_unused_hot_units() -> None:
    settings = Settings()
    settings.maintenance.decay_archive_after_days = 90
    storage = InMemoryStorage()

    old_unused = _mem(content="old", last_accessed=NOW - timedelta(days=200))
    recent = _mem(content="recent", last_accessed=NOW - timedelta(days=5))
    never = _mem(content="never", last_accessed=None, valid_from=NOW - timedelta(days=200))
    for m in (old_unused, recent, never):
        await storage.upsert_memory(m)

    job = DecayJob(storage, settings=settings, now_fn=lambda: NOW)
    report = await job.run()

    # old_unused and never -> archived; recent stays hot. Never hard-deleted.
    assert old_unused.tier is Tier.ARCHIVE
    assert never.tier is Tier.ARCHIVE
    assert recent.tier is Tier.HOT
    assert report.archived == 2
    assert all(m.id in storage.memories for m in (old_unused, recent, never))


@pytest.mark.asyncio
async def test_fresh_never_recalled_hot_unit_not_selected_or_archived() -> None:
    """A freshly-ingested, NEVER-recalled HOT unit must NOT be swept (regression).

    The decay sweep selects via ``iter_memories(tier=HOT, unused_since=cutoff)``.
    BEFORE the fix, ``unused_since`` treated a null ``last_accessed`` as infinitely
    unused, so a fresh memory (``valid_from=now``, never recalled) was archived on
    the first maintenance run after a big ingest — starving hot-only recall. The
    fix anchors a null ``last_accessed`` on ``valid_from`` (ingestion time), matching
    ``decay_score``'s recency anchor, so a fresh unit is kept until its creation time
    itself ages past the cutoff. This test fails on the OLD clause, passes on the new.
    """

    settings = Settings()
    settings.maintenance.decay_archive_after_days = 90
    storage = InMemoryStorage()

    # Fresh: valid_from = now, never recalled (last_accessed is None).
    fresh = _mem(content="fresh", last_accessed=None, valid_from=NOW)
    await storage.upsert_memory(fresh)

    cutoff = NOW - timedelta(days=settings.maintenance.decay_archive_after_days)
    # iter_memories must NOT select the fresh, never-recalled unit.
    selected = [
        m.id
        async for m in storage.iter_memories(tier=Tier.HOT, unused_since=cutoff)
    ]
    assert fresh.id not in selected

    job = DecayJob(storage, settings=settings, now_fn=lambda: NOW)
    report = await job.run()

    # The fresh unit stays hot; nothing is archived (never hard-deleted either).
    assert fresh.tier is Tier.HOT
    assert report.archived == 0
    assert fresh.id in storage.memories


@pytest.mark.asyncio
async def test_old_never_recalled_hot_unit_still_archived() -> None:
    """A genuinely old, never-recalled HOT unit IS still swept (real-stale path).

    The fix anchors a null ``last_accessed`` on ``valid_from``; a unit ingested
    long before the cutoff (and never recalled) still ages past it, so the decay
    sweep keeps working for genuinely stale memories.
    """

    settings = Settings()
    settings.maintenance.decay_archive_after_days = 90
    storage = InMemoryStorage()

    old = _mem(
        content="old", last_accessed=None, valid_from=NOW - timedelta(days=200)
    )
    await storage.upsert_memory(old)

    cutoff = NOW - timedelta(days=settings.maintenance.decay_archive_after_days)
    selected = [
        m.id
        async for m in storage.iter_memories(tier=Tier.HOT, unused_since=cutoff)
    ]
    assert old.id in selected

    job = DecayJob(storage, settings=settings, now_fn=lambda: NOW)
    report = await job.run()

    assert old.tier is Tier.ARCHIVE
    assert report.archived == 1


@pytest.mark.asyncio
async def test_recently_recalled_old_unit_not_archived() -> None:
    """A recently-recalled unit is NOT archived even if ``valid_from`` is old.

    The anchor prefers ``last_accessed`` when present, so a recent recall protects
    an old-ingested unit from the sweep (existing behaviour, pinned here).
    """

    settings = Settings()
    settings.maintenance.decay_archive_after_days = 90
    storage = InMemoryStorage()

    recalled = _mem(
        content="recalled",
        last_accessed=NOW - timedelta(days=2),
        valid_from=NOW - timedelta(days=300),
    )
    await storage.upsert_memory(recalled)

    cutoff = NOW - timedelta(days=settings.maintenance.decay_archive_after_days)
    selected = [
        m.id
        async for m in storage.iter_memories(tier=Tier.HOT, unused_since=cutoff)
    ]
    assert recalled.id not in selected

    job = DecayJob(storage, settings=settings, now_fn=lambda: NOW)
    report = await job.run()

    assert recalled.tier is Tier.HOT
    assert report.archived == 0


@pytest.mark.asyncio
async def test_sweep_is_idempotent() -> None:
    settings = Settings()
    settings.maintenance.decay_archive_after_days = 90
    storage = InMemoryStorage()
    old = _mem(content="old", last_accessed=NOW - timedelta(days=200))
    await storage.upsert_memory(old)

    job = DecayJob(storage, settings=settings, now_fn=lambda: NOW)
    first = await job.run()
    second = await job.run()

    assert first.archived == 1
    # Re-run demotes nothing twice (already archived, off the hot-tier scan).
    assert second.archived == 0
    assert old.tier is Tier.ARCHIVE
