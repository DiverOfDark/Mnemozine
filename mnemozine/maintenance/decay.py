"""FR-MNT-3 — decay & expiry: rank by recency + access, sink + archive, never delete.

Old, never-retrieved memories must *sink* (rank low) and eventually move to the
cold archive tier (FR-STO-4) so the hot retrieval path stays small and precise.
Superseded units (closed validity window) already leave the hot path via the
temporal model (FR-MNT-1); this job handles the *decay* dimension on top.

Two pieces:

* :func:`decay_score` / :func:`rank_by_decay` — pure ranking functions
  (recency half-life + access frequency), unit-testable with no I/O.
* :class:`DecayJob` — a :class:`~mnemozine.interfaces.MaintenanceJob` that sweeps
  the hot tier and **archives** (never hard-deletes) units unused longer than
  ``decay.archive_after``. Idempotent: already-archived units are skipped, so
  re-running demotes nothing twice.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import MaintenanceReport, StorageBackend
from mnemozine.schema.models import MemoryUnit, Tier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure ranking (recency half-life + access frequency)
# ---------------------------------------------------------------------------


def _age_days(reference: datetime, when: datetime | None) -> float:
    """Age in days of ``when`` relative to ``reference``.

    A ``None`` timestamp (never accessed) is treated as maximally old via a large
    finite age, so never-retrieved units sink without raising.
    """

    if when is None:
        return float("inf")
    # Normalize naive datetimes to UTC so subtraction never raises.
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    ref = reference if reference.tzinfo is not None else reference.replace(tzinfo=UTC)
    delta = ref - when
    return max(delta.total_seconds() / 86400.0, 0.0)


def decay_score(
    memory: MemoryUnit,
    *,
    now: datetime | None = None,
    half_life_days: float = 30.0,
    access_weight: float = 0.1,
) -> float:
    """Rank score for a memory by recency + access frequency (FR-MNT-3).

    Higher = more worth keeping hot. The recency term decays exponentially with
    the configured half-life off ``last_accessed`` (falling back to ``valid_from``
    for never-accessed units), and access frequency adds a saturating bonus so a
    frequently-recalled-but-older unit still ranks above a stale never-touched
    one. A never-accessed unit's recency term still decays off its creation time,
    so it sinks over time as required.
    """

    now = now or datetime.now(UTC)
    # Recency anchor: last access if any, else creation time.
    anchor = memory.last_accessed or memory.valid_from
    age = _age_days(now, anchor)
    if math.isinf(age):
        recency = 0.0
    elif half_life_days <= 0:
        recency = 1.0
    else:
        recency = math.pow(0.5, age / half_life_days)
    # Saturating access bonus: log-scaled so a few accesses matter, many saturate.
    access_bonus = access_weight * math.log1p(max(memory.access_count, 0))
    return recency + access_bonus


def rank_by_decay(
    memories: list[MemoryUnit],
    *,
    now: datetime | None = None,
    half_life_days: float = 30.0,
    access_weight: float = 0.1,
) -> list[MemoryUnit]:
    """Return ``memories`` sorted by :func:`decay_score`, highest (keep) first.

    Stable on the id as a tiebreak so the ordering is deterministic for tests.
    """

    now = now or datetime.now(UTC)
    return sorted(
        memories,
        key=lambda m: (
            decay_score(
                m, now=now, half_life_days=half_life_days, access_weight=access_weight
            ),
            m.id,
        ),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# The decay/archive sweep job
# ---------------------------------------------------------------------------


class DecayJob:
    """FR-MNT-3 decay/archive sweep — sink old never-retrieved units to archive.

    Walks the hot tier (the only enumeration entry point is
    :meth:`StorageBackend.iter_memories`; ``unused_since`` selects units whose
    ``last_accessed`` is older than the cutoff — a ``None`` ``last_accessed``
    counts as never used) and demotes each unused unit via
    :meth:`StorageBackend.archive`. Never hard-deletes (PRD FR-MNT-3 "archive,
    never hard-delete by default").

    Idempotent (FR-MNT-5): the sweep is restricted to the ``hot`` tier and skips
    anything already archived, so a re-run demotes nothing twice.
    """

    def __init__(
        self,
        storage: StorageBackend,
        *,
        settings: Settings | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._storage = storage
        self._settings = settings or get_settings()
        # Injectable clock for deterministic tests; defaults to real UTC now.
        self._now_fn: Callable[[], datetime] = now_fn or (lambda: datetime.now(UTC))

    @property
    def name(self) -> str:
        return "decay"

    async def run(self) -> MaintenanceReport:
        m = self._settings.maintenance
        now: datetime = self._now_fn()
        cutoff = now - timedelta(days=m.decay_archive_after_days)
        report = MaintenanceReport(job_name=self.name)

        # Only hot-tier units unused since the cutoff are demotion candidates.
        candidates: list[MemoryUnit] = []
        async for mem in self._storage.iter_memories(
            tier=Tier.HOT, unused_since=cutoff
        ):
            candidates.append(mem)

        for mem in candidates:
            # Defensive: skip if it raced to archive already (idempotent re-run).
            if mem.tier is Tier.ARCHIVE:
                continue
            await self._storage.archive(mem.id)
            report.archived += 1

        report.notes.append(
            f"archived {report.archived} hot unit(s) unused since "
            f"{cutoff.isoformat()} (half_life={m.decay_half_life_days}d, "
            f"archive_after={m.decay_archive_after_days}d)"
        )
        return report
