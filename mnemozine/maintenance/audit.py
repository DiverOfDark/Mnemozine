"""R5 audit — a read-only integrity walk over the whole store.

Memory drift / poisoning (R5) accumulates below the threshold where any single
entry looks wrong, so the maintenance job includes a periodic audit. This pass is
**read-only** (it mutates nothing, hence trivially idempotent and safe to
re-run, FR-MNT-5) and records counts + anomalies into a
:class:`~mnemozine.interfaces.MaintenanceReport` for the operator/log:

* total / active / archived / superseded unit counts,
* units with the classify-sentinel provenance still attached (a provenance gap —
  every persisted unit should carry a real source, FR-EXT-4),
* units with zero linked entities (un-traversable, FR-EXT-2),
* low-confidence units below an audit floor (candidate poisoning, R5).
"""

from __future__ import annotations

import logging

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import MaintenanceReport, StorageBackend
from mnemozine.schema.models import Tier

logger = logging.getLogger(__name__)


class AuditJob:
    """R5 audit walk. Read-only; depends only on the storage Protocol."""

    def __init__(
        self,
        storage: StorageBackend,
        *,
        settings: Settings | None = None,
        confidence_floor: float = 0.2,
    ) -> None:
        self._storage = storage
        self._settings = settings or get_settings()
        self._confidence_floor = confidence_floor

    @property
    def name(self) -> str:
        return "audit"

    async def run(self) -> MaintenanceReport:
        report = MaintenanceReport(job_name=self.name)
        total = active = archived = superseded = 0
        no_provenance = no_entities = low_confidence = 0

        async for mem in self._storage.iter_memories():
            total += 1
            if mem.is_active:
                active += 1
            else:
                superseded += 1
            if mem.tier is Tier.ARCHIVE:
                archived += 1
            if mem.provenance.is_classify_sentinel:
                no_provenance += 1
            if not mem.entities:
                no_entities += 1
            if mem.confidence < self._confidence_floor:
                low_confidence += 1

        report.notes.append(
            f"audit: total={total} active={active} archived={archived} "
            f"superseded={superseded}"
        )
        if no_provenance:
            report.notes.append(
                f"WARN {no_provenance} unit(s) carry the classify-sentinel "
                f"provenance (missing real source, FR-EXT-4)"
            )
        if no_entities:
            report.notes.append(
                f"WARN {no_entities} unit(s) have no linked entities (FR-EXT-2)"
            )
        if low_confidence:
            report.notes.append(
                f"WARN {low_confidence} unit(s) below confidence floor "
                f"{self._confidence_floor} (possible drift, R5)"
            )
        return report
