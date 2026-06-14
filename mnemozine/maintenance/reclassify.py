"""Offline re-extract / reclassify passes — apply model/prompt changes after the fact.

Two offline maintenance passes that re-apply the *current* extractor/classifier
to memories that were written by an *older* one, without needing the original raw
transcript (which Claude cleans up after 30 days, R4):

* :class:`ReExtractJob` — re-runs the **whole extractor** over the retained raw
  tier (:class:`~mnemozine.schema.models.RawChunk`). It delegates to
  :meth:`StorageBackend.re_extract_from_raw_chunks`, which iterates the stored
  raw chunks, feeds each chunk's normalized ``content`` back through the current
  :class:`~mnemozine.interfaces.Extractor`, upserts the fresh units through the
  FR-MNT-1 write path, and (when ``supersede_existing``) closes the validity
  windows of the memories the chunk previously produced. This is how a model /
  prompt change is applied offline to already-ingested data.

* :class:`ReclassifyJob` — a lighter pass that **re-scopes + re-categorizes
  existing memories from their stored content + provenance** (no raw text). It
  iterates the hot tier, asks the current classifier
  (:meth:`Extractor.classify`) for a fresh scope / category / cross-ref decision
  per unit, and writes only the *changed* tags through
  :meth:`StorageBackend.reclassify_memory`. Because it reads the already-stored
  ``content``, it works long after the raw transcript is gone, and it never
  re-embeds or duplicates a node — it only re-tags.

Both are :class:`~mnemozine.interfaces.MaintenanceJob`s (``name`` / ``run``),
idempotent (FR-MNT-5): once a memory's tags match the current classifier a
re-run leaves it unchanged, and an unchanged extractor re-produces equivalent
units that reinforce rather than duplicate. They depend only on the
:class:`~mnemozine.interfaces.StorageBackend` and
:class:`~mnemozine.interfaces.Extractor` Protocols, so they are unit-testable
offline against the conftest fakes.
"""

from __future__ import annotations

import logging

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import (
    Extractor,
    MaintenanceReport,
    RetrievalContext,
    StorageBackend,
)
from mnemozine.schema.models import MemoryUnit, Scope

logger = logging.getLogger(__name__)


class ReExtractJob:
    """Re-run the current extractor over the retained raw tier (offline reindex).

    A thin :class:`~mnemozine.interfaces.MaintenanceJob` wrapper over
    :meth:`StorageBackend.re_extract_from_raw_chunks`: that storage seam owns the
    re-extraction loop (iterate raw chunks -> extract -> upsert -> supersede the
    prior memories) so a backend can re-parse ``RawChunk.content`` into events the
    way it originally did. The job exposes it as a scheduled/operator pass and
    optionally narrows the sweep to one ``scope`` / ``session_id`` so a single
    project can be reprocessed in isolation (the storage filter is EXACT — a
    re-extraction must not widen across scopes).

    Idempotent (FR-MNT-5): an unchanged extractor re-produces equivalent units
    that reinforce rather than duplicate.
    """

    def __init__(
        self,
        storage: StorageBackend,
        extractor: Extractor,
        *,
        scope: Scope | None = None,
        session_id: str | None = None,
        supersede_existing: bool = True,
        settings: Settings | None = None,
    ) -> None:
        self._storage = storage
        self._extractor = extractor
        self._scope = scope
        self._session_id = session_id
        self._supersede_existing = supersede_existing
        self._settings = settings or get_settings()

    @property
    def name(self) -> str:
        return "re_extract"

    async def run(self) -> MaintenanceReport:
        """Re-extract over the (optionally scope/session-filtered) raw tier."""

        report = await self._storage.re_extract_from_raw_chunks(
            self._extractor,
            scope=self._scope,
            session_id=self._session_id,
            supersede_existing=self._supersede_existing,
        )
        # Normalize the report's job_name to this job (the backend stamps its own)
        # and surface the filters for the audit log.
        report.job_name = self.name
        filters = []
        if self._scope is not None:
            filters.append(f"scope={self._scope.as_str()}")
        if self._session_id is not None:
            filters.append(f"session_id={self._session_id}")
        report.notes.append(
            f"re-extracted {report.re_extracted} raw chunk(s)"
            + (f" ({', '.join(filters)})" if filters else "")
            + f"; supersede_existing={self._supersede_existing}"
        )
        return report


class ReclassifyJob:
    """Re-scope + re-categorize existing memories from stored content (R1, no raw text).

    Iterates the hot tier and, for each active memory, asks the current
    classifier (:meth:`Extractor.classify`) for a fresh scope / category /
    cross-ref decision over the memory's already-stored ``content``. Only the
    fields that actually changed are written, through
    :meth:`StorageBackend.reclassify_memory` (a re-tag, not a new node, and no raw
    transcript needed). This is the offline path for applying a *classifier*
    change (prompt/model) to historical memories that have outlived their raw
    transcript (R4).

    A re-scope must still obey the hierarchical no-leak rule; the classifier
    returns a :class:`~mnemozine.schema.models.Scope`, and the storage
    :meth:`reclassify_memory` is responsible for persisting it as the unit's new
    stored scope. Idempotent (FR-MNT-5): a memory whose stored tags already match
    the classifier is left untouched, so a re-run is a no-op.
    """

    def __init__(
        self,
        storage: StorageBackend,
        extractor: Extractor,
        *,
        scope: Scope | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._storage = storage
        self._extractor = extractor
        # Optional scope filter so a single project can be reclassified in
        # isolation (exact scope, no ancestor composition).
        self._scope = scope
        self._settings = settings or get_settings()

    @property
    def name(self) -> str:
        return "reclassify"

    async def run(self) -> MaintenanceReport:
        """Reclassify every active hot memory whose tags drifted from the classifier."""

        report = MaintenanceReport(job_name=self.name)
        reclassified = 0
        scanned = 0
        async for memory in self._storage.iter_memories(
            scope=self._scope, active_only=True
        ):
            scanned += 1
            changed = await self._reclassify_one(memory)
            if changed:
                reclassified += 1
        report.notes.append(
            f"reclassified {reclassified}/{scanned} active memor(ies) "
            "from stored content (no raw text)"
        )
        # Reuse the consolidated counter so the runner's summary line surfaces it
        # (the report has no dedicated reclassified field — see integration note).
        report.consolidated = reclassified
        return report

    async def _reclassify_one(self, memory: MemoryUnit) -> bool:
        """Reclassify a single memory; return True iff any tag was changed.

        Builds the minimal :class:`RetrievalContext` from the memory's stored
        scope/entities (so the classifier sees the same working context it would
        at retrieval), runs :meth:`Extractor.classify` over the stored content,
        and writes only the fields that differ. A classifier failure leaves the
        memory untouched (a bad classify must never corrupt a stored tag).
        """

        context = RetrievalContext(
            project=memory.scope.project_id,
            scopes=memory.scope.ancestors(),
            entities=list(memory.entities),
            recent_text=None,
        )
        try:
            decision = await self._extractor.classify(memory.content, context)
        except Exception:  # noqa: BLE001 - a flaky classifier must not corrupt tags
            logger.warning(
                "reclassify: classify failed for memory %s; leaving it unchanged",
                memory.id,
                exc_info=True,
            )
            return False

        # Compute only the deltas so an unchanged classify is a true no-op
        # (idempotency, FR-MNT-5). Category is normalized both sides for the
        # comparison to match the stored (already-normalized) value.
        new_scope: Scope | None = (
            decision.scope if decision.scope.as_str() != memory.scope.as_str() else None
        )
        new_category: str | None = (
            decision.category
            if decision.category.strip().lower() != memory.category
            else None
        )
        new_cross_ref: bool | None = (
            decision.cross_ref_candidate
            if decision.cross_ref_candidate != memory.cross_ref_candidate
            else None
        )
        if new_scope is None and new_category is None and new_cross_ref is None:
            return False

        await self._storage.reclassify_memory(
            memory.id,
            scope=new_scope,
            category=new_category,
            cross_ref_candidate=new_cross_ref,
        )
        return True
