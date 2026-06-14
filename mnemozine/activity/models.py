"""The :class:`ActivityEvent` record + query filter + typed builders (Q3).

These pydantic models are the durable shape of one row in the append-only
activity log and the filter used to read it back. They are deliberately storage
agnostic (an :class:`~mnemozine.activity.log.ActivityLog` impl maps them onto
FalkorDB nodes, an in-memory list, or nothing) and JSON-serializable so the same
record is both stored and returned over the WebUI wire.

An :class:`ActivityEvent` answers the Logs-screen questions: *what kind* of thing
happened (:class:`ActivityKind`), *which source/session* it came from, a
one-line human ``summary``, *which memories* it touched (so the UI can link to
them), and *when*. ``detail`` carries kind-specific structured extras (e.g. the
write ``decision``, the maintenance counts) without bloating the core schema.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid4().hex


class ActivityKind(str, Enum):
    """The four kinds of activity the pipeline records (Q3, PRD WEBUI §4.6).

    * ``ingest``          — a source chunk/session was ingested (FR-ING-*).
    * ``extract_decision`` — the FR-MNT-1 4-way write decision fired for a unit
      (add / reinforce / supersede / no-op).
    * ``maintenance``     — a maintenance pass ran (FR-MNT-* consolidate / decay /
      entity-resolution / migrate-index).
    * ``injection``       — a memory index/recall was surfaced into a session
      (FR-RET-3/5).
    """

    INGEST = "ingest"
    EXTRACT_DECISION = "extract_decision"
    MAINTENANCE = "maintenance"
    INJECTION = "injection"


class ActivityEvent(BaseModel):
    """One append-only activity record (Q3).

    Stored verbatim by the persisted log and returned verbatim over the WebUI
    wire. ``ts`` is the event time (defaults to now), ``ref_memory_ids`` lets the
    Logs screen link an entry to the memories it affected, and ``detail`` carries
    kind-specific structured extras.
    """

    id: str = Field(default_factory=_new_id, description="Stable event id.")
    kind: ActivityKind = Field(description="The kind of activity (Q3).")
    source: str | None = Field(
        default=None,
        description="Originating source/subsystem, e.g. 'claude_code' or 'maintenance'.",
    )
    summary: str = Field(description="One-line human-readable summary for the feed.")
    ref_memory_ids: list[str] = Field(
        default_factory=list,
        description="Ids of memory units this event affected (UI links to them).",
    )
    session_id: str | None = Field(
        default=None,
        description="Originating session id when applicable (ingest/injection).",
    )
    project: str | None = Field(
        default=None,
        description="Project/scope context for the event when applicable.",
    )
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description="Kind-specific structured extras (decision, counts, scores).",
    )
    ts: datetime = Field(default_factory=_utcnow, description="Event timestamp (UTC).")


class ActivityQuery(BaseModel):
    """Filters for reading the activity log back (the Logs screen, PRD §4.6).

    All filters are optional and AND-combined. ``kinds`` restricts to a subset of
    event kinds; ``source`` / ``session_id`` / ``project`` narrow by origin;
    ``since`` / ``until`` bound the time window; ``ref_memory_id`` selects events
    that touched a given memory (the memory-detail "related activity" view).
    ``limit`` / ``offset`` page the result, newest first.
    """

    kinds: list[ActivityKind] | None = Field(
        default=None, description="Restrict to these event kinds (None = all)."
    )
    source: str | None = Field(default=None, description="Filter by originating source.")
    session_id: str | None = Field(default=None, description="Filter by session id.")
    project: str | None = Field(default=None, description="Filter by project/scope.")
    ref_memory_id: str | None = Field(
        default=None, description="Only events that touched this memory id."
    )
    since: datetime | None = Field(default=None, description="Inclusive lower time bound.")
    until: datetime | None = Field(default=None, description="Exclusive upper time bound.")
    limit: int = Field(default=100, ge=1, le=1000, description="Max events to return.")
    offset: int = Field(default=0, ge=0, description="Result offset for pagination.")

    def matches(self, event: ActivityEvent) -> bool:
        """True if ``event`` passes every set filter (used by the in-memory impl)."""

        if self.kinds is not None and event.kind not in self.kinds:
            return False
        if self.source is not None and event.source != self.source:
            return False
        if self.session_id is not None and event.session_id != self.session_id:
            return False
        if self.project is not None and event.project != self.project:
            return False
        if self.ref_memory_id is not None and self.ref_memory_id not in event.ref_memory_ids:
            return False
        if self.since is not None and event.ts < self.since:
            return False
        if self.until is not None and event.ts >= self.until:
            return False
        return True


# ---------------------------------------------------------------------------
# Typed builders — call sites use these rather than hand-rolling the record.
# ---------------------------------------------------------------------------


def ingest_event(
    *,
    source: str,
    session_id: str | None,
    project: str | None,
    summary: str,
    ref_memory_ids: Sequence[str] = (),
    detail: dict[str, Any] | None = None,
) -> ActivityEvent:
    """Build an ``ingest`` activity event (FR-ING-* seam)."""

    return ActivityEvent(
        kind=ActivityKind.INGEST,
        source=source,
        session_id=session_id,
        project=project,
        summary=summary,
        ref_memory_ids=list(ref_memory_ids),
        detail=detail or {},
    )


def write_decision_event(
    *,
    decision: str,
    memory_id: str,
    source: str | None = None,
    summary: str | None = None,
    superseded_ids: Sequence[str] = (),
    detail: dict[str, Any] | None = None,
) -> ActivityEvent:
    """Build an ``extract_decision`` event for one FR-MNT-1 4-way write.

    ``decision`` is the :class:`~mnemozine.interfaces.WriteDecision` value
    (add/reinforce/supersede/no-op); ``superseded_ids`` are the units whose
    validity windows the write closed, included in ``ref_memory_ids`` so the Logs
    screen links them all.
    """

    extra = dict(detail or {})
    extra.setdefault("decision", decision)
    if superseded_ids:
        extra.setdefault("superseded_ids", list(superseded_ids))
    refs = [memory_id, *[s for s in superseded_ids if s != memory_id]]
    return ActivityEvent(
        kind=ActivityKind.EXTRACT_DECISION,
        source=source,
        summary=summary or f"write decision: {decision}",
        ref_memory_ids=refs,
        detail=extra,
    )


def maintenance_event(
    *,
    job_name: str,
    summary: str | None = None,
    detail: dict[str, Any] | None = None,
) -> ActivityEvent:
    """Build a ``maintenance`` event for one maintenance pass (FR-MNT-*)."""

    extra = dict(detail or {})
    extra.setdefault("job_name", job_name)
    return ActivityEvent(
        kind=ActivityKind.MAINTENANCE,
        source="maintenance",
        summary=summary or f"maintenance job: {job_name}",
        detail=extra,
    )


def injection_event(
    *,
    session_id: str | None,
    project: str | None,
    summary: str,
    ref_memory_ids: Sequence[str] = (),
    detail: dict[str, Any] | None = None,
) -> ActivityEvent:
    """Build an ``injection`` event for a surfaced index/recall (FR-RET-3/5)."""

    return ActivityEvent(
        kind=ActivityKind.INJECTION,
        source="retrieval",
        session_id=session_id,
        project=project,
        summary=summary,
        ref_memory_ids=list(ref_memory_ids),
        detail=detail or {},
    )


__all__ = [
    "ActivityKind",
    "ActivityEvent",
    "ActivityQuery",
    "ingest_event",
    "write_decision_event",
    "maintenance_event",
    "injection_event",
]
