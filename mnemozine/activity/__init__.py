"""The append-only Activity log (WEBUI PRD §3 / Q3).

The WebUI Logs screen and Dashboard feed need a chronological record of what the
memory layer *did*: every ingestion, every 4-way write decision (FR-MNT-1), every
maintenance pass (FR-MNT-*), and every injection surfaced into a session
(FR-RET-3/5). The pipeline never had a place to record this, so Q3 adds a
lightweight, append-only :class:`ActivityEvent` log.

Design constraints that shape this module:

* **Opt-in / non-invasive.** The existing pipeline (ingest, maintenance, recall)
  and its 442 tests must be unaffected. So the log is *injected through the
  Container* and defaults to :class:`NullActivityLog` (a no-op). Pipeline call
  sites emit through the safe :func:`emit` seam, which swallows the no-op and any
  backend error — recording activity must never break a write or a recall.
* **Protocol-first.** Like every other layer, call sites code against the
  :class:`ActivityLog` Protocol (added to :mod:`mnemozine.interfaces`), never a
  concrete impl. Three impls are provided: :class:`NullActivityLog` (default),
  :class:`InMemoryActivityLog` (tests / dev), and
  :class:`FalkorDBActivityLog` (the persisted WebUI backend, FR-STO store reuse).
* **Append-only.** There is no update/delete — an activity record is history.

Public surface (imported by the Container, the web routes, and the pipeline emit
seam):

* :class:`ActivityKind`  — the four event kinds.
* :class:`ActivityEvent` — the wire/storage record.
* :class:`ActivityQuery` — query filters for the Logs screen.
* :class:`NullActivityLog` / :class:`InMemoryActivityLog` / :class:`FalkorDBActivityLog`.
* :func:`emit`            — the safe, fire-and-forget pipeline seam.
* helper constructors :func:`ingest_event`, :func:`write_decision_event`,
  :func:`maintenance_event`, :func:`injection_event` — typed builders so call
  sites do not hand-roll the record.
"""

from __future__ import annotations

from mnemozine.activity.log import (
    FalkorDBActivityLog,
    InMemoryActivityLog,
    NullActivityLog,
    emit,
)
from mnemozine.activity.models import (
    ActivityEvent,
    ActivityKind,
    ActivityQuery,
    ingest_event,
    injection_event,
    maintenance_event,
    write_decision_event,
)

__all__ = [
    "ActivityEvent",
    "ActivityKind",
    "ActivityQuery",
    "NullActivityLog",
    "InMemoryActivityLog",
    "FalkorDBActivityLog",
    "emit",
    "ingest_event",
    "write_decision_event",
    "maintenance_event",
    "injection_event",
]
