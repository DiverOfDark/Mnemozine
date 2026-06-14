"""The :class:`ActivityLog` impls + the safe :func:`emit` pipeline seam (Q3).

Three concrete logs implement the :class:`mnemozine.interfaces.ActivityLog`
Protocol:

* :class:`NullActivityLog`     — the **default**. ``append`` is a no-op and
  ``query`` is always empty, so a process that never enables the activity log
  (every existing CLI/MCP/ingest/maintenance path and all 442 tests) behaves
  exactly as before. This is the whole reason the log is opt-in.
* :class:`InMemoryActivityLog` — a list-backed log for tests and `--no-persist`
  dev: real append + filter/paging via :meth:`ActivityQuery.matches`.
* :class:`FalkorDBActivityLog` — the persisted WebUI backend. Reuses the existing
  storage connection (any object exposing ``execute_query``, i.e. the
  :class:`~mnemozine.storage.GraphitiClient` already inside the storage backend)
  so the activity log is **not a new source of truth / not a new connection** — it
  is an append-only ``(:ActivityEvent)`` node label in the same FalkorDB graph.

The :func:`emit` seam is what pipeline call sites use. It is deliberately
forgiving: it accepts ``None`` (so a site that was handed no log just returns), it
fast-paths the :class:`NullActivityLog`, and it **never propagates an error** —
recording activity must never break an ingest write or a recall. It schedules the
append on the running event loop when possible, else awaits inline, so a sync or
async call site can both fire it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from mnemozine.activity.models import ActivityEvent, ActivityKind, ActivityQuery

logger = logging.getLogger(__name__)


class NullActivityLog:
    """No-op :class:`~mnemozine.interfaces.ActivityLog` (the Container default).

    Append discards; query returns empty. Lets every non-WebUI path run with an
    activity log wired in without paying for it or changing behavior — the opt-in
    contract that keeps the existing tests green.
    """

    @property
    def enabled(self) -> bool:
        return False

    async def append(self, event: ActivityEvent) -> None:
        return None

    async def query(self, query: ActivityQuery | None = None) -> list[ActivityEvent]:
        return []

    async def close(self) -> None:
        return None


class InMemoryActivityLog:
    """List-backed activity log for tests / offline dev.

    Stores events in insertion order; :meth:`query` filters with
    :meth:`ActivityQuery.matches`, returns newest-first, and applies
    ``offset``/``limit``. Bounded by ``max_events`` (oldest dropped) so a
    long-running dev process does not grow unbounded.
    """

    def __init__(self, *, max_events: int = 10_000) -> None:
        self._events: list[ActivityEvent] = []
        self._max = max_events

    @property
    def enabled(self) -> bool:
        return True

    async def append(self, event: ActivityEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._max:
            del self._events[: len(self._events) - self._max]

    async def query(self, query: ActivityQuery | None = None) -> list[ActivityEvent]:
        q = query or ActivityQuery()
        matched = [e for e in self._events if q.matches(e)]
        matched.sort(key=lambda e: e.ts, reverse=True)
        return matched[q.offset : q.offset + q.limit]

    async def close(self) -> None:
        return None


class FalkorDBActivityLog:
    """Persisted activity log over the existing FalkorDB connection (Q3).

    Takes any object exposing ``async execute_query(cypher, **params)`` — in
    production the :class:`~mnemozine.storage.GraphitiClient` already held by the
    storage backend — so it reuses the one store/connection rather than opening a
    new one. Events are append-only ``(:ActivityEvent {...})`` nodes; reads filter
    in Cypher and page newest-first.

    The record's ``ref_memory_ids`` / ``detail`` are stored as JSON strings (a
    FalkorDB property is a scalar/array of scalars, not a nested map), and rebuilt
    on read.
    """

    LABEL = "ActivityEvent"

    def __init__(self, client: Any) -> None:
        self._client = client

    @property
    def enabled(self) -> bool:
        return True

    @staticmethod
    def _rows(result: Any) -> list[list[Any]]:
        """Normalize a driver result to row-lists (mirrors the storage backend)."""

        if result is None:
            return []
        header: list[str] | None = None
        if isinstance(result, tuple):
            records = result[0]
            if len(result) > 1 and result[1]:
                header = [str(h) for h in result[1]]
        else:
            records = getattr(result, "result_set", result)
        if not records:
            return []
        out: list[list[Any]] = []
        for r in records:
            if isinstance(r, dict):
                keys = header if header is not None else list(r.keys())
                out.append([r.get(k) for k in keys])
            else:
                out.append(list(r))
        return out

    @staticmethod
    def _props(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        props = getattr(value, "properties", None)
        if isinstance(props, dict):
            return dict(props)
        return dict(value)

    def _to_props(self, event: ActivityEvent) -> dict[str, Any]:
        return {
            "id": event.id,
            "kind": event.kind.value,
            "source": event.source,
            "summary": event.summary,
            "ref_memory_ids": json.dumps(event.ref_memory_ids),
            "session_id": event.session_id,
            "project": event.project,
            "detail": json.dumps(event.detail),
            "ts": event.ts.isoformat(),
        }

    def _from_props(self, props: dict[str, Any]) -> ActivityEvent:
        def _loads(value: Any, default: Any) -> Any:
            if value is None:
                return default
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (ValueError, TypeError):
                    return default
            return value

        ts = props.get("ts")
        return ActivityEvent(
            id=props["id"],
            kind=ActivityKind(props["kind"]),
            source=props.get("source"),
            summary=props.get("summary", ""),
            ref_memory_ids=list(_loads(props.get("ref_memory_ids"), [])),
            session_id=props.get("session_id"),
            project=props.get("project"),
            detail=dict(_loads(props.get("detail"), {})),
            ts=datetime.fromisoformat(ts) if isinstance(ts, str) else ts,
        )

    async def append(self, event: ActivityEvent) -> None:
        # FalkorDB does not support ``CREATE (a:Label $props)`` with a
        # *parameterized map* for inline properties ("Encountered unhandled type
        # in inlined properties"); each property must be set individually. We
        # CREATE the bare node then SET each scalar prop from a named parameter
        # (NULL-valued params are accepted by SET, unlike inline map-create).
        props = self._to_props(event)
        assignments = ", ".join(f"a.{key} = ${key}" for key in props)
        cypher = f"CREATE (a:{self.LABEL}) SET {assignments} RETURN a.id"
        await self._client.execute_query(cypher, **props)

    async def query(self, query: ActivityQuery | None = None) -> list[ActivityEvent]:
        q = query or ActivityQuery()
        where: list[str] = []
        params: dict[str, Any] = {}
        if q.kinds is not None:
            where.append("a.kind IN $kinds")
            params["kinds"] = [k.value for k in q.kinds]
        if q.source is not None:
            where.append("a.source = $source")
            params["source"] = q.source
        if q.session_id is not None:
            where.append("a.session_id = $session_id")
            params["session_id"] = q.session_id
        if q.project is not None:
            where.append("a.project = $project")
            params["project"] = q.project
        if q.since is not None:
            where.append("a.ts >= $since")
            params["since"] = q.since.isoformat()
        if q.until is not None:
            where.append("a.ts < $until")
            params["until"] = q.until.isoformat()
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        cypher = (
            f"MATCH (a:{self.LABEL}) {clause} "
            f"RETURN a ORDER BY a.ts DESC SKIP {int(q.offset)} LIMIT {int(q.limit)}"
        )
        rows = self._rows(await self._client.execute_query(cypher, **params))
        events = [self._from_props(self._props(row[0])) for row in rows if row and row[0]]
        # ref_memory_id is filtered in Python: a JSON-string array can't be
        # matched in portable Cypher without unpacking it.
        if q.ref_memory_id is not None:
            events = [e for e in events if q.ref_memory_id in e.ref_memory_ids]
        return events

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# The safe pipeline emit seam.
# ---------------------------------------------------------------------------


async def _safe_append(log: Any, event: ActivityEvent) -> None:
    try:
        await log.append(event)
    except Exception:  # noqa: BLE001 - recording activity must never break a write
        logger.debug("activity append failed (ignored)", exc_info=True)


def emit(log: Any | None, event: ActivityEvent) -> None:
    """Fire-and-forget append of one activity event from a pipeline seam (Q3).

    Safe and non-invasive by construction:

    * ``log is None`` or a :class:`NullActivityLog` -> returns immediately (the
      default path; existing call sites that were handed nothing pay nothing).
    * Otherwise schedules ``log.append(event)`` on the running event loop as a
      background task when one is running (the common case — pipeline seams are
      inside ``async`` code), or runs it to completion via ``asyncio.run`` when
      called from sync code with no loop.
    * Any error is swallowed: recording activity must never break the write/recall
      it is observing.
    """

    if log is None or getattr(log, "enabled", False) is False:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        loop.create_task(_safe_append(log, event))
        return
    try:
        asyncio.run(_safe_append(log, event))
    except Exception:  # noqa: BLE001 - never let the seam raise
        logger.debug("activity emit failed (ignored)", exc_info=True)


async def emit_async(log: Any | None, event: ActivityEvent) -> None:
    """Awaitable variant of :func:`emit` for call sites that want to await it.

    Same null/error safety as :func:`emit` but awaits the append inline so the
    record is durable before the caller proceeds (used where ordering matters,
    e.g. a test asserting the event landed). Still swallows backend errors.
    """

    if log is None or getattr(log, "enabled", False) is False:
        return
    await _safe_append(log, event)


def build_activity_log(
    *,
    enable: bool,
    client: Any | None = None,
) -> Any:
    """Construct the activity log the Container should hold (Q3 wiring helper).

    Returns a :class:`NullActivityLog` when ``enable`` is false (the default) so
    the pipeline is unaffected; a :class:`FalkorDBActivityLog` over ``client``
    when persistence is on and a connection is available; or an
    :class:`InMemoryActivityLog` when enabled with no client (offline dev/tests).
    """

    if not enable:
        return NullActivityLog()
    if client is not None:
        return FalkorDBActivityLog(client)
    return InMemoryActivityLog()


__all__ = [
    "NullActivityLog",
    "InMemoryActivityLog",
    "FalkorDBActivityLog",
    "emit",
    "emit_async",
    "build_activity_log",
]
