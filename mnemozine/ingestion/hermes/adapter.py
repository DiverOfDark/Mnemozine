"""Hermes ingestion adapter (FR-ING-4).

Hermes is the self-hosted Nous Research Hermes agent
(``https://hermes-agent.nousresearch.com/``) running on a homelab VM. Because it
is self-owned, the **preferred** path (PRD §6.1 / FR-ING-4, resolved OQ1) is to
*instrument the VM deployment directly* so it emits turns into the FR-ING-1
common schema, rather than scraping an external surface.

This module provides:

* :func:`events_from_hermes_turn` — the pure, testable normalization of one
  Hermes-native turn payload (a ``{"conversation_id", "messages": [...]}`` style
  dict, tolerant of the field-name variations a homelab build might use) into
  common-schema :class:`~mnemozine.schema.events.IngestEvent`s, stripping
  ``tool_calls`` per FR-ING-7.
* :class:`HermesAdapter` — an :class:`~mnemozine.interfaces.IngestSource` that
  wraps a feed of those native payloads (the instrumented VM pushes turns into an
  in-process :class:`asyncio.Queue`; recorded turns can be replayed via
  ``backfill``).
* :func:`hermes_gateway_source` — the fallback when direct instrumentation is
  impractical: a :class:`~mnemozine.ingestion.gateway.callback.GatewayCallback`
  configured with ``source=hermes`` to front Hermes' OpenAI-compatible endpoint
  through the same FR-ING-3 LiteLLM logging gateway.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from mnemozine.config import IngestSettings, get_settings
from mnemozine.ingestion.gateway.callback import GatewayCallback
from mnemozine.schema.events import IngestEvent, Role, Source
from mnemozine.schema.models import SourceSession

_TOOL_ROLE = "tool"


def _coerce_text(content: Any) -> str:
    """Flatten a Hermes message ``content`` (string or content-part list) to text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return str(content)


def _role_of(raw_role: str | None) -> Role | None:
    """Map a Hermes message role onto a common-schema :class:`Role`.

    ``tool`` -> ``None`` (FR-ING-7 strip); ``system``/``developer`` -> ``None``
    (background, not conversational memory); unknown -> ``None``.
    """

    if raw_role is None:
        return None
    role = raw_role.lower()
    if role == "user":
        return Role.USER
    if role in ("assistant", "ai", "hermes"):
        return Role.ASSISTANT
    return None


def _get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Return the first present key from a mapping (tolerant of name variants)."""

    if not isinstance(obj, Mapping):
        return default
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return default


def _parse_timestamp(value: Any, *, fallback: datetime) -> datetime:
    """Parse a Hermes timestamp (ISO-8601 str / epoch number) into aware UTC."""

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return fallback
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return fallback


def events_from_hermes_turn(
    payload: Mapping[str, Any],
    *,
    default_project: str = "hermes",
    strip_tool_calls: bool = True,
) -> list[IngestEvent]:
    """Normalize one Hermes-native turn payload into common-schema events (FR-ING-4).

    ``payload`` is the Hermes deployment's own turn/conversation object. The
    shape is taken from the instrumented VM and is intentionally tolerant of the
    common field-name variants a homelab build emits:

    * conversation id: ``conversation_id`` | ``session_id`` | ``id``
    * project:         ``project`` | ``metadata.project`` (else ``default_project``)
    * messages:        ``messages`` | ``turns`` — each with ``role`` + ``content``
      and an optional per-message ``timestamp``/``created_at``.
    * a single-turn shape (``role`` + ``content`` at the top level) is also
      accepted.

    ``tool_calls`` are stripped (FR-ING-7): ``role="tool"`` messages and any
    message carrying ``tool_calls`` keep only their text content, and the event's
    ``tool_calls`` field is never populated. Returns ``[]`` when nothing durable
    remains.
    """

    session_id = str(
        _get(payload, "conversation_id", "session_id", "id", default="hermes-unknown-session")
    )
    metadata_obj = _get(payload, "metadata", default={})
    project = (
        _get(payload, "project")
        or (metadata_obj.get("project") if isinstance(metadata_obj, Mapping) else None)
        or default_project
    )
    base_meta: dict[str, Any] = {"source_agent": "hermes"}
    model = _get(payload, "model")
    if isinstance(model, str) and model:
        base_meta["model"] = model

    raw_messages = _get(payload, "messages", "turns")
    if raw_messages is None:
        # Single-turn payload shape: treat the payload itself as one message.
        if _get(payload, "role") is not None:
            raw_messages = [payload]
        else:
            raw_messages = []

    if not isinstance(raw_messages, Iterable) or isinstance(raw_messages, (str, bytes)):
        return []

    fallback_ts = _parse_timestamp(
        _get(payload, "timestamp", "created_at", "ended_at"),
        fallback=datetime.now(UTC),
    )

    events: list[IngestEvent] = []
    for msg in raw_messages:
        if not isinstance(msg, Mapping):
            continue
        raw_role = _get(msg, "role")
        if strip_tool_calls and raw_role == _TOOL_ROLE:
            continue
        role = _role_of(raw_role)
        if role is None:
            continue
        text = _coerce_text(_get(msg, "content", "text")).strip()
        if not text:
            # e.g. a pure tool-call assistant message -> dropped (FR-ING-7).
            continue
        meta = dict(base_meta)
        if strip_tool_calls and _get(msg, "tool_calls"):
            meta["tool_calls_stripped"] = True
        ts = _parse_timestamp(
            _get(msg, "timestamp", "created_at"), fallback=fallback_ts
        )
        events.append(
            IngestEvent(
                source=Source.HERMES,
                project=str(project),
                session_id=session_id,
                timestamp=ts,
                role=role,
                content=text,
                tool_calls=None,
                metadata=meta,
            )
        )
    return events


class HermesAdapter:
    """Direct-instrumentation Hermes :class:`~mnemozine.interfaces.IngestSource` (FR-ING-4).

    The preferred FR-ING-4 path: the self-hosted Hermes VM is instrumented to
    push each completed turn (its native payload) into this adapter, which
    normalizes them into the common schema via :func:`events_from_hermes_turn`.

    Live capture: the instrumentation calls :meth:`feed` (sync) or
    :meth:`afeed` (async) per turn; :meth:`stream` yields the resulting events
    near-real-time. Backlog: :meth:`backfill` replays a provided iterable of
    recorded native payloads (e.g. a JSONL dump from the VM) so the Phase-1
    historical import works the same way (FR-ING-6); downstream de-dups on the
    FR-ING-5 idempotency key.
    """

    def __init__(
        self,
        *,
        settings: IngestSettings | None = None,
        default_project: str = "hermes",
        recorded: Iterable[Mapping[str, Any]] | None = None,
        max_queue: int = 10_000,
    ) -> None:
        self._settings = settings or get_settings().ingest
        self._default_project = default_project
        self._recorded: list[Mapping[str, Any]] = list(recorded or [])
        self._queue: asyncio.Queue[IngestEvent] = asyncio.Queue(maxsize=max_queue)
        self._pending: list[IngestEvent] = []

    @property
    def source_name(self) -> str:
        return Source.HERMES.value

    # --- mapping ---------------------------------------------------------

    def map_turn(self, payload: Mapping[str, Any]) -> list[IngestEvent]:
        """Normalize one native Hermes payload to events (FR-ING-4/7)."""

        return events_from_hermes_turn(
            payload,
            default_project=self._default_project,
            strip_tool_calls=self._settings.strip_tool_calls,
        )

    # --- live capture ----------------------------------------------------

    def feed(self, payload: Mapping[str, Any]) -> list[IngestEvent]:
        """Ingest one native turn (sync push from the instrumented VM).

        Maps + buffers the events and returns them (handy for tests). Safe to
        call with no running event loop: events land in an internal buffer that
        :meth:`stream` drains first.
        """

        events = self.map_turn(payload)
        for event in events:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                self._pending.append(event)
        return events

    async def afeed(self, payload: Mapping[str, Any]) -> list[IngestEvent]:
        """Async variant of :meth:`feed` (awaits queue space under backpressure)."""

        events = self.map_turn(payload)
        for event in events:
            await self._queue.put(event)
        return events

    # --- IngestSource ----------------------------------------------------

    async def stream(self) -> AsyncIterator[IngestEvent]:
        """Yield events as the VM pushes turns (near-real-time, FR-ING-4)."""

        while self._pending:
            yield self._pending.pop(0)
        while True:
            yield await self._queue.get()

    async def backfill(
        self, *, since: SourceSession | None = None
    ) -> AsyncIterator[IngestEvent]:
        """Replay recorded native payloads for backlog import (FR-ING-6).

        Iterates the ``recorded`` payloads supplied at construction (e.g. a JSONL
        export of historical Hermes conversations), normalizing each. If
        ``since`` is given, payloads for the same session that pre-date
        ``since.started_at`` are skipped. Safe to re-run: downstream de-dups on
        the FR-ING-5 key.
        """

        cutoff = since.started_at if since is not None else None
        for payload in self._recorded:
            events = self.map_turn(payload)
            for event in events:
                if cutoff is not None and event.timestamp < cutoff:
                    continue
                yield event


def hermes_gateway_source(
    *,
    settings: IngestSettings | None = None,
    default_project: str = "hermes",
) -> GatewayCallback:
    """Front Hermes' OpenAI-compatible endpoint via the FR-ING-3 gateway (FR-ING-4).

    The fallback when direct VM instrumentation is impractical: returns a
    :class:`~mnemozine.ingestion.gateway.callback.GatewayCallback` stamped with
    ``source=hermes``. Register it on a LiteLLM proxy whose backend points at
    Hermes' OpenAI-compatible URL (see ``gateway/config.yaml``'s Hermes variant),
    and every Hermes completion routed through that proxy is captured as a
    common-schema event with ``source=hermes`` — reusing the exact same mapping
    and ``tool_calls`` stripping as the OpenAI path, differing only in ``source``.
    """

    return GatewayCallback(
        source=Source.HERMES,
        settings=settings,
        default_project=default_project,
    )
