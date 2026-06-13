"""LiteLLM custom logging callback -> common-schema IngestEvents (FR-ING-3).

The operator's OpenAI-format agents are repointed at a LiteLLM proxy (see the
``config.yaml`` next to this module). LiteLLM invokes a *custom logger* on every
completion; this module provides :class:`GatewayCallback`, a
:class:`litellm.integrations.custom_logger.CustomLogger` subclass that captures
each completion turn and normalizes it into FR-ING-1
:class:`~mnemozine.schema.events.IngestEvent`s.

The mapping itself is factored into the pure, dependency-free function
:func:`events_from_completion` so it can be unit-tested against mocked LiteLLM
payloads with no live proxy, no Qwen, and no network (the call signature mirrors
LiteLLM's ``log_success_event(kwargs, response_obj, start_time, end_time)``).

Design points:

* **Source is configurable** (``source=openai`` by default, ``source=hermes``
  when the same callback fronts Hermes' OpenAI-compatible endpoint, FR-ING-4).
* **``tool_calls`` are stripped** (FR-ING-7): assistant ``tool_calls`` and
  ``role="tool"`` result turns are dropped entirely; the event's ``tool_calls``
  field is never populated.
* The callback is also an :class:`~mnemozine.interfaces.IngestSource`: emitted
  events are buffered on an :class:`asyncio.Queue` and replayed by
  :meth:`GatewayCallback.stream`. ``backfill`` is a no-op generator — a live
  proxy callback has no historical backlog to replay (that path belongs to the
  Claude Code JSONL watcher), but the method exists so the Protocol is satisfied.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from mnemozine.config import IngestSettings, get_settings
from mnemozine.schema.events import IngestEvent, Role, Source
from mnemozine.schema.models import SourceSession

logger = logging.getLogger(__name__)

# Roles we never emit as memory: tool-call results are the highest-density
# source of raw credentials/command output (FR-ING-7) and carry little durable
# memory value, so they are dropped on ingest along with assistant tool_calls.
_TOOL_ROLE = "tool"

# A monotonically increasing per-turn offset is added to the response timestamp
# so that the (request) messages and the (response) assistant turn from a single
# completion keep a stable chronological order even when LiteLLM hands us only a
# single ``end_time`` for the whole call.
_MICROSECOND = 1  # microseconds, used only to break ties deterministically


def _coerce_text(content: Any) -> str:
    """Flatten an OpenAI message ``content`` into plain text.

    OpenAI/LiteLLM messages may carry ``content`` as a plain string or as a list
    of content parts (``{"type": "text", "text": ...}`` / image parts). We keep
    only text parts; non-text parts (images, audio) contribute nothing to
    durable textual memory and are skipped.
    """

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return str(content)


def _role_of(raw_role: str | None) -> Role | None:
    """Map an OpenAI message role onto a common-schema :class:`Role`.

    Returns ``None`` for roles that must not become events: ``tool`` (FR-ING-7)
    and any unrecognized role. ``system`` and ``developer`` messages are treated
    as background instructions, not conversational memory, so they are also
    dropped (returning ``None``).
    """

    if raw_role is None:
        return None
    role = raw_role.lower()
    if role == "user":
        return Role.USER
    if role in ("assistant", "ai"):
        return Role.ASSISTANT
    # tool / system / developer / function -> not emitted (FR-ING-7 + noise).
    return None


def _extract_request_messages(kwargs: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Pull the request ``messages`` list out of a LiteLLM ``kwargs`` payload."""

    messages = kwargs.get("messages")
    if messages is None:
        # LiteLLM nests the literal call params under ``optional_params`` /
        # ``litellm_params`` in some code paths; fall back defensively.
        litellm_params = kwargs.get("litellm_params") or {}
        if isinstance(litellm_params, Mapping):
            messages = litellm_params.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        return [m for m in messages if isinstance(m, Mapping)]
    return []


def _response_choices(response_obj: Any) -> list[Any]:
    """Return the list of choices from a LiteLLM/OpenAI response object.

    Tolerates both attribute-style objects (``ModelResponse``) and plain dicts.
    """

    if response_obj is None:
        return []
    choices = None
    if isinstance(response_obj, Mapping):
        choices = response_obj.get("choices")
    else:
        choices = getattr(response_obj, "choices", None)
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        return list(choices)
    return []


def _message_of_choice(choice: Any) -> Mapping[str, Any] | Any | None:
    """Return the ``message`` of a single response choice (dict or object)."""

    if isinstance(choice, Mapping):
        return choice.get("message")
    return getattr(choice, "message", None)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict-or-object uniformly."""

    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _resolve_session_id(kwargs: Mapping[str, Any]) -> str:
    """Resolve a stable session id for this completion.

    Prefers an explicit operator-supplied id passed through LiteLLM
    ``metadata`` (``mnemozine_session_id`` or a generic ``session_id`` /
    OpenAI-style ``user``), then LiteLLM's own ``litellm_call_id``. The chosen
    id only affects FR-ING-5 idempotency grouping; content-hash de-dup means a
    wrong/duplicate id can never create duplicate memories, only group turns.
    """

    metadata = kwargs.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("mnemozine_session_id", "session_id", "user"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    # OpenAI ``user`` field is sometimes passed at top level.
    user = kwargs.get("user")
    if isinstance(user, str) and user:
        return user
    call_id = kwargs.get("litellm_call_id")
    if isinstance(call_id, str) and call_id:
        return call_id
    return "gateway-unknown-session"


def _resolve_project(kwargs: Mapping[str, Any], *, default: str) -> str:
    """Resolve the ``project`` for this completion (FR-ING-1).

    Pulled from LiteLLM ``metadata`` (``mnemozine_project`` / ``project``) when
    the operator threads it through their agent; otherwise the configured
    default. Project derivation for a stateless proxy turn is necessarily
    metadata-driven — there is no cwd/git-remote like the Claude Code path.
    """

    metadata = kwargs.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("mnemozine_project", "project"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    return default


def _resolve_timestamp(end_time: Any) -> datetime:
    """Normalize the completion ``end_time`` into a timezone-aware UTC datetime."""

    if isinstance(end_time, datetime):
        return end_time if end_time.tzinfo else end_time.replace(tzinfo=UTC)
    if isinstance(end_time, (int, float)):
        return datetime.fromtimestamp(end_time, tz=UTC)
    return datetime.now(UTC)


def events_from_completion(
    kwargs: Mapping[str, Any],
    response_obj: Any,
    *,
    source: Source = Source.OPENAI,
    end_time: Any | None = None,
    default_project: str = "default",
    strip_tool_calls: bool = True,
) -> list[IngestEvent]:
    """Map one LiteLLM completion into common-schema events (FR-ING-3, FR-ING-1).

    This is the pure, side-effect-free heart of the gateway. Given the
    ``kwargs``/``response_obj`` LiteLLM passes to ``log_success_event`` (here
    accepted as plain mappings/objects so tests can mock them), it returns the
    ordered list of :class:`~mnemozine.schema.events.IngestEvent`s for the turn:
    the new request turns (typically the latest ``user`` message) followed by the
    assistant's reply.

    Behavior:

    * **Only the *new* request turns are emitted, not the whole history.** OpenAI
      chat calls resend the full prior transcript on every turn; emitting all of
      it on each completion would re-ingest every earlier turn N times. We emit
      only the trailing contiguous run of request messages after the last
      assistant message (i.e. the turns the operator added since the previous
      reply). FR-ING-5 content-hash de-dup is the safety net, but trimming here
      avoids a per-turn re-ingest storm (R3).
    * **``tool_calls`` are stripped** (FR-ING-7) when ``strip_tool_calls``:
      ``role="tool"`` request turns are dropped, assistant ``tool_calls`` are
      never carried onto the event, and an assistant turn that is *purely* a tool
      call (empty textual content) is dropped.
    * ``source`` selects ``openai`` (default) vs ``hermes`` (FR-ING-4 fronting).

    Empty-content turns are skipped. Returns ``[]`` when nothing durable remains.
    """

    project = _resolve_project(kwargs, default=default_project)
    session_id = _resolve_session_id(kwargs)
    base_ts = _resolve_timestamp(end_time)

    request_messages = _extract_request_messages(kwargs)
    new_request_messages = _trailing_request_turns(request_messages)

    model = kwargs.get("model")
    metadata_common: dict[str, Any] = {"gateway": source.value}
    if isinstance(model, str) and model:
        metadata_common["model"] = model

    events: list[IngestEvent] = []
    # Request turns first, each nudged 1us apart so ordering is stable and they
    # precede the assistant reply that shares this completion's end_time.
    seq = 0
    for msg in new_request_messages:
        role = _role_of(_get(msg, "role"))
        if role is None:
            continue
        if strip_tool_calls and _get(msg, "role") == _TOOL_ROLE:
            continue
        text = _coerce_text(_get(msg, "content")).strip()
        if not text:
            continue
        events.append(
            IngestEvent(
                source=source,
                project=project,
                session_id=session_id,
                timestamp=_offset(base_ts, seq - len(new_request_messages) - 1),
                role=role,
                content=text,
                tool_calls=None,
                metadata=dict(metadata_common),
            )
        )
        seq += 1

    # Assistant reply turn(s) from the response.
    for choice in _response_choices(response_obj):
        message = _message_of_choice(choice)
        if message is None:
            continue
        role = _role_of(_get(message, "role") or "assistant")
        if role is None:
            continue
        text = _coerce_text(_get(message, "content")).strip()
        has_tool_calls = bool(_get(message, "tool_calls"))
        if not text:
            # Pure tool-call reply (no textual content) -> dropped (FR-ING-7).
            continue
        meta = dict(metadata_common)
        if has_tool_calls and strip_tool_calls:
            # Record that tool calls were present and stripped, without keeping
            # their (credential-dense) payloads.
            meta["tool_calls_stripped"] = True
        events.append(
            IngestEvent(
                source=source,
                project=project,
                session_id=session_id,
                timestamp=base_ts,
                role=role,
                content=text,
                tool_calls=None,
                metadata=meta,
            )
        )

    return events


def _trailing_request_turns(
    messages: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return only the request turns added since the previous assistant reply.

    OpenAI chat completions resend the entire prior conversation each turn. To
    avoid re-ingesting every earlier message on every completion (R3), we keep
    only the contiguous trailing run of messages that follows the last
    ``assistant`` message in the request — i.e. the new user (and tool) turns the
    caller appended for this round.
    """

    last_assistant_idx = -1
    for i, msg in enumerate(messages):
        if _get(msg, "role") == "assistant":
            last_assistant_idx = i
    return list(messages[last_assistant_idx + 1 :])


def _offset(base: datetime, micros: int) -> datetime:
    """Return ``base`` shifted by ``micros`` microseconds (deterministic order)."""

    from datetime import timedelta

    return base + timedelta(microseconds=micros * _MICROSECOND)


class GatewayCallback:
    """LiteLLM custom logger emitting common-schema events (FR-ING-3, IngestSource).

    Register an instance with LiteLLM (``litellm.callbacks = [GatewayCallback()]``
    in the proxy, or via ``config.yaml``'s ``litellm_settings.callbacks``). On
    every successful completion LiteLLM calls :meth:`log_success_event` /
    :meth:`async_log_success_event`; both delegate to
    :func:`events_from_completion` and enqueue the resulting events.

    The instance is itself an :class:`~mnemozine.interfaces.IngestSource`: the
    ingestion service consumes emitted events with ``async for e in
    callback.stream()``. The queue decouples the (sync-or-async) LiteLLM logging
    thread from the async consumer.

    ``source`` distinguishes the FR-ING-3 OpenAI gateway (default
    ``Source.OPENAI``) from the FR-ING-4 Hermes-fronting reuse
    (``Source.HERMES``) — the only behavioral difference between the two.

    Subclassing note: the class is written to subclass LiteLLM's ``CustomLogger``
    when LiteLLM is importable, but it does **not** require LiteLLM at import time
    so it (and its tests) load offline. The mixin is resolved lazily in
    ``__init_subclass__``-free fashion via :func:`make_gateway_callback`.
    """

    def __init__(
        self,
        *,
        source: Source = Source.OPENAI,
        settings: IngestSettings | None = None,
        default_project: str = "default",
        max_queue: int = 10_000,
    ) -> None:
        self._source = source
        self._settings = settings or get_settings().ingest
        self._default_project = default_project
        self._queue: asyncio.Queue[IngestEvent] = asyncio.Queue(maxsize=max_queue)
        # Buffer used when no event loop is running at emit time (sync logging
        # path); drained into the queue lazily by :meth:`stream`.
        self._pending: list[IngestEvent] = []

    # --- IngestSource protocol ------------------------------------------

    @property
    def source_name(self) -> str:
        return self._source.value

    async def stream(self) -> AsyncIterator[IngestEvent]:
        """Yield events as completions are logged (near-real-time, FR-ING-3)."""

        # Drain anything captured before the loop existed.
        while self._pending:
            yield self._pending.pop(0)
        while True:
            event = await self._queue.get()
            yield event

    async def backfill(
        self, *, since: SourceSession | None = None
    ) -> AsyncIterator[IngestEvent]:
        """No historical backlog for a live proxy callback (FR-ING-6 N/A).

        The gateway only sees live completions; there is no on-disk transcript to
        replay (that is the Claude Code watcher's job). This is an empty async
        generator so the :class:`~mnemozine.interfaces.IngestSource` Protocol is
        satisfied and ``async for e in callback.backfill(...)`` simply yields
        nothing.
        """

        return
        yield  # pragma: no cover - makes this an async generator

    # --- event capture / mapping ----------------------------------------

    def map_events(
        self, kwargs: Mapping[str, Any], response_obj: Any, end_time: Any | None = None
    ) -> list[IngestEvent]:
        """Map one completion to events using the configured source/settings."""

        return events_from_completion(
            kwargs,
            response_obj,
            source=self._source,
            end_time=end_time,
            default_project=self._default_project,
            strip_tool_calls=self._settings.strip_tool_calls,
        )

    def _enqueue(self, events: Sequence[IngestEvent]) -> None:
        for event in events:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "gateway event queue full (source=%s); dropping oldest",
                    self._source.value,
                )
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(event)
                except asyncio.QueueEmpty:  # pragma: no cover - race only
                    self._pending.append(event)

    # --- LiteLLM CustomLogger hooks -------------------------------------

    def log_success_event(
        self, kwargs: Mapping[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """Sync LiteLLM success hook: map + enqueue (FR-ING-3)."""

        try:
            events = self.map_events(kwargs, response_obj, end_time=end_time)
        except Exception:  # never let logging break the proxy
            logger.exception("gateway callback failed to map a completion")
            return
        self._enqueue(events)

    async def async_log_success_event(
        self, kwargs: Mapping[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """Async LiteLLM success hook: map + enqueue (FR-ING-3)."""

        try:
            events = self.map_events(kwargs, response_obj, end_time=end_time)
        except Exception:
            logger.exception("gateway callback failed to map a completion")
            return
        self._enqueue(events)


def make_gateway_callback(
    *,
    source: Source = Source.OPENAI,
    settings: IngestSettings | None = None,
    default_project: str = "default",
) -> GatewayCallback:
    """Construct a :class:`GatewayCallback` wired as a LiteLLM ``CustomLogger``.

    LiteLLM accepts any object exposing the ``log_success_event`` /
    ``async_log_success_event`` hooks as a callback; subclassing
    ``CustomLogger`` is the documented path and gives forward-compatibility with
    new hook points. We resolve the base class lazily so importing this module
    (and unit-testing the pure mapping) never requires LiteLLM.

    When LiteLLM is importable this returns an instance of a dynamically-created
    subclass ``GatewayCallback(CustomLogger)``; otherwise a plain
    :class:`GatewayCallback`. The behavior of the hooks is identical either way.
    """

    try:
        from litellm.integrations.custom_logger import CustomLogger
    except Exception:  # LiteLLM not installed -> plain instance still works.
        return GatewayCallback(
            source=source, settings=settings, default_project=default_project
        )

    if isinstance(GatewayCallback(), CustomLogger):  # pragma: no cover - rare
        return GatewayCallback(
            source=source, settings=settings, default_project=default_project
        )

    bases: tuple[type, ...] = (GatewayCallback, CustomLogger)
    cls = type("GatewayCallbackLogger", bases, {})
    instance = cls(source=source, settings=settings, default_project=default_project)
    return cast(GatewayCallback, instance)
