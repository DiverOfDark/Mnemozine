"""Shared hook runtime: payload decoding, service protocols, stdout emission.

This module holds the glue every Claude Code hook shares (deliverable #5):

* :class:`HookPayload` — the decoded stdin JSON Claude Code passes a hook
  (``session_id``, ``cwd``, ``transcript_path``, ``hook_event_name``, and the
  prompt for ``UserPromptSubmit``).
* :class:`HookServices` — a **structural** bundle of the services a hook needs:
  a :class:`~mnemozine.interfaces.Retriever` (for FR-RET-3/5 injection) and an
  :class:`IngestService` (for the FR-ING-6 chunk flush). The integration pass
  constructs and injects this; the hooks never import a sibling module's concrete
  code (INTERFACES.md rule). A loader hook (``services_loader``) lets the console
  entrypoints obtain a wired bundle lazily.
* :class:`IngestService` — the minimal Protocol the ``Stop``/``PreCompact`` hooks
  call to flush a session's accumulated chunk into the pipeline. The ingestion
  service (watcher process / integration pass) implements it; defined here as a
  Protocol so the hook depends on a contract, not an implementation.
* :func:`context_from_payload` — derive a
  :class:`~mnemozine.interfaces.RetrievalContext` (project, composed scopes,
  recent-turn text) from the payload + transcript (FR-RET-3 context detection).
* :func:`emit_additional_context` — render the hook's stdout response in Claude
  Code's ``hookSpecificOutput.additionalContext`` shape, clearly delimited so the
  model treats injected memory as advisory background (FR-RET-3 format contract).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol, runtime_checkable

from mnemozine.config import Settings, get_settings
from mnemozine.ingestion.claude_code.parser import derive_project, read_transcript
from mnemozine.interfaces import InjectionIndex, RetrievalContext, Retriever
from mnemozine.schema.models import Scope

# Delimiters that mark injected memory as advisory background (FR-RET-3): the
# injection "must be clearly delimited so the model treats it as background".
INJECTION_OPEN = "<mnemozine-memory>"
INJECTION_CLOSE = "</mnemozine-memory>"

# How many recent transcript turns feed RetrievalContext.recent_text.
_RECENT_TURNS = 6


@dataclass(slots=True)
class HookPayload:
    """The decoded stdin payload Claude Code passes a hook.

    Only the fields the memory hooks use are surfaced; ``extra`` keeps the rest so
    nothing is lost. ``hook_event_name`` is ``SessionStart`` / ``UserPromptSubmit``
    / ``Stop`` / ``PreCompact``; ``prompt`` is populated for ``UserPromptSubmit``.
    """

    session_id: str | None = None
    cwd: str | None = None
    transcript_path: str | None = None
    hook_event_name: str | None = None
    prompt: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HookPayload:
        """Build a payload from the decoded JSON, tolerating field-name variants."""

        known = {
            "session_id",
            "cwd",
            "transcript_path",
            "hook_event_name",
            "prompt",
        }
        return cls(
            session_id=data.get("session_id"),
            cwd=data.get("cwd"),
            transcript_path=data.get("transcript_path"),
            hook_event_name=data.get("hook_event_name"),
            # UserPromptSubmit carries the prompt under "prompt".
            prompt=data.get("prompt") or data.get("user_prompt"),
            extra={k: v for k, v in data.items() if k not in known},
        )


def read_payload(stream: IO[str] | None = None) -> HookPayload:
    """Decode the hook payload from ``stream`` (defaults to stdin).

    A hook is invoked with its JSON payload on stdin. An empty/invalid stream
    yields an empty :class:`HookPayload` rather than raising, so a hook degrades
    gracefully (it should never break the user's session).
    """

    src = stream if stream is not None else sys.stdin
    try:
        raw = src.read()
    except Exception:  # noqa: BLE001 - a broken stdin must not crash the session
        return HookPayload()
    if not raw or not raw.strip():
        return HookPayload()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return HookPayload()
    if not isinstance(data, dict):
        return HookPayload()
    return HookPayload.from_dict(data)


@runtime_checkable
class IngestService(Protocol):
    """The flush contract the Stop/PreCompact hooks call (FR-ING-6).

    The ingestion service (the watcher process or the integration-pass wiring)
    implements this. ``flush_session`` parses the named transcript, normalizes +
    chunks it (the FR-ING-6 episode unit), and submits the resulting chunk(s) for
    extraction — de-duplicated on the FR-ING-5 content hash so flushing a session
    the watcher already tailed is a no-op. Returns the number of chunks newly
    submitted (0 if everything was already ingested).
    """

    async def flush_session(
        self, *, session_id: str, transcript_path: str | None, project: str | None
    ) -> int:
        """Flush one session's accumulated chunk(s) into the pipeline (FR-ING-6)."""
        ...


@dataclass(slots=True)
class HookServices:
    """The services a hook needs, injected by the integration pass.

    Holding both as optionals lets a single bundle serve every hook: the
    injection hooks need only ``retriever``; the flush hooks need only ``ingest``.
    ``settings`` defaults to the process-wide cached instance.
    """

    retriever: Retriever | None = None
    ingest: IngestService | None = None
    settings: Settings = field(default_factory=get_settings)


# The integration pass sets this so the console entrypoints can lazily obtain a
# wired :class:`HookServices` without these modules importing concrete wiring.
services_loader: Callable[[], HookServices] | None = None


def load_services() -> HookServices:
    """Return the wired :class:`HookServices`, via ``services_loader`` if set.

    Until the integration pass installs a loader, this returns an empty bundle
    (no retriever/ingest). The hooks treat missing services as a no-op so they are
    safe to install before the backend is wired (a hook must never break a
    session).
    """

    if services_loader is not None:
        return services_loader()
    return HookServices()


def _recent_text_from_transcript(
    transcript_path: str | None, *, strip_tool_calls: bool, limit: int = _RECENT_TURNS
) -> str | None:
    """Join the last few conversational turns into ``RetrievalContext.recent_text``."""

    if not transcript_path:
        return None
    events = read_transcript(transcript_path, strip_tool_calls=strip_tool_calls)
    if not events:
        return None
    tail = events[-limit:]
    return "\n".join(f"{e.role.value}: {e.content}" for e in tail) or None


def context_from_payload(
    payload: HookPayload, settings: Settings | None = None
) -> RetrievalContext:
    """Derive the retrieval context from a hook payload (FR-RET-3 context detection).

    Builds a :class:`RetrievalContext` whose ``project`` comes from the payload
    ``cwd`` (falling back to the transcript path), whose ``scopes`` compose the
    current project scope with the global scope (FR-RET-2: current project +
    global preferences), and whose ``recent_text`` is the tail of the transcript
    (FR-RET-3 "recent turns"). Entities are left to the retriever/cross-ref layer
    to derive from this context — the hook does not duplicate that logic.
    """

    cfg = settings or get_settings()
    project = derive_project(
        payload.transcript_path or (payload.cwd or "."),
        cwd=payload.cwd,
    )
    scopes = [Scope.global_()]
    if project:
        scopes.append(Scope.project(project))
    recent = _recent_text_from_transcript(
        payload.transcript_path,
        strip_tool_calls=cfg.ingest.strip_tool_calls,
    )
    # For UserPromptSubmit, the current prompt is the most relevant recent text.
    if payload.prompt:
        recent = f"{recent}\n{payload.prompt}" if recent else payload.prompt
    return RetrievalContext(
        project=project or None,
        scopes=scopes,
        entities=[],
        recent_text=recent,
    )


def render_injection(index: InjectionIndex) -> str:
    """Wrap the index text in the advisory-background delimiters (FR-RET-3).

    The index ``text`` is already truncated to ``inject.token_budget`` by the
    retriever; this only frames it as clearly-delimited background context so the
    model treats it as advisory, not a directive (FR-RET-3 format contract).
    """

    body = index.text.strip()
    if not body:
        return ""
    return f"{INJECTION_OPEN}\n{body}\n{INJECTION_CLOSE}"


def emit_additional_context(
    text: str, *, hook_event_name: str, stream: IO[str] | None = None
) -> None:
    """Emit injected context in Claude Code's hook-output JSON shape.

    Writes ``{"hookSpecificOutput": {"hookEventName": ...,
    "additionalContext": text}}`` to stdout, which Claude Code splices into the
    session as additional context. An empty ``text`` writes nothing (no injection
    this turn) so the hook stays silent when there is nothing to surface.
    """

    out = stream if stream is not None else sys.stdout
    if not text:
        return
    payload = {
        "hookSpecificOutput": {
            "hookEventName": hook_event_name,
            "additionalContext": text,
        }
    }
    out.write(json.dumps(payload))
    out.flush()


def session_and_project(payload: HookPayload, settings: Settings) -> tuple[str | None, str | None]:
    """Resolve ``(session_id, project)`` for the flush hooks (FR-ING-6).

    Falls back to the transcript filename stem for the session id and the
    path/cwd-derived project so a flush works even if Claude Code omits a field.
    """

    session_id = payload.session_id
    project: str | None = None
    if payload.transcript_path:
        if not session_id:
            session_id = Path(payload.transcript_path).stem
        project = derive_project(payload.transcript_path, cwd=payload.cwd)
    elif payload.cwd:
        project = derive_project(payload.cwd, cwd=payload.cwd)
    return session_id, (project or None)
