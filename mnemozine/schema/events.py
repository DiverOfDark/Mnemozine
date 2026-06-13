"""The common ingest event schema (FR-ING-1) and idempotency helpers (FR-ING-5).

Every ingestion source — Claude Code JSONL transcripts (FR-ING-2), the OpenAI
LiteLLM gateway (FR-ING-3), and Hermes (FR-ING-4) — normalizes its turns into
:class:`IngestEvent` *before* anything downstream sees them. The extraction
layer only ever consumes ``IngestEvent`` chunks, never raw source formats.

Idempotency (FR-ING-5) keys on ``(source, session_id, content-hash)`` where the
hash is computed over the *normalized content*, not the byte/line offset, so
that a resumed or rewound Claude Code session that rewrites line offsets still
de-duplicates correctly.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Source(str, Enum):
    """The originating agent surface for an event (FR-ING-1)."""

    CLAUDE_CODE = "claude_code"
    OPENAI = "openai"
    HERMES = "hermes"


class Role(str, Enum):
    """The speaker role of an event (FR-ING-1)."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class IngestEvent(BaseModel):
    """One normalized conversational turn (FR-ING-1).

    This is the single schema all sources converge on. ``tool_calls`` is carried
    on the model for fidelity at the boundary, but FR-ING-7 requires it to be
    stripped before storage; the ingestion layer is responsible for honoring
    ``IngestSettings.strip_tool_calls``.
    """

    source: Source = Field(description="Originating agent surface.")
    project: str = Field(
        description="Project identifier — derived from cwd/transcript path, or explicit (FR-ING-2)."
    )
    session_id: str = Field(description="Stable id of the originating session.")
    timestamp: datetime = Field(description="ISO-8601 timestamp of the turn.")
    role: Role = Field(description="Speaker role.")
    content: str = Field(description="Normalized message content (text).")
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional raw tool calls; stripped before storage per FR-ING-7.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific metadata (e.g. git remote, cwd, model).",
    )

    def normalized_content(self) -> str:
        """Return the content used for hashing — role-prefixed, whitespace-trimmed.

        Keeping this deterministic and offset-free is what makes FR-ING-5
        resume-safe. The role is included so that an identical string spoken by
        different roles hashes distinctly.
        """

        return f"{self.role.value}:{self.content.strip()}"

    def content_hash(self) -> str:
        """Content hash of this single event (FR-ING-5)."""

        return content_hash(self.normalized_content())

    def idempotency_key(self) -> tuple[str, str, str]:
        """The ``(source, session_id, content-hash)`` idempotency key (FR-ING-5)."""

        return idempotency_key(self.source, self.session_id, self.normalized_content())


def content_hash(content: str) -> str:
    """Return a stable hex digest of ``content`` (FR-ING-5, hash-on-content).

    SHA-256 over the UTF-8 bytes of the *content*. Callers must pass already
    normalized text (see :meth:`IngestEvent.normalized_content`) so the hash is
    invariant to byte/line offset changes from session resume/rewind.
    """

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def chunk_content_hash(events: list[IngestEvent]) -> str:
    """Return a stable content hash for an ordered chunk of events (FR-ING-5/6).

    A chunk/session is the unit of extraction (FR-ING-6); this hashes the
    concatenation of per-event normalized content so re-ingesting the same chunk
    de-duplicates regardless of line offsets.
    """

    joined = "\n".join(e.normalized_content() for e in events)
    return content_hash(joined)


def idempotency_key(
    source: Source | str, session_id: str, normalized_content: str
) -> tuple[str, str, str]:
    """Build the FR-ING-5 idempotency key ``(source, session_id, content-hash)``."""

    source_value = source.value if isinstance(source, Source) else source
    return (source_value, session_id, content_hash(normalized_content))
