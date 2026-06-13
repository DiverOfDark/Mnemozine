"""Claude Code ingestion (FR-ING-2/5/6/7).

Claude Code records each session as an append-only JSONL transcript at
``$CLAUDE_CONFIG_DIR/projects/<project>/<session-id>.jsonl`` (one JSON object per
line). This subpackage turns those transcripts into common-schema
:class:`~mnemozine.schema.events.IngestEvent`s and groups them into
Graphiti-episode-sized chunks for extraction.

Components
----------
* :mod:`~mnemozine.ingestion.claude_code.parser` — parse a single JSONL line /
  whole transcript into normalized :class:`IngestEvent`s, deriving the
  ``project`` field from the transcript path and stripping ``tool_calls`` / tool
  results (FR-ING-7).
* :mod:`~mnemozine.ingestion.claude_code.chunker` — accumulate events into
  chunk/session-sized episodes bounded by ``ingest.chunk_max_chars`` /
  ``chunk_max_messages`` (FR-ING-6), with hash-on-content idempotency so a
  resumed/rewound session does not re-emit duplicate chunks (FR-ING-5).
* :mod:`~mnemozine.ingestion.claude_code.source` — the
  :class:`~mnemozine.interfaces.IngestSource`: a ``watchfiles``-based watcher
  tailing the transcript tree (``stream``) plus a backlog replay (``backfill``).
* :mod:`~mnemozine.ingestion.claude_code.hooks` — Claude Code hook entrypoints
  (``SessionStart``, ``UserPromptSubmit``, ``Stop``, ``PreCompact``) that flush
  the current chunk and trigger injection by delegating to the retrieval +
  ingestion service protocols (the integration pass wires concrete services).

.. note::
   ``INTERFACES.md`` names the ingestion-layer root ``mnemozine/ingest/**``; the
   owning task assigns this path ``mnemozine/ingestion/claude_code``. The public
   symbols below are stable regardless of the final root name — flagged for the
   integration pass to reconcile.
"""

from __future__ import annotations

from mnemozine.ingestion.claude_code.chunker import Chunk, ChunkAccumulator
from mnemozine.ingestion.claude_code.parser import (
    derive_project,
    parse_transcript_line,
    parse_transcript_lines,
    session_id_from_path,
)
from mnemozine.ingestion.claude_code.source import ClaudeCodeSource

__all__ = [
    "Chunk",
    "ChunkAccumulator",
    "ClaudeCodeSource",
    "derive_project",
    "parse_transcript_line",
    "parse_transcript_lines",
    "session_id_from_path",
]
