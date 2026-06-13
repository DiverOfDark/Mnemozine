"""watchfiles-based Claude Code transcript source (FR-ING-2/5/6/7, R4).

:class:`ClaudeCodeSource` is the concrete
:class:`~mnemozine.interfaces.IngestSource` for Claude Code. It tails the JSONL
transcript tree under ``$CLAUDE_CONFIG_DIR/projects/`` in near-real-time
(``stream``, FR-ING-2/R4) and replays already-existing transcripts for the
backlog import (``backfill``, FR-ING-6).

Both ``stream`` and ``backfill`` are **async generators** of normalized
:class:`~mnemozine.schema.events.IngestEvent`s (per the
:class:`~mnemozine.interfaces.IngestSource` call convention): iterate with
``async for e in source.stream()`` — never ``await`` them.

Resume-safe tailing (FR-ING-5)
------------------------------
Claude Code appends to transcripts, but a session resume/rewind can *rewrite*
earlier lines, shifting byte/line offsets. So the watcher never tracks byte
offsets: on every change it re-parses the whole file and de-dups events on their
``(session_id, content_hash)`` key (the same hash-on-content key the
``ChunkAccumulator`` uses), which is invariant to offset changes. The watcher
emits per-event; chunking into episodes is the caller's job via
:class:`~mnemozine.ingestion.claude_code.chunker.ChunkAccumulator` (so the same
accumulator the ``Stop``/``PreCompact`` hooks flush is authoritative).

``tool_calls`` are stripped per FR-ING-7 when ``settings.ingest.strip_tool_calls``
is set (the default).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

from watchfiles import awatch

from mnemozine.config import Settings, get_settings
from mnemozine.ingestion.claude_code.parser import (
    PROJECTS_DIRNAME,
    read_transcript,
)
from mnemozine.schema.events import IngestEvent, Source
from mnemozine.schema.models import SourceSession

# Claude Code's own env var (NOT prefixed with MNEMOZINE_) overriding the config
# root if the transcripts were relocated (FR-ING-2).
CLAUDE_CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"

_TRANSCRIPT_SUFFIX = ".jsonl"


def resolve_config_dir(settings: Settings) -> Path:
    """Resolve the Claude Code config root, honoring ``CLAUDE_CONFIG_DIR`` (FR-ING-2).

    Precedence: the raw ``CLAUDE_CONFIG_DIR`` environment variable (Claude Code's
    own override, used when transcripts are relocated) wins, then the
    ``ingest.claude_config_dir`` setting (which itself defaults to ``~/.claude``).
    ``~`` is expanded.
    """

    env = os.environ.get(CLAUDE_CONFIG_DIR_ENV)
    if env:
        return Path(env).expanduser()
    return Path(settings.ingest.claude_config_dir).expanduser()


def projects_dir(settings: Settings) -> Path:
    """The ``<config>/projects`` directory holding per-project transcript dirs."""

    return resolve_config_dir(settings) / PROJECTS_DIRNAME


class ClaudeCodeSource:
    """Claude Code JSONL ingest source (FR-ING-2). See module docstring.

    Implements :class:`mnemozine.interfaces.IngestSource` structurally.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        # (session_id, content_hash) of every event already yielded, so a
        # re-parsed file after an append/rewind does not re-emit a prefix.
        self._seen: set[tuple[str, str]] = set()

    @property
    def source_name(self) -> str:
        """The :class:`~mnemozine.schema.events.Source` value this produces."""

        return Source.CLAUDE_CODE.value

    @property
    def config_dir(self) -> Path:
        """The resolved Claude Code config root (FR-ING-2)."""

        return resolve_config_dir(self._settings)

    @property
    def projects_dir(self) -> Path:
        """The resolved ``projects`` directory under the config root."""

        return projects_dir(self._settings)

    # --- discovery -------------------------------------------------------

    def discover_transcripts(self) -> list[Path]:
        """List existing transcript files under the projects tree (sorted)."""

        root = self.projects_dir
        if not root.is_dir():
            return []
        return sorted(root.rglob(f"*{_TRANSCRIPT_SUFFIX}"))

    def session_for(self, path: str | Path) -> SourceSession:
        """Build a :class:`SourceSession` record for a transcript (provenance)."""

        events = read_transcript(
            path, strip_tool_calls=self._settings.ingest.strip_tool_calls
        )
        first = events[0] if events else None
        last = events[-1] if events else None
        p = Path(path)
        return SourceSession(
            source=Source.CLAUDE_CODE.value,
            session_id=first.session_id if first else p.stem,
            project=first.project if first else p.parent.name,
            started_at=first.timestamp if first else None,
            ended_at=last.timestamp if last else None,
            raw_path=str(p),
        )

    # --- event production ------------------------------------------------

    def _events_from(self, path: Path) -> list[IngestEvent]:
        """Parse a transcript, dropping events already yielded this run (FR-ING-5)."""

        out: list[IngestEvent] = []
        for event in read_transcript(
            path, strip_tool_calls=self._settings.ingest.strip_tool_calls
        ):
            key = (event.session_id, event.content_hash())
            if key in self._seen:
                continue
            self._seen.add(key)
            out.append(event)
        return out

    async def backfill(
        self, *, since: SourceSession | None = None
    ) -> AsyncIterator[IngestEvent]:
        """Replay existing transcripts for the backlog import (FR-ING-6).

        Yields the conversational events of every existing transcript, in path
        order, de-duplicated on content hash so a re-run is safe (FR-ING-5). When
        ``since`` is given, transcripts whose session id matches an already-ingested
        session can still be replayed cheaply — downstream de-dups on the
        idempotency key — but transcripts older than ``since.ended_at`` are skipped
        as a coarse optimization.
        """

        cutoff = since.ended_at if since is not None else None
        for path in self.discover_transcripts():
            if cutoff is not None:
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = None
                if mtime is not None and mtime < cutoff.timestamp():
                    continue
            for event in self._events_from(path):
                yield event

    async def stream(self) -> AsyncIterator[IngestEvent]:
        """Tail the transcript tree in near-real-time (FR-ING-2/R4).

        First replays existing transcripts (so a freshly-started watcher does not
        miss in-flight sessions), then blocks on :func:`watchfiles.awatch`,
        re-parsing each changed ``*.jsonl`` and yielding only events not seen
        before (FR-ING-5, resume/rewind safe). Runs indefinitely. The watched
        directory is created if absent so the watcher does not crash on a fresh
        install before Claude Code has written anything.
        """

        root = self.projects_dir
        root.mkdir(parents=True, exist_ok=True)

        # Initial catch-up so nothing already on disk is missed (R4).
        for path in self.discover_transcripts():
            for event in self._events_from(path):
                yield event

        async for changes in awatch(root):
            # changes is a set of (Change, path) tuples; re-parse touched files.
            touched: set[Path] = set()
            for _change, raw_path in changes:
                p = Path(raw_path)
                if p.suffix == _TRANSCRIPT_SUFFIX:
                    touched.add(p)
            for p in sorted(touched):
                if not p.exists():
                    # Deleted (e.g. 30-day cleanup, R4); nothing to re-parse.
                    continue
                for event in self._events_from(p):
                    yield event
