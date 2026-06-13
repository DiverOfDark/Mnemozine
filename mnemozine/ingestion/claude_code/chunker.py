"""Chunk-level batching + hash-on-content idempotency (FR-ING-5/6).

The unit of extraction is a **chunk/session**, not a single message (FR-ING-6) —
this matches Graphiti's episode model and avoids a per-message LLM storm (R3).
:class:`ChunkAccumulator` groups normalized :class:`IngestEvent`s (per session)
into chunks bounded by ``ingest.chunk_max_chars`` / ``chunk_max_messages``, and
the ``Stop`` / ``PreCompact`` hooks flush the in-flight chunk at session end and
before compaction.

Idempotency (FR-ING-5) is **hash-on-content**: every chunk carries
``chunk_content_hash(events)`` so re-ingesting the same transcript — after a
crash, or a session resume/rewind that rewrites line offsets — produces the same
hash and de-duplicates, because the hash is over normalized content, never byte
or line offsets. The accumulator tracks the per-event content hashes it has
already absorbed so a replayed prefix is skipped, and tracks already-emitted
chunk hashes so a re-emitted identical chunk is suppressed.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from mnemozine.config import IngestSettings
from mnemozine.schema.events import (
    IngestEvent,
    chunk_content_hash,
)


@dataclass(slots=True)
class Chunk:
    """A flushed chunk/episode: the ordered events plus their content hash.

    ``content_hash`` is :func:`~mnemozine.schema.events.chunk_content_hash` over
    the contained events (FR-ING-5/6) — stable across resume/rewind because it
    hashes normalized content, not offsets. ``source``/``session_id``/``project``
    are taken from the first event for convenience (a chunk is single-session).
    """

    events: list[IngestEvent]
    content_hash: str
    source: str
    session_id: str
    project: str

    @property
    def char_count(self) -> int:
        """Total characters of normalized content across the chunk's events."""

        return sum(len(e.normalized_content()) for e in self.events)

    def __len__(self) -> int:
        return len(self.events)


def _make_chunk(events: list[IngestEvent]) -> Chunk:
    first = events[0]
    return Chunk(
        events=list(events),
        content_hash=chunk_content_hash(events),
        source=first.source.value,
        session_id=first.session_id,
        project=first.project,
    )


@dataclass(slots=True)
class _SessionBuffer:
    """In-flight events for one session, with running size bookkeeping."""

    events: list[IngestEvent] = field(default_factory=list)
    chars: int = 0


class ChunkAccumulator:
    """Accumulate per-session events into episode-sized chunks (FR-ING-6).

    Add events with :meth:`add` (which may yield zero or more completed chunks as
    size bounds are crossed) and force-emit the in-flight remainder with
    :meth:`flush` (which the ``Stop`` / ``PreCompact`` hooks call at session end /
    before compaction). Idempotency is enforced two ways (FR-ING-5):

    * **per-event de-dup** — an event whose ``(session_id, content_hash)`` was
      already seen is dropped, so replaying a transcript prefix after a
      resume/rewind does not re-accumulate it.
    * **per-chunk de-dup** — a completed chunk whose content hash was already
      emitted is suppressed, so an identical re-flushed chunk is not re-ingested.

    The accumulator keeps one buffer per ``session_id`` so interleaved sessions
    (multiple concurrent transcripts under the watcher) do not bleed into one
    chunk.
    """

    def __init__(self, settings: IngestSettings | None = None) -> None:
        self._max_chars = settings.chunk_max_chars if settings is not None else 8000
        self._max_messages = settings.chunk_max_messages if settings is not None else 40
        self._buffers: dict[str, _SessionBuffer] = {}
        # (session_id, content_hash) of every event already absorbed (FR-ING-5).
        self._seen_events: set[tuple[str, str]] = set()
        # content hashes of every chunk already emitted (FR-ING-5).
        self._emitted_chunks: set[str] = set()

    @property
    def emitted_chunk_hashes(self) -> frozenset[str]:
        """The set of chunk content hashes emitted so far (for inspection/tests)."""

        return frozenset(self._emitted_chunks)

    def seen_chunk(self, content_hash: str) -> bool:
        """True if a chunk with this content hash was already emitted (FR-ING-5)."""

        return content_hash in self._emitted_chunks

    def mark_emitted(self, content_hash: str) -> None:
        """Record a chunk hash as already ingested (resume-safe restart, FR-ING-5).

        Lets a restarting watcher seed the accumulator from a persisted high-water
        set so previously-ingested chunks are not re-emitted.
        """

        self._emitted_chunks.add(content_hash)

    def add(self, event: IngestEvent) -> list[Chunk]:
        """Absorb one event; return any chunks completed by crossing a size bound.

        De-dups the event on its ``(session_id, content_hash)`` key first
        (FR-ING-5). A chunk is flushed when adding the event would exceed
        ``chunk_max_chars`` (the event that overflows starts the next chunk) or
        when ``chunk_max_messages`` is reached (inclusive of the new event).
        """

        key = (event.session_id, event.content_hash())
        if key in self._seen_events:
            return []
        self._seen_events.add(key)

        buf = self._buffers.setdefault(event.session_id, _SessionBuffer())
        size = len(event.normalized_content())

        completed: list[Chunk] = []
        # Flush BEFORE adding when the event would overflow the char budget and
        # the buffer is non-empty — keep the chunk under budget, start fresh.
        if buf.events and buf.chars + size > self._max_chars:
            chunk = self._emit(event.session_id)
            if chunk is not None:
                completed.append(chunk)
            buf = self._buffers[event.session_id]

        buf.events.append(event)
        buf.chars += size

        if len(buf.events) >= self._max_messages:
            chunk = self._emit(event.session_id)
            if chunk is not None:
                completed.append(chunk)

        return completed

    def add_many(self, events: Iterable[IngestEvent]) -> list[Chunk]:
        """Absorb many events, returning all chunks completed along the way."""

        out: list[Chunk] = []
        for event in events:
            out.extend(self.add(event))
        return out

    def flush(self, session_id: str | None = None) -> list[Chunk]:
        """Force-emit in-flight buffers (Stop / PreCompact hook, FR-ING-6).

        With ``session_id`` given, flushes only that session's buffer; otherwise
        flushes every buffered session (e.g. on watcher shutdown). Returns the
        completed chunks, suppressing any whose content hash was already emitted
        (FR-ING-5). An empty buffer flushes to nothing.
        """

        if session_id is not None:
            chunk = self._emit(session_id)
            return [chunk] if chunk is not None else []

        out: list[Chunk] = []
        for sid in list(self._buffers.keys()):
            chunk = self._emit(sid)
            if chunk is not None:
                out.append(chunk)
        return out

    def _emit(self, session_id: str) -> Chunk | None:
        buf = self._buffers.get(session_id)
        if buf is None or not buf.events:
            return None
        chunk = _make_chunk(buf.events)
        # Reset the buffer regardless of whether the chunk is a duplicate.
        self._buffers[session_id] = _SessionBuffer()
        if chunk.content_hash in self._emitted_chunks:
            return None
        self._emitted_chunks.add(chunk.content_hash)
        return chunk


def chunk_events(
    events: Iterable[IngestEvent], settings: IngestSettings | None = None
) -> Iterator[Chunk]:
    """Eagerly chunk a finite event stream (backlog import, FR-ING-6).

    Convenience wrapper over :class:`ChunkAccumulator` for the backfill path: feed
    a whole transcript's events and get back every chunk (size-bounded chunks plus
    a final flush of the remainder), de-duplicated on content hash.
    """

    acc = ChunkAccumulator(settings)
    for event in events:
        yield from acc.add(event)
    yield from acc.flush()
