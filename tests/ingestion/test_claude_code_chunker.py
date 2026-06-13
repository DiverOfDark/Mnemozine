"""Unit tests for the chunker (FR-ING-6) and hash-on-content idempotency (FR-ING-5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from mnemozine.config import IngestSettings
from mnemozine.ingestion.claude_code.chunker import (
    Chunk,
    ChunkAccumulator,
    chunk_events,
)
from mnemozine.ingestion.claude_code.parser import read_transcript
from mnemozine.schema.events import (
    IngestEvent,
    Role,
    Source,
    chunk_content_hash,
)

FIXTURES = Path(__file__).parent / "fixtures"
RUST_TRANSCRIPT = FIXTURES / "-home-op-Projects-rust-cli" / "sess-rust-1.jsonl"


def _event(content: str, *, session_id: str = "s1", n: int = 0) -> IngestEvent:
    base = datetime(2026, 6, 13, 10, 0, 0, tzinfo=UTC)
    return IngestEvent(
        source=Source.CLAUDE_CODE,
        project="rust-cli",
        session_id=session_id,
        timestamp=base + timedelta(seconds=n),
        role=Role.USER,
        content=content,
    )


# --- FR-ING-6 chunk-level batching ----------------------------------------


def test_flush_emits_single_chunk() -> None:
    acc = ChunkAccumulator(IngestSettings())
    completed = acc.add_many([_event("a", n=0), _event("b", n=1), _event("c", n=2)])
    assert completed == []  # under both bounds -> nothing auto-emitted
    chunks = acc.flush()
    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
    assert len(chunk) == 3
    assert chunk.session_id == "s1"
    assert chunk.project == "rust-cli"
    assert chunk.source == Source.CLAUDE_CODE.value


def test_chunk_max_messages_bound() -> None:
    settings = IngestSettings(chunk_max_messages=3, chunk_max_chars=100_000)
    acc = ChunkAccumulator(settings)
    completed: list[Chunk] = []
    for i in range(7):
        completed.extend(acc.add(_event(f"msg-{i}", n=i)))
    # 7 messages, bound 3 -> two full chunks auto-emitted (3 + 3).
    assert len(completed) == 2
    assert all(len(c) == 3 for c in completed)
    # remainder of 1 flushes out.
    tail = acc.flush()
    assert len(tail) == 1 and len(tail[0]) == 1


def test_chunk_max_chars_bound() -> None:
    # Distinct content per event (so per-event de-dup does not drop them).
    # Each normalized content ~ "user:" (5) + 11 chars = 16 chars.
    settings = IngestSettings(chunk_max_chars=40, chunk_max_messages=1000)
    acc = ChunkAccumulator(settings)
    completed: list[Chunk] = []
    for i in range(5):
        completed.extend(acc.add(_event(f"payload-{i:03d}", n=i)))
    # ~16 chars each, budget 40 -> 2 fit per chunk before overflow.
    assert completed  # at least one chunk auto-emitted
    for c in completed:
        assert c.char_count <= 40
    acc.flush()


def test_per_session_buffers_do_not_bleed() -> None:
    acc = ChunkAccumulator(IngestSettings())
    acc.add(_event("alpha", session_id="A", n=0))
    acc.add(_event("beta", session_id="B", n=0))
    chunks = acc.flush()
    by_session = {c.session_id: c for c in chunks}
    assert set(by_session) == {"A", "B"}
    assert by_session["A"].events[0].content == "alpha"
    assert by_session["B"].events[0].content == "beta"


def test_chunk_events_helper_finite_stream() -> None:
    events = read_transcript(RUST_TRANSCRIPT)
    chunks = list(chunk_events(events, IngestSettings()))
    assert len(chunks) == 1
    assert len(chunks[0]) == len(events) == 5


# --- FR-ING-5 hash-on-content idempotency ---------------------------------


def test_chunk_content_hash_is_offset_invariant() -> None:
    # Two events with the SAME normalized content but different timestamps/ids
    # hash identically: the hash is on content, not offset (FR-ING-5).
    a = [_event("hello", n=0), _event("world", n=1)]
    b = [_event("hello", n=99), _event("world", n=100)]
    assert chunk_content_hash(a) == chunk_content_hash(b)
    c = [_event("hello", n=0), _event("DIFFERENT", n=1)]
    assert chunk_content_hash(a) != chunk_content_hash(c)


def test_duplicate_event_deduped() -> None:
    acc = ChunkAccumulator(IngestSettings())
    acc.add(_event("same content", n=0))
    # Re-adding the same normalized content (different timestamp) is dropped.
    completed = acc.add(_event("same content", n=50))
    assert completed == []
    chunks = acc.flush()
    assert len(chunks) == 1
    assert len(chunks[0]) == 1  # the duplicate did not accumulate


def test_reflushed_identical_chunk_suppressed() -> None:
    acc = ChunkAccumulator(IngestSettings())
    acc.add(_event("a", n=0))
    acc.add(_event("b", n=1))
    first = acc.flush()
    assert len(first) == 1
    h = first[0].content_hash
    # Re-feeding the exact same events: per-event de-dup drops them, so flush is
    # empty; even if they were new events with the same content, the chunk hash
    # would be suppressed.
    acc.add(_event("a", n=0))
    acc.add(_event("b", n=1))
    second = acc.flush()
    assert second == []
    assert acc.seen_chunk(h)


def test_resume_rewind_idempotent_across_accumulator() -> None:
    # Simulate a resume/rewind: the same transcript re-read after offsets shift.
    # Feeding it twice through one accumulator yields the chunk once (FR-ING-5).
    events = read_transcript(RUST_TRANSCRIPT)
    acc = ChunkAccumulator(IngestSettings())
    first = acc.add_many(events) + acc.flush()
    # Re-read (offsets could differ on disk; content identical) and feed again.
    events2 = read_transcript(RUST_TRANSCRIPT)
    second = acc.add_many(events2) + acc.flush()
    assert len(first) == 1
    assert second == []


def test_mark_emitted_seeds_high_water() -> None:
    # A restarting watcher can seed already-ingested chunk hashes (FR-ING-5).
    events = read_transcript(RUST_TRANSCRIPT)
    h = chunk_content_hash(events)
    acc = ChunkAccumulator(IngestSettings())
    acc.mark_emitted(h)
    out = acc.add_many(events) + acc.flush()
    assert out == []  # already known -> not re-emitted
    assert acc.seen_chunk(h)
