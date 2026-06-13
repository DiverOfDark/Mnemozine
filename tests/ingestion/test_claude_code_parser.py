"""Unit tests for the Claude Code transcript parser (FR-ING-2/7).

Covers project derivation, line-type filtering, content flattening, timestamp
parsing, and the FR-ING-7 tool_calls / tool_result stripping — all against the
fixture JSONL so the parser is exercised on a realistic transcript shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemozine.ingestion.claude_code.parser import (
    derive_project,
    parse_transcript_line,
    parse_transcript_lines,
    read_transcript,
    session_id_from_path,
)
from mnemozine.schema.events import Role, Source

FIXTURES = Path(__file__).parent / "fixtures"
RUST_DIR = FIXTURES / "-home-op-Projects-rust-cli"
RUST_TRANSCRIPT = RUST_DIR / "sess-rust-1.jsonl"


def test_session_id_from_path() -> None:
    assert session_id_from_path(RUST_TRANSCRIPT) == "sess-rust-1"
    assert session_id_from_path("/x/y/abc-123.jsonl") == "abc-123"


def test_derive_project_prefers_cwd() -> None:
    # cwd basename wins over the path-encoded directory.
    assert derive_project(RUST_TRANSCRIPT, cwd="/home/op/Projects/rust-cli") == "rust-cli"


def test_derive_project_falls_back_to_path() -> None:
    # No cwd: the trailing segment of the path-encoded dir name is the project.
    assert derive_project(RUST_TRANSCRIPT) == "cli"
    # The encoded dir uses '-' separators; the leaf segment is taken.
    p = "/root/projects/-var-home-op-Projects-mnemozine/sess.jsonl"
    assert derive_project(p) == "mnemozine"


def test_parse_user_line_string_content() -> None:
    raw = (
        '{"type":"user","sessionId":"s1","cwd":"/home/op/proj",'
        '"timestamp":"2026-06-13T10:00:01.000Z",'
        '"message":{"role":"user","content":"hello world"}}'
    )
    event = parse_transcript_line(raw, path="/cfg/projects/-home-op-proj/s1.jsonl")
    assert event is not None
    assert event.source is Source.CLAUDE_CODE
    assert event.role is Role.USER
    assert event.content == "hello world"
    assert event.session_id == "s1"
    assert event.project == "proj"
    assert event.timestamp.tzinfo is not None  # Z -> tz-aware
    assert event.metadata["cwd"] == "/home/op/proj"


def test_bookkeeping_lines_are_dropped() -> None:
    for raw in [
        '{"type":"file-history-snapshot","messageId":"m"}',
        '{"type":"permission-mode","mode":"default"}',
        '{"type":"ai-title","title":"x"}',
        '{"type":"last-prompt"}',
        '{"type":"attachment"}',
    ]:
        assert parse_transcript_line(raw, path=RUST_TRANSCRIPT) is None


def test_malformed_line_returns_none_not_raises() -> None:
    assert parse_transcript_line("not json at all {{{", path=RUST_TRANSCRIPT) is None
    assert parse_transcript_line("", path=RUST_TRANSCRIPT) is None
    assert parse_transcript_line("   ", path=RUST_TRANSCRIPT) is None


def test_assistant_thinking_block_is_stripped() -> None:
    raw = (
        '{"type":"assistant","sessionId":"s1",'
        '"timestamp":"2026-06-13T10:00:02.000Z",'
        '"message":{"role":"assistant","content":['
        '{"type":"thinking","thinking":"secret reasoning"},'
        '{"type":"text","text":"the visible answer"}]}}'
    )
    event = parse_transcript_line(raw, path=RUST_TRANSCRIPT)
    assert event is not None
    assert event.content == "the visible answer"
    assert "secret reasoning" not in event.content


def test_full_transcript_parse_counts() -> None:
    events = read_transcript(RUST_TRANSCRIPT)
    # Conversational turns with surviving text only:
    #   u: "I prefer thiserror..."          -> kept
    #   a: thinking + text                  -> kept (text only)
    #   u: "This project pins tokio 1.38..."-> kept
    #   a: text + tool_use                  -> kept (text only)
    #   u: tool_result only                 -> dropped (no text)
    #   a: "All 12 tests passed."           -> kept
    #   u: tool_result only (string)        -> dropped
    #   a: tool_use only                    -> dropped (no text)
    assert len(events) == 5
    contents = [e.content for e in events]
    assert contents[0].startswith("I prefer thiserror")
    assert "Understood" in contents[1]
    assert "tokio 1.38" in contents[2]
    assert "Running the test suite" in contents[3]
    assert contents[4] == "All 12 tests passed."


def test_parse_transcript_lines_skips_none() -> None:
    lines = RUST_TRANSCRIPT.read_text().splitlines()
    events = list(parse_transcript_lines(lines, path=RUST_TRANSCRIPT))
    assert len(events) == 5
    # File order preserved.
    assert events[0].timestamp < events[-1].timestamp


def test_missing_file_returns_empty() -> None:
    assert read_transcript(FIXTURES / "does-not-exist.jsonl") == []


@pytest.mark.parametrize(
    "ts,expect_tz",
    [
        ("2026-06-13T10:00:01.000Z", True),
        ("2026-06-13T10:00:01+00:00", True),
        ("2026-06-13T10:00:01", True),  # naive -> coerced to UTC
        ("garbage", True),  # falls back to now(), still tz-aware
        (None, True),
    ],
)
def test_timestamp_parsing_is_tolerant(ts: str | None, expect_tz: bool) -> None:
    record = {
        "type": "user",
        "sessionId": "s1",
        "timestamp": ts,
        "message": {"role": "user", "content": "hi"},
    }
    event = parse_transcript_line(record, path="/c/projects/-x/s1.jsonl")
    assert event is not None
    assert (event.timestamp.tzinfo is not None) == expect_tz
