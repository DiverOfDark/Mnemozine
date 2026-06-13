"""Unit tests for FR-ING-7 tool-call / tool-result stripping.

The highest-density source of raw credentials is tool traffic. These tests assert
that ``tool_use`` and ``tool_result`` blocks never survive into an event's
``content``, that ``tool_calls`` is ``None`` under the default
(``strip_tool_calls=True``), and that leaked secrets inside those blocks do not
reach the normalized event.
"""

from __future__ import annotations

from pathlib import Path

from mnemozine.ingestion.claude_code.parser import (
    parse_transcript_line,
    read_transcript,
)

FIXTURES = Path(__file__).parent / "fixtures"
RUST_TRANSCRIPT = FIXTURES / "-home-op-Projects-rust-cli" / "sess-rust-1.jsonl"


def test_tool_use_block_stripped_from_content_default() -> None:
    # Assistant turn with text + a tool_use carrying a leaked credential.
    raw = (
        '{"type":"assistant","sessionId":"s1",'
        '"timestamp":"2026-06-13T10:00:04.000Z",'
        '"message":{"role":"assistant","content":['
        '{"type":"text","text":"Running the test suite now."},'
        '{"type":"tool_use","id":"tu_1","name":"Bash",'
        '"input":{"command":"cargo test","secret_token":"sk-LEAKED-123"}}]}}'
    )
    event = parse_transcript_line(raw, path=RUST_TRANSCRIPT)
    assert event is not None
    assert event.content == "Running the test suite now."
    assert "sk-LEAKED-123" not in event.content
    assert "cargo test" not in event.content
    # Default strip_tool_calls=True -> no tool_calls retained.
    assert event.tool_calls is None


def test_tool_result_only_turn_is_dropped() -> None:
    raw = (
        '{"type":"user","sessionId":"s1",'
        '"timestamp":"2026-06-13T10:00:05.000Z",'
        '"message":{"role":"user","content":['
        '{"type":"tool_result","tool_use_id":"tu_1","content":['
        '{"type":"text","text":"AWS_SECRET_ACCESS_KEY=AKIAEXAMPLE"}]}]}}'
    )
    # A turn whose only content was tool traffic yields nothing durable.
    assert parse_transcript_line(raw, path=RUST_TRANSCRIPT) is None


def test_no_leaked_secret_anywhere_in_transcript_events() -> None:
    events = read_transcript(RUST_TRANSCRIPT)
    blob = "\n".join(e.content for e in events)
    # Credentials present in the fixture's tool blocks must not survive.
    assert "sk-LEAKED-CREDENTIAL-123" not in blob
    assert "AWS_SECRET_ACCESS_KEY" not in blob
    assert "AKIAEXAMPLE" not in blob
    assert "/etc/passwd" not in blob
    # And no event retains tool_calls under the default strip.
    assert all(e.tool_calls is None for e in events)


def test_tool_use_retained_when_strip_disabled() -> None:
    # With stripping off, tool_use blocks attach to tool_calls but are STILL not
    # inlined into content (FR-ING-7: never inline tool traffic into content).
    raw = (
        '{"type":"assistant","sessionId":"s1",'
        '"timestamp":"2026-06-13T10:00:04.000Z",'
        '"message":{"role":"assistant","content":['
        '{"type":"text","text":"visible"},'
        '{"type":"tool_use","id":"tu_1","name":"Bash",'
        '"input":{"command":"ls"}}]}}'
    )
    event = parse_transcript_line(raw, path=RUST_TRANSCRIPT, strip_tool_calls=False)
    assert event is not None
    assert event.content == "visible"
    assert event.tool_calls is not None
    assert event.tool_calls[0]["name"] == "Bash"
    assert "ls" not in event.content


def test_settings_strip_tool_calls_default_is_true() -> None:
    from mnemozine.config import IngestSettings

    assert IngestSettings().strip_tool_calls is True
