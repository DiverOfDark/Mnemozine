"""Unit tests for the Hermes ingestion adapter (FR-ING-4, FR-ING-7).

Fully offline with **mocked** Hermes-native turn payloads — no VM, no network.
Covers the pure normalization (:func:`events_from_hermes_turn`), the
:class:`HermesAdapter` ``IngestSource`` (feed / stream / backfill), and the
OpenAI-fronting fallback (:func:`hermes_gateway_source`).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from mnemozine.config import IngestSettings
from mnemozine.ingestion.hermes import (
    HermesAdapter,
    events_from_hermes_turn,
    hermes_gateway_source,
)
from mnemozine.interfaces import IngestSource
from mnemozine.schema.events import IngestEvent, Role, Source


def _turn(messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"conversation_id": "conv-1", "messages": messages}
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# events_from_hermes_turn — core mapping
# ---------------------------------------------------------------------------


def test_basic_conversation_maps_all_turns() -> None:
    payload = _turn(
        [
            {"role": "user", "content": "I floated an idea for a CLI graph tool."},
            {"role": "assistant", "content": "Interesting — that's an idea_seed."},
        ],
        project="ideas",
        model="hermes-3",
    )

    events = events_from_hermes_turn(payload)

    assert len(events) == 2
    assert [e.role for e in events] == [Role.USER, Role.ASSISTANT]
    assert all(e.source is Source.HERMES for e in events)
    assert all(e.project == "ideas" for e in events)
    assert all(e.session_id == "conv-1" for e in events)
    assert all(e.metadata.get("source_agent") == "hermes" for e in events)
    assert all(e.metadata.get("model") == "hermes-3" for e in events)


def test_tolerates_field_name_variants_turns_and_text() -> None:
    # session_id instead of conversation_id; "turns" instead of "messages";
    # "text" instead of "content".
    payload = {
        "session_id": "sess-9",
        "turns": [
            {"role": "user", "text": "variant fields"},
            {"role": "hermes", "text": "mapped as assistant"},
        ],
    }

    events = events_from_hermes_turn(payload)

    assert [e.content for e in events] == ["variant fields", "mapped as assistant"]
    assert [e.role for e in events] == [Role.USER, Role.ASSISTANT]
    assert all(e.session_id == "sess-9" for e in events)


def test_single_turn_top_level_shape() -> None:
    payload = {"id": "c2", "role": "user", "content": "single turn payload"}

    events = events_from_hermes_turn(payload)

    assert len(events) == 1
    assert events[0].role is Role.USER
    assert events[0].session_id == "c2"
    assert events[0].content == "single turn payload"


def test_project_from_metadata_when_not_top_level() -> None:
    payload = _turn(
        [{"role": "user", "content": "hi"}],
        metadata={"project": "from-meta"},
    )

    events = events_from_hermes_turn(payload)

    assert all(e.project == "from-meta" for e in events)


def test_default_project_used_when_absent() -> None:
    payload = _turn([{"role": "user", "content": "hi"}])

    events = events_from_hermes_turn(payload, default_project="hermes-default")

    assert all(e.project == "hermes-default" for e in events)


def test_per_message_timestamp_is_parsed() -> None:
    payload = _turn(
        [
            {"role": "user", "content": "first", "timestamp": "2026-06-13T10:00:00Z"},
            {"role": "assistant", "content": "second", "created_at": "2026-06-13T10:00:05Z"},
        ]
    )

    events = events_from_hermes_turn(payload)

    assert events[0].timestamp == datetime(2026, 6, 13, 10, 0, 0, tzinfo=UTC)
    assert events[1].timestamp == datetime(2026, 6, 13, 10, 0, 5, tzinfo=UTC)
    assert events[0].timestamp < events[1].timestamp


def test_epoch_timestamp_is_parsed() -> None:
    payload = _turn([{"role": "user", "content": "x"}], timestamp=1_700_000_000)

    events = events_from_hermes_turn(payload)

    assert events[0].timestamp == datetime.fromtimestamp(1_700_000_000, tz=UTC)


def test_content_parts_list_flattened() -> None:
    payload = _turn(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            }
        ]
    )

    events = events_from_hermes_turn(payload)

    assert events[0].content == "a\nb"


# ---------------------------------------------------------------------------
# FR-ING-7 — tool_calls stripping
# ---------------------------------------------------------------------------


def test_tool_role_messages_are_stripped() -> None:
    payload = _turn(
        [
            {"role": "user", "content": "use a tool"},
            {"role": "tool", "content": "{'api_key': 'SECRET'}"},
            {"role": "assistant", "content": "Done."},
        ]
    )

    events = events_from_hermes_turn(payload, strip_tool_calls=True)

    assert [e.role for e in events] == [Role.USER, Role.ASSISTANT]
    assert all("SECRET" not in e.content for e in events)
    assert all(e.tool_calls is None for e in events)


def test_assistant_message_with_tool_calls_keeps_text_and_marks_stripped() -> None:
    payload = _turn(
        [
            {
                "role": "assistant",
                "content": "Calling the search tool now.",
                "tool_calls": [{"id": "t1", "function": {"name": "search"}}],
            }
        ]
    )

    events = events_from_hermes_turn(payload, strip_tool_calls=True)

    assert len(events) == 1
    assert events[0].content == "Calling the search tool now."
    assert events[0].tool_calls is None
    assert events[0].metadata.get("tool_calls_stripped") is True


def test_pure_tool_call_message_with_no_text_dropped() -> None:
    payload = _turn(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "t1", "function": {"name": "f"}}],
            },
        ]
    )

    events = events_from_hermes_turn(payload, strip_tool_calls=True)

    assert [e.role for e in events] == [Role.USER]


def test_system_messages_dropped() -> None:
    payload = _turn(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "ok"},
        ]
    )

    events = events_from_hermes_turn(payload)

    assert [e.role for e in events] == [Role.USER]


def test_empty_or_malformed_payload_yields_nothing() -> None:
    assert events_from_hermes_turn({}) == []
    assert events_from_hermes_turn({"messages": "not-a-list"}) == []
    assert events_from_hermes_turn({"messages": [42, "garbage", {}]}) == []


# ---------------------------------------------------------------------------
# HermesAdapter — IngestSource
# ---------------------------------------------------------------------------


def test_adapter_is_ingest_source() -> None:
    adapter = HermesAdapter()
    assert isinstance(adapter, IngestSource)
    assert adapter.source_name == "hermes"


def test_feed_maps_and_returns_events() -> None:
    adapter = HermesAdapter()
    events = adapter.feed(_turn([{"role": "user", "content": "fed in"}]))
    assert [e.content for e in events] == ["fed in"]
    assert events[0].source is Source.HERMES


@pytest.mark.asyncio
async def test_feed_then_stream_yields_events() -> None:
    adapter = HermesAdapter()
    adapter.feed(
        _turn(
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ]
        )
    )

    stream = adapter.stream()
    first = await anext(stream)
    second = await anext(stream)
    await stream.aclose()

    assert [first.content, second.content] == ["q1", "a1"]


@pytest.mark.asyncio
async def test_afeed_then_stream() -> None:
    adapter = HermesAdapter()
    await adapter.afeed(_turn([{"role": "user", "content": "async fed"}]))
    stream = adapter.stream()
    first = await anext(stream)
    await stream.aclose()
    assert first.content == "async fed"


@pytest.mark.asyncio
async def test_backfill_replays_recorded_payloads() -> None:
    recorded = [
        {"conversation_id": "h1", "messages": [{"role": "user", "content": "old one"}]},
        {"conversation_id": "h2", "messages": [{"role": "assistant", "content": "old two"}]},
    ]
    adapter = HermesAdapter(recorded=recorded)

    out = [e async for e in adapter.backfill()]

    assert [e.content for e in out] == ["old one", "old two"]
    assert all(isinstance(e, IngestEvent) for e in out)
    assert {e.session_id for e in out} == {"h1", "h2"}


@pytest.mark.asyncio
async def test_backfill_respects_since_cutoff() -> None:
    recorded = [
        {
            "conversation_id": "h1",
            "messages": [
                {"role": "user", "content": "before", "timestamp": "2026-06-01T00:00:00Z"},
                {"role": "user", "content": "after", "timestamp": "2026-06-10T00:00:00Z"},
            ],
        },
    ]
    adapter = HermesAdapter(recorded=recorded)

    from mnemozine.schema.models import SourceSession

    since = SourceSession(
        source="hermes",
        session_id="h1",
        project="hermes",
        started_at=datetime(2026, 6, 5, tzinfo=UTC),
    )
    out = [e async for e in adapter.backfill(since=since)]

    assert [e.content for e in out] == ["after"]


def test_adapter_honors_strip_tool_calls_setting() -> None:
    adapter = HermesAdapter(settings=IngestSettings(strip_tool_calls=False))
    events = adapter.feed(
        _turn(
            [
                {
                    "role": "assistant",
                    "content": "calling tool",
                    "tool_calls": [{"id": "t", "function": {"name": "f"}}],
                }
            ]
        )
    )
    # stripping disabled -> not marked stripped, but still never carries tool_calls.
    assert events[0].tool_calls is None
    assert "tool_calls_stripped" not in events[0].metadata


# ---------------------------------------------------------------------------
# Gateway-fronting fallback
# ---------------------------------------------------------------------------


def test_hermes_gateway_source_stamps_source_hermes() -> None:
    src = hermes_gateway_source()
    assert isinstance(src, IngestSource)
    assert src.source_name == "hermes"

    kwargs = {"model": "openai/hermes", "messages": [{"role": "user", "content": "via gateway"}]}
    resp = {"choices": [{"message": {"role": "assistant", "content": "reply"}}]}
    events = src.map_events(kwargs, resp, end_time=datetime(2026, 6, 13, tzinfo=UTC))

    assert events and all(e.source is Source.HERMES for e in events)
    assert [e.content for e in events] == ["via gateway", "reply"]
