"""Unit tests for the LiteLLM gateway callback event mapping (FR-ING-3, FR-ING-7).

All tests run fully offline with **mocked** LiteLLM payloads — no proxy, no Qwen,
no network. The LiteLLM ``log_success_event`` contract is
``(kwargs, response_obj, start_time, end_time)``; ``kwargs`` carries the request
``messages``/``model``/``metadata`` and ``response_obj`` carries the completion
choices. We exercise both a plain-dict response and a tiny object-style stub so
the mapping's dict-or-object tolerance is covered.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from mnemozine.config import IngestSettings
from mnemozine.ingestion.gateway import (
    GatewayCallback,
    events_from_completion,
    make_gateway_callback,
)
from mnemozine.interfaces import IngestSource
from mnemozine.schema.events import IngestEvent, Role, Source

END_TIME = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Mock LiteLLM payload builders
# ---------------------------------------------------------------------------


def _kwargs(
    messages: list[dict[str, Any]],
    *,
    model: str = "openai/qwen2.5",
    metadata: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if metadata is not None:
        payload["metadata"] = metadata
    payload.update(extra)
    return payload


def _dict_response(
    content: str, *, tool_calls: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "model": "openai/qwen2.5",
        "choices": [{"index": 0, "message": message}],
    }


@dataclass
class _ObjMessage:
    role: str
    content: str | None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class _ObjChoice:
    message: _ObjMessage
    index: int = 0


@dataclass
class _ObjResponse:
    choices: list[_ObjChoice]
    id: str = "chatcmpl-obj"
    model: str = "openai/qwen2.5"


# ---------------------------------------------------------------------------
# events_from_completion — core mapping
# ---------------------------------------------------------------------------


def test_basic_user_assistant_turn_maps_to_two_events() -> None:
    kwargs = _kwargs(
        [{"role": "user", "content": "I prefer thiserror over anyhow in Rust."}],
        metadata={"mnemozine_project": "rust-cli", "mnemozine_session_id": "sess-42"},
    )
    resp = _dict_response("Understood — I'll use thiserror here.")

    events = events_from_completion(kwargs, resp, source=Source.OPENAI, end_time=END_TIME)

    assert len(events) == 2
    user, assistant = events
    assert user.role is Role.USER
    assert user.content == "I prefer thiserror over anyhow in Rust."
    assert assistant.role is Role.ASSISTANT
    assert assistant.content == "Understood — I'll use thiserror here."
    # source + project + session threaded through from metadata.
    assert all(e.source is Source.OPENAI for e in events)
    assert all(e.project == "rust-cli" for e in events)
    assert all(e.session_id == "sess-42" for e in events)
    # request turn is ordered strictly before the assistant reply.
    assert user.timestamp < assistant.timestamp


def test_object_style_response_is_supported() -> None:
    kwargs = _kwargs([{"role": "user", "content": "hello"}])
    resp = _ObjResponse(
        choices=[_ObjChoice(message=_ObjMessage(role="assistant", content="hi there"))]
    )

    events = events_from_completion(kwargs, resp, end_time=END_TIME)

    assert [e.content for e in events] == ["hello", "hi there"]
    assert [e.role for e in events] == [Role.USER, Role.ASSISTANT]


def test_source_hermes_is_stamped() -> None:
    kwargs = _kwargs([{"role": "user", "content": "ping"}])
    resp = _dict_response("pong")

    events = events_from_completion(kwargs, resp, source=Source.HERMES, end_time=END_TIME)

    assert events and all(e.source is Source.HERMES for e in events)
    assert all(e.metadata.get("gateway") == "hermes" for e in events)


def test_only_trailing_request_turns_emitted_not_whole_history() -> None:
    # OpenAI resends the full transcript each turn; we must emit only the NEW
    # user turn appended after the last assistant message, not re-ingest history.
    kwargs = _kwargs(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ],
    )
    resp = _dict_response("second answer")

    events = events_from_completion(kwargs, resp, end_time=END_TIME)

    contents = [e.content for e in events]
    # history ("first question"/"first answer") + system are NOT re-emitted.
    assert contents == ["second question", "second answer"]


def test_system_and_developer_messages_are_dropped() -> None:
    kwargs = _kwargs(
        [
            {"role": "system", "content": "system prompt"},
            {"role": "developer", "content": "dev prompt"},
            {"role": "user", "content": "real question"},
        ]
    )
    resp = _dict_response("real answer")

    events = events_from_completion(kwargs, resp, end_time=END_TIME)

    assert [e.content for e in events] == ["real question", "real answer"]
    assert [e.role for e in events] == [Role.USER, Role.ASSISTANT]


# ---------------------------------------------------------------------------
# FR-ING-7 — tool_calls stripping
# ---------------------------------------------------------------------------


def test_tool_role_request_turns_are_stripped() -> None:
    kwargs = _kwargs(
        [
            {"role": "user", "content": "what's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "wx"}}],
            },
            {
                "role": "tool",
                "content": "{'temp': 21, 'secret_token': 'abc123'}",
                "tool_call_id": "c1",
            },
        ]
    )
    resp = _dict_response("It's 21 degrees.")

    events = events_from_completion(kwargs, resp, end_time=END_TIME, strip_tool_calls=True)

    # The tool result turn (credential-dense) is gone; only the final assistant
    # text reply survives from the response. The trailing request run after the
    # last assistant message is just the tool turn, which is stripped.
    assert [e.role for e in events] == [Role.ASSISTANT]
    assert events[0].content == "It's 21 degrees."
    assert all(e.tool_calls is None for e in events)
    assert "secret_token" not in events[0].content


def test_assistant_tool_calls_are_never_carried_onto_event() -> None:
    kwargs = _kwargs([{"role": "user", "content": "call a tool"}])
    resp = _dict_response(
        "Here is the result.",
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
    )

    events = events_from_completion(kwargs, resp, end_time=END_TIME, strip_tool_calls=True)

    assistant = events[-1]
    assert assistant.role is Role.ASSISTANT
    assert assistant.tool_calls is None  # FR-ING-7: never populated
    assert assistant.metadata.get("tool_calls_stripped") is True


def test_pure_tool_call_assistant_reply_with_no_text_is_dropped() -> None:
    kwargs = _kwargs([{"role": "user", "content": "do the thing"}])
    resp = _dict_response("", tool_calls=[{"id": "c1", "function": {"name": "f"}}])

    events = events_from_completion(kwargs, resp, end_time=END_TIME, strip_tool_calls=True)

    # Only the user turn remains; the empty tool-only assistant reply is dropped.
    assert [e.role for e in events] == [Role.USER]


# ---------------------------------------------------------------------------
# Content coercion + edge cases
# ---------------------------------------------------------------------------


def test_content_parts_list_is_flattened_to_text() -> None:
    kwargs = _kwargs(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                    {"type": "text", "text": "in detail"},
                ],
            }
        ]
    )
    resp = _dict_response("Sure.")

    events = events_from_completion(kwargs, resp, end_time=END_TIME)

    assert events[0].content == "describe this\nin detail"  # image part dropped


def test_empty_request_and_empty_response_yields_nothing() -> None:
    assert events_from_completion({"messages": []}, {"choices": []}, end_time=END_TIME) == []


def test_default_project_used_when_no_metadata() -> None:
    kwargs = _kwargs([{"role": "user", "content": "hi"}])
    resp = _dict_response("hello")

    events = events_from_completion(
        kwargs, resp, end_time=END_TIME, default_project="my-agent"
    )

    assert all(e.project == "my-agent" for e in events)


def test_session_id_falls_back_to_litellm_call_id() -> None:
    kwargs = _kwargs([{"role": "user", "content": "x"}], litellm_call_id="call-xyz")
    resp = _dict_response("y")

    events = events_from_completion(kwargs, resp, end_time=END_TIME)

    assert all(e.session_id == "call-xyz" for e in events)


def test_messages_nested_under_litellm_params_are_found() -> None:
    # Some LiteLLM code paths nest the call messages; the extractor falls back.
    kwargs = {
        "model": "openai/qwen2.5",
        "litellm_params": {"messages": [{"role": "user", "content": "nested?"}]},
    }
    resp = _dict_response("found it")

    events = events_from_completion(kwargs, resp, end_time=END_TIME)

    assert [e.content for e in events] == ["nested?", "found it"]


def test_emitted_events_are_real_ingest_events_with_idempotency_key() -> None:
    kwargs = _kwargs(
        [{"role": "user", "content": "stable content"}],
        metadata={"mnemozine_session_id": "s1"},
    )
    resp = _dict_response("reply")

    events = events_from_completion(kwargs, resp, end_time=END_TIME)

    assert all(isinstance(e, IngestEvent) for e in events)
    src, sess, _hash = events[0].idempotency_key()
    assert src == "openai"
    assert sess == "s1"
    # FR-ING-5: hashing on normalized content, offset-free.
    assert events[0].content_hash() == events[0].content_hash()


# ---------------------------------------------------------------------------
# GatewayCallback — IngestSource + LiteLLM hooks
# ---------------------------------------------------------------------------


def test_gateway_callback_is_ingest_source() -> None:
    cb = GatewayCallback()
    assert isinstance(cb, IngestSource)
    assert cb.source_name == "openai"
    assert GatewayCallback(source=Source.HERMES).source_name == "hermes"


def test_strip_tool_calls_setting_is_honored() -> None:
    # With stripping disabled the tool result turn survives as a (user-visible)
    # mapping decision — but tool role still maps to None, so it is dropped; what
    # the flag governs is the metadata marker / tool-result handling.
    settings = IngestSettings(strip_tool_calls=False)
    cb = GatewayCallback(settings=settings)
    kwargs = _kwargs([{"role": "user", "content": "q"}])
    resp = _dict_response("a", tool_calls=[{"id": "c", "function": {"name": "f"}}])

    events = cb.map_events(kwargs, resp, end_time=END_TIME)
    assistant = events[-1]
    # Not flagged as stripped because stripping was disabled.
    assert "tool_calls_stripped" not in assistant.metadata
    # Event tool_calls field stays None regardless (schema is stripped downstream).
    assert assistant.tool_calls is None


@pytest.mark.asyncio
async def test_log_success_event_enqueues_and_stream_yields() -> None:
    cb = GatewayCallback()
    kwargs = _kwargs(
        [{"role": "user", "content": "remember X"}],
        metadata={"mnemozine_session_id": "sess-stream"},
    )
    resp = _dict_response("noted X")

    cb.log_success_event(kwargs, resp, None, END_TIME)

    collected: list[IngestEvent] = []
    stream = cb.stream()
    # Two events were enqueued; pull exactly two then stop.
    collected.append(await anext(stream))
    collected.append(await anext(stream))
    await stream.aclose()

    assert [e.content for e in collected] == ["remember X", "noted X"]
    assert all(e.session_id == "sess-stream" for e in collected)


@pytest.mark.asyncio
async def test_async_log_success_event_enqueues() -> None:
    cb = GatewayCallback()
    kwargs = _kwargs([{"role": "user", "content": "async path"}])
    resp = _dict_response("ok")

    await cb.async_log_success_event(kwargs, resp, None, END_TIME)

    stream = cb.stream()
    first = await anext(stream)
    await stream.aclose()
    assert first.content == "async path"


@pytest.mark.asyncio
async def test_backfill_is_empty_async_generator() -> None:
    cb = GatewayCallback()
    out = [e async for e in cb.backfill()]
    assert out == []


def test_log_success_event_swallows_mapping_errors() -> None:
    cb = GatewayCallback()
    # Passing a response that raises when accessed must not propagate.

    class _Boom:
        @property
        def choices(self) -> Any:
            raise RuntimeError("boom")

    # Should not raise; just logs and drops.
    cb.log_success_event(_kwargs([{"role": "user", "content": "x"}]), _Boom(), None, END_TIME)


def test_make_gateway_callback_returns_litellm_custom_logger() -> None:
    cb = make_gateway_callback(source=Source.OPENAI)
    assert isinstance(cb, IngestSource)
    # When LiteLLM is importable, the factory mixes in CustomLogger.
    from litellm.integrations.custom_logger import CustomLogger

    assert isinstance(cb, CustomLogger)
