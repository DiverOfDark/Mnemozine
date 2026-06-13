"""Unit tests for the Claude Code hook entrypoints (deliverable #5).

Covers payload decoding, context derivation, the SessionStart/UserPromptSubmit
injection path (FR-RET-3/5) via a fake Retriever, and the Stop/PreCompact flush
path (FR-ING-6) via a fake IngestService — all offline, no subprocess.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from mnemozine.config import Settings
from mnemozine.ingestion.claude_code.hooks import (
    pre_compact,
    session_start,
    stop,
    user_prompt_submit,
)
from mnemozine.ingestion.claude_code.hooks.runtime import (
    INJECTION_CLOSE,
    INJECTION_OPEN,
    HookPayload,
    HookServices,
    context_from_payload,
    emit_additional_context,
    read_payload,
)
from mnemozine.interfaces import InjectionIndex, RetrievalContext
from mnemozine.schema.models import Scope

FIXTURES = Path(__file__).parent / "fixtures"
RUST_TRANSCRIPT = FIXTURES / "-home-op-Projects-rust-cli" / "sess-rust-1.jsonl"


# --- fakes -----------------------------------------------------------------


class _FakeRetriever:
    """Minimal Retriever returning a canned index; records the context seen."""

    def __init__(self, text: str = "Relevant: 1 preference (rust)") -> None:
        self._text = text
        self.contexts: list[RetrievalContext] = []
        self.access_recorded = False  # build_index must NOT record access

    async def scoped_retrieve(self, query, context, *, top_k=10):  # pragma: no cover
        raise AssertionError("hook should not call scoped_retrieve")

    async def build_index(self, context, *, token_budget=None):
        self.contexts.append(context)
        return InjectionIndex(
            text=self._text,
            token_estimate=len(self._text) // 4,
            preference_count=1,
            entity_tags=["rust"],
        )

    async def recall(self, query, scope=None, *, top_k=10):  # pragma: no cover
        raise AssertionError("hook should not call recall")


class _FakeIngest:
    """Minimal IngestService recording flush calls."""

    def __init__(self, returns: int = 1) -> None:
        self.calls: list[dict[str, object]] = []
        self._returns = returns

    async def flush_session(self, *, session_id, transcript_path, project):
        self.calls.append(
            {
                "session_id": session_id,
                "transcript_path": transcript_path,
                "project": project,
            }
        )
        return self._returns


# --- payload decoding ------------------------------------------------------


def test_read_payload_decodes_stdin_shape() -> None:
    data = {
        "session_id": "sess-rust-1",
        "cwd": "/home/op/Projects/rust-cli",
        "transcript_path": str(RUST_TRANSCRIPT),
        "hook_event_name": "SessionStart",
    }
    payload = read_payload(io.StringIO(json.dumps(data)))
    assert payload.session_id == "sess-rust-1"
    assert payload.cwd == "/home/op/Projects/rust-cli"
    assert payload.hook_event_name == "SessionStart"


def test_read_payload_tolerates_garbage() -> None:
    assert read_payload(io.StringIO("")).session_id is None
    assert read_payload(io.StringIO("not json")).session_id is None
    assert read_payload(io.StringIO("[1,2,3]")).session_id is None


def test_user_prompt_field() -> None:
    payload = read_payload(
        io.StringIO(json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": "do X"}))
    )
    assert payload.prompt == "do X"


# --- context derivation (FR-RET-3) -----------------------------------------


def test_context_from_payload_composes_scopes() -> None:
    payload = HookPayload(
        session_id="sess-rust-1",
        cwd="/home/op/Projects/rust-cli",
        transcript_path=str(RUST_TRANSCRIPT),
    )
    ctx = context_from_payload(payload, Settings())
    assert ctx.project == "rust-cli"
    scope_strs = {s.as_str() for s in ctx.scopes}
    assert scope_strs == {"global", "project:rust-cli"}
    assert Scope.global_() in ctx.scopes
    # recent_text is drawn from the transcript tail.
    assert ctx.recent_text is not None
    assert "All 12 tests passed." in ctx.recent_text


def test_context_includes_prompt_for_user_prompt_submit() -> None:
    payload = HookPayload(
        cwd="/home/op/Projects/rust-cli",
        transcript_path=str(RUST_TRANSCRIPT),
        prompt="What about clap for arg parsing?",
    )
    ctx = context_from_payload(payload, Settings())
    assert "clap for arg parsing" in (ctx.recent_text or "")


# --- injection rendering ---------------------------------------------------


def test_emit_additional_context_shape() -> None:
    out = io.StringIO()
    emit_additional_context(
        f"{INJECTION_OPEN}\nbody\n{INJECTION_CLOSE}",
        hook_event_name="SessionStart",
        stream=out,
    )
    parsed = json.loads(out.getvalue())
    assert parsed["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "body" in parsed["hookSpecificOutput"]["additionalContext"]


def test_emit_additional_context_empty_writes_nothing() -> None:
    out = io.StringIO()
    emit_additional_context("", hook_event_name="SessionStart", stream=out)
    assert out.getvalue() == ""


# --- SessionStart / UserPromptSubmit injection (FR-RET-3/5) ----------------


@pytest.mark.asyncio
async def test_session_start_injects_index() -> None:
    retriever = _FakeRetriever("Relevant: 3 preferences (rust/error-handling)")
    services = HookServices(retriever=retriever, settings=Settings())
    payload = HookPayload(
        session_id="sess-rust-1",
        cwd="/home/op/Projects/rust-cli",
        transcript_path=str(RUST_TRANSCRIPT),
    )
    text = await session_start.run(payload, services)
    assert INJECTION_OPEN in text and INJECTION_CLOSE in text
    assert "3 preferences" in text
    # The retriever saw a composed-scope context.
    assert retriever.contexts
    assert {s.as_str() for s in retriever.contexts[0].scopes} == {
        "global",
        "project:rust-cli",
    }


@pytest.mark.asyncio
async def test_session_start_no_retriever_is_noop() -> None:
    services = HookServices(retriever=None, settings=Settings())
    text = await session_start.run(HookPayload(cwd="/x"), services)
    assert text == ""


@pytest.mark.asyncio
async def test_user_prompt_submit_injects_prompt_scoped() -> None:
    retriever = _FakeRetriever("Relevant: 1 idea (project C)")
    services = HookServices(retriever=retriever, settings=Settings())
    payload = HookPayload(
        cwd="/home/op/Projects/rust-cli",
        transcript_path=str(RUST_TRANSCRIPT),
        prompt="anything about async runtimes?",
    )
    text = await user_prompt_submit.run(payload, services)
    assert "project C" in text
    assert "async runtimes" in (retriever.contexts[0].recent_text or "")


# --- Stop / PreCompact flush (FR-ING-6) ------------------------------------


@pytest.mark.asyncio
async def test_stop_flushes_session() -> None:
    ingest = _FakeIngest(returns=2)
    services = HookServices(ingest=ingest, settings=Settings())
    payload = HookPayload(
        session_id="sess-rust-1",
        cwd="/home/op/Projects/rust-cli",
        transcript_path=str(RUST_TRANSCRIPT),
    )
    n = await stop.run(payload, services)
    assert n == 2
    assert ingest.calls[0]["session_id"] == "sess-rust-1"
    assert ingest.calls[0]["project"] == "rust-cli"
    assert ingest.calls[0]["transcript_path"] == str(RUST_TRANSCRIPT)


@pytest.mark.asyncio
async def test_pre_compact_flushes_session() -> None:
    ingest = _FakeIngest(returns=1)
    services = HookServices(ingest=ingest, settings=Settings())
    payload = HookPayload(
        transcript_path=str(RUST_TRANSCRIPT),  # session_id derived from filename
        cwd="/home/op/Projects/rust-cli",
    )
    n = await pre_compact.run(payload, services)
    assert n == 1
    assert ingest.calls[0]["session_id"] == "sess-rust-1"


@pytest.mark.asyncio
async def test_flush_noop_without_ingest_service() -> None:
    services = HookServices(ingest=None, settings=Settings())
    assert await stop.run(HookPayload(session_id="s"), services) == 0
    assert await pre_compact.run(HookPayload(session_id="s"), services) == 0


@pytest.mark.asyncio
async def test_flush_noop_without_session_id() -> None:
    ingest = _FakeIngest()
    services = HookServices(ingest=ingest, settings=Settings())
    # No session_id and no transcript_path -> cannot resolve a session.
    assert await stop.run(HookPayload(), services) == 0
    assert ingest.calls == []
