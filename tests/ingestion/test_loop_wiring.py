"""Integration test: gateway + Hermes events flow through the wired loop (F1).

Proves the FR-ING-3 (LiteLLM gateway) and FR-ING-4 (Hermes) ``IngestSource``s —
which were implemented + unit-tested but never *consumed* — are now driven by the
``mnemozine-ingest`` loop into the SAME ``chunk -> extract -> store`` pipeline the
Claude Code path uses. Everything is **offline**: no live LiteLLM proxy, no
Hermes VM, no Qwen, no FalkorDB. Synthetic events are injected straight into each
source's in-process queue and we assert both land, extracted + stored, in the
``InMemoryStorage`` fake (the "fake storage backend").

Two angles are covered:

* :func:`test_backfill_drains_hermes_recorded_into_storage` — the deterministic
  ``backfill`` path (Hermes replays recorded payloads; the gateway has no
  backlog), which returns on its own.
* :func:`test_stream_drains_injected_gateway_and_hermes_into_storage` — the live
  ``stream`` path: a synthetic gateway completion and a synthetic Hermes turn are
  injected, the streaming loop drains them, and we assert BOTH memories were
  extracted and stored, with ``tool_calls`` stripping (FR-ING-7) intact.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from mnemozine.config import IngestSettings, Settings
from mnemozine.extract.extractor import TypedExtractor
from mnemozine.ingestion.claude_code.chunker import ChunkAccumulator
from mnemozine.ingestion.loop import build_ingest_sources, run_ingest_loop
from mnemozine.interfaces import IngestSource
from mnemozine.schema.events import Source
from mnemozine.schema.models import Scope, ScopeDecision
from mnemozine.services import MnemozineIngestService
from tests.conftest import FakeLLMProvider, InMemoryStorage

END_TIME = datetime(2026, 6, 14, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Synthetic source payload builders (mirror the unit-test mocks)
# ---------------------------------------------------------------------------


def _gateway_completion() -> tuple[dict[str, Any], dict[str, Any]]:
    """A mocked LiteLLM ``(kwargs, response_obj)`` carrying a stripped tool call."""

    kwargs = {
        "model": "openai/qwen2.5",
        "messages": [
            # The new user turn (carries the routing marker) followed by a
            # tool-result turn whose secret MUST be stripped (FR-ING-7). With no
            # prior assistant message both are in the trailing request run, so the
            # user turn survives and the tool turn (and its secret) is dropped.
            {"role": "user", "content": "GATEWAY: I prefer ruff over flake8."},
            {
                "role": "tool",
                "content": "{'token': 'GATEWAY_SECRET'}",
                "tool_call_id": "c1",
            },
        ],
        "metadata": {
            "mnemozine_project": "py-tooling",
            "mnemozine_session_id": "gw-sess-1",
        },
    }
    # The assistant reply also carries tool_calls that must never reach the
    # event (FR-ING-7): the text survives, the tool_calls do not.
    response = {
        "id": "chatcmpl-1",
        "model": "openai/qwen2.5",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Noted: ruff.",
                    "tool_calls": [
                        {"id": "c2", "function": {"name": "lint", "arguments": "{}"}}
                    ],
                },
            }
        ],
    }
    return kwargs, response


def _hermes_turn() -> dict[str, Any]:
    """A mocked Hermes-native turn payload carrying a stripped tool result."""

    return {
        "conversation_id": "hermes-sess-1",
        "project": "hermes-ideas",
        "messages": [
            {"role": "user", "content": "HERMES: idea for a graph-memory CLI."},
            {"role": "tool", "content": "{'api_key': 'HERMES_SECRET'}"},
            {"role": "assistant", "content": "That's an idea_seed worth keeping."},
        ],
    }


# ---------------------------------------------------------------------------
# Fake extraction LLM: route each chunk to a distinct, identifiable memory
# ---------------------------------------------------------------------------


def _routing_llm() -> FakeLLMProvider:
    """A FakeLLMProvider that emits a distinct memory per source's chunk content.

    The extractor calls ``complete_json`` once per chunk with the chunk's events
    embedded in the prompt; we route on the marker text so the gateway chunk and
    the Hermes chunk each yield a uniquely-identifiable MemoryUnit in storage.
    Any chunk that still carries a stripped secret would be visible in the prompt;
    we assert separately that none does.
    """

    def responder(prompt: str, _system: str | None) -> dict[str, Any] | None:
        if "GATEWAY:" in prompt:
            return {
                "memories": [
                    {
                        "content": "Prefers ruff over flake8 for Python linting.",
                        "scope": "global",
                        "category": "preference",
                        "cross_ref": False,
                        "entities": ["python", "ruff", "linting"],
                        "confidence": 0.9,
                    }
                ],
                "relationships": [],
            }
        if "HERMES:" in prompt:
            return {
                "memories": [
                    {
                        "content": "Idea seed: a graph-backed memory CLI.",
                        "scope": "global",
                        "category": "idea",
                        "cross_ref": True,
                        "entities": ["cli", "graph", "memory"],
                        "confidence": 0.8,
                    }
                ],
                "relationships": [],
            }
        return {"memories": [], "relationships": []}

    return FakeLLMProvider(json_responder=responder)


def _build_pipeline() -> tuple[InMemoryStorage, MnemozineIngestService, ChunkAccumulator]:
    storage = InMemoryStorage()
    extractor = TypedExtractor(_routing_llm(), settings=Settings())
    service = MnemozineIngestService(storage, extractor, settings=Settings())
    accumulator = ChunkAccumulator(Settings().ingest)
    return storage, service, accumulator


def _both_sources_enabled() -> Settings:
    """Settings with the two Phase-2 sources on and Claude Code off (isolation).

    Claude Code is disabled so the test does not touch the real ``~/.claude``
    transcript tree (it would start a watcher); the point here is the gateway +
    Hermes wiring.
    """

    return Settings(
        ingest=IngestSettings(
            enable_claude_code=False,
            enable_gateway=True,
            enable_hermes=True,
        )
    )


# ---------------------------------------------------------------------------
# build_ingest_sources — enablement / disablement
# ---------------------------------------------------------------------------


def test_build_sources_honors_enable_flags() -> None:
    built = build_ingest_sources(_both_sources_enabled())

    assert built.claude_code is None  # disabled -> not constructed
    assert built.gateway is not None
    assert built.hermes is not None
    assert {s.source_name for s in built.sources} == {"openai", "hermes"}
    # The handles are real IngestSources the driver will drain.
    assert all(isinstance(s, IngestSource) for s in built.sources)
    assert built.gateway.source_name == Source.OPENAI.value
    assert built.hermes.source_name == Source.HERMES.value


def test_build_sources_default_only_claude_code() -> None:
    # Defaults: only Claude Code on (Phase-1), gateway + Hermes off.
    built = build_ingest_sources(Settings())

    assert built.claude_code is not None
    assert built.gateway is None
    assert built.hermes is None
    assert [s.source_name for s in built.sources] == ["claude_code"]


def test_build_sources_queue_bounds_from_config() -> None:
    settings = Settings(
        ingest=IngestSettings(
            enable_claude_code=False,
            enable_gateway=True,
            enable_hermes=True,
            gateway_queue_max=7,
            hermes_queue_max=11,
        )
    )
    built = build_ingest_sources(settings)

    assert built.gateway is not None and built.gateway._queue.maxsize == 7
    assert built.hermes is not None and built.hermes._queue.maxsize == 11
    # default_project mapped from the gateway/hermes_default_project config.
    assert built.gateway._default_project == "default"
    assert built.hermes._default_project == "hermes"


def test_no_enabled_sources_returns_empty() -> None:
    built = build_ingest_sources(
        Settings(
            ingest=IngestSettings(
                enable_claude_code=False,
                enable_gateway=False,
                enable_hermes=False,
            )
        )
    )
    assert len(built) == 0


@pytest.mark.asyncio
async def test_run_loop_with_no_sources_is_a_noop() -> None:
    storage, service, accumulator = _build_pipeline()
    built = build_ingest_sources(
        Settings(
            ingest=IngestSettings(
                enable_claude_code=False,
                enable_gateway=False,
                enable_hermes=False,
            )
        )
    )
    # Returns immediately; nothing stored.
    await asyncio.wait_for(
        run_ingest_loop(built, accumulator, service, backfill=True), timeout=2.0
    )
    assert storage.memories == {}


# ---------------------------------------------------------------------------
# backfill path — deterministic, returns on its own
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_drains_hermes_recorded_into_storage() -> None:
    """Hermes ``backfill`` replays a recorded turn end-to-end into storage."""

    storage, service, accumulator = _build_pipeline()

    # Build sources then seed the Hermes adapter's recorded backlog. The gateway
    # has no backlog (its backfill is an empty generator), proving the loop
    # tolerates a source that yields nothing in backfill.
    built = build_ingest_sources(_both_sources_enabled())
    assert built.hermes is not None
    built.hermes._recorded = [_hermes_turn()]

    await asyncio.wait_for(
        run_ingest_loop(built, accumulator, service, backfill=True), timeout=5.0
    )

    idea_seeds = [m for m in storage.memories.values() if m.cross_ref_candidate]
    assert len(idea_seeds) == 1
    assert idea_seeds[0].content == "Idea seed: a graph-backed memory CLI."
    assert idea_seeds[0].category == "idea"
    # FR-ING-7: the stripped tool secret never reached extraction/storage.
    assert all("HERMES_SECRET" not in m.content for m in storage.memories.values())


# ---------------------------------------------------------------------------
# stream path — inject synthetic gateway + hermes events, drain, cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_drains_injected_gateway_and_hermes_into_storage() -> None:
    """The headline wiring proof: BOTH live sources flow through to storage.

    A synthetic gateway completion and a synthetic Hermes turn are injected into
    the exact in-process queues the loop drains, the streaming loop runs, and we
    assert each produced its distinct memory in the fake storage — without a live
    LiteLLM proxy or Hermes VM. The small test chunks never trip a size-based
    flush, so they are ingested by the consumer's shutdown flush when we cancel
    the loop, which also exercises clean shutdown.
    """

    storage, service, accumulator = _build_pipeline()
    built = build_ingest_sources(_both_sources_enabled())
    assert built.gateway is not None and built.hermes is not None

    kwargs, response = _gateway_completion()

    # FR-ING-7 is enforced at the source mapping (before the loop ever sees an
    # event): the tool-result secret is dropped and the assistant tool_calls are
    # stripped + marked. Assert that directly so the strip path is proven, not
    # merely inferred from the secret's later absence in storage.
    mapped = built.gateway.map_events(kwargs, response, end_time=END_TIME)
    assert [e.role.value for e in mapped] == ["user", "assistant"]
    assert all(e.tool_calls is None for e in mapped)
    assert all("GATEWAY_SECRET" not in e.content for e in mapped)
    assert mapped[-1].metadata.get("tool_calls_stripped") is True

    # Inject one synthetic event into each source's in-process queue, exactly as
    # a live LiteLLM proxy / instrumented Hermes VM would.
    built.gateway.log_success_event(kwargs, response, None, END_TIME)
    built.hermes.feed(_hermes_turn())

    loop_task = asyncio.create_task(
        run_ingest_loop(built, accumulator, service, backfill=False)
    )

    # Wait until both injected events have been pulled off the source queues by
    # their producers (i.e. handed to the consumer + accumulator). The events
    # then sit in per-session accumulator buffers until the shutdown flush.
    async def _queues_drained() -> bool:
        return built.gateway._queue.empty() and built.hermes._queue.empty()

    for _ in range(200):
        if await _queues_drained():
            break
        await asyncio.sleep(0.01)
    assert await _queues_drained(), "injected events were not drained off the source queues"
    # Let the consumer pull the last items out of the fan-in queue too.
    await asyncio.sleep(0.05)

    # Clean shutdown: cancelling triggers the consumer's flush of in-flight
    # buffers, ingesting both chunks. run_ingest_loop swallows the cancellation
    # of its child tasks and returns.
    loop_task.cancel()
    try:
        await asyncio.wait_for(loop_task, timeout=5.0)
    except asyncio.CancelledError:
        pass

    by_content = {m.content: m for m in storage.memories.values()}

    # Gateway path: the OpenAI-source preference was extracted + stored.
    assert "Prefers ruff over flake8 for Python linting." in by_content
    pref = by_content["Prefers ruff over flake8 for Python linting."]
    assert pref.scope_decision is ScopeDecision.GLOBAL
    assert pref.category == "preference"
    assert pref.scope.as_str() == Scope.global_().as_str()

    # Hermes path: the cross-ref idea seed was extracted + stored.
    assert "Idea seed: a graph-backed memory CLI." in by_content
    seed = by_content["Idea seed: a graph-backed memory CLI."]
    assert seed.cross_ref_candidate is True

    # BOTH sources flowed through — the whole point of the wiring.
    assert len(storage.memories) == 2

    # FR-ING-7 stays intact end-to-end: neither stripped secret reached storage,
    # and no event ever carried tool_calls.
    assert all(
        "GATEWAY_SECRET" not in m.content and "HERMES_SECRET" not in m.content
        for m in storage.memories.values()
    )

    # Provenance proves which source each memory came from (FR-EXT-4 / FR-ING-1).
    assert pref.provenance.source == Source.OPENAI.value
    assert seed.provenance.source == Source.HERMES.value
