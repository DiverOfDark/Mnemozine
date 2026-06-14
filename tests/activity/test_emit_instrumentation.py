"""Pipeline emit-instrumentation tests (WEBUI Q3).

Asserts that the four pipeline seams record :class:`ActivityEvent`s through the
safe :func:`mnemozine.activity.emit` seam **when** an activity log is wired, and
that with the default (no log / ``NullActivityLog``) nothing is recorded and the
pipeline behaves exactly as before — the contract that keeps the existing 442
tests green.

Seams under test:

* **ingestion** — ``ingestion.loop`` records one ``ingest`` event per newly
  ingested chunk, tagged with the source.
* **extraction** — ``maintenance.decision.WriteDecider`` records one
  ``extract_decision`` event carrying the 4-way add/reinforce/supersede/no-op
  decision.
* **maintenance** — ``maintenance.runner.MaintenanceRunner`` records one
  ``maintenance`` event per job run with the report counts.
* **retrieval** — ``retrieval.retriever.ScopedRetriever.build_index`` records one
  ``injection`` event for the surfaced index.

The :func:`emit` seam schedules the append on the running loop, so each test
drains pending tasks with ``await asyncio.sleep(0)`` before asserting.
"""

from __future__ import annotations

import asyncio

from mnemozine.activity import (
    ActivityKind,
    InMemoryActivityLog,
    NullActivityLog,
)
from mnemozine.config import Settings
from mnemozine.ingestion.claude_code.chunker import ChunkAccumulator
from mnemozine.ingestion.loop import run_ingest_loop
from mnemozine.interfaces import (
    MaintenanceReport,
    RetrievalContext,
    WriteDecision,
)
from mnemozine.maintenance.decision import WriteDecider
from mnemozine.maintenance.runner import MaintenanceRunner
from mnemozine.retrieval.retriever import ScopedRetriever
from mnemozine.schema.events import IngestEvent, Role, Source
from mnemozine.schema.models import MemoryUnit, Provenance, Scope
from tests.conftest import FakeEmbeddingProvider, FakeLLMProvider, InMemoryStorage


async def _drain() -> None:
    """Let emit()'s scheduled append tasks run to completion before asserting."""

    for _ in range(3):
        await asyncio.sleep(0)


def _pref(content: str, *, entities: list[str], confidence: float = 0.9) -> MemoryUnit:
    return MemoryUnit(
        category="preference",
        content=content,
        scope=Scope.global_(),
        entities=entities,
        confidence=confidence,
        provenance=Provenance(source="claude_code", session_id="s1"),
    )


# ---------------------------------------------------------------------------
# extraction — the 4-way write decision (WriteDecider)
# ---------------------------------------------------------------------------


async def test_write_decider_emits_add_decision() -> None:
    log = InMemoryActivityLog()
    storage = InMemoryStorage()
    decider = WriteDecider(
        storage,
        FakeLLMProvider(),
        embeddings=FakeEmbeddingProvider(),
        settings=Settings(),
        activity_log=log,
    )

    result = await decider.decide(_pref("Prefers thiserror.", entities=["rust"]))
    await _drain()

    assert result.decision is WriteDecision.ADD
    events = await log.query()
    assert len(events) == 1
    ev = events[0]
    assert ev.kind is ActivityKind.EXTRACT_DECISION
    assert ev.detail["decision"] == "add"
    assert ev.source == "claude_code"
    assert result.memory.id in ev.ref_memory_ids


async def test_write_decider_emits_supersede_with_superseded_ids() -> None:
    log = InMemoryActivityLog()
    storage = InMemoryStorage()
    # Seed an existing preference that the new one will contradict.
    old = _pref("Prefers anyhow.", entities=["rust", "error-handling"])
    storage.memories[old.id] = old

    # Drive the supersede branch by scripting the contradiction LLM call true.
    llm = FakeLLMProvider(json_responder=lambda p, s: {"contradicts": True, "reason": "reversal"})
    decider = WriteDecider(
        storage, llm, embeddings=FakeEmbeddingProvider(), settings=Settings(), activity_log=log
    )

    new = _pref("Prefers thiserror now.", entities=["rust", "error-handling"])
    result = await decider.decide(new)
    await _drain()

    assert result.decision is WriteDecision.SUPERSEDE
    events = await log.query()
    assert len(events) == 1
    ev = events[0]
    assert ev.detail["decision"] == "supersede"
    # Both the new unit and the superseded one are linked for the Logs screen.
    assert new.id in ev.ref_memory_ids
    assert old.id in ev.ref_memory_ids
    assert old.id in ev.detail["superseded_ids"]


async def test_write_decider_null_log_records_nothing() -> None:
    log = NullActivityLog()
    storage = InMemoryStorage()
    decider = WriteDecider(
        storage,
        FakeLLMProvider(),
        embeddings=FakeEmbeddingProvider(),
        settings=Settings(),
        activity_log=log,
    )

    result = await decider.decide(_pref("Prefers thiserror.", entities=["rust"]))
    await _drain()

    assert result.decision is WriteDecision.ADD
    assert await log.query() == []


async def test_write_decider_default_no_log_is_unchanged() -> None:
    # No activity_log passed at all — the existing call convention. Must still
    # make the decision and not error.
    storage = InMemoryStorage()
    decider = WriteDecider(storage, FakeLLMProvider(), embeddings=FakeEmbeddingProvider())
    result = await decider.decide(_pref("Prefers thiserror.", entities=["rust"]))
    assert result.decision is WriteDecision.ADD


# ---------------------------------------------------------------------------
# maintenance — job runs (MaintenanceRunner)
# ---------------------------------------------------------------------------


class _RecordingJob:
    def __init__(self, name: str, report: MaintenanceReport) -> None:
        self._name = name
        self._report = report

    @property
    def name(self) -> str:
        return self._name

    async def run(self) -> MaintenanceReport:
        return self._report


async def test_maintenance_runner_emits_per_job() -> None:
    log = InMemoryActivityLog()
    jobs = [
        _RecordingJob("consolidate", MaintenanceReport(job_name="consolidate", consolidated=3)),
        _RecordingJob("decay", MaintenanceReport(job_name="decay", archived=2)),
    ]
    runner = MaintenanceRunner(jobs, settings=Settings(), activity_log=log)

    reports = await runner.run_once()
    await _drain()

    assert [r.job_name for r in reports] == ["consolidate", "decay"]
    events = await log.query()
    assert len(events) == 2
    assert {e.kind for e in events} == {ActivityKind.MAINTENANCE}
    by_job = {e.detail["job_name"]: e for e in events}
    assert by_job["consolidate"].detail["consolidated"] == 3
    assert by_job["decay"].detail["archived"] == 2
    assert all(e.source == "maintenance" for e in events)


async def test_maintenance_runner_emits_for_failed_job() -> None:
    log = InMemoryActivityLog()

    class _Exploding:
        name = "boom"

        async def run(self) -> MaintenanceReport:
            raise RuntimeError("kaboom")

    runner = MaintenanceRunner([_Exploding()], settings=Settings(), activity_log=log)
    reports = await runner.run_once()
    await _drain()

    # The runner converts the failure to a report; an event is still recorded.
    assert reports[0].job_name == "boom"
    events = await log.query()
    assert len(events) == 1
    assert events[0].detail["job_name"] == "boom"


async def test_maintenance_runner_default_no_log_is_unchanged() -> None:
    jobs = [_RecordingJob("audit", MaintenanceReport(job_name="audit"))]
    runner = MaintenanceRunner(jobs, settings=Settings())  # no activity_log
    reports = await runner.run_once()
    assert reports[0].job_name == "audit"


# ---------------------------------------------------------------------------
# retrieval — injection surfaced (ScopedRetriever.build_index)
# ---------------------------------------------------------------------------


async def test_build_index_emits_injection() -> None:
    log = InMemoryActivityLog()
    storage = InMemoryStorage()
    # Content lexically overlaps the query so the fake's lexical scoped_query
    # surfaces it (the in-memory fake scores by word overlap).
    pref = _pref("rust error handling: prefer thiserror", entities=["rust", "error-handling"])
    storage.memories[pref.id] = pref

    retriever = ScopedRetriever(storage, settings=Settings(), activity_log=log)
    context = RetrievalContext(
        project="rust-cli",
        scopes=[Scope.global_()],
        entities=["rust"],
        recent_text="rust error handling",
    )
    index = await retriever.build_index(context)
    await _drain()

    events = await log.query()
    assert len(events) == 1
    ev = events[0]
    assert ev.kind is ActivityKind.INJECTION
    assert ev.source == "retrieval"
    assert ev.project == "rust-cli"
    assert ev.detail["global_count"] == index.global_count
    assert pref.id in ev.ref_memory_ids


async def test_build_index_does_not_record_access() -> None:
    # The injection emit is observability only — it must NOT bump access_count
    # (the FR-MNT-3 decay-ranking contract that build_index must not violate).
    log = InMemoryActivityLog()
    storage = InMemoryStorage()
    pref = _pref("Prefers thiserror.", entities=["rust"])
    storage.memories[pref.id] = pref

    retriever = ScopedRetriever(storage, settings=Settings(), activity_log=log)
    context = RetrievalContext(scopes=[Scope.global_()], entities=["rust"], recent_text="rust")
    await retriever.build_index(context)
    await _drain()

    assert pref.access_count == 0
    assert pref.last_accessed is None


async def test_build_index_null_log_records_nothing() -> None:
    log = NullActivityLog()
    storage = InMemoryStorage()
    storage.memories["m1"] = _pref("Prefers thiserror.", entities=["rust"])

    retriever = ScopedRetriever(storage, settings=Settings(), activity_log=log)
    context = RetrievalContext(scopes=[Scope.global_()], entities=["rust"], recent_text="rust")
    await retriever.build_index(context)
    await _drain()

    assert await log.query() == []


async def test_build_index_default_no_log_is_unchanged() -> None:
    storage = InMemoryStorage()
    storage.memories["m1"] = _pref("rust prefer thiserror", entities=["rust"])
    retriever = ScopedRetriever(storage, settings=Settings())  # no activity_log
    context = RetrievalContext(scopes=[Scope.global_()], entities=["rust"], recent_text="rust")
    index = await retriever.build_index(context)
    assert index.global_count == 1


# ---------------------------------------------------------------------------
# ingestion — chunk ingested, per source (ingestion.loop)
# ---------------------------------------------------------------------------


class _FakeIngestService:
    """Minimal ingest service: records ingested chunks, reports newly-ingested."""

    def __init__(self, *, already_seen: set[str] | None = None) -> None:
        self.ingested: list[str] = []
        self._seen = already_seen or set()

    async def ingest_chunk(self, chunk) -> bool:  # type: ignore[no-untyped-def]
        self.ingested.append(chunk.content_hash)
        if chunk.content_hash in self._seen:
            return False
        self._seen.add(chunk.content_hash)
        return True


def _events(session: str, *, source: Source = Source.CLAUDE_CODE) -> list[IngestEvent]:
    from datetime import UTC, datetime

    base = datetime(2026, 6, 14, 10, 0, 0, tzinfo=UTC)
    return [
        IngestEvent(
            source=source,
            project="rust-cli",
            session_id=session,
            timestamp=base,
            role=Role.USER,
            content="I prefer thiserror over anyhow for error handling in Rust.",
        ),
    ]


async def test_ingest_loop_emits_per_chunk() -> None:
    log = InMemoryActivityLog()
    service = _FakeIngestService()
    accumulator = ChunkAccumulator(Settings().ingest)

    # A finite backfill source that yields one session's events then completes.
    class _Source:
        source_name = "claude_code"

        def stream(self):  # pragma: no cover - not used in backfill
            raise NotImplementedError

        async def _gen(self):
            for e in _events("sess-1"):
                yield e

        def backfill(self, *, since=None):  # type: ignore[no-untyped-def]
            return self._gen()

    await run_ingest_loop(
        [_Source()], accumulator, service, backfill=True, activity_log=log
    )
    await _drain()

    assert service.ingested  # at least one chunk flushed
    events = await log.query()
    assert events, "expected at least one ingest activity event"
    assert all(e.kind is ActivityKind.INGEST for e in events)
    assert all(e.source == "claude_code" for e in events)
    assert events[0].session_id == "sess-1"
    assert events[0].project == "rust-cli"


async def test_ingest_loop_skips_event_for_already_seen_chunk() -> None:
    # An FR-ING-5 idempotent skip (ingest_chunk returns False) must NOT emit.
    log = InMemoryActivityLog()
    accumulator = ChunkAccumulator(Settings().ingest)

    # Pre-seed the service so its single flushed chunk is treated as already seen.
    # Compute the chunk hash the accumulator will produce for this session.
    from mnemozine.schema.events import chunk_content_hash

    evs = _events("sess-1")
    seen_hash = chunk_content_hash(evs)
    service = _FakeIngestService(already_seen={seen_hash})

    class _Source:
        source_name = "claude_code"

        def stream(self):  # pragma: no cover
            raise NotImplementedError

        async def _gen(self):
            for e in evs:
                yield e

        def backfill(self, *, since=None):  # type: ignore[no-untyped-def]
            return self._gen()

    await run_ingest_loop(
        [_Source()], accumulator, service, backfill=True, activity_log=log
    )
    await _drain()

    assert service.ingested  # the chunk was processed
    assert await log.query() == []  # but no event, because it was an idempotent skip


async def test_ingest_loop_default_no_log_is_unchanged() -> None:
    service = _FakeIngestService()
    accumulator = ChunkAccumulator(Settings().ingest)

    class _Source:
        source_name = "claude_code"

        def stream(self):  # pragma: no cover
            raise NotImplementedError

        async def _gen(self):
            for e in _events("sess-1"):
                yield e

        def backfill(self, *, since=None):  # type: ignore[no-untyped-def]
            return self._gen()

    # No activity_log passed — existing call convention.
    await run_ingest_loop([_Source()], accumulator, service, backfill=True)
    assert service.ingested
