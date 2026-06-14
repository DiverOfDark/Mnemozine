"""Multi-source ingest driver: enabled sources -> chunk -> extract -> store.

This module is the FR-ING-2/3/4 fan-in for ``mnemozine-ingest``. Phase 1 wired
only :class:`~mnemozine.ingestion.claude_code.source.ClaudeCodeSource`; the
gateway (FR-ING-3) and Hermes (FR-ING-4) :class:`~mnemozine.interfaces.IngestSource`
implementations existed and were unit-tested but were never *consumed*. This
module builds the enabled set from the ``ingest.enable_*`` config flags and runs
them **concurrently** into the single shared
``ChunkAccumulator -> MnemozineIngestService`` pipeline the Claude Code path
already used.

Concurrency model
-----------------
The :class:`~mnemozine.ingestion.claude_code.chunker.ChunkAccumulator` and the
:class:`~mnemozine.services.MnemozineIngestService` are **not** safe to drive
from two coroutines at once (the accumulator mutates per-session buffers; the
service awaits storage between mutations of its ``_seen`` set). So this driver
uses a **fan-in**: one producer task per source pumps that source's events onto
a shared :class:`asyncio.Queue`, and a *single* consumer task drains the queue
through the accumulator + ingest service. Sources therefore run concurrently
(their I/O — watchfiles tailing, queue waits — overlaps) while all chunking and
storage writes are serialized through the one consumer, so the accumulator's
per-``session_id`` buffering stays correct even when events from different
sources interleave (chunks are keyed on ``session_id``, which is unique per
source session).

``tool_calls`` stripping (FR-ING-7) is unchanged: it happens inside each source's
event mapping before an event ever reaches this driver.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mnemozine.activity import emit, ingest_event
from mnemozine.config import Settings, get_settings
from mnemozine.ingestion.claude_code.chunker import ChunkAccumulator
from mnemozine.ingestion.claude_code.source import ClaudeCodeSource
from mnemozine.ingestion.gateway.callback import GatewayCallback, make_gateway_callback
from mnemozine.ingestion.hermes.adapter import HermesAdapter
from mnemozine.interfaces import ActivityLog, IngestSource
from mnemozine.schema.events import IngestEvent, Source

if TYPE_CHECKING:  # pragma: no cover
    from mnemozine.ingestion.claude_code.chunker import Chunk

logger = logging.getLogger(__name__)

# Sentinel pushed onto the fan-in queue when a producer finishes (backfill mode)
# so the consumer can tell "no event right now" from "this producer is done".
_PRODUCER_DONE = object()


@runtime_checkable
class IngestService(Protocol):
    """Minimal duck-type the driver needs from the ingest service.

    The concrete implementation is
    :class:`mnemozine.services.MnemozineIngestService`; this Protocol keeps the
    ingestion package from importing the cross-layer ``services`` module, while
    still giving the driver a typed handle on the one method it calls.
    """

    async def ingest_chunk(self, chunk: Chunk) -> bool:
        """Extract + persist one chunk; True if newly ingested (FR-ING-5)."""
        ...


@dataclass(slots=True)
class IngestSources:
    """The set of enabled :class:`~mnemozine.interfaces.IngestSource`s for a run.

    Built by :func:`build_ingest_sources` from the ``ingest.enable_*`` flags. The
    ``sources`` list is what the driver iterates; the concrete ``gateway`` /
    ``hermes`` handles are also surfaced so an in-process producer (the LiteLLM
    callback registration, or a test injecting a synthetic event) can reach the
    exact instance the driver is draining.
    """

    sources: list[IngestSource] = field(default_factory=list)
    claude_code: ClaudeCodeSource | None = None
    gateway: GatewayCallback | None = None
    hermes: HermesAdapter | None = None

    def __len__(self) -> int:
        return len(self.sources)


def build_ingest_sources(settings: Settings | None = None) -> IngestSources:
    """Construct the enabled ingest sources from config (FR-ING-2/3/4).

    Honors ``ingest.enable_claude_code`` / ``enable_gateway`` / ``enable_hermes``
    — a disabled source is never constructed, so a fresh install with the Phase-2
    sources off needs neither a LiteLLM proxy nor a Hermes VM. The gateway
    callback is built via :func:`~mnemozine.ingestion.gateway.callback.make_gateway_callback`
    so it subclasses LiteLLM's ``CustomLogger`` when LiteLLM is importable (and is
    the same instance a proxy would register), mapping the
    ``gateway_default_project`` / ``gateway_queue_max`` config onto its ctor; the
    Hermes adapter maps ``hermes_default_project`` / ``hermes_queue_max``.
    """

    settings = settings or get_settings()
    ingest = settings.ingest
    built = IngestSources()

    if ingest.enable_claude_code:
        built.claude_code = ClaudeCodeSource(settings)
        built.sources.append(built.claude_code)

    if ingest.enable_gateway:
        gateway = make_gateway_callback(
            source=Source.OPENAI,
            settings=ingest,
            default_project=ingest.gateway_default_project,
        )
        # make_gateway_callback does not thread max_queue through (it keeps the
        # CustomLogger-subclassing path simple); rebuild the queue at the
        # configured bound so gateway_queue_max is honored.
        gateway._queue = asyncio.Queue(maxsize=ingest.gateway_queue_max)
        built.gateway = gateway
        built.sources.append(gateway)

    if ingest.enable_hermes:
        hermes = HermesAdapter(
            settings=ingest,
            default_project=ingest.hermes_default_project,
            max_queue=ingest.hermes_queue_max,
        )
        built.hermes = hermes
        built.sources.append(hermes)

    return built


def _events_iter(source: IngestSource, *, backfill: bool) -> AsyncIterator[IngestEvent]:
    """Return the event stream for one source (backfill replays then completes)."""

    return source.backfill() if backfill else source.stream()


async def _produce(
    source: IngestSource,
    queue: asyncio.Queue[object],
    *,
    backfill: bool,
) -> None:
    """Pump one source's events onto the shared fan-in queue.

    Runs as its own task so every enabled source streams concurrently. On
    ``backfill`` the source's generator completes after replaying its backlog and
    a ``_PRODUCER_DONE`` sentinel is enqueued; ``stream`` sources run until the
    task is cancelled at shutdown. A failure in one source is logged and that
    producer ends (sentinel enqueued) without taking the others down.
    """

    name = source.source_name
    try:
        async for event in _events_iter(source, backfill=backfill):
            await queue.put(event)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - one source must not crash the whole loop
        logger.exception("ingest source %s failed; stopping that producer", name)
    finally:
        await queue.put(_PRODUCER_DONE)


async def _ingest_one(
    chunk: Chunk,
    ingest_service: IngestService,
    activity_log: ActivityLog | None,
) -> None:
    """Ingest one chunk and (WEBUI Q3) record the ingestion on the activity feed.

    The :func:`emit` seam is null-safe + error-swallowing, so when no activity log
    is wired (the default ``None`` / ``NullActivityLog``) nothing is recorded and
    the ingest path is unchanged. Only chunks that were *newly* ingested
    (``ingest_chunk`` returned True — not an FR-ING-5 idempotent skip) are
    recorded, so the feed reflects real ingestion, not re-flushes.
    """

    newly = await ingest_service.ingest_chunk(chunk)
    if newly:
        emit(
            activity_log,
            ingest_event(
                source=chunk.source,
                session_id=chunk.session_id,
                project=chunk.project,
                summary=(
                    f"ingested chunk from {chunk.source} "
                    f"({len(chunk.events)} event(s), session {chunk.session_id})"
                ),
                detail={
                    "content_hash": chunk.content_hash,
                    "event_count": len(chunk.events),
                    "char_count": chunk.char_count,
                },
            ),
        )


async def _consume(
    queue: asyncio.Queue[object],
    accumulator: ChunkAccumulator,
    ingest_service: IngestService,
    *,
    producer_count: int,
    drain_to_completion: bool,
    activity_log: ActivityLog | None = None,
) -> None:
    """Single consumer: drain the fan-in queue through chunk -> extract -> store.

    Serializes all accumulator + ingest access (see module docstring). Each event
    is fed to the accumulator; any completed chunks are ingested immediately. In
    ``drain_to_completion`` mode (backfill) the consumer stops once every producer
    has signalled done and then flushes the accumulator remainder; otherwise it
    runs until cancelled (the streaming watcher loop), and a final flush still
    runs in the cancellation handler so in-flight buffers are not lost.

    ``activity_log`` is the optional WEBUI Q3 observability seam: when wired, each
    newly-ingested chunk is recorded on the feed (per source). Defaults to None so
    the existing ingest path is unaffected.
    """

    finished = 0
    try:
        while True:
            item = await queue.get()
            if item is _PRODUCER_DONE:
                finished += 1
                if drain_to_completion and finished >= producer_count:
                    break
                continue
            assert isinstance(item, IngestEvent)  # narrows the object queue
            for chunk in accumulator.add(item):
                await _ingest_one(chunk, ingest_service, activity_log)
    finally:
        # Flush any in-flight remainder so the last partial chunk per session is
        # not dropped (always for backfill; on cancellation for the stream loop).
        for chunk in accumulator.flush():
            await _ingest_one(chunk, ingest_service, activity_log)


async def run_ingest_loop(
    sources: IngestSources | Sequence[IngestSource],
    accumulator: ChunkAccumulator,
    ingest_service: IngestService,
    *,
    backfill: bool = False,
    activity_log: ActivityLog | None = None,
) -> None:
    """Run the enabled sources concurrently into the shared pipeline (FR-ING-*).

    Starts one producer task per source plus the single serializing consumer.

    * ``backfill=True`` — every source replays its backlog and the call returns
      once all producers have completed and the accumulator has been flushed.
    * ``backfill=False`` — the streaming watcher loop: producers tail their
      sources indefinitely and the consumer runs until cancelled (``KeyboardInterrupt``
      / task cancellation at shutdown), flushing in-flight buffers on the way out.

    With no enabled sources this returns immediately (nothing to drive).

    ``activity_log`` is the optional WEBUI Q3 observability seam threaded into the
    consumer so each newly-ingested chunk is recorded on the activity feed.
    Defaults to None (the existing pipeline behavior is unchanged).
    """

    source_list = list(
        sources.sources if isinstance(sources, IngestSources) else sources
    )
    if not source_list:
        logger.warning("ingest loop started with no enabled sources; nothing to do")
        return

    queue: asyncio.Queue[object] = asyncio.Queue()
    producers = [
        asyncio.create_task(
            _produce(s, queue, backfill=backfill),
            name=f"ingest-src-{s.source_name}",
        )
        for s in source_list
    ]
    consumer = asyncio.create_task(
        _consume(
            queue,
            accumulator,
            ingest_service,
            producer_count=len(producers),
            drain_to_completion=backfill,
            activity_log=activity_log,
        ),
        name="ingest-consumer",
    )

    try:
        if backfill:
            # Consumer returns once all producers are done + flushed; producers
            # finish on their own (finite backlog), so just await the consumer.
            await consumer
            await asyncio.gather(*producers)
        else:
            # Streaming: run until cancelled (Ctrl-C). The consumer never returns
            # on its own here, so await the producers (which also never return)
            # and let cancellation propagate to the finally block.
            await asyncio.gather(*producers)
    finally:
        for task in (*producers, consumer):
            if not task.done():
                task.cancel()
        await asyncio.gather(*producers, consumer, return_exceptions=True)


__all__ = [
    "IngestService",
    "IngestSources",
    "build_ingest_sources",
    "run_ingest_loop",
]
