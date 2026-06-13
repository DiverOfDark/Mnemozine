"""Composition root + console-script entrypoints.

This module is where the integration pass wires concrete implementations of the
:mod:`mnemozine.interfaces` Protocols together:

* :class:`Container` — the dependency container / composition root. Each
  ``build_*`` method constructs a wired concrete component from :class:`Settings`
  and the already-built lower layers. Pure (config-only) builders are sync;
  :meth:`Container.build_storage` is **async** because it opens the
  Graphiti/FalkorDB connection and builds indices.
* Four console-script entrypoints backing ``[project.scripts]``:
  ``mnemozine-mcp`` (the single MCP server, FR-RET-1), ``mnemozine-ingest`` (the
  source -> chunk -> extract -> store loop, FR-ING-*), ``mnemozine-maintenance``
  (the scheduled consolidate/resolve/decay/audit pass, FR-MNT-*), and
  ``mnemozine-eval`` (the §9 eval harness). Each delegates to its module's real
  entrypoint with the live container wired in.

Layering rule: this is the *only* module that imports across every layer. The
modules themselves code against the Protocols and never import each other's
internals.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import typer

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import (
    CrossReferencer,
    EmbeddingProvider,
    Extractor,
    LLMProvider,
    Retriever,
    StorageBackend,
)

if TYPE_CHECKING:
    from mnemozine.services import MnemozineIngestService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Container:
    """Dependency container / composition root.

    Holds the wired, concrete implementations of each layer Protocol. Builders are
    memoized on the container so a process shares one of each component (one
    storage connection, one LLM/embedding client). :meth:`build_storage` is async
    because constructing the real backend opens the FalkorDB connection and builds
    indices; everything else is pure construction.
    """

    settings: Settings
    _embedding: EmbeddingProvider | None = field(default=None, init=False, repr=False)
    _llm: LLMProvider | None = field(default=None, init=False, repr=False)
    _storage: StorageBackend | None = field(default=None, init=False, repr=False)
    _extractor: Extractor | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(cls) -> Container:
        """Build a container from environment/`.env` configuration."""

        return cls(settings=get_settings())

    # -- providers (pure config; no DI) -----------------------------------

    def build_embedding_provider(self) -> EmbeddingProvider:
        """Wire the bge-m3/Ollama embedding provider (FR-STO-2)."""

        if self._embedding is None:
            from mnemozine.storage import OllamaEmbeddingProvider

            self._embedding = OllamaEmbeddingProvider(self.settings.embedding)
        return self._embedding

    def build_llm_provider(self) -> LLMProvider:
        """Wire the OpenAI-format extraction LLM provider (local Qwen by default)."""

        if self._llm is None:
            from mnemozine.llm import LiteLLMProvider

            self._llm = LiteLLMProvider(self.settings.extraction)
        return self._llm

    # -- storage (async: opens the FalkorDB connection) -------------------

    async def build_storage(self) -> StorageBackend:
        """Wire and connect the Graphiti/FalkorDB storage backend (FR-STO-*).

        Constructs the :class:`GraphitiClient`, opens the connection (lazy-imports
        ``graphiti_core``/``FalkorDriver``, builds Graphiti's native indices and
        the FalkorDB vector index over ``MnemozineMemory.embedding``), then wraps
        it in :class:`GraphitiStorageBackend` with the FR-MNT-1 contradiction
        predicate wired to the LLM provider. Memoized: a second call returns the
        same connected backend.
        """

        if self._storage is None:
            from mnemozine.services import make_contradiction_fn
            from mnemozine.storage import GraphitiClient, GraphitiStorageBackend

            embeddings = self.build_embedding_provider()
            client = GraphitiClient(
                self.settings.falkordb,
                embedding_dimensions=self.settings.embedding.dimensions,
            )
            await client.connect()
            self._storage = GraphitiStorageBackend(
                client,
                embeddings,
                contradicts=make_contradiction_fn(self.build_llm_provider()),
                maintenance=self.settings.maintenance,
            )
        return self._storage

    # -- higher layers (need an already-built, connected storage) ---------

    def build_extractor(self) -> Extractor:
        """Wire the typed extraction pipeline (FR-EXT-*)."""

        if self._extractor is None:
            from mnemozine.extract import TypedExtractor

            self._extractor = TypedExtractor(self.build_llm_provider(), settings=self.settings)
        return self._extractor

    async def build_retriever(self) -> Retriever:
        """Wire the scoped retriever over the connected storage backend (FR-RET-*)."""

        from mnemozine.retrieval import ScopedRetriever

        storage = await self.build_storage()
        return ScopedRetriever(storage, settings=self.settings)

    async def build_cross_referencer(self) -> CrossReferencer:
        """Wire the cross-reference engine (FR-RET-6)."""

        from mnemozine.crossref import CrossReferenceEngine

        storage = await self.build_storage()
        return CrossReferenceEngine(
            storage, self.build_embedding_provider(), settings=self.settings
        )

    def build_ingest_service(self, storage: StorageBackend) -> MnemozineIngestService:
        """Wire the chunk -> extract -> store ingest service (FR-ING-*, FR-EXT-2).

        Uses the concrete :class:`~mnemozine.extract.TypedExtractor` (built fresh
        here rather than via :meth:`build_extractor`, which returns the bare
        ``Extractor`` Protocol) because the ingest pipeline needs ``extract_full``
        to write entity nodes + relationship edges, not just MemoryUnits.
        """

        from mnemozine.extract import TypedExtractor

        extractor = TypedExtractor(self.build_llm_provider(), settings=self.settings)
        self._extractor = extractor
        return MnemozineIngestService(storage, extractor, settings=self.settings)

    async def close(self) -> None:
        """Close the storage connection if one was opened."""

        if self._storage is not None:
            await self._storage.close()
            self._storage = None


# ---------------------------------------------------------------------------
# mnemozine-mcp — the single MCP server (FR-RET-1)
# ---------------------------------------------------------------------------

mcp_app = typer.Typer(help="Mnemozine MCP server (FR-RET-1).", add_completion=False)


@mcp_app.callback(invoke_without_command=True)
def _mcp_main(
    transport: str = typer.Option(
        "stdio", help="MCP transport: stdio | streamable-http | sse."
    ),
) -> None:
    """Serve memory to all agents via one MCP server (FR-RET-1, FR-RET-4)."""

    from mnemozine.retrieval.server import run as run_server

    container = Container.from_env()
    retriever = asyncio.run(container.build_retriever())
    # ``server.run`` is blocking and owns its own event loop, so the retriever
    # (over a now-connected backend) is built first, then handed to the server.
    run_server(retriever, transport=transport, settings=container.settings)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# mnemozine-ingest — source -> chunk -> extract -> store loop (FR-ING-*)
# ---------------------------------------------------------------------------

ingest_app = typer.Typer(help="Mnemozine ingestion service (FR-ING-*).", add_completion=False)


async def _run_ingest(container: Container, *, backfill: bool) -> None:
    """Consume the Claude Code source, chunking + extracting + storing (FR-ING-*).

    Installs the hook ``services_loader`` (so the Stop/PreCompact/SessionStart
    hooks share this process's wired retriever + ingest service) and then drives
    the source: ``backfill`` replays existing transcripts once and exits;
    otherwise ``stream`` tails new turns indefinitely. Each completed chunk is
    extracted and persisted via the FR-MNT-1 4-way write.
    """

    from mnemozine.ingestion.claude_code import ClaudeCodeSource
    from mnemozine.ingestion.claude_code.chunker import ChunkAccumulator
    from mnemozine.ingestion.claude_code.hooks import runtime as hook_runtime

    storage = await container.build_storage()
    ingest_service = container.build_ingest_service(storage)
    retriever = await container.build_retriever()

    # Share this process's wired services with the hook entrypoints.
    hook_runtime.services_loader = lambda: hook_runtime.HookServices(
        retriever=retriever,
        ingest=ingest_service,
        settings=container.settings,
    )

    source = ClaudeCodeSource(container.settings)
    accumulator = ChunkAccumulator(container.settings.ingest)
    events = source.backfill() if backfill else source.stream()

    async for event in events:
        for chunk in accumulator.add(event):
            await ingest_service.ingest_chunk(chunk)
    # Flush any in-flight remainder (always for backfill; stream() never returns).
    for chunk in accumulator.flush():
        await ingest_service.ingest_chunk(chunk)


@ingest_app.callback(invoke_without_command=True)
def _ingest_main(
    backfill: bool = typer.Option(
        False, help="Replay existing transcripts once and exit (FR-ING-6 backlog)."
    ),
) -> None:
    """Watch sources and ingest conversations (FR-ING-*)."""

    container = Container.from_env()
    try:
        asyncio.run(_run_ingest(container, backfill=backfill))
    except KeyboardInterrupt:  # graceful Ctrl-C on the tailing watcher
        typer.echo("mnemozine-ingest: stopped.")
    finally:
        asyncio.run(container.close())


# ---------------------------------------------------------------------------
# mnemozine-maintenance — scheduled consolidate/resolve/decay/audit (FR-MNT-*)
# ---------------------------------------------------------------------------

maintenance_app = typer.Typer(
    help="Mnemozine scheduled maintenance (FR-MNT-*).", add_completion=False
)


async def _maintenance_run_once(container: Container) -> None:
    from mnemozine.maintenance.runner import MaintenanceRunner, build_default_jobs

    storage = await container.build_storage()
    jobs = build_default_jobs(
        storage,
        container.build_llm_provider(),
        container.build_embedding_provider(),
        settings=container.settings,
    )
    reports = await MaintenanceRunner(jobs, settings=container.settings).run_once()
    for r in reports:
        typer.echo(
            f"[{r.job_name}] consolidated={r.consolidated} merged={r.entities_merged} "
            f"archived={r.archived} pruned={r.edges_pruned}"
        )
        for note in r.notes:
            typer.echo(f"    - {note}")


async def _maintenance_serve(container: Container) -> None:
    from mnemozine.maintenance.runner import MaintenanceRunner, build_default_jobs

    storage = await container.build_storage()
    jobs = build_default_jobs(
        storage,
        container.build_llm_provider(),
        container.build_embedding_provider(),
        settings=container.settings,
    )
    await MaintenanceRunner(jobs, settings=container.settings).serve_forever()


@maintenance_app.command("run")
def _maintenance_run_cmd() -> None:
    """Run the full maintenance pass once and exit (idempotent, FR-MNT-5)."""

    container = Container.from_env()
    try:
        asyncio.run(_maintenance_run_once(container))
    finally:
        asyncio.run(container.close())


@maintenance_app.command("serve")
def _maintenance_serve_cmd() -> None:
    """Run maintenance on the configured cron schedule until interrupted (FR-MNT-5)."""

    container = Container.from_env()
    try:
        asyncio.run(_maintenance_serve(container))
    except (KeyboardInterrupt, asyncio.CancelledError):
        typer.echo("mnemozine-maintenance: scheduler stopped.")
    finally:
        asyncio.run(container.close())


# ---------------------------------------------------------------------------
# mnemozine-eval — the §9 eval harness
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Console-script entrypoints (declared in pyproject [project.scripts]).
# ---------------------------------------------------------------------------


def run_mcp() -> None:
    """Console-script entrypoint for ``mnemozine-mcp`` (FR-RET-1)."""

    mcp_app()


def run_ingest() -> None:
    """Console-script entrypoint for ``mnemozine-ingest`` (FR-ING-*)."""

    ingest_app()


def run_maintenance() -> None:
    """Console-script entrypoint for ``mnemozine-maintenance`` (FR-MNT-*)."""

    maintenance_app()


def run_eval() -> None:
    """Console-script entrypoint for ``mnemozine-eval`` (§9 eval harness)."""

    from mnemozine.evals.cli import app as eval_app

    eval_app()
