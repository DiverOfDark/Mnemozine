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
    ActivityLog,
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
    _activity: ActivityLog | None = field(default=None, init=False, repr=False)

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
                retrieval=self.settings.retrieval,
            )
        return self._storage

    # -- higher layers (need an already-built, connected storage) ---------

    def build_extractor(self) -> Extractor:
        """Wire the typed extraction pipeline (FR-EXT-*)."""

        if self._extractor is None:
            from mnemozine.extract import TypedExtractor

            self._extractor = TypedExtractor(self.build_llm_provider(), settings=self.settings)
        return self._extractor

    # -- activity log (WEBUI Q3: injectable, NullActivityLog default) ------

    async def build_activity_log(self) -> ActivityLog:
        """Wire the append-only activity log (WEBUI Q3).

        Returns a :class:`~mnemozine.activity.NullActivityLog` (a no-op) **by
        default**, so every existing pipeline call site and the 442-test suite are
        unaffected — the log is strictly opt-in. When ``web.enable_activity_log``
        is set it returns a persisted :class:`~mnemozine.activity.FalkorDBActivityLog`
        over the **same** storage connection (the ``GraphitiClient`` the backend
        already holds), never a new source of truth; if that connection cannot be
        reached it falls back to an in-memory log. Memoized like the other layers.
        """

        if self._activity is None:
            from mnemozine.activity.log import build_activity_log

            client: object | None = None
            if self.settings.web.enable_activity_log:
                storage = await self.build_storage()
                client = getattr(storage, "_client", None)
            self._activity = build_activity_log(
                enable=self.settings.web.enable_activity_log,
                client=client,
            )
        return self._activity

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
    """Drive all enabled ingest sources into the shared pipeline (FR-ING-2/3/4).

    Installs the hook ``services_loader`` (so the Stop/PreCompact/SessionStart
    hooks share this process's wired retriever + ingest service), builds the
    enabled sources from the ``ingest.enable_*`` config flags (Claude Code,
    FR-ING-3 LiteLLM gateway, FR-ING-4 Hermes), and runs them **concurrently**
    through one shared ``ChunkAccumulator -> ingest_service`` pipeline (see
    :mod:`mnemozine.ingestion.loop`): ``backfill`` replays each source's backlog
    once and exits; otherwise the sources tail/stream indefinitely. Each completed
    chunk is extracted and persisted via the FR-MNT-1 4-way write.

    The gateway callback built here is the same in-process
    :class:`~mnemozine.ingestion.gateway.callback.GatewayCallback` instance a
    co-hosted LiteLLM proxy should register (its in-process queue is what the loop
    drains), and the Hermes adapter is the instance the instrumented VM feeds.
    """

    from mnemozine.ingestion.claude_code.chunker import ChunkAccumulator
    from mnemozine.ingestion.claude_code.hooks import runtime as hook_runtime
    from mnemozine.ingestion.loop import build_ingest_sources, run_ingest_loop

    storage = await container.build_storage()
    ingest_service = container.build_ingest_service(storage)
    retriever = await container.build_retriever()

    # Share this process's wired services with the hook entrypoints.
    hook_runtime.services_loader = lambda: hook_runtime.HookServices(
        retriever=retriever,
        ingest=ingest_service,
        settings=container.settings,
    )

    sources = build_ingest_sources(container.settings)
    if not sources:
        logger.warning("mnemozine-ingest: no ingest sources enabled; exiting")
        return
    logger.info(
        "mnemozine-ingest: %d source(s) enabled: %s",
        len(sources),
        ", ".join(s.source_name for s in sources.sources),
    )

    accumulator = ChunkAccumulator(container.settings.ingest)
    await run_ingest_loop(sources, accumulator, ingest_service, backfill=backfill)


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


@maintenance_app.command("migrate-index")
def _maintenance_migrate_index_cmd(
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Re-embed the hot tier even when the index dimension is unchanged "
            "(use after an embedding MODEL change that kept the same width)."
        ),
    ),
) -> None:
    """Migrate the vector index + re-embed on an embedding dimension change (OQ3).

    Mirrors :func:`mnemozine.maintenance.runner._run_migrate_index` onto the live
    ``mnemozine-maintenance`` console app so the OQ3 migration is reachable through
    the real entrypoint (the runner's own Typer app is not wired to a script). The
    job detects a configured-vs-actual vector-index dimension mismatch, drops +
    recreates the FalkorDB vector index at the configured width, and re-embeds all
    hot memories. Idempotent: a no-op when the dimension already matches (unless
    ``--force``).
    """

    from mnemozine.maintenance.runner import _run_migrate_index

    report = asyncio.run(_run_migrate_index(force=force))
    typer.echo(f"[{report.job_name}] reembedded={report.consolidated}")
    for note in report.notes:
        typer.echo(f"    - {note}")


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


# ---------------------------------------------------------------------------
# mnemozine-web — the operator console WebUI (WEBUI PRD)
# ---------------------------------------------------------------------------


def run_web() -> None:
    """Console-script entrypoint for ``mnemozine-web`` (the operator console).

    Wires the live :class:`Container` into the FastAPI app
    (:func:`mnemozine.web.create_app`) and serves it with uvicorn, bound to the
    configured ``web.host`` / ``web.port`` (localhost by default — never expose
    publicly, Q5). The container is shared with the app so every route goes
    through the existing ``StorageBackend`` / retriever / maintenance / evals —
    the UI is never a new source of truth (WEBUI PRD §2).
    """

    import uvicorn

    from mnemozine.web import create_app

    container = Container.from_env()
    app = create_app(container)
    cfg = container.settings.web
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=container.settings.log_level.lower())
