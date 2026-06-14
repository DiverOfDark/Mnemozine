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
import contextlib
import logging
import signal
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

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
    from fastapi import FastAPI

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
        from mnemozine.services import MnemozineIngestService

        extractor = TypedExtractor(self.build_llm_provider(), settings=self.settings)
        self._extractor = extractor
        return MnemozineIngestService(storage, extractor, settings=self.settings)

    async def close(self) -> None:
        """Close the storage connection if one was opened."""

        if self._storage is not None:
            await self._storage.close()
            self._storage = None


async def _run_then_close(
    container: Container, coro: Coroutine[Any, Any, None]
) -> None:
    """Await ``coro`` and then close the container in the SAME event loop.

    The standalone scripts must NOT close the container in a *second*
    ``asyncio.run`` after the work loop has already finished: the FalkorDB async
    redis connection is bound to the loop that opened it (in ``build_storage``), so
    closing it from a fresh loop raises ``RuntimeError: Event loop is closed`` (the
    redis pool's transport tears down against the original, now-closed loop) — a
    crash on every ``mnemozine-ingest`` / ``mnemozine-maintenance`` exit, only ever
    seen against a *live* store. Running the work and the close under one loop keeps
    the connection lifecycle on a single loop, so ``close`` runs cleanly.
    """

    try:
        await coro
    finally:
        await container.close()


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
        # Run the ingest loop AND close the container under one event loop so the
        # FalkorDB connection is finalized on the loop that opened it (a second
        # asyncio.run for close raises "Event loop is closed" against a live store).
        asyncio.run(_run_then_close(container, _run_ingest(container, backfill=backfill)))
    except KeyboardInterrupt:  # graceful Ctrl-C on the tailing watcher
        typer.echo("mnemozine-ingest: stopped.")


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
    # Single-loop run+close (see _run_then_close): a second asyncio.run for close
    # would raise "Event loop is closed" against a live FalkorDB connection.
    asyncio.run(_run_then_close(container, _maintenance_run_once(container)))


@maintenance_app.command("serve")
def _maintenance_serve_cmd() -> None:
    """Run maintenance on the configured cron schedule until interrupted (FR-MNT-5)."""

    container = Container.from_env()
    try:
        asyncio.run(_run_then_close(container, _maintenance_serve(container)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        typer.echo("mnemozine-maintenance: scheduler stopped.")


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


@maintenance_app.command("merge-categories")
def _maintenance_merge_categories_cmd(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only print the proposed (source -> canonical) merges; apply nothing.",
    ),
) -> None:
    """Merge near-duplicate emergent categories into a canonical one (FR-MNT-2/4).

    Mirrors :func:`mnemozine.maintenance.runner._run_merge_categories` onto the
    live ``mnemozine-maintenance`` console app (the runner's own Typer app is not
    wired to a script). Clusters the free-form ``MemoryUnit.category`` registry by
    name/embedding similarity above ``category.merge_similarity_threshold`` and
    folds each cluster into its highest-count canonical category. Idempotent; use
    ``--dry-run`` to review the proposals first.
    """

    from mnemozine.maintenance.runner import _echo_report, _run_merge_categories

    report = asyncio.run(_run_merge_categories(dry_run=dry_run))
    _echo_report(report)


@maintenance_app.command("reclassify")
def _maintenance_reclassify_cmd(
    scope: str | None = typer.Option(
        None,
        "--scope",
        help=(
            "Restrict to one scope (canonical form, e.g. 'global' or "
            "'project:Mnemozine'); default: all scopes."
        ),
    ),
) -> None:
    """Re-scope + re-categorize stored memories with the current classifier (R1).

    Mirrors :func:`mnemozine.maintenance.runner._run_reclassify`. Reads each
    memory's already-stored content + provenance (no raw transcript) and re-applies
    the current scope/category/cross-ref decision, writing only the fields that
    drifted. Idempotent: a memory already matching the classifier is untouched.
    """

    from mnemozine.maintenance.runner import _echo_report, _run_reclassify

    report = asyncio.run(_run_reclassify(scope=scope))
    _echo_report(report)


@maintenance_app.command("re-extract")
def _maintenance_re_extract_cmd(
    scope: str | None = typer.Option(
        None,
        "--scope",
        help="Restrict to one EXACT scope (e.g. 'project:Mnemozine'); default: all.",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Restrict to one originating session id; default: all sessions.",
    ),
    keep_existing: bool = typer.Option(
        False,
        "--keep-existing",
        help=(
            "Do NOT close the validity windows of the memories each chunk "
            "previously produced (default: supersede them)."
        ),
    ),
) -> None:
    """Re-run the current extractor over the retained raw tier (offline reindex).

    Mirrors :func:`mnemozine.maintenance.runner._run_re_extract`. Re-processes
    stored RawChunks through the current extractor/classifier to apply a model or
    prompt change to already-ingested data without the original transcript. By
    default the memories each chunk previously produced are superseded;
    ``--keep-existing`` leaves them active. Idempotent.
    """

    from mnemozine.maintenance.runner import _echo_report, _run_re_extract

    report = asyncio.run(
        _run_re_extract(
            scope=scope,
            session_id=session_id,
            supersede_existing=not keep_existing,
        )
    )
    _echo_report(report)


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


# ---------------------------------------------------------------------------
# mnemozine — the all-in-one entrypoint (build Container once, run every ENABLED
# component concurrently under one event loop, graceful shutdown).
# ---------------------------------------------------------------------------


def _build_no_signal_server(config: Any) -> Any:
    """Build a uvicorn ``Server`` that never installs its own signal handlers.

    The all-in-one coordinator (:func:`_run_all`) owns SIGINT/SIGTERM via
    ``loop.add_signal_handler`` so a stop signal cancels *all* components together.
    uvicorn's default ``capture_signals`` would override those handlers (it uses
    ``signal.signal``); overriding it to a no-op context manager leaves the
    coordinator in control. uvicorn is imported lazily here so the non-HTTP
    entrypoints (e.g. ``mnemozine-ingest``) never import the web stack.
    """

    import uvicorn

    class _NoSignalServer(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self) -> Iterator[None]:
            yield

    return _NoSignalServer(config)


async def _serve_uvicorn(app: Any, *, host: str, port: int, log_level: str) -> None:
    """Serve an ASGI ``app`` with uvicorn as a cancellable coroutine.

    Unlike ``uvicorn.run`` (which installs signal handlers and owns the loop), this
    awaits ``serve()`` on the *current* loop using a no-signal server so the
    coordinator owns SIGINT/SIGTERM. On cancellation we set ``should_exit`` and
    await the server's graceful shutdown before re-raising.
    """

    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level=log_level, lifespan="on")
    server = _build_no_signal_server(config)
    serve_task = asyncio.ensure_future(server.serve())
    try:
        await serve_task
    except asyncio.CancelledError:
        server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task
        raise


async def _ingest_component(container: Container) -> None:
    """Run the streaming ingest loop (FR-ING-*) as a background component.

    Equivalent to ``mnemozine-ingest`` (no ``--backfill``): tails every enabled
    source forever. Reuses :func:`_run_ingest` so the all-in-one path and the
    standalone script wire ingest identically (same hook ``services_loader``, same
    fan-in pipeline).

    Resilience: in the all-in-one, :func:`_run_all` shuts every component down as
    soon as *any* one finishes (``FIRST_COMPLETED``). A streaming ingest loop is
    meant to run forever, but it returns early if no source is enabled OR every
    source producer exits (e.g. a Claude Code watcher hitting an unreadable mount
    raises, is caught per-source, and that producer ends). Without the guard below
    that early return would tear down the WebUI + MCP + maintenance too. So when
    the loop returns we log it and *park* indefinitely instead of returning, which
    keeps the rest of the all-in-one serving; a real SIGINT/SIGTERM still cancels
    this parked await. (The standalone ``mnemozine-ingest`` script is unaffected —
    it calls :func:`_run_ingest` directly and simply exits when the loop returns.)
    """

    await _run_ingest(container, backfill=False)
    logger.warning(
        "mnemozine: ingest loop ended (no enabled source or all sources stopped); "
        "keeping the rest of the all-in-one running"
    )
    await asyncio.Event().wait()  # park until cancelled at shutdown


async def _maintenance_component(container: Container) -> None:
    """Run the maintenance cron scheduler (FR-MNT-5) as a background component.

    Equivalent to ``mnemozine-maintenance serve``: builds the default job set over
    the shared container and runs the APScheduler cron loop until cancelled.
    """

    await _maintenance_serve(container)


async def _mcp_standalone_component(container: Container) -> None:
    """Serve the MCP server over streamable-HTTP on its own port (web disabled).

    Used when ``run.mcp`` is enabled but ``run.web`` is not, so there is no FastAPI
    app to mount into. Binds ``mcp_host`` / ``mcp_port`` and runs the streamable
    HTTP ASGI app under uvicorn; the sub-app's own lifespan drives the session
    manager here (no mounting), so no extra lifespan wiring is needed.
    """

    from mnemozine.retrieval.retriever import ScopedRetriever
    from mnemozine.retrieval.server import build_mcp_http_app

    retriever = cast(ScopedRetriever, await container.build_retriever())
    _server, asgi_app = build_mcp_http_app(retriever, settings=container.settings)
    cfg = container.settings
    await _serve_uvicorn(
        asgi_app,
        host=cfg.mcp_host,
        port=cfg.mcp_port,
        log_level=cfg.log_level.lower(),
    )


async def _build_web_app(container: Container, *, mount_mcp: bool) -> FastAPI:
    """Build the FastAPI WebUI app, optionally mounting the MCP app at ``/mcp``.

    When ``mount_mcp`` is True the MCP streamable-HTTP ASGI app is mounted under
    the WebUI app so both are served from the single ``web.port`` (default 8765),
    resolving the historical web/MCP port clash. The MCP session manager (created
    lazily by ``streamable_http_app()``) must run for the lifetime of the parent
    app; a Starlette ``Mount`` does **not** invoke the sub-app's lifespan, so we
    splice ``server.session_manager.run()`` into the FastAPI app's lifespan.
    """

    from contextlib import AsyncExitStack
    from contextlib import asynccontextmanager as _acm

    from starlette.routing import Mount

    from mnemozine.retrieval.retriever import ScopedRetriever
    from mnemozine.retrieval.server import build_mcp_http_app
    from mnemozine.web import create_app

    app = create_app(container)

    if not mount_mcp:
        return app

    retriever = cast(ScopedRetriever, await container.build_retriever())
    # Build the sub-app with its route at "/" so the ``Mount("/mcp", ...)`` below
    # supplies the prefix and the endpoint lands at exactly ``/mcp`` (not /mcp/mcp).
    server, asgi_app = build_mcp_http_app(
        retriever, settings=container.settings, streamable_http_path="/"
    )

    # Insert the MCP mount at index 0 so it is matched BEFORE the SPA catch-all
    # (``/{full_path:path}``, added last by create_app) and is not swallowed by the
    # SPA fallback. A ``Mount("/mcp")`` only matches the ``/mcp`` prefix, so the
    # ``/api`` routers are untouched.
    #
    # A Starlette ``Mount("/mcp")`` serves the sub-app at ``/mcp/...`` but does NOT
    # answer the bare, slash-less ``/mcp`` itself — without the route below that
    # exact path falls through to the SPA catch-all and a client configured with
    # ``http://host:8765/mcp`` (the documented endpoint, and the standalone server's
    # default path) would get the HTML SPA instead of the MCP transport. Insert an
    # explicit redirect ``/mcp`` -> ``/mcp/`` ahead of both so the contracted
    # slash-less URL reaches the transport. Use 307 so the client re-issues the same
    # method/body (a POST initialize must stay a POST).
    from starlette.responses import RedirectResponse
    from starlette.routing import Route

    async def _mcp_redirect(_request: Any) -> RedirectResponse:
        return RedirectResponse(url="/mcp/", status_code=307)

    app.router.routes.insert(0, Mount("/mcp", app=asgi_app))
    app.router.routes.insert(
        0, Route("/mcp", _mcp_redirect, methods=["GET", "POST", "DELETE"])
    )

    prior_lifespan = app.router.lifespan_context

    @_acm
    async def _combined_lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            # Drive the MCP StreamableHTTP session manager for the app's lifetime.
            await stack.enter_async_context(server.session_manager.run())
            await stack.enter_async_context(prior_lifespan(fastapi_app))
            yield

    app.router.lifespan_context = _combined_lifespan
    return app


async def _web_component(container: Container, *, mount_mcp: bool) -> None:
    """Serve the WebUI (and, when ``mount_mcp``, the MCP app at /mcp) on one port."""

    app = await _build_web_app(container, mount_mcp=mount_mcp)
    cfg = container.settings.web
    await _serve_uvicorn(
        app,
        host=cfg.host,
        port=cfg.port,
        log_level=container.settings.log_level.lower(),
    )


def _select_components(
    container: Container,
) -> dict[str, Callable[[], Coroutine[Any, Any, None]]]:
    """Map each ENABLED component to its coroutine factory (disabled -> absent).

    The selection rule for the web/MCP pair implements the contract:

    * web + mcp  -> one ``web`` component serving the WebUI with the MCP app mounted
      at ``/mcp`` on the single ``web.port`` (no separate MCP server).
    * web only   -> ``web`` component, no MCP mount.
    * mcp only   -> ``mcp`` component, MCP standalone on ``mcp_port``.
    * neither    -> no HTTP server.

    ``ingest`` and ``maintenance`` are independent background tasks. A component
    absent from the returned mapping is never started.
    """

    run = container.settings.run
    components: dict[str, Callable[[], Coroutine[Any, Any, None]]] = {}

    if run.web:
        mount_mcp = run.mcp
        components["web"] = lambda: _web_component(container, mount_mcp=mount_mcp)
    elif run.mcp:
        components["mcp"] = lambda: _mcp_standalone_component(container)

    if run.ingest:
        components["ingest"] = lambda: _ingest_component(container)
    if run.maintenance:
        components["maintenance"] = lambda: _maintenance_component(container)

    return components


async def _run_all(container: Container) -> None:
    """Run every ENABLED component concurrently with graceful shutdown.

    Builds the component set from ``settings.run.*`` (see :func:`_select_components`)
    and launches each as an asyncio task under this one event loop. SIGINT/SIGTERM
    (or any task crashing) triggers shutdown: every task is cancelled and awaited,
    then ``container.close()`` releases the shared storage connection. Disabled
    components are never created, so this is a no-op-safe superset of every
    standalone script (e.g. ``run.ingest`` only == ``mnemozine-ingest``).
    """

    components = _select_components(container)
    if not components:
        logger.warning("mnemozine: no components enabled (settings.run.*); exiting")
        return

    logger.info("mnemozine: starting components: %s", ", ".join(sorted(components)))

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _request_stop() -> None:
        logger.info("mnemozine: shutdown signal received")
        stop.set()

    installed: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, _request_stop)
            installed.append(signum)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - non-main thread / Windows
            pass

    tasks: dict[str, asyncio.Task[None]] = {
        name: asyncio.create_task(factory(), name=f"mnemozine-{name}")
        for name, factory in components.items()
    }
    stop_task = asyncio.create_task(stop.wait(), name="mnemozine-stop")

    try:
        # Wake on first of: a stop signal, or any component finishing/crashing.
        await asyncio.wait(
            [stop_task, *tasks.values()],
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for signum in installed:
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.remove_signal_handler(signum)
        stop_task.cancel()
        for task in tasks.values():
            if not task.done():
                task.cancel()
        results = await asyncio.gather(
            stop_task, *tasks.values(), return_exceptions=True
        )
        for name, result in zip(tasks, results[1:], strict=True):
            if isinstance(result, Exception) and not isinstance(
                result, asyncio.CancelledError
            ):
                logger.error("mnemozine: component %s exited with error", name, exc_info=result)
        await container.close()


def run_all() -> None:
    """Console-script entrypoint for ``mnemozine`` (the all-in-one process).

    Builds the :class:`Container` **once** and runs every component enabled in
    ``settings.run.*`` concurrently under one event loop (see :func:`_run_all`):
    the WebUI + MCP (mounted at ``/mcp`` on the single ``web.port`` 8765), the
    ingest loop, and the maintenance scheduler. Toggle components with
    ``MNEMOZINE_RUN__MCP`` / ``__INGEST`` / ``__MAINTENANCE`` / ``__WEB`` (all
    default true). To split ingest onto another machine, run this with only
    ``MNEMOZINE_RUN__INGEST=true`` pointed at a remote ``MNEMOZINE_FALKORDB__URL``
    plus remote embedding/extraction endpoints — equivalent to ``mnemozine-ingest``.
    """

    container = Container.from_env()
    logging.basicConfig(level=container.settings.log_level.upper())
    try:
        asyncio.run(_run_all(container))
    except KeyboardInterrupt:  # pragma: no cover - asyncio.run usually handles SIGINT first
        typer.echo("mnemozine: stopped.")
