"""FR-MNT-5 — the scheduled maintenance runner (APScheduler) + console app.

Runs the maintenance jobs (consolidation FR-MNT-2, entity resolution FR-MNT-4,
decay/archive FR-MNT-3, audit R5) either **once** or on a **cron schedule**
(``maintenance.cron``, default ``0 3 * * *``), via APScheduler's
``AsyncIOScheduler`` + ``CronTrigger``.

Idempotency / re-run safety (FR-MNT-5) is a property of each individual job (see
their docstrings); the runner adds:

* ``max_instances=1`` + ``coalesce=True`` on the scheduled job so an overrunning
  pass never overlaps itself or stacks missed runs, and
* a fixed, deterministic job order so a run is reproducible.

The console_script ``mnemozine-maintenance`` (declared in ``pyproject.toml`` as
``mnemozine.app:run_maintenance``) should delegate to :func:`run_maintenance`
here — see the integration note. This module also exposes :data:`maintenance_cli`
(a Typer app) and :func:`run_maintenance` directly so the script can be wired to
either without this module owning ``app.py``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import typer
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import (
    EmbeddingProvider,
    LLMProvider,
    MaintenanceJob,
    MaintenanceReport,
    StorageBackend,
)
from mnemozine.maintenance.audit import AuditJob
from mnemozine.maintenance.consolidation import ConsolidationJob
from mnemozine.maintenance.decay import DecayJob
from mnemozine.maintenance.entity_resolution import EntityResolutionJob
from mnemozine.maintenance.migrate_index import MigrateIndexJob

logger = logging.getLogger(__name__)


def build_default_jobs(
    storage: StorageBackend,
    llm: LLMProvider,
    embeddings: EmbeddingProvider,
    *,
    settings: Settings | None = None,
) -> list[MaintenanceJob]:
    """Construct the standard maintenance job set in a deterministic run order.

    Order matters for a single pass: consolidate first (collapses duplicate
    facts), resolve entities next (the merged graph), then decay/archive (demote
    the now-quiet hot tier), and finally audit (report on the settled state).
    """

    settings = settings or get_settings()
    return [
        ConsolidationJob(storage, llm, embeddings, settings=settings),
        EntityResolutionJob(storage, llm=llm, settings=settings),
        DecayJob(storage, settings=settings),
        AuditJob(storage, settings=settings),
    ]


class MaintenanceRunner:
    """Runs a set of :class:`~mnemozine.interfaces.MaintenanceJob`s, once or on cron.

    Holds no storage/LLM details itself — it is handed already-constructed jobs
    (see :func:`build_default_jobs`) so it depends purely on the
    :class:`~mnemozine.interfaces.MaintenanceJob` Protocol. This keeps the runner
    trivially testable: a fake job that records that it ran is enough.
    """

    def __init__(
        self,
        jobs: Sequence[MaintenanceJob],
        *,
        settings: Settings | None = None,
    ) -> None:
        self._jobs = list(jobs)
        self._settings = settings or get_settings()

    async def run_once(self) -> list[MaintenanceReport]:
        """Run every job once, in order, and return their reports.

        Each job is isolated: a failure in one is logged and recorded as a note
        on a failure report, but does not abort the rest of the pass (a botched
        consolidation must not skip the decay sweep). Safe to call repeatedly —
        every job is individually idempotent (FR-MNT-5).
        """

        reports: list[MaintenanceReport] = []
        for job in self._jobs:
            try:
                report = await job.run()
            except Exception as exc:  # noqa: BLE001 - one bad job must not abort the pass
                logger.exception("maintenance job %s failed", job.name)
                report = MaintenanceReport(
                    job_name=job.name, notes=[f"ERROR: {exc!r}"]
                )
            reports.append(report)
            logger.info(
                "maintenance job %s: consolidated=%d merged=%d archived=%d pruned=%d",
                report.job_name,
                report.consolidated,
                report.entities_merged,
                report.archived,
                report.edges_pruned,
            )
        return reports

    def schedule(self, scheduler: AsyncIOScheduler) -> None:
        """Register the pass on ``scheduler`` using ``maintenance.cron`` (FR-MNT-5).

        ``max_instances=1`` + ``coalesce=True`` keep an overrunning pass from
        overlapping itself or stacking missed runs. The runner does not start or
        own the scheduler's event loop — the caller does — so it can be embedded
        in a larger service.
        """

        trigger = CronTrigger.from_crontab(self._settings.maintenance.cron)
        scheduler.add_job(
            self.run_once,
            trigger=trigger,
            id="mnemozine-maintenance",
            name="mnemozine maintenance pass",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )

    async def serve_forever(self) -> None:
        """Start the cron scheduler and block until cancelled (FR-MNT-5 daemon mode)."""

        scheduler = AsyncIOScheduler()
        self.schedule(scheduler)
        scheduler.start()
        logger.info(
            "mnemozine maintenance scheduled: cron=%r jobs=%s",
            self._settings.maintenance.cron,
            [j.name for j in self._jobs],
        )
        stop = asyncio.Event()
        try:
            await stop.wait()
        finally:
            scheduler.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Console app (mnemozine-maintenance)
# ---------------------------------------------------------------------------

maintenance_cli = typer.Typer(
    help="Mnemozine scheduled maintenance: consolidate, resolve, decay, audit (FR-MNT-*).",
    add_completion=False,
)


def _build_runner_from_env() -> MaintenanceRunner:
    """Wire a runner from the live composition root (Container.build_*).

    Imported lazily so the maintenance package never imports ``app.py`` at module
    load (avoids an import cycle and keeps unit tests offline). The integration
    pass fills in ``Container.build_*``; until then this raises a clear
    ``NotImplementedError`` from those builders rather than mis-wiring.
    """

    from mnemozine.app import Container  # local import: avoid cycle / keep tests offline

    container = Container.from_env()
    settings = container.settings
    # build_storage() is async (opens the FalkorDB connection); connect it here so
    # this sync helper still returns a ready runner.
    storage = asyncio.run(container.build_storage())
    llm = container.build_llm_provider()
    embeddings = container.build_embedding_provider()
    jobs = build_default_jobs(storage, llm, embeddings, settings=settings)
    return MaintenanceRunner(jobs, settings=settings)


@maintenance_cli.command("run")
def _cmd_run() -> None:
    """Run the full maintenance pass once and exit (idempotent, FR-MNT-5)."""

    runner = _build_runner_from_env()
    reports = asyncio.run(runner.run_once())
    for r in reports:
        typer.echo(
            f"[{r.job_name}] consolidated={r.consolidated} merged={r.entities_merged} "
            f"archived={r.archived} pruned={r.edges_pruned}"
        )
        for note in r.notes:
            typer.echo(f"    - {note}")


@maintenance_cli.command("serve")
def _cmd_serve() -> None:
    """Run maintenance on the configured cron schedule until interrupted (FR-MNT-5)."""

    runner = _build_runner_from_env()
    try:
        asyncio.run(runner.serve_forever())
    except (KeyboardInterrupt, asyncio.CancelledError):  # graceful Ctrl-C
        typer.echo("mnemozine-maintenance: scheduler stopped.")


async def _run_migrate_index(*, force: bool) -> MaintenanceReport:
    """Build the wired migrate-index job from the live container and run it (OQ3).

    The job needs the FalkorDB vector-index admin seam, which lives on the
    :class:`~mnemozine.storage.graphiti_client.GraphitiClient` composed inside the
    storage backend. ``GraphitiStorageBackend`` exposes that client as ``_client``;
    we read it back here rather than threading a new public accessor, keeping the
    OQ3 path within the existing contract (no StorageBackend Protocol change).
    """

    from mnemozine.app import Container  # local import: avoid cycle / keep tests offline

    container = Container.from_env()
    settings = container.settings
    storage = await container.build_storage()
    embeddings = container.build_embedding_provider()
    index_admin = getattr(storage, "_client", None)
    if index_admin is None:  # pragma: no cover - only a mis-wired backend hits this
        raise RuntimeError(
            "migrate-index requires the Graphiti/FalkorDB backend (no vector-index "
            "admin seam on the configured storage backend)."
        )
    job = MigrateIndexJob(
        storage, index_admin, embeddings, settings=settings, force=force
    )
    try:
        return await job.run()
    finally:
        await container.close()


@maintenance_cli.command("migrate-index")
def _cmd_migrate_index(
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

    Detects a configured-vs-actual vector-index dimension mismatch, drops +
    recreates the FalkorDB vector index at the configured width, and re-embeds all
    hot memories through the embedding provider. Idempotent and safe to re-run: a
    no-op when the dimension already matches (unless ``--force``).
    """

    report = asyncio.run(_run_migrate_index(force=force))
    typer.echo(f"[{report.job_name}] reembedded={report.consolidated}")
    for note in report.notes:
        typer.echo(f"    - {note}")


def run_maintenance() -> None:
    """Console-script entrypoint for ``mnemozine-maintenance`` (FR-MNT-5).

    Target for the ``[project.scripts]`` ``mnemozine-maintenance`` entry. The
    integration pass should repoint that script (currently
    ``mnemozine.app:run_maintenance``, a Phase-0 stub) here, or call this from the
    stub. Subcommands: ``run`` (once) and ``serve`` (cron daemon).
    """

    maintenance_cli()
