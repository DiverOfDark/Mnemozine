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

from mnemozine.activity import emit, maintenance_event
from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import (
    ActivityLog,
    EmbeddingProvider,
    LLMProvider,
    MaintenanceJob,
    MaintenanceReport,
    StorageBackend,
)
from mnemozine.maintenance.audit import AuditJob
from mnemozine.maintenance.category_merge import CategoryMergeJob
from mnemozine.maintenance.co_mention import CoMentionJob
from mnemozine.maintenance.consolidation import ConsolidationJob
from mnemozine.maintenance.decay import DecayJob
from mnemozine.maintenance.entity_dedup import EntityDedupJob
from mnemozine.maintenance.entity_resolution import EntityResolutionJob
from mnemozine.maintenance.mentions import MentionsJob
from mnemozine.maintenance.migrate_index import MigrateIndexJob
from mnemozine.maintenance.reclassify import ReclassifyJob, ReExtractJob
from mnemozine.maintenance.relation_norm import RelationNormJob
from mnemozine.schema.models import Scope

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
    facts), resolve entities next (the merged graph), then persist the
    memory->entity mention edges (graph connectivity — the substrate later
    graph-connectivity jobs derive from), then derive the weighted entity-entity
    co-mention layer from those mention edges (so it MUST run AFTER mentions),
    then normalize the fragmented relation-label vocabulary into a controlled set
    (independent of the mention/co-mention layers — it operates on the
    LLM-extracted RELATES edges), then dedup true-duplicate entity nodes (LAST
    among the graph-connectivity jobs, AFTER mentions + co-mention so the
    survivor-repoint covers all three edge types), then merge near-duplicate
    emergent categories (the category analogue of entity resolution), then
    decay/archive (demote the now-quiet hot tier), and finally audit (report on
    the settled state).

    The re-extract / reclassify passes are intentionally **not** in this default
    scheduled set: re-running the extractor/classifier over the whole store is an
    explicit, operator-triggered offline migration (applied after a model/prompt
    change), exposed as the ``re-extract`` / ``reclassify`` subcommands rather
    than run on every cron tick.
    """

    settings = settings or get_settings()
    return [
        ConsolidationJob(storage, llm, embeddings, settings=settings),
        EntityResolutionJob(storage, llm=llm, settings=settings),
        MentionsJob(storage, settings=settings),
        CoMentionJob(storage, settings=settings),
        RelationNormJob(storage, settings=settings),
        EntityDedupJob(storage, embeddings=embeddings, settings=settings),
        CategoryMergeJob(storage, embeddings=embeddings, settings=settings),
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
        activity_log: ActivityLog | None = None,
    ) -> None:
        self._jobs = list(jobs)
        self._settings = settings or get_settings()
        # Optional WEBUI Q3 observability seam. Defaults to None so every existing
        # caller (CLI, tests) is unaffected; emit() fast-paths None / NullActivityLog.
        self._activity_log = activity_log

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
                "maintenance job %s: consolidated=%d merged=%d cat_merged=%d "
                "re_extracted=%d archived=%d pruned=%d edges_added=%d "
                "relations_merged=%d",
                report.job_name,
                report.consolidated,
                report.entities_merged,
                report.categories_merged,
                report.re_extracted,
                report.archived,
                report.edges_pruned,
                report.edges_added,
                report.relations_merged,
            )
            # WEBUI Q3 observability: record each job run on the activity feed.
            # Null-safe + error-swallowing (emit); a no-op unless a log is wired.
            emit(
                self._activity_log,
                maintenance_event(
                    job_name=report.job_name,
                    summary=(
                        f"maintenance {report.job_name}: "
                        f"consolidated={report.consolidated} merged={report.entities_merged} "
                        f"cat_merged={report.categories_merged} "
                        f"re_extracted={report.re_extracted} "
                        f"archived={report.archived} pruned={report.edges_pruned} "
                        f"edges_added={report.edges_added} "
                        f"relations_merged={report.relations_merged}"
                    ),
                    detail={
                        "consolidated": report.consolidated,
                        "entities_merged": report.entities_merged,
                        "categories_merged": report.categories_merged,
                        "re_extracted": report.re_extracted,
                        "archived": report.archived,
                        "edges_pruned": report.edges_pruned,
                        "edges_added": report.edges_added,
                        "relations_merged": report.relations_merged,
                        "notes": list(report.notes),
                    },
                ),
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
        _echo_report(r)


def _echo_report(r: MaintenanceReport) -> None:
    """Print a one-line summary of a report plus its notes (shared by subcommands)."""

    typer.echo(
        f"[{r.job_name}] consolidated={r.consolidated} merged={r.entities_merged} "
        f"cat_merged={r.categories_merged} re_extracted={r.re_extracted} "
        f"archived={r.archived} pruned={r.edges_pruned} "
        f"edges_added={r.edges_added} relations_merged={r.relations_merged}"
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


# ---------------------------------------------------------------------------
# Category merge (mnemozine-maintenance merge-categories)
# ---------------------------------------------------------------------------


async def _run_merge_categories(*, dry_run: bool = False) -> MaintenanceReport:
    """Build the wired :class:`CategoryMergeJob` from the live container and run it.

    The category-merge job needs only the storage backend (the category registry)
    and the embedding provider (to compare category *names*). When ``dry_run`` is
    set, the read-only :meth:`CategoryMergeJob.propose_merges` proposals are folded
    into a report's notes without applying any merge — the CLI preview of what the
    pass would do. Lazily imports the composition root to keep tests offline.
    """

    from mnemozine.app import Container  # local import: avoid cycle / keep tests offline

    container = Container.from_env()
    settings = container.settings
    storage = await container.build_storage()
    embeddings = container.build_embedding_provider()
    job = CategoryMergeJob(storage, embeddings=embeddings, settings=settings)
    try:
        if dry_run:
            proposals = await job.propose_merges()
            report = MaintenanceReport(job_name=job.name)
            for source, target in proposals:
                report.notes.append(f"would merge category '{source}' -> '{target}'")
            report.notes.append(f"dry-run: {len(proposals)} proposed merge(s)")
            return report
        return await job.run()
    finally:
        await container.close()


@maintenance_cli.command("merge-categories")
def _cmd_merge_categories(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only print the proposed (source -> canonical) merges; apply nothing.",
    ),
) -> None:
    """Merge near-duplicate emergent categories into a canonical one (FR-MNT-2/4).

    The category analogue of entity resolution: clusters the free-form
    ``MemoryUnit.category`` registry by name/embedding similarity above
    ``category.merge_similarity_threshold`` and folds each cluster into its
    highest-count canonical category. Idempotent: a re-run finds nothing left to
    merge. Use ``--dry-run`` to review the proposals first.
    """

    report = asyncio.run(_run_merge_categories(dry_run=dry_run))
    _echo_report(report)


# ---------------------------------------------------------------------------
# Persist mentions (mnemozine-maintenance persist-mentions) — graph connectivity
# ---------------------------------------------------------------------------


async def _run_persist_mentions(*, dry_run: bool = False) -> MaintenanceReport:
    """Build the wired :class:`MentionsJob` from the live container and run it.

    The mentions job needs only the storage backend (it MERGEs the
    memory->entity edges from each memory's ``m.entities`` name list). When
    ``dry_run`` is set we report the would-be assertion count via a read-only
    preview rather than writing edges — but because the persist is a single
    idempotent set-based MERGE there is no cheap pure proposal to enumerate, so
    the dry-run simply notes that no edges were written. Lazily imports the
    composition root to keep tests offline.
    """

    from mnemozine.app import Container  # local import: avoid cycle / keep tests offline

    container = Container.from_env()
    settings = container.settings
    storage = await container.build_storage()
    job = MentionsJob(storage, settings=settings)
    try:
        if dry_run:
            report = MaintenanceReport(job_name=job.name)
            report.notes.append(
                "dry-run: would MERGE memory->entity mention edges from m.entities "
                "(no edges written)"
            )
            return report
        return await job.run()
    finally:
        await container.close()


@maintenance_cli.command("persist-mentions")
def _cmd_persist_mentions(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Do not write any edges; only report that the pass would run.",
    ),
) -> None:
    """Persist (memory)-[:MNEMOZINE_MENTIONS]->(entity) edges from m.entities.

    Turns each memory's ``m.entities`` name list into real, traversable mention
    edges so the graph becomes navigable memory<->entity<->memory (the substrate
    the co-mention layer derives from). Idempotent (MERGE, never CREATE): a
    re-run asserts the same edges and adds nothing new.
    """

    report = asyncio.run(_run_persist_mentions(dry_run=dry_run))
    _echo_report(report)


# ---------------------------------------------------------------------------
# Co-mention (mnemozine-maintenance co-mention) — graph connectivity
# ---------------------------------------------------------------------------


async def _run_co_mention(
    *, dry_run: bool = False, min_shared: int | None = None
) -> MaintenanceReport:
    """Build the wired :class:`CoMentionJob` from the live container and run it.

    The co-mention job needs only the storage backend (it reads the
    mention-derived co-occurrence enumeration and upserts the weighted
    entity-entity edges). ``min_shared`` overrides ``graph.co_mention_min_shared``
    for this run. When ``dry_run`` is set we report what the pass *would* assert —
    the ranked/down-weighted/capped surviving pair count — via the read-only
    enumeration seams WITHOUT writing any edge. Lazily imports the composition
    root to keep tests offline.
    """

    from mnemozine.app import Container  # local import: avoid cycle / keep tests offline

    container = Container.from_env()
    settings = container.settings
    if min_shared is not None:
        # Override the configured threshold for this run only (CLI --min-shared).
        settings = settings.model_copy(
            update={
                "graph": settings.graph.model_copy(
                    update={"co_mention_min_shared": min_shared}
                )
            }
        )
    storage = await container.build_storage()
    job = CoMentionJob(storage, settings=settings)
    try:
        if dry_run:
            pairs = await storage.co_mention_pairs(
                min_shared=settings.graph.co_mention_min_shared
            )
            df = await storage.entity_mention_counts()
            kept = job._rank_downweight_and_cap(pairs, df)  # noqa: SLF001 - read-only preview
            report = MaintenanceReport(job_name=job.name)
            report.notes.append(
                f"dry-run: would assert {len(kept)} co-mention edge(s) "
                f"from {len(pairs)} co-occurring pair(s) (no edges written)"
            )
            return report
        return await job.run()
    finally:
        await container.close()


@maintenance_cli.command("co-mention")
def _cmd_co_mention(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Do not write any edges; only report how many would be asserted.",
    ),
    min_shared: int | None = typer.Option(
        None,
        "--min-shared",
        help=(
            "Minimum shared memories for a co-mention edge "
            "(overrides graph.co_mention_min_shared for this run)."
        ),
    ),
) -> None:
    """Derive weighted entity-entity co-mention edges from the mention layer.

    Reads the (memory)-[:MNEMOZINE_MENTIONS]->(entity) edges, links entities
    mentioned by the same memory, TF-IDF-style down-weights ultra-frequent hub
    entities, and caps the edges added per node so the layer does not become a
    hairball. Idempotent (MERGE, weight re-asserted not summed): a re-run writes
    the same edges. Use ``--dry-run`` to preview the count first.
    """

    report = asyncio.run(_run_co_mention(dry_run=dry_run, min_shared=min_shared))
    _echo_report(report)


# ---------------------------------------------------------------------------
# Relation normalization (mnemozine-maintenance normalize-relations)
# ---------------------------------------------------------------------------


async def _run_normalize_relations(*, dry_run: bool = False) -> MaintenanceReport:
    """Build the wired :class:`RelationNormJob` from the live container and run it.

    The relation-normalization job needs only the storage backend (it reads the
    relation registry and relabels the ``MNEMOZINE_RELATES`` edges through the
    controlled vocabulary). When ``dry_run`` is set, the read-only
    :meth:`RelationNormJob.propose_merges` proposals are folded into a report's
    notes without applying any merge — the CLI preview, modeled exactly on
    ``merge-categories`` --dry-run. Lazily imports the composition root to keep
    tests offline.
    """

    from mnemozine.app import Container  # local import: avoid cycle / keep tests offline

    container = Container.from_env()
    settings = container.settings
    storage = await container.build_storage()
    job = RelationNormJob(storage, settings=settings)
    try:
        if dry_run:
            proposals = await job.propose_merges()
            report = MaintenanceReport(job_name=job.name)
            for source, target in proposals:
                report.notes.append(
                    f"would normalize relation '{source}' -> '{target}'"
                )
            report.notes.append(f"dry-run: {len(proposals)} proposed merge(s)")
            return report
        return await job.run()
    finally:
        await container.close()


@maintenance_cli.command("normalize-relations")
def _cmd_normalize_relations(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only print the proposed (source -> canonical) relabels; apply nothing.",
    ),
) -> None:
    """Collapse the fragmented MNEMOZINE_RELATES label vocabulary (FR-MNT-2/4).

    The relation analogue of ``merge-categories``: maps each in-use relation label
    through a controlled vocabulary (``uses`` / ``used-in`` / ``used_in`` -> uses;
    ``depends-on`` / ``requires`` -> depends_on; …) and folds every non-canonical
    label into its canonical one, combining the parallel edges' weights and never
    leaving a duplicate parallel edge. Deterministic and embedding-free. Idempotent:
    a re-run finds every label already canonical and merges nothing. Use
    ``--dry-run`` to review the proposed relabels first.
    """

    report = asyncio.run(_run_normalize_relations(dry_run=dry_run))
    _echo_report(report)


# ---------------------------------------------------------------------------
# Entity dedup (mnemozine-maintenance dedup-entities) — graph connectivity
# ---------------------------------------------------------------------------


async def _run_dedup_entities(*, mode: str | None = None) -> MaintenanceReport:
    """Build the wired :class:`EntityDedupJob` from the live container and run it.

    Merges true-duplicate ENTITY nodes by driving the existing
    :meth:`StorageBackend.merge_entities` path (which repoints RELATES + MENTIONS +
    CO_MENTIONS onto the survivor). ``mode`` (CLI ``--mode``) overrides
    ``graph.entity_dedup_mode`` for this run: ``exact`` (default,
    ``lower(canonical_name)`` collisions only), ``alias``, or ``embedding`` (the
    fuzzier near-dup mode, which needs the embedding provider). Lazily imports the
    composition root to keep tests offline. No memory is ever deleted; only true
    duplicate entities are merged.
    """

    from mnemozine.app import Container  # local import: avoid cycle / keep tests offline

    container = Container.from_env()
    settings = container.settings
    storage = await container.build_storage()
    embeddings = container.build_embedding_provider()
    job = EntityDedupJob(storage, embeddings=embeddings, settings=settings, mode=mode)
    try:
        return await job.run()
    finally:
        await container.close()


@maintenance_cli.command("dedup-entities")
def _cmd_dedup_entities(
    mode: str = typer.Option(
        "exact",
        "--mode",
        help=(
            "Duplicate-detection mode: 'exact' (lower(canonical_name) collisions, "
            "default), 'alias' (also fold alias-linked entities), or 'embedding' "
            "(also fold near-dup names above graph.entity_dedup_similarity_threshold)."
        ),
    ),
) -> None:
    """Merge true-duplicate entities, repointing ALL edge types (FR-MNT-4).

    Groups duplicate ENTITY nodes (case/spacing drift in ``exact`` mode; also
    alias- or embedding-linked in the fuzzier modes), picks a deterministic
    survivor, and folds each duplicate into it via the existing
    ``merge_entities`` path — which now repoints the source's RELATES, MENTIONS,
    AND CO_MENTIONS edges onto the survivor so no edge type is orphaned.
    Idempotent: a re-run finds no duplicate group and merges nothing. No memory is
    ever deleted.
    """

    report = asyncio.run(_run_dedup_entities(mode=mode))
    _echo_report(report)


# ---------------------------------------------------------------------------
# Reclassify (mnemozine-maintenance reclassify) — re-tag from stored content
# ---------------------------------------------------------------------------


async def _run_reclassify(*, scope: str | None = None) -> MaintenanceReport:
    """Build the wired :class:`ReclassifyJob` and re-tag stored memories (R1).

    Re-scopes + re-categorizes existing memories from their stored content +
    provenance through the *current* classifier — no raw transcript needed, so it
    works after Claude's 30-day cleanup (R4). An optional ``scope`` string
    (canonical form, e.g. ``project:Mnemozine``) narrows the pass to one scope.
    """

    from mnemozine.app import Container  # local import: avoid cycle / keep tests offline

    container = Container.from_env()
    settings = container.settings
    storage = await container.build_storage()
    extractor = container.build_extractor()
    scope_obj = Scope.parse(scope, settings.scope.delimiter) if scope else None
    job = ReclassifyJob(storage, extractor, scope=scope_obj, settings=settings)
    try:
        return await job.run()
    finally:
        await container.close()


@maintenance_cli.command("reclassify")
def _cmd_reclassify(
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

    Reads each memory's already-stored content + provenance (no raw transcript)
    and re-applies the current scope/category/cross-ref decision, writing only the
    fields that drifted. Idempotent: a memory already matching the classifier is
    left untouched. Use this to apply a classifier prompt/model change to
    historical data offline.
    """

    report = asyncio.run(_run_reclassify(scope=scope))
    _echo_report(report)


# ---------------------------------------------------------------------------
# Re-extract (mnemozine-maintenance re-extract) — re-run extractor over raw tier
# ---------------------------------------------------------------------------


async def _run_re_extract(
    *,
    scope: str | None = None,
    session_id: str | None = None,
    supersede_existing: bool = True,
) -> MaintenanceReport:
    """Build the wired :class:`ReExtractJob` and re-run extraction over the raw tier.

    Re-processes the retained :class:`~mnemozine.schema.models.RawChunk` tier
    through the *current* extractor (applies a model/prompt change offline). The
    raw tier survives Claude's 30-day cleanup (R4). Optional ``scope`` /
    ``session_id`` narrow the sweep (EXACT scope — a re-extraction must not widen).
    """

    from mnemozine.app import Container  # local import: avoid cycle / keep tests offline

    container = Container.from_env()
    settings = container.settings
    storage = await container.build_storage()
    extractor = container.build_extractor()
    scope_obj = Scope.parse(scope, settings.scope.delimiter) if scope else None
    job = ReExtractJob(
        storage,
        extractor,
        scope=scope_obj,
        session_id=session_id,
        supersede_existing=supersede_existing,
        settings=settings,
    )
    try:
        return await job.run()
    finally:
        await container.close()


@maintenance_cli.command("re-extract")
def _cmd_re_extract(
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
            "previously produced (default: supersede them so the new extraction "
            "replaces the old)."
        ),
    ),
) -> None:
    """Re-run the current extractor over the retained raw tier (offline reindex).

    Re-processes stored RawChunks through the current extractor/classifier to
    apply a model or prompt change to already-ingested data, without the original
    transcript. By default the memories each chunk previously produced are
    superseded so the new extraction replaces the old; ``--keep-existing`` leaves
    them active. Idempotent: an unchanged extractor reinforces rather than
    duplicates.
    """

    report = asyncio.run(
        _run_re_extract(
            scope=scope,
            session_id=session_id,
            supersede_existing=not keep_existing,
        )
    )
    _echo_report(report)


def run_maintenance() -> None:
    """Console-script entrypoint for ``mnemozine-maintenance`` (FR-MNT-5).

    Target for the ``[project.scripts]`` ``mnemozine-maintenance`` entry. The
    integration pass should repoint that script (currently
    ``mnemozine.app:run_maintenance``, a Phase-0 stub) here, or call this from the
    stub. Subcommands: ``run`` (once) and ``serve`` (cron daemon).
    """

    maintenance_cli()
