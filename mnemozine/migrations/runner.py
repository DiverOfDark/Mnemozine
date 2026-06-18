"""The in-place migration RUNNER — drive the pending :data:`MIGRATIONS` set.

This is the engine the ``mnemozine-maintenance migrate`` subcommand and the
:func:`mnemozine.app._apply_startup_migrations` startup hook both code against. It
turns the declarative pieces in :mod:`mnemozine.migrations` (the
:data:`~mnemozine.migrations.CURRENT_DATA_VERSION` target, the
:data:`~mnemozine.migrations.MIGRATIONS` registry, and
:func:`~mnemozine.migrations.pending_migrations`) into one idempotent, resumable
apply pass over a live :class:`~mnemozine.interfaces.StorageBackend`:

* it reads the store's
  :meth:`~mnemozine.interfaces.StorageBackend.min_data_version` (the lowest
  ``data_version`` across BOTH the memory and raw-chunk tiers),
* selects every registered :class:`~mnemozine.migrations.Migration` whose
  :attr:`~mnemozine.migrations.Migration.version` is in
  ``(min_data_version, CURRENT_DATA_VERSION]`` (the "pending" set, ascending),
* runs each in order via :meth:`~mnemozine.migrations.Migration.run`, and
* aggregates a :class:`MigrationRunReport` (per-step
  :class:`~mnemozine.interfaces.MaintenanceReport`s plus the from/to versions).

RESUMABLE + SAFE TO RE-RUN (FR-MNT-5): the runner re-reads
:meth:`~mnemozine.interfaces.StorageBackend.min_data_version` BEFORE every step
and SKIPS any migration whose version the store already reached (e.g. a prior
interrupted run, a concurrent process, or a finished migration). A migration that
re-stamps the records it selects is therefore a true no-op on a re-run — both the
``--dry-run`` plan and a real apply over a fully-migrated store report zero work.

COST CONTRACT: each migration declares
:attr:`~mnemozine.migrations.Migration.requires_reextract`. The startup hook runs
only CHEAP (reclassify) migrations automatically; this runner honors the same
split via ``include_reextract`` (default ``True`` for the explicit
operator-triggered CLI, set ``False`` from the auto-on-startup path) and requires
an ``extractor`` only when a selected heavy migration needs one.

This module imports the heavier :mod:`mnemozine.interfaces` /
:mod:`mnemozine.schema` types at module scope (unlike
:mod:`mnemozine.migrations.__init__`, which stays import-light for the
schema->migrations one-way import). The runner is downstream of the schema, so it
is free to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from mnemozine.interfaces import Extractor, MaintenanceReport, StorageBackend
from mnemozine.migrations import (
    CURRENT_DATA_VERSION,
    Migration,
    pending_migrations,
)

# Importing the baseline module self-registers BASELINE_MIGRATION into the
# (otherwise empty, import-light) MIGRATIONS registry. The runner is imported by
# every apply path (the CLI subcommand and the startup hook), so importing it here
# guarantees the registry is seeded before any pending/run query. Kept here rather
# than in migrations/__init__ to preserve the schema -> migrations import-light
# one-way (no cycle through schema.models).
from mnemozine.migrations import baseline as _baseline  # noqa: F401
from mnemozine.migrations import entity_name_key as _entity_name_key  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MigrationStepPlan:
    """One planned migration step in a :meth:`MigrationRunner.plan` preview.

    Pure description (no writes): the migration's
    :attr:`~mnemozine.migrations.Migration.version` / ``description`` and whether
    it is a heavy re-extract pass (:attr:`requires_reextract`) or a cheap
    reclassify. ``skipped`` is ``True`` for a step the runner would NOT apply on
    this pass — a heavy migration excluded by ``include_reextract`` — with the
    reason in :attr:`skip_reason`.
    """

    version: int
    description: str
    requires_reextract: bool
    skipped: bool = False
    skip_reason: str | None = None


@dataclass(slots=True)
class MigrationRunReport:
    """Aggregate report of one runner pass over the pending migrations.

    ``from_version`` is the store's :meth:`min_data_version` at the start of the
    pass and ``to_version`` is the version it reached at the end (== the highest
    applied migration's version, or ``from_version`` when nothing ran). ``steps``
    holds the per-migration :class:`~mnemozine.interfaces.MaintenanceReport`s (in
    apply order), ``migrated`` is the total records touched across them, and
    ``dry_run`` records whether this was a plan-only pass.
    """

    from_version: int
    to_version: int
    dry_run: bool = False
    plan: list[MigrationStepPlan] = field(default_factory=list)
    steps: list[MaintenanceReport] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def migrated(self) -> int:
        """Total records re-derived / re-stamped across every applied step."""

        return sum(s.re_extracted for s in self.steps)

    @property
    def applied(self) -> int:
        """Number of migrations actually run on this pass (0 on a dry-run)."""

        return len(self.steps)


class MigrationRunner:
    """Apply the pending :data:`~mnemozine.migrations.MIGRATIONS` over a backend.

    Holds no storage state itself — it is handed a
    :class:`~mnemozine.interfaces.StorageBackend` (and, for heavy re-extract
    migrations, an :class:`~mnemozine.interfaces.Extractor`) so it depends purely
    on the Protocols and is trivially testable against the conftest fakes.

    A fresh instance per pass is fine; the runner keeps no cross-pass state. The
    registry it applies is the module-level
    :data:`~mnemozine.migrations.MIGRATIONS` (resolved through
    :func:`~mnemozine.migrations.pending_migrations`, which validates the registry
    invariant), so appending a migration there is the entire surface for shipping a
    new step.
    """

    def __init__(
        self,
        storage: StorageBackend,
        *,
        extractor: Extractor | None = None,
        target_version: int = CURRENT_DATA_VERSION,
    ) -> None:
        self._storage = storage
        self._extractor = extractor
        self._target = target_version

    async def pending(self) -> list[Migration]:
        """Return the migrations still needed, ascending (pure selection).

        Reads the store's :meth:`min_data_version` and returns every registered
        migration with a strictly-greater version (capped at the runner's target).
        An empty list means the store is fully migrated.
        """

        current = await self._storage.min_data_version()
        return [m for m in pending_migrations(current) if m.version <= self._target]

    async def plan(self, *, include_reextract: bool = True) -> MigrationRunReport:
        """Compute the apply plan WITHOUT touching the store (the ``--dry-run``).

        Returns a :class:`MigrationRunReport` whose :attr:`~MigrationRunReport.plan`
        lists each pending step and whether it would be applied or skipped (a heavy
        re-extract migration is marked skipped when ``include_reextract`` is
        ``False``). No :attr:`~MigrationRunReport.steps` are produced because
        nothing runs. Safe to call repeatedly; pure with respect to the store.
        """

        current = await self._storage.min_data_version()
        pending = await self.pending()
        report = MigrationRunReport(
            from_version=current, to_version=current, dry_run=True
        )
        if not pending:
            report.notes.append(
                f"store already at data_version {current} "
                f"(target {self._target}); nothing pending"
            )
            return report
        for migration in pending:
            skipped = migration.requires_reextract and not include_reextract
            report.plan.append(
                MigrationStepPlan(
                    version=migration.version,
                    description=migration.description,
                    requires_reextract=migration.requires_reextract,
                    skipped=skipped,
                    skip_reason=(
                        "heavy re-extract migration excluded from this pass"
                        if skipped
                        else None
                    ),
                )
            )
        applicable = [step for step in report.plan if not step.skipped]
        if applicable:
            report.to_version = max(step.version for step in applicable)
        report.notes.append(
            f"dry-run: {len(applicable)} migration(s) to apply, "
            f"{len(report.plan) - len(applicable)} skipped "
            f"(data_version {current} -> {report.to_version})"
        )
        return report

    async def run(self, *, include_reextract: bool = True) -> MigrationRunReport:
        """Apply the pending migrations in order, idempotently (FR-MNT-5).

        Re-reads :meth:`min_data_version` before EACH step and skips any migration
        whose version the store already reached, so the pass is resumable and a
        re-run over a migrated store is a no-op. A heavy re-extract migration is
        skipped (and noted) when ``include_reextract`` is ``False`` (the
        auto-on-startup contract). Raises :class:`MigrationExtractorRequired` if a
        selected heavy migration has no ``extractor`` to run with. Returns the
        aggregated :class:`MigrationRunReport`.
        """

        current = await self._storage.min_data_version()
        report = MigrationRunReport(from_version=current, to_version=current)
        pending = await self.pending()
        if not pending:
            report.notes.append(
                f"store already at data_version {current} "
                f"(target {self._target}); nothing to do"
            )
            return report

        for migration in pending:
            # Resumable gate: re-read the floor and skip anything already reached
            # (a prior partial run, a concurrent process, or a no-op migration).
            current = await self._storage.min_data_version()
            if migration.version <= current:
                report.notes.append(
                    f"v{migration.version} ({migration.description}): "
                    f"already at data_version {current}; skipped"
                )
                continue
            if migration.requires_reextract and not include_reextract:
                report.notes.append(
                    f"v{migration.version} ({migration.description}): "
                    "heavy re-extract migration skipped (not auto-applied)"
                )
                continue
            if migration.requires_reextract and self._extractor is None:
                raise MigrationExtractorRequired(
                    f"migration v{migration.version} ({migration.description}) "
                    "requires an extractor (re-extract from raw chunks)"
                )
            logger.info(
                "migration v%d (%s): applying (data_version %d -> %d)",
                migration.version,
                migration.description,
                current,
                migration.version,
            )
            step = await migration.run(self._storage, extractor=self._extractor)
            report.steps.append(step)
            report.to_version = migration.version
            logger.info(
                "migration v%d (%s): re_extracted=%d notes=%s",
                migration.version,
                migration.description,
                step.re_extracted,
                step.notes,
            )
        report.notes.append(
            f"applied {report.applied} migration(s); "
            f"data_version {report.from_version} -> {report.to_version}"
        )
        return report


class MigrationExtractorRequired(RuntimeError):
    """A selected heavy re-extract migration was run without an ``extractor``.

    Raised by :meth:`MigrationRunner.run` when a pending migration has
    :attr:`~mnemozine.migrations.Migration.requires_reextract` set but no
    :class:`~mnemozine.interfaces.Extractor` was provided. Heavy migrations need
    the extractor to re-run over the raw tier; failing loudly here is better than
    a confusing ``None`` deref inside the migration.
    """


__all__ = [
    "MigrationExtractorRequired",
    "MigrationRunReport",
    "MigrationRunner",
    "MigrationStepPlan",
]
