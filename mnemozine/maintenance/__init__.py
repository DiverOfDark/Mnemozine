"""Maintenance layer (FR-MNT-1..5) — the scheduled "consolidate, don't accumulate" passes.

This subpackage owns everything the PRD §6.5 maintenance layer covers, built
strictly against the :mod:`mnemozine.interfaces` Protocols (never another
module's concrete code):

* :class:`~mnemozine.maintenance.decision.WriteDecider` — the FR-MNT-1 4-way
  write decision (add / reinforce / **supersede** / no-op). The supersede branch
  runs a single narrowly-scoped cheap LLM contradiction check over
  ``type=preference`` candidates sharing >=1 entity in the same scope, and closes
  the loser's validity window (delivers UC-2 / Goal 2).
* :class:`~mnemozine.maintenance.consolidation.ConsolidationJob` — FR-MNT-2
  tiered raw -> fact -> theme periodic merge.
* :class:`~mnemozine.maintenance.decay.DecayJob` — FR-MNT-3 decay/expiry ranking
  by recency + access frequency; sink + archive, **never hard-delete**.
* :class:`~mnemozine.maintenance.entity_resolution.EntityResolutionJob` —
  FR-MNT-4 duplicate-entity merge + low-weight edge pruning + node-degree cap.
* :class:`~mnemozine.maintenance.migrate_index.MigrateIndexJob` — OQ3 vector
  index/re-embed migration on an embedding-dimension change (the
  ``mnemozine-maintenance migrate-index`` subcommand); not in the default
  scheduled set (operator-triggered).
* :class:`~mnemozine.maintenance.audit.AuditJob` — the R5 audit walk.
* :class:`~mnemozine.maintenance.runner.MaintenanceRunner` — the APScheduler
  cron runner (FR-MNT-5), idempotent and safe to re-run, plus the
  ``mnemozine-maintenance`` Typer console app.

Each job implements :class:`mnemozine.interfaces.MaintenanceJob`.
"""

from __future__ import annotations

from mnemozine.maintenance.audit import AuditJob
from mnemozine.maintenance.consolidation import ConsolidationJob
from mnemozine.maintenance.decay import DecayJob, decay_score, rank_by_decay
from mnemozine.maintenance.decision import WriteDecider, WriteDecisionConfig
from mnemozine.maintenance.entity_resolution import EntityResolutionJob
from mnemozine.maintenance.migrate_index import MigrateIndexJob, needs_migration
from mnemozine.maintenance.runner import (
    MaintenanceRunner,
    build_default_jobs,
    maintenance_cli,
    run_maintenance,
)

__all__ = [
    "AuditJob",
    "ConsolidationJob",
    "DecayJob",
    "EntityResolutionJob",
    "MaintenanceRunner",
    "MigrateIndexJob",
    "WriteDecider",
    "WriteDecisionConfig",
    "build_default_jobs",
    "decay_score",
    "maintenance_cli",
    "needs_migration",
    "rank_by_decay",
    "run_maintenance",
]
