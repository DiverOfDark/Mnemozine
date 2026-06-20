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
* :class:`~mnemozine.maintenance.mentions.MentionsJob` — persists
  (memory)-[:MNEMOZINE_MENTIONS]->(entity) edges from each memory's ``m.entities``
  name list (the graph-connectivity substrate the co-mention layer derives from).
* :class:`~mnemozine.maintenance.co_mention.CoMentionJob` — derives the weighted
  entity-entity (entity)-[:MNEMOZINE_CO_MENTIONS]->(entity) co-mention layer from
  the mention edges, TF-IDF-style down-weighting ultra-frequent hub entities and
  capping the edges added per node so the layer does not become a hairball.
* :class:`~mnemozine.maintenance.relation_norm.RelationNormJob` — the relation
  analogue of category merge: collapses the fragmented ``MNEMOZINE_RELATES``
  relation-label vocabulary (``uses`` / ``used-in`` / ``used_in`` …) into a
  controlled vocabulary (:func:`~mnemozine.maintenance.relation_norm.normalize_relation`
  + :data:`~mnemozine.maintenance.relation_norm.RELATION_SYNONYMS`) via
  :meth:`~mnemozine.interfaces.StorageBackend.merge_relations` (the
  ``normalize-relations`` subcommand).
* :class:`~mnemozine.maintenance.entity_dedup.EntityDedupJob` — merges
  true-duplicate ENTITY nodes (case/spacing drift, alias, or — behind the flag —
  embedding near-dups) by driving the existing
  :meth:`~mnemozine.interfaces.StorageBackend.merge_entities` path, which repoints
  ALL three edge types (RELATES / MENTIONS / CO_MENTIONS) onto the survivor so no
  edge is orphaned (the ``dedup-entities`` subcommand; runs AFTER mentions +
  co-mention so all three layers exist to repoint). No memory is ever deleted.
* :class:`~mnemozine.maintenance.category_merge.CategoryMergeJob` — the category
  analogue of entity resolution: clusters near-duplicate emergent
  ``MemoryUnit.category`` strings (by name/embedding similarity) and folds each
  cluster into one canonical category (the ``merge-categories`` subcommand). Also
  satisfies the :class:`~mnemozine.interfaces.CategoryMerger` Protocol.
* :class:`~mnemozine.maintenance.reclassify.ReExtractJob` /
  :class:`~mnemozine.maintenance.reclassify.ReclassifyJob` — offline migration
  passes that re-apply the current extractor/classifier to already-ingested data:
  re-extract over the retained raw tier (``re-extract``) or re-tag stored
  memories from their content+provenance (``reclassify``). Operator-triggered, not
  in the default scheduled set.
* :class:`~mnemozine.maintenance.provenance_rescope.ProvenanceRescopeJob` — a
  DETERMINISTIC (no-LLM) offline pass that repairs mis-globalized project memos:
  it streams the active global memos and, for each whose category is not a
  cross-project kind, parses the source project from ``provenance.raw_path`` and
  re-scopes it global -> project:<its own source project> via ``reclassify_memory``
  (the ``rescope-global`` subcommand). Operator-triggered, not in the default set.
* :class:`~mnemozine.maintenance.memory_dedup.MemoryDedupJob` — a DETERMINISTIC
  (no-LLM) offline pass that collapses BYTE-FOR-BYTE duplicate active memos: it
  streams the active hot memos, buckets them by ``(normalized content, scope)``,
  and for each cluster of >=2 keeps one deterministic survivor and supersedes the
  rest via ``close_validity_window`` (retained, never deleted) — the
  ``dedup-memories`` subcommand. Operator-triggered, not in the default set.
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
from mnemozine.maintenance.category_merge import (
    CategoryMergeJob,
    name_similarity,
    normalize_category,
)
from mnemozine.maintenance.co_mention import CoMentionJob, co_mention_weight
from mnemozine.maintenance.consolidation import ConsolidationJob
from mnemozine.maintenance.decay import DecayJob, decay_score, rank_by_decay
from mnemozine.maintenance.decision import WriteDecider, WriteDecisionConfig
from mnemozine.maintenance.entity_dedup import DEDUP_MODES, EntityDedupJob
from mnemozine.maintenance.entity_resolution import EntityResolutionJob
from mnemozine.maintenance.memory_dedup import MemoryDedupJob
from mnemozine.maintenance.mentions import MentionsJob
from mnemozine.maintenance.migrate_index import MigrateIndexJob, needs_migration
from mnemozine.maintenance.provenance_rescope import ProvenanceRescopeJob
from mnemozine.maintenance.reclassify import ReclassifyJob, ReExtractJob
from mnemozine.maintenance.relation_norm import (
    RELATION_SYNONYMS,
    RelationNormJob,
    normalize_relation,
)
from mnemozine.maintenance.runner import (
    MaintenanceRunner,
    build_default_jobs,
    maintenance_cli,
    run_maintenance,
)

__all__ = [
    "DEDUP_MODES",
    "RELATION_SYNONYMS",
    "AuditJob",
    "CategoryMergeJob",
    "CoMentionJob",
    "ConsolidationJob",
    "DecayJob",
    "EntityDedupJob",
    "EntityResolutionJob",
    "MaintenanceRunner",
    "MemoryDedupJob",
    "MentionsJob",
    "MigrateIndexJob",
    "ProvenanceRescopeJob",
    "ReExtractJob",
    "ReclassifyJob",
    "RelationNormJob",
    "WriteDecider",
    "WriteDecisionConfig",
    "build_default_jobs",
    "co_mention_weight",
    "decay_score",
    "maintenance_cli",
    "name_similarity",
    "needs_migration",
    "normalize_category",
    "normalize_relation",
    "rank_by_decay",
    "run_maintenance",
]
