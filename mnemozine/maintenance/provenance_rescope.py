"""Deterministic provenance re-scope of mis-globalized project memos (no LLM).

A historical classifier over-emitted ``scope=global`` for memories that are in
fact specific to ONE project's codebase (its code/architecture/build/bugs/…).
The classifier *prompt* fix corrects future ingests, but the ~700+ already-stored
global memos cannot be re-run through the LLM on the now-CPU-only Ollama (minutes
per call). This pass repairs them **deterministically** — no LLM — by reading
each memo's own *provenance* (the originating transcript path) and moving it to
its own source project's scope.

:class:`ProvenanceRescopeJob` streams the active GLOBAL memos
(:meth:`StorageBackend.iter_memories` ``scope=global``), and for each one:

* SKIPS the genuinely cross-project kinds — categories in
  ``maintenance.rescope_keep_global_categories`` (preference / convention / rule
  / idea), the operator preferences/conventions/rules/ideas that stay true in
  ANY project — they correctly belong at global.
* Otherwise parses the source PROJECT from ``provenance.raw_path`` by reusing the
  existing ingestion seam
  :func:`~mnemozine.ingestion.claude_code.derive_scope_from_transcript`, which
  already (a) finds the encoded-cwd project dir under ``projects/``, (b) decodes
  it to the friendly project name, and (c) ROLLS UP a subagent/workflow/worktree
  transcript to the PARENT project (never an opaque ``project:agent-XXXX``).
* Re-scopes the memo to ``project:<that one source project>`` via the scope-only
  :meth:`StorageBackend.reclassify_memory` (category / cross_ref left ``None``) —
  global -> project:<its own source project> ONLY (no-leak: never an unrelated
  project).

A memo is re-scopable only when its ``raw_path`` is present, non-empty, resolves
*through* a ``projects/`` ancestor to exactly one clear non-empty project segment,
and is not the classify sentinel. If the path is missing / ambiguous /
unparseable the memo is LEFT at global (skipped).

Idempotent (FR-MNT-5): once a memo is moved to project scope it no longer appears
in the global iteration, so a re-run touches nothing. Operator-triggered (the
``rescope-global`` subcommand), kept OUT of the default scheduled pass.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mnemozine.config import Settings, get_settings
from mnemozine.ingestion.claude_code import (
    PROJECTS_DIRNAME,
    derive_scope_from_transcript,
)
from mnemozine.interfaces import MaintenanceReport, StorageBackend
from mnemozine.schema.models import MemoryUnit, Scope

logger = logging.getLogger(__name__)

# How many moved-memo samples to surface in the report notes (cap so a large
# pass does not flood the audit log).
_SAMPLE_LIMIT = 10
# Content prefix length in the per-memo sample line.
_CONTENT_PREFIX = 60


class ProvenanceRescopeJob:
    """Deterministically re-scope mis-globalized project memos from provenance (R1, no LLM).

    A :class:`~mnemozine.interfaces.MaintenanceJob` that depends only on the
    :class:`~mnemozine.interfaces.StorageBackend` Protocol (enumerate global memos
    + scope-only :meth:`reclassify_memory`) and the deterministic
    :func:`derive_scope_from_transcript` parser seam — no LLM, no embeddings, no
    raw transcript text. Safe to re-run (moved memos leave the global iteration).
    """

    def __init__(
        self,
        storage: StorageBackend,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._storage = storage
        self._settings = settings or get_settings()
        # Normalize the keep-global set once (case-insensitive compare against the
        # already-normalized stored category).
        self._keep_global = {
            c.strip().lower()
            for c in self._settings.maintenance.rescope_keep_global_categories
            if c and c.strip()
        }

    @property
    def name(self) -> str:
        return "rescope_global"

    async def run(self) -> MaintenanceReport:
        """Re-scope every active global memo whose provenance resolves to one project."""

        report = MaintenanceReport(job_name=self.name)
        moved = 0
        kept_category = 0
        skipped_unparseable = 0
        scanned = 0
        samples: list[str] = []

        async for memory in self._storage.iter_memories(
            scope=Scope.global_(), active_only=True
        ):
            scanned += 1
            # Cross-project kinds (preferences/conventions/rules/ideas) stay global.
            if memory.category in self._keep_global:
                kept_category += 1
                continue
            target = self._source_project_scope(memory)
            if target is None:
                skipped_unparseable += 1
                continue
            await self._storage.reclassify_memory(memory.id, scope=target)
            moved += 1
            if len(samples) < _SAMPLE_LIMIT:
                prefix = memory.content[:_CONTENT_PREFIX].replace("\n", " ")
                samples.append(
                    f"{memory.id}: {prefix!r} global -> {target.as_str()}"
                )

        report.notes.append(
            f"re-scoped {moved}/{scanned} active global memor(ies) to their source "
            f"project from provenance (no LLM); kept {kept_category} cross-project "
            f"categor(ies) global, skipped {skipped_unparseable} unparseable/ambiguous"
        )
        for s in samples:
            report.notes.append(f"  moved {s}")
        # Reuse the consolidated counter so the runner's summary line surfaces it
        # (mirrors ReclassifyJob, which has no dedicated field either).
        report.consolidated = moved
        return report

    def _source_project_scope(self, memory: MemoryUnit) -> Scope | None:
        """Resolve a memo's source PROJECT scope from its provenance, or ``None``.

        Returns ``project:<source>`` only when the provenance ``raw_path`` is
        present, non-empty, not the classify sentinel, and resolves *through* a
        ``projects/`` ancestor (via the existing
        :func:`derive_scope_from_transcript` seam, which rolls subagent/workflow/
        worktree transcripts up to the parent project) to exactly ONE clear
        non-empty project segment. Otherwise ``None`` (leave the memo at global).
        """

        prov = memory.provenance
        if prov.is_classify_sentinel or prov.source == "classify":
            return None
        raw_path = (prov.raw_path or "").strip()
        if not raw_path:
            return None
        # Require a `projects/` ancestor on the path so derive_scope_from_transcript
        # resolves a REAL encoded-cwd project dir rather than falling back to the
        # bare parent-dir/stem name (which would be ambiguous).
        if PROJECTS_DIRNAME not in Path(raw_path).parts:
            return None
        try:
            scope = derive_scope_from_transcript(raw_path, self._settings)
        except Exception:  # noqa: BLE001 - a bad path must never crash the pass
            logger.warning(
                "rescope_global: could not parse provenance path %r for memory %s; "
                "leaving it at global",
                raw_path,
                memory.id,
                exc_info=True,
            )
            return None
        # The derived scope must be a clear project scope (never global, and with a
        # non-empty project segment). Roll-up already collapsed any subagent/
        # workflow path to the parent project; take the bare project scope so the
        # move is global -> project:<source> (no-leak, never a sub-scope leak).
        project_id = scope.project_id
        if scope.is_global or not project_id or not project_id.strip():
            return None
        return Scope.project(project_id)
