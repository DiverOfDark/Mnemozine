"""The BASELINE migration to data_version 1 — a cheap, no-LLM RECLASSIFY.

This is the first concrete :class:`~mnemozine.migrations.Migration`, appended to
the :data:`~mnemozine.migrations.MIGRATIONS` registry at import time. It both
SEEDS the registry (so :data:`~mnemozine.migrations.CURRENT_DATA_VERSION` = 1 is
reachable) and DEMONSTRATES the in-place upgrade the framework exists for.

WHAT IT FIXES (the worktree-scope bug, in place)
------------------------------------------------
A subagent / workflow run inside a git worktree at
``<project>/.claude/worktrees/<id>`` historically got its memory scope stamped
from the opaque worktree leaf instead of rolling up to ``project:<project>``
(FR-EXT-3 no-leak). The classifier was later fixed
(:func:`~mnemozine.ingestion.claude_code.parser.derive_scope_from_transcript`
now strips the worktree suffix), but memories already written carry the stale
scope. This migration re-derives the correct scope for every below-v1 memory
**from its already-stored provenance + content** — the cheap path
(:meth:`~mnemozine.interfaces.StorageBackend.reclassify_memory`), no raw
transcript and no LLM needed, so it works long after Claude's 30-day local
cleanup (R4). It is exactly the in-place fix that would have corrected the bug
without a wipe + re-ingest.

HOW IT STAMPS (both tiers, FR-MNT-5 idempotency)
------------------------------------------------
Per the :class:`~mnemozine.migrations.Migration` contract, a migration must
advance EVERY tier
:meth:`~mnemozine.interfaces.StorageBackend.min_data_version` mins over, or the
floor never reaches the target and the migration re-runs on every boot:

* MEMORY tier — selected via
  :meth:`~mnemozine.interfaces.StorageBackend.iter_memories_below_version`; each
  selected memory is re-derived and stamped to
  :data:`~mnemozine.migrations.CURRENT_DATA_VERSION`. When the re-derived scope
  *changed* it is written through
  :meth:`~mnemozine.interfaces.StorageBackend.reclassify_memory` (which
  re-stamps implicitly); when it did NOT change the memory is stamped explicitly
  via :meth:`~mnemozine.interfaces.StorageBackend.set_data_version` (so an
  already-correct memory still advances its version — same ids, no new node, no
  delete: a true in-place upgrade).
* RAW-CHUNK tier — selected via
  :meth:`~mnemozine.interfaces.StorageBackend.iter_chunks_below_version` and
  advanced via
  :meth:`~mnemozine.interfaces.StorageBackend.set_chunk_data_version` (the cheap
  path: re-stamp without re-extracting).

Re-running over an already-migrated store finds nothing below v1 in either tier
and is a no-op.
"""

from __future__ import annotations

import logging

from mnemozine.interfaces import Extractor, MaintenanceReport, StorageBackend
from mnemozine.migrations.report import MigrationReport
from mnemozine.schema.models import MemoryUnit, Scope

logger = logging.getLogger(__name__)

#: The data-model version this migration produces (the BASELINE target).
_BASELINE_VERSION = 1

#: Git-worktree path marker (mirrors the un-exported local in
#: ``ingestion.claude_code.parser``). A subagent/workflow cwd of
#: ``<project>/.claude/worktrees/<id>`` must roll up to ``<project>`` (FR-EXT-3);
#: presence of this marker in a stored ``raw_path`` is the worktree-scope signal.
_WORKTREE_MARKER = "/.claude/worktrees/"


def rederive_scope(memory: MemoryUnit) -> Scope | None:
    """Re-derive a memory's correct scope from its STORED provenance + content.

    The cheap, no-LLM, no-raw-transcript path: the originating transcript path is
    retained on :attr:`~mnemozine.schema.models.Provenance.raw_path`, so the same
    deterministic mapping the (fixed) classifier uses
    (:func:`~mnemozine.ingestion.claude_code.parser.derive_scope_from_transcript`)
    can be replayed off stored data alone. That mapping strips a
    ``<project>/.claude/worktrees/<id>`` suffix, so a memory wrongly scoped to an
    opaque worktree id rolls back up to ``project:<project>`` — the in-place fix
    for the worktree-scope bug.

    Returns the re-derived :class:`~mnemozine.schema.models.Scope`, or ``None``
    when there is no usable provenance to re-derive from (no ``raw_path``, or a
    classify-sentinel / global memory whose scope is not transcript-derived) — in
    which case the caller leaves the stored scope untouched and only re-stamps the
    version. ``None`` is also returned if the re-derived scope equals the stored
    one (no change needed), so the caller can take the explicit-stamp fast path.
    """

    prov = memory.provenance
    # A classify-sentinel unit (Extractor.classify path) has no real session /
    # transcript to re-derive from; leave its scope as-is.
    if prov.is_classify_sentinel:
        return None
    raw_path = prov.raw_path
    if not raw_path:
        return None
    # Local import to avoid a heavy ingestion import at module load (and any cycle
    # through config); only needed when there is a raw_path to map.
    from mnemozine.ingestion.claude_code.parser import derive_scope_from_transcript

    try:
        # The stored raw_path IS the literal working-directory transcript path, so
        # pass it as the ``cwd`` hint: that drives the same git-worktree roll-up the
        # FIXED classifier applies (strip a ``<project>/.claude/worktrees/<id>``
        # suffix -> ``project:<project>``). A memory wrongly scoped to the opaque
        # worktree id therefore rolls back up to its real project. When the path has
        # no worktree marker the cwd hint is harmless (same result as path-only).
        cwd_hint = raw_path if _WORKTREE_MARKER in raw_path else None
        rederived = derive_scope_from_transcript(raw_path, cwd=cwd_hint)
    except Exception:  # noqa: BLE001 - a bad stored path must not abort the migration
        logger.warning(
            "baseline migration: could not re-derive scope for memory %s from "
            "raw_path %r; leaving scope unchanged",
            memory.id,
            raw_path,
            exc_info=True,
        )
        return None
    if rederived.as_str() == memory.scope.as_str():
        return None
    return rederived


class BaselineReclassifyMigration:
    """Cheap RECLASSIFY migration to data_version 1 (the worktree-scope fix).

    Satisfies the :class:`~mnemozine.migrations.Migration` Protocol structurally
    (``version`` / ``description`` / ``requires_reextract`` properties + an async
    :meth:`run`). It is a CHEAP migration — :attr:`requires_reextract` is
    ``False`` — so the startup hook may auto-apply it and it ignores any
    ``extractor`` passed to :meth:`run`.
    """

    @property
    def version(self) -> int:
        return _BASELINE_VERSION

    @property
    def description(self) -> str:
        return (
            "baseline v1: re-derive scope/category from stored provenance+content "
            "(cheap reclassify; fixes worktree-scoped memories in place)"
        )

    @property
    def requires_reextract(self) -> bool:
        # Cheap reclassify path: re-derives from already-stored data; no extractor
        # / GPU needed, safe to auto-run at startup.
        return False

    async def run(
        self, backend: StorageBackend, *, extractor: Extractor | None = None
    ) -> MaintenanceReport:
        """Re-derive scope for every below-v1 memory and stamp both tiers to v1.

        Idempotent (FR-MNT-5): selects only records below
        :attr:`version`, re-derives + stamps them, and is a no-op on a re-run
        (nothing left below v1). ``extractor`` is ignored — this is the cheap,
        no-LLM path. Returns a :class:`~mnemozine.interfaces.MaintenanceReport`
        (built from a :class:`~mnemozine.migrations.report.MigrationReport` so the
        from/to version is recorded in the notes).
        """

        del extractor  # cheap path: no re-extraction.
        report = MigrationReport(
            migration="migrate_baseline_v1",
            from_version=0,
            to_version=self.version,
        )

        rescoped = 0
        stamped_only: list[str] = []
        scanned = 0
        async for memory in backend.iter_memories_below_version(self.version):
            scanned += 1
            new_scope = rederive_scope(memory)
            if new_scope is not None:
                # Scope drifted (e.g. the worktree bug): re-scope through the cheap
                # reclassify seam, which ALSO re-stamps data_version implicitly.
                old = memory.scope.as_str()
                await backend.reclassify_memory(memory.id, scope=new_scope)
                rescoped += 1
                report.notes.append(
                    f"memory {memory.id}: scope {old} -> {new_scope.as_str()}"
                )
            else:
                # Already-correct (or non-transcript) memory: re-stamp in place so
                # the version floor still advances. Batched below.
                stamped_only.append(memory.id)

        stamped = 0
        if stamped_only:
            stamped = await backend.set_data_version(stamped_only, self.version)

        # RAW-CHUNK tier: even on the cheap path we MUST advance the chunks we
        # selected, or min_data_version() never reaches v1 and the migration
        # re-runs on every boot (the chunk-version seam, see the Migration
        # contract). Re-stamp without re-extracting.
        chunk_hashes = [
            chunk.content_hash
            async for chunk in backend.iter_chunks_below_version(self.version)
        ]
        chunks_stamped = 0
        if chunk_hashes:
            chunks_stamped = await backend.set_chunk_data_version(
                chunk_hashes, self.version
            )

        report.migrated = rescoped + stamped + chunks_stamped
        report.notes.append(
            f"scanned {scanned} memor(ies) below v{self.version}: "
            f"{rescoped} re-scoped, {stamped} stamped-in-place, "
            f"{chunks_stamped} raw chunk(s) stamped"
        )
        return report.to_maintenance()


#: The singleton baseline migration instance registered in MIGRATIONS.
BASELINE_MIGRATION = BaselineReclassifyMigration()


def register() -> None:
    """Append :data:`BASELINE_MIGRATION` to the registry (idempotent).

    The :data:`~mnemozine.migrations.MIGRATIONS` registry is intentionally empty
    in the import-light :mod:`mnemozine.migrations.__init__` (so
    :mod:`mnemozine.schema.models` can read
    :data:`~mnemozine.migrations.CURRENT_DATA_VERSION` without pulling this module
    and creating an import cycle). Concrete migrations register themselves here,
    away from that import path. Importing this module performs the registration;
    calling :func:`register` again is a no-op (the registry is matched by version
    so a re-import / repeated call never duplicates the step). After registering it
    re-validates the registry invariant so a mis-version fails loudly.
    """

    from mnemozine.migrations import MIGRATIONS, validate_migrations

    if any(m.version == BASELINE_MIGRATION.version for m in MIGRATIONS):
        return
    MIGRATIONS.append(BASELINE_MIGRATION)
    # Re-run the registry guard now that the concrete migration is present so a
    # mis-registration (gap / over-current version) fails loudly at import time.
    validate_migrations()


# Register on import: importing the runner (or this module) seeds the registry.
register()


__all__ = [
    "BASELINE_MIGRATION",
    "BaselineReclassifyMigration",
    "register",
    "rederive_scope",
]
