"""Collapse exact-duplicate active memos to one survivor (supersede the rest).

Repeated ingests of the same conversation (or the same operator preference stated
verbatim in many sessions) can leave several BYTE-FOR-BYTE identical active memos
in the same scope — e.g. three live copies of ``If any build is failing - retry
it`` at ``global``. They all match a recall and crowd the top-k with redundant
rows. This pass collapses each such exact-duplicate cluster to ONE active survivor
and supersedes the rest — it never destroys unique content (no memo is deleted;
the redundant copies are retained with a closed validity window, exactly like a
FR-MNT-1 supersede).

:class:`MemoryDedupJob` streams the active HOT memos
(:meth:`StorageBackend.iter_memories` ``active_only=True, tier=hot``) and buckets
them by the EXACT-duplicate key ``(normalized content, final scope)`` where the
content is ``content.strip().lower()`` and the scope is its canonical
:meth:`Scope.as_str`. For each bucket of >=2 it picks a DETERMINISTIC survivor
(highest confidence, then earliest ``valid_from``, then smallest ``id``) and
closes the other copies' validity windows via the existing
:meth:`StorageBackend.close_validity_window` — supersede, never delete.

Buckets are keyed on BOTH content AND scope, so a memo that merely shares wording
with a memo in a DIFFERENT scope is left untouched (the scope split is preserved);
only true same-scope exact duplicates collapse.

Idempotent (FR-MNT-5): a superseded copy leaves the active set, so after one pass
every key is single-membered and a re-run finds nothing to collapse. Depends only
on the :class:`~mnemozine.interfaces.StorageBackend` Protocol (enumerate +
``close_validity_window``) — no LLM, no embeddings — so it is unit-testable
offline against the conftest fakes. Operator-triggered (the ``dedup-memories``
subcommand), kept OUT of the default scheduled pass.
"""

from __future__ import annotations

import logging

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import MaintenanceReport, StorageBackend
from mnemozine.schema.models import MemoryUnit, Tier

logger = logging.getLogger(__name__)

# How many collapsed-group samples to surface in the report notes (cap so a large
# pass does not flood the audit log).
_SAMPLE_LIMIT = 10
# Content prefix length in the per-group sample line.
_CONTENT_PREFIX = 60


def _dedup_key(memory: MemoryUnit) -> tuple[str, str]:
    """The EXACT-duplicate bucket key: (normalized content, canonical scope).

    Content is ``strip().lower()`` so whitespace/case-only differences collapse;
    scope is its canonical persisted string so two copies collapse ONLY within the
    same scope (a cross-scope wording match is never merged — the scope split is
    load-bearing for the no-leak retrieval boundary).
    """

    return (memory.content.strip().lower(), memory.scope.as_str())


def _survivor_sort_key(memory: MemoryUnit) -> tuple[float, str, str]:
    """Deterministic survivor ordering: max confidence, earliest valid_from, min id.

    Returned so ``min(bucket, key=...)`` selects the survivor: negate confidence so
    the HIGHEST confidence sorts first; ``valid_from`` ISO string sorts earliest
    first (timezone-aware datetimes compare chronologically); ``id`` is the final
    stable tie-break so the choice is fully deterministic across runs.
    """

    return (-memory.confidence, memory.valid_from.isoformat(), memory.id)


class MemoryDedupJob:
    """Collapse exact-duplicate active memos to one survivor (supersede the rest).

    A :class:`~mnemozine.interfaces.MaintenanceJob` that depends only on the
    :class:`~mnemozine.interfaces.StorageBackend` Protocol (stream active hot
    memos + :meth:`close_validity_window`) — no LLM, no embeddings. Safe to re-run
    (superseded copies leave the active set, so each key is single-membered on a
    second pass).
    """

    def __init__(
        self,
        storage: StorageBackend,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._storage = storage
        self._settings = settings or get_settings()

    @property
    def name(self) -> str:
        return "dedup_memories"

    async def run(self) -> MaintenanceReport:
        """Collapse each exact-duplicate (content, scope) cluster to one survivor."""

        report = MaintenanceReport(job_name=self.name)
        # Bucket the whole active hot set by exact-duplicate key. Preserve insertion
        # order within a bucket only for stability; the survivor is chosen by an
        # explicit deterministic key below, independent of iteration order.
        buckets: dict[tuple[str, str], list[MemoryUnit]] = {}
        scanned = 0
        async for memory in self._storage.iter_memories(
            active_only=True, tier=Tier.HOT
        ):
            scanned += 1
            buckets.setdefault(_dedup_key(memory), []).append(memory)

        collapsed_groups = 0
        superseded = 0
        samples: list[str] = []
        # Iterate buckets in a deterministic order (by key) so the report sample is
        # reproducible regardless of store iteration order.
        for key in sorted(buckets):
            bucket = buckets[key]
            if len(bucket) < 2:
                continue
            survivor = min(bucket, key=_survivor_sort_key)
            redundant = [m for m in bucket if m.id != survivor.id]
            for memo in redundant:
                await self._storage.close_validity_window(memo.id)
                superseded += 1
            collapsed_groups += 1
            if len(samples) < _SAMPLE_LIMIT:
                content_key, scope_str = key
                prefix = content_key[:_CONTENT_PREFIX].replace("\n", " ")
                samples.append(
                    f"{prefix!r} @ {scope_str}: kept {survivor.id}, "
                    f"superseded {len(redundant)} duplicate(s)"
                )

        report.notes.append(
            f"collapsed {collapsed_groups} exact-duplicate group(s) over {scanned} "
            f"active hot memor(ies); superseded {superseded} redundant cop(ies) "
            "(retained, not deleted)"
        )
        for s in samples:
            report.notes.append(f"  collapsed {s}")
        # Reuse the consolidated counter so the runner's summary line surfaces it
        # (mirrors ReclassifyJob / ProvenanceRescopeJob, which have no dedicated
        # field either). Count the redundant copies removed from the active set.
        report.consolidated = superseded
        return report
