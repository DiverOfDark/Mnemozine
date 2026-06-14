"""Maintenance / Ops read routes (PRD §4.7 / §6 GET /api/maintenance[/merge-candidates]).

The **read** half of the Ops screen: the scheduler status + the per-job set, and
the FR-MNT-4 entity-resolution merge-candidate review queue.

The maintenance scheduler is a separate ``mnemozine-maintenance serve`` process
(not the web process), so the console reports the configured cron + the
deterministic job list (the ``build_default_jobs`` order plus the OQ3
migrate-index op) and leaves ``scheduler_running`` False — the web process does not
host the cron daemon. Merge candidates are computed the way the entity-resolution
job groups duplicates (normalized-key grouping over ``iter_entities``) so the
review queue matches what a run would merge.

The job-trigger *write* (``POST /api/maintenance/{job}/run``) lives in
``mutations.py`` (the single auditable write surface, PRD §2); this module is
read-only. Runs against the in-memory fake in tests.
"""

from __future__ import annotations

from itertools import combinations

from fastapi import APIRouter

from mnemozine.maintenance.entity_resolution import normalize_entity_key
from mnemozine.schema.models import Entity
from mnemozine.web.deps import SettingsDep, StorageDep
from mnemozine.web.schemas import (
    MaintenanceJobStatus,
    MaintenanceStatusResponse,
    MergeCandidate,
    MergeCandidatesResponse,
)

router = APIRouter(prefix="/api/maintenance", tags=["maintenance"])

# The deterministic job set surfaced in the UI. Mirrors build_default_jobs order
# (consolidation, entity-resolution, decay, audit) plus the OQ3 migrate-index op,
# using the hyphenated wire names the API contract froze.
_JOB_NAMES = ["consolidate", "entity-resolution", "decay", "audit", "migrate-index"]


@router.get("", response_model=MaintenanceStatusResponse, summary="Scheduler + job status")
async def maintenance_status(settings: SettingsDep) -> MaintenanceStatusResponse:
    """Scheduler status + per-job set (PRD §4.7).

    Reports the configured cron and the deterministic job list. The cron daemon is
    a separate ``mnemozine-maintenance serve`` process, so ``scheduler_running`` is
    False here — the web process triggers jobs on demand rather than hosting the
    schedule.
    """

    return MaintenanceStatusResponse(
        cron=settings.maintenance.cron,
        scheduler_running=False,
        jobs=[
            MaintenanceJobStatus(name=name, enabled=True, last_run=None, next_run=None)
            for name in _JOB_NAMES
        ],
    )


def _name_similarity(a: str, b: str) -> float:
    """A cheap, deterministic 0..1 name-similarity for the review row.

    Character Jaccard over the lowercased names — enough to rank obvious duplicates
    for the HITL queue without an embedding round-trip. The operator decides.
    """

    sa, sb = set(a.lower()), set(b.lower())
    if not sa and not sb:
        return 1.0
    union = len(sa | sb)
    return round(len(sa & sb) / union, 4) if union else 0.0


@router.get(
    "/merge-candidates",
    response_model=MergeCandidatesResponse,
    summary="Entity-resolution review queue",
)
async def merge_candidates(storage: StorageDep) -> MergeCandidatesResponse:
    """Entity-resolution merge candidates for HITL review (PRD §4.7, FR-MNT-4).

    Groups entities by the same normalized key the entity-resolution job uses
    (``rust`` / ``rust-lang`` / "the Rust work" collapse to one key) and emits a
    candidate per duplicate pair within a group, scored by name similarity and the
    count of shared graph neighbors. This is exactly the set a resolution run would
    merge, surfaced for the operator to confirm before it does.
    """

    groups: dict[str, list[Entity]] = {}
    async for entity in storage.iter_entities():
        key = normalize_entity_key(entity.canonical_name)
        groups.setdefault(key, []).append(entity)

    candidates: list[MergeCandidate] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        # Pre-fetch each entity's neighbor id set for the shared-neighbor count.
        neighbor_ids: dict[str, set[str]] = {}
        for ent in group:
            nbrs = await storage.neighbors(ent.canonical_name, active_only=True)
            neighbor_ids[ent.id] = {n.entity.id for n in nbrs}
        for a, b in combinations(group, 2):
            shared = len(neighbor_ids.get(a.id, set()) & neighbor_ids.get(b.id, set()))
            candidates.append(
                MergeCandidate(
                    source_id=a.id,
                    source_name=a.canonical_name,
                    target_id=b.id,
                    target_name=b.canonical_name,
                    similarity=_name_similarity(a.canonical_name, b.canonical_name),
                    shared_neighbors=shared,
                )
            )
    return MergeCandidatesResponse(candidates=candidates)


__all__ = ["router"]
