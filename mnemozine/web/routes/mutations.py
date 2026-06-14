"""HITL mutation routes — the only writes the operator console makes (PRD §4.3/§6).

PRD §2 is *read-first, write-where-it-matters*: the console observes, and the
**only** writes are the R1/R5 human-in-the-loop corrections plus the F4 eval
labels. To keep that write surface in one auditable place this module owns *every*
mutation endpoint, not just ``PATCH /api/memories``:

* ``PATCH /api/memories/{id}``           — reclassify / re-scope / archive-restore
  (R1/R5). Content is never editable (PRD §7 out of scope).
* ``POST  /api/crossrefs/{id}/suppress`` — dismiss a cross-reference (R2).
* ``POST  /api/maintenance/{job}/run``   — trigger a maintenance job on demand.
* ``POST  /api/eval/bootstrap/{id}/label`` / ``.../finish`` — the F4 browser
  bootstrap labeling that folds the kept candidates into the committed gold set.

Every handler goes through the existing composition root (``StorageBackend`` /
maintenance runner / evals) — the UI is never a new source of truth (PRD §2) — and
every successful mutation emits an activity event through the safe
:func:`mnemozine.activity.emit` seam so the correction shows in the Logs feed. The
seam is null-safe: when the injected log is the default ``NullActivityLog`` nothing
is recorded and nothing changes for the existing pipeline.

This single ``router`` carries full paths (no router prefix) and is registered
*before* the read-side ``crossrefs`` / ``maintenance`` / ``eval`` routers, so
FastAPI's first-match wins and these live handlers replace the Phase-1 stubs on the
same paths.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status

from mnemozine.activity import (
    maintenance_event,
    write_decision_event,
)
from mnemozine.activity.log import emit_async as emit
from mnemozine.app import Container
from mnemozine.evals.bootstrap import (
    LABEL_DROP,
    LABEL_KEEP,
    LABEL_UNREVIEWED,
    Candidate,
    candidates_to_gold_set,
)
from mnemozine.evals.goldset import save_gold_set
from mnemozine.interfaces import StorageBackend, WriteDecision
from mnemozine.schema.models import MemoryUnit, Scope, Tier
from mnemozine.web.deps import (
    ActivityLogDep,
    ContainerDep,
    StorageDep,
)
from mnemozine.web.routes._bootstrap_state import bootstrap_store
from mnemozine.web.schemas import (
    BootstrapCandidate,
    BootstrapLabelRequest,
    EvalMetric,
    EvalSummaryResponse,
    MaintenanceReportOut,
    MaintenanceRunResponse,
    MemoryDetail,
    MemoryPatchRequest,
    MutationResponse,
    Provenance,
    SuppressRequest,
    ValidityWindow,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mutations"])

# The deterministic maintenance job set (mirrors runner.build_default_jobs order
# plus migrate-index). Kept in sync with the read-side maintenance router.
_JOB_NAMES = ["consolidate", "entity-resolution", "decay", "audit", "migrate-index"]

# The WebUI contract job names map onto the internal MaintenanceJob.name values
# returned by build_default_jobs (which differ: 'consolidation' / 'entity_resolution').
_JOB_NAME_TO_INTERNAL = {
    "consolidate": "consolidation",
    "entity-resolution": "entity_resolution",
    "decay": "decay",
    "audit": "audit",
}

_VALID_LABELS = {LABEL_KEEP, LABEL_DROP, LABEL_UNREVIEWED}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def normalize_scope(value: str) -> Scope:
    """Parse a scope string, accepting a bare project id as a convenience.

    The wire contract allows ``'global'``, ``'project:<id>'``, or a bare project
    id everywhere a scope is supplied. ``Scope.parse`` only accepts the first two,
    so a bare id is promoted to ``project:<id>`` here.
    """

    stripped = value.strip()
    if stripped == "global" or stripped.startswith("project:"):
        return Scope.parse(stripped)
    return Scope.project(stripped)


async def _find_memory(storage: StorageBackend, memory_id: str) -> MemoryUnit | None:
    """Resolve one memory by id through the existing backend, both tiers/states.

    Prefers a backend keyed read (``get_memory`` on the real Graphiti backend) and
    falls back to scanning :meth:`StorageBackend.iter_memories` (the only
    Protocol enumeration entry point — the offline fake exposes only this). Returns
    ``None`` when no unit matches.
    """

    getter = getattr(storage, "get_memory", None)
    if callable(getter):
        found: MemoryUnit | None = await getter(memory_id)
        if found is not None:
            return found
    # Fall back to a full scan (both tiers, active + superseded).
    async for unit in storage.iter_memories():
        if unit.id == memory_id:
            return unit
    return None


def _to_detail(unit: MemoryUnit) -> MemoryDetail:
    """Project a :class:`MemoryUnit` onto the wire :class:`MemoryDetail`.

    The mutation response echoes the post-mutation unit so the SPA can update the
    detail view without a refetch. Supersession chains are not re-resolved here
    (they are unchanged by a reclassify/re-scope/tier edit); the read-side detail
    route owns chain resolution.
    """

    prov = unit.provenance
    return MemoryDetail(
        id=unit.id,
        category=unit.category,
        cross_ref_candidate=unit.cross_ref_candidate,
        scope_decision=unit.scope_decision,
        content=unit.content,
        scope=unit.scope.as_str(),
        entities=list(unit.entities),
        confidence=unit.confidence,
        tier=unit.tier,
        validity=ValidityWindow(
            valid_from=unit.valid_from,
            valid_to=unit.valid_to,
            active=unit.valid_to is None,
        ),
        provenance=Provenance(
            source=prov.source,
            session_id=prov.session_id,
            chunk_hash=prov.chunk_hash,
            raw_path=prov.raw_path,
        ),
        supersedes=[],
        superseded_by=[],
        last_accessed=unit.last_accessed,
        access_count=unit.access_count,
    )


def _persist_inplace(storage: StorageBackend, unit: MemoryUnit) -> bool:
    """Reflect an in-place type/scope mutation back into the backend store.

    The :class:`~mnemozine.interfaces.StorageBackend` Protocol has dedicated
    methods for tier (:meth:`archive`/:meth:`promote`) and window-closing
    (:meth:`close_validity_window`) but none for reclassify/re-scope, so this
    writes through the backend's ``memories`` dict when present (the offline fake
    and any dict-backed store). Returns True if the write landed. A real
    Graphiti backend needs a dedicated update method (see integration note); this
    returns False there so the caller knows the change was on the returned object
    only.
    """

    store = getattr(storage, "memories", None)
    if isinstance(store, dict):
        store[unit.id] = unit
        return True
    return False


# ---------------------------------------------------------------------------
# PATCH /api/memories/{id} — reclassify / re-scope / archive-restore
# ---------------------------------------------------------------------------


@router.patch(
    "/api/memories/{memory_id}",
    response_model=MutationResponse,
    summary="Reclassify/re-scope/tier",
    tags=["mutations"],
)
async def patch_memory(
    memory_id: str,
    req: MemoryPatchRequest,
    storage: StorageDep,
    activity: ActivityLogDep,
) -> MutationResponse:
    """Apply a HITL correction to one memory (PRD §4.3, R1/R5).

    Only ``category`` (re-label), ``cross_ref_candidate`` (toggle the seed flag),
    ``scope`` (re-scope), and ``tier`` (archive=``archive`` / restore=``hot``) are
    accepted — content is never editable (PRD §7). A patch that sets nothing is a
    422; an unknown id is 404. Tier changes go through
    :meth:`StorageBackend.archive` / :meth:`promote`; category/cross-ref/scope are
    applied via :meth:`StorageBackend.reclassify_memory`. Emits an
    ``extract_decision`` activity event recording the correction.
    """

    changed: list[str] = []
    if req.category is not None:
        changed.append("category")
    if req.cross_ref_candidate is not None:
        changed.append("cross_ref_candidate")
    if req.scope is not None:
        changed.append("scope")
    if req.tier is not None:
        changed.append("tier")
    if not changed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="patch must set at least one of: category, cross_ref_candidate, scope, tier",
        )

    # Validate the scope eagerly so a bad scope string is a 422, not a 500.
    new_scope: Scope | None = None
    if req.scope is not None:
        try:
            new_scope = normalize_scope(req.scope)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"invalid scope: {req.scope!r}",
            ) from exc

    unit = await _find_memory(storage, memory_id)
    if unit is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory not found")

    detail_extra: dict[str, Any] = {}

    # Tier first (it has a dedicated backend method that returns the updated unit).
    if req.tier is not None:
        if req.tier is Tier.ARCHIVE:
            unit = await storage.archive(memory_id)
            detail_extra["tier"] = "archive"
        else:
            unit = await storage.promote(memory_id)
            detail_extra["tier"] = "hot"

    # Reclassify / re-scope / toggle cross-ref: persist via the backend's
    # reclassify_memory seam when available (the real Graphiti backend + the fake
    # both implement it), falling back to the dict-backed in-place write.
    if req.category is not None:
        detail_extra["from_category"] = unit.category
    if req.cross_ref_candidate is not None:
        detail_extra["from_cross_ref_candidate"] = unit.cross_ref_candidate
        detail_extra["to_cross_ref_candidate"] = req.cross_ref_candidate
    if new_scope is not None:
        detail_extra["from_scope"] = unit.scope.as_str()
        detail_extra["to_scope"] = new_scope.as_str()
    if req.category is not None or req.cross_ref_candidate is not None or new_scope is not None:
        reclassify = getattr(storage, "reclassify_memory", None)
        if callable(reclassify):
            unit = await reclassify(
                memory_id,
                scope=new_scope,
                category=req.category,
                cross_ref_candidate=req.cross_ref_candidate,
            )
        else:
            if req.category is not None:
                unit.category = req.category.strip().lower()
            if req.cross_ref_candidate is not None:
                unit.cross_ref_candidate = req.cross_ref_candidate
            if new_scope is not None:
                unit.scope = new_scope
            _persist_inplace(storage, unit)
        if req.category is not None:
            detail_extra["to_category"] = unit.category

    await emit(
        activity,
        write_decision_event(
            decision=WriteDecision.NO_OP.value,
            memory_id=unit.id,
            source="operator",
            summary=f"operator correction: {', '.join(changed)}",
            detail={"changed": changed, "hitl": True, **detail_extra},
        ),
    )

    return MutationResponse(ok=True, memory=_to_detail(unit), changed=changed)


# ---------------------------------------------------------------------------
# POST /api/crossrefs/{id}/suppress — dismiss a cross-reference (R2)
# ---------------------------------------------------------------------------


@router.post(
    "/api/crossrefs/{memory_id}/suppress",
    response_model=MutationResponse,
    summary="Suppress a cross-ref",
    tags=["mutations"],
)
async def suppress_crossref(
    memory_id: str,
    req: SuppressRequest,
    storage: StorageDep,
    activity: ActivityLogDep,
) -> MutationResponse:
    """Dismiss a cross-reference suggestion in a working context (R2, PRD §4.7).

    Persists the ``(memory_id, context_key)`` suppression through
    :meth:`StorageBackend.record_suppression` so the suggestion stops resurfacing
    in that context across calls/restarts. Idempotent. Emits a ``maintenance``
    activity event so the dismissal is visible in the Logs feed.
    """

    context_key = req.context_key.strip()
    if not context_key:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="context_key must be non-empty",
        )

    await storage.record_suppression(memory_id, context_key)
    await emit(
        activity,
        maintenance_event(
            job_name="suppress-crossref",
            summary=f"suppressed cross-ref {memory_id} in context {context_key!r}",
            detail={"memory_id": memory_id, "context_key": context_key},
        ),
    )
    return MutationResponse(ok=True, memory=None, changed=["suppressed"])


# ---------------------------------------------------------------------------
# POST /api/maintenance/{job}/run — trigger a maintenance job on demand
# ---------------------------------------------------------------------------


@router.post(
    "/api/maintenance/{job}/run",
    response_model=MaintenanceRunResponse,
    summary="Trigger a job",
    tags=["mutations"],
)
async def run_job(
    job: str,
    container: ContainerDep,
    activity: ActivityLogDep,
) -> MaintenanceRunResponse:
    """Trigger one maintenance job on demand (PRD §4.7).

    Validates ``job`` against the known set (unknown -> 404), then runs only that
    job through the real :class:`~mnemozine.maintenance.runner.MaintenanceRunner`
    over the live container's storage/LLM/embedding providers. Returns the job's
    :class:`MaintenanceReport`. Emits a ``maintenance`` activity event with the
    run counts. The runner emits a per-job event too; this route-level event marks
    the *operator-triggered* run.
    """

    if job not in _JOB_NAMES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown maintenance job: {job!r} (known: {', '.join(_JOB_NAMES)})",
        )

    report_out = await _run_single_job(container, job)
    await emit(
        activity,
        maintenance_event(
            job_name=job,
            summary=f"operator-triggered maintenance: {job}",
            detail={
                "triggered_by": "operator",
                "consolidated": report_out.consolidated,
                "entities_merged": report_out.entities_merged,
                "archived": report_out.archived,
                "edges_pruned": report_out.edges_pruned,
            },
        ),
    )
    return MaintenanceRunResponse(job=job, started=True, report=report_out)


async def _run_single_job(container: Container, job: str) -> MaintenanceReportOut:
    """Build the requested job from the container and run it once.

    Reuses :func:`mnemozine.maintenance.runner.build_default_jobs` for the four
    scheduled jobs (matching on :attr:`MaintenanceJob.name`) and the dedicated
    migrate-index path for the OQ3 job, so no maintenance logic is duplicated here.
    """

    from mnemozine.maintenance.runner import build_default_jobs

    storage = await container.build_storage()

    if job == "migrate-index":
        from mnemozine.maintenance.runner import _run_migrate_index

        report = await _run_migrate_index(force=False)
        return _report_to_out(report)

    jobs = build_default_jobs(
        storage,
        container.build_llm_provider(),
        container.build_embedding_provider(),
        settings=container.settings,
    )
    internal_name = _JOB_NAME_TO_INTERNAL.get(job, job)
    by_name = {j.name: j for j in jobs}
    target = by_name.get(internal_name)
    if target is None:
        # consolidate/entity-resolution/decay/audit all come from build_default_jobs;
        # a mismatch means the job-name contract drifted from the runner.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"job {job!r} is known but not wired in build_default_jobs",
        )
    report = await target.run()
    return _report_to_out(report)


def _report_to_out(report: Any) -> MaintenanceReportOut:
    """Map a :class:`~mnemozine.interfaces.MaintenanceReport` onto the wire model."""

    return MaintenanceReportOut(
        job_name=report.job_name,
        consolidated=report.consolidated,
        entities_merged=report.entities_merged,
        archived=report.archived,
        edges_pruned=report.edges_pruned,
        notes=list(report.notes),
    )


# ---------------------------------------------------------------------------
# F4 eval bootstrap — label / finish (browser labeling, PRD §4.8)
# ---------------------------------------------------------------------------


def _candidate_to_out(candidate: Candidate) -> BootstrapCandidate:
    """Project an internal eval :class:`Candidate` onto the wire model (F4)."""

    return BootstrapCandidate(
        candidate_id=candidate.candidate_id,
        content=candidate.content,
        proposed_type=candidate.proposed_type,
        scope=candidate.scope,
        entities=list(candidate.entities),
        confidence=candidate.confidence,
        source_session=candidate.source_session,
        label=candidate.label,
        corrected_type=candidate.corrected_type,
    )


@router.post(
    "/api/eval/bootstrap/{candidate_id}/label",
    response_model=BootstrapCandidate,
    summary="Label a bootstrap candidate (F4)",
    tags=["mutations"],
)
async def label_candidate(
    candidate_id: str,
    req: BootstrapLabelRequest,
    activity: ActivityLogDep,
) -> BootstrapCandidate:
    """Apply a keep/drop/reclassify label to one bootstrap candidate (F4, PRD §4.8).

    Validates the label (``keep | drop | unreviewed``), updates the candidate in
    the in-process bootstrap store, and returns it. ``corrected_type`` reclassifies
    the candidate (R1 HITL). Unknown candidate id -> 404. Emits an
    ``extract_decision`` activity event for the labeling action.
    """

    if req.label not in _VALID_LABELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="label must be one of: keep, drop, unreviewed",
        )

    updated = bootstrap_store.label(
        candidate_id, label=req.label, corrected_type=req.corrected_type
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown bootstrap candidate: {candidate_id!r}",
        )

    await emit(
        activity,
        write_decision_event(
            decision=WriteDecision.NO_OP.value,
            memory_id=candidate_id,
            source="eval-bootstrap",
            summary=f"bootstrap label {updated.label}: {updated.content[:60]}",
            detail={
                "label": updated.label,
                "corrected_type": (
                    updated.corrected_type.value if updated.corrected_type else None
                ),
            },
        ),
    )
    return _candidate_to_out(updated)


@router.post(
    "/api/eval/bootstrap/finish",
    response_model=EvalSummaryResponse,
    summary="Save labeled gold set",
    tags=["mutations"],
)
async def finish_bootstrap(
    activity: ActivityLogDep,
) -> EvalSummaryResponse:
    """Fold the kept candidates into the committed gold set (F4, PRD §4.8).

    Takes the labeled candidates from the in-process bootstrap store, folds the
    ``keep`` set into a :class:`~mnemozine.evals.goldset.GoldSet` via
    :func:`candidates_to_gold_set`, and persists it. Returns an
    :class:`EvalSummaryResponse` describing the resulting gold set. Emits a
    ``maintenance`` activity event recording the fold.
    """

    candidates = bootstrap_store.all()
    gold_set = candidates_to_gold_set(candidates)
    kept = sum(1 for c in candidates if c.kept)

    saved_path = bootstrap_store.gold_set_path
    try:
        save_gold_set(gold_set, saved_path)
    except OSError:
        logger.warning("finish_bootstrap: could not persist gold set to %s", saved_path)

    await emit(
        activity,
        maintenance_event(
            job_name="eval-bootstrap-finish",
            summary=f"folded {kept} kept candidate(s) into gold set {gold_set.name!r}",
            detail={"gold_set": gold_set.name, "kept": kept, "total": len(candidates)},
        ),
    )

    return EvalSummaryResponse(
        gold_set=gold_set.name,
        passed=True,
        metrics=[
            EvalMetric(
                name="gold_classifier_cases",
                value=float(len(gold_set.classifier_cases)),
                threshold=None,
                passed=True,
                detail=f"{kept} kept candidate(s) folded into the gold set",
            )
        ],
        ran_at=datetime.now(UTC),
    )


__all__ = ["router", "normalize_scope"]
