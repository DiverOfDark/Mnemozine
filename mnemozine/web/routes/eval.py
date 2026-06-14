"""Eval read routes (PRD §4.8 / §6 GET /api/eval + GET /api/eval/bootstrap).

The **read** half of the Eval screen:

* ``GET /api/eval`` runs the offline §9 eval harness
  (:func:`mnemozine.evals.runner.default_inmemory_runner`) over the committed gold
  set and projects each :class:`~mnemozine.evals.metrics.MetricResult` onto an
  :class:`~mnemozine.web.schemas.EvalMetric`. The harness seeds its *own* store
  from the gold set (it is a graded benchmark, not a scan of the live store), so it
  runs with no FalkorDB/Ollama/Qwen.
* ``GET /api/eval/bootstrap`` lists the F4 bootstrap-labeling queue from the shared
  in-process :data:`mnemozine.web.routes._bootstrap_state.bootstrap_store` — the
  same store the write-side ``POST .../{id}/label`` / ``.../finish`` handlers (in
  ``mutations.py``) mutate, so a label applied there is visible here.

The ``label`` / ``finish`` *writes* live in ``mutations.py`` (the single auditable
write surface, PRD §2); this module is read-only. Everything runs offline.
"""

from __future__ import annotations

from fastapi import APIRouter

from mnemozine.evals.bootstrap import Candidate
from mnemozine.evals.goldset import GoldSet
from mnemozine.evals.metrics import MetricResult
from mnemozine.evals.runner import EvalReport, default_inmemory_runner
from mnemozine.web.routes._bootstrap_state import bootstrap_store
from mnemozine.web.schemas import (
    BootstrapCandidate,
    BootstrapCandidatesResponse,
    EvalMetric,
    EvalSummaryResponse,
)

router = APIRouter(prefix="/api/eval", tags=["eval"])


def _metric_out(result: MetricResult) -> EvalMetric:
    """Project a §9 metric result onto the wire EvalMetric."""

    return EvalMetric(
        name=result.name,
        value=result.score,
        threshold=result.threshold,
        passed=result.passed,
        detail="; ".join(result.notes) or None,
    )


def _candidate_out(c: Candidate) -> BootstrapCandidate:
    """Project an internal eval :class:`Candidate` onto the wire model (F4)."""

    return BootstrapCandidate(
        candidate_id=c.candidate_id,
        content=c.content,
        proposed_type=c.proposed_type,
        scope=c.scope,
        entities=list(c.entities),
        confidence=c.confidence,
        source_session=c.source_session,
        label=c.label,
        corrected_type=c.corrected_type,
    )


async def _run_summary(gold_set: GoldSet | None = None) -> EvalSummaryResponse:
    """Run the offline eval harness and project the report onto the wire summary."""

    runner = default_inmemory_runner(gold_set=gold_set)
    report: EvalReport = await runner.run_all()
    return EvalSummaryResponse(
        gold_set=runner.gold_set.name,
        passed=report.passed,
        metrics=[_metric_out(r) for r in report.results],
        ran_at=None,
    )


@router.get("", response_model=EvalSummaryResponse, summary="Eval results summary")
async def eval_summary() -> EvalSummaryResponse:
    """Latest eval metrics (precision/classifier/latency/no-leak/...) — PRD §4.8.

    Runs the offline §9 harness over the committed gold set and returns the metric
    results (each with its measured value, threshold, and pass flag).
    """

    return await _run_summary()


@router.get(
    "/bootstrap",
    response_model=BootstrapCandidatesResponse,
    summary="Bootstrap labeling queue (F4)",
)
async def bootstrap_candidates() -> BootstrapCandidatesResponse:
    """Auto-proposed eval candidates to label in the browser (F4, PRD §4.8).

    Reads the shared in-process bootstrap store (self-seeded on first access). The
    write-side label/finish handlers mutate the *same* store, so labels applied
    there are reflected in this list.
    """

    return BootstrapCandidatesResponse(
        candidates=[_candidate_out(c) for c in bootstrap_store.all()]
    )


__all__ = ["router"]
