"""The §9 metric runners + their pure calculation helpers.

PRD §9 success metrics, each a runner here:

* **injection precision@k** (FR-RET-3/5) — :func:`injection_precision`
* **changed-preference correctness** (UC-2) — :func:`changed_preference_correctness`
* **cross-reference precision** (FR-RET-6) — :func:`crossref_precision`
* **classifier accuracy** preference vs project_fact (R1) — :func:`classifier_accuracy`
* **retrieval p95 latency** — :func:`retrieval_latency`
* **no-leak check** project_fact never leaks across projects — :func:`no_leak_check`

Design split so the harness is *unit-testable per metric*:

* **Pure helpers** (``precision_at_k``, ``percentile``, ``mean``, ...) do the
  arithmetic and have no I/O — directly unit-tested with hand-built inputs.
* **Async runners** drive a live (or fake) ``Retriever`` / ``StorageBackend`` /
  ``CrossReferencer`` / ``Extractor`` against the gold set and return a
  :class:`MetricResult`.

Every runner returns a :class:`MetricResult` with a numeric ``score``, an
optional ``threshold`` + ``passed`` flag, and per-case detail for debugging.
Nothing here imports a sibling module's internals — only the Protocols in
:mod:`mnemozine.interfaces` and the schema.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from mnemozine.config import Settings
from mnemozine.evals.goldset import GoldSet
from mnemozine.interfaces import (
    CrossReferencer,
    Extractor,
    RetrievalContext,
    Retriever,
    StorageBackend,
)
from mnemozine.schema.models import Scope

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CaseResult:
    """Per-case outcome inside a metric, for debugging a failure."""

    case_id: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricResult:
    """The outcome of one §9 metric run."""

    name: str
    score: float
    passed: bool
    threshold: float | None = None
    n: int = 0
    cases: list[CaseResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line human summary for the CLI / report."""

        status = "PASS" if self.passed else "FAIL"
        thr = "" if self.threshold is None else f" (threshold {self.threshold:.3f})"
        return f"[{status}] {self.name}: {self.score:.3f}{thr} over n={self.n}"


# ---------------------------------------------------------------------------
# Pure calculation helpers (no I/O — directly unit-tested)
# ---------------------------------------------------------------------------


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Precision@k: fraction of the top-``k`` retrieved that are relevant.

    With fewer than ``k`` retrieved, the denominator is the number actually
    retrieved (standard precision@k with a short list). An empty retrieval scores
    1.0 *only if* nothing was supposed to surface is the caller's concern; here an
    empty top-k yields 1.0 (no false positives) — the runner separately checks
    recall of the should-surface set so a vacuous "return nothing" can't pass.
    """

    if k <= 0:
        return 0.0
    top = list(retrieved_ids)[:k]
    if not top:
        return 1.0
    hits = sum(1 for mid in top if mid in relevant_ids)
    return hits / len(top)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Recall@k: fraction of the relevant set present in the top-``k``."""

    if not relevant_ids:
        return 1.0
    top = set(list(retrieved_ids)[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; empty -> 1.0 (a vacuous metric is treated as a pass)."""

    if not values:
        return 1.0
    return sum(values) / len(values)


def percentile(values: Sequence[float], pct: float) -> float:
    """The ``pct`` (0–100) percentile via linear interpolation between ranks.

    Matches numpy's default ('linear') method so the p95 latency metric is
    comparable to ad-hoc numpy checks, but without the dependency.
    """

    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def accuracy(correct: int, total: int) -> float:
    """Classification accuracy; ``total == 0`` -> 1.0 (vacuous pass)."""

    if total <= 0:
        return 1.0
    return correct / total


# ---------------------------------------------------------------------------
# Metric runners
# ---------------------------------------------------------------------------


async def injection_precision(
    retriever: Retriever,
    gold_set: GoldSet,
    *,
    threshold: float = 1.0,
) -> MetricResult:
    """Injection precision@k for the SessionStart/mid-session path (FR-RET-3/5).

    For each injection case, runs scoped retrieval and computes precision@k over
    the case's ``top_k``: the fraction of top-k retrieved that are gold ids the
    case says SHOULD surface. A retrieved id appearing in ``should_not_surface``
    (or any non-relevant id) counts against precision. The metric also requires
    full recall of the should-surface set (a vacuous empty return cannot pass),
    so a case passes only when precision is perfect AND every should-surface id
    appears. The headline score is the mean precision@k across cases.
    """

    precisions: list[float] = []
    cases: list[CaseResult] = []
    for case in gold_set.injection_cases:
        scopes = [Scope.parse(s) for s in case.scopes]
        ctx = RetrievalContext(
            project=case.project,
            scopes=scopes,
            entities=list(case.entities),
            recent_text=case.query,
        )
        retrieved = await retriever.scoped_retrieve(case.query, ctx, top_k=case.top_k)
        retrieved_ids = [r.memory.id for r in retrieved]
        relevant = {gold_set.runtime_id(g) for g in case.should_surface}
        p = precision_at_k(retrieved_ids, relevant, case.top_k)
        r = recall_at_k(retrieved_ids, relevant, case.top_k)
        precisions.append(p)
        cases.append(
            CaseResult(
                case_id=case.case_id,
                passed=p >= 1.0 and r >= 1.0,
                detail={
                    "precision": p,
                    "recall": r,
                    "retrieved": retrieved_ids,
                    "relevant": sorted(relevant),
                },
            )
        )
    score = mean(precisions)
    return MetricResult(
        name="injection_precision_at_k",
        score=score,
        passed=all(c.passed for c in cases) if cases else True,
        threshold=threshold,
        n=len(cases),
        cases=cases,
    )


async def changed_preference_correctness(
    retriever: Retriever,
    gold_set: GoldSet,
    *,
    threshold: float = 1.0,
) -> MetricResult:
    """Changed-preference correctness (UC-2, PRD §9 preference correctness).

    For each preference case: query and assert the *current* value surfaces as
    active and the *stale* (superseded) value does NOT. A case passes only when
    both hold. Score is the fraction of cases that pass.
    """

    cases: list[CaseResult] = []
    for case in gold_set.preference_cases:
        scopes = [Scope.parse(s) for s in case.scopes]
        ctx = RetrievalContext(
            project=case.project,
            scopes=scopes,
            entities=list(case.entities),
            recent_text=case.query,
        )
        retrieved = await retriever.scoped_retrieve(case.query, ctx, top_k=case.top_k)
        retrieved_ids = {r.memory.id for r in retrieved}
        current_id = gold_set.runtime_id(case.current_gold_id)
        stale_id = gold_set.runtime_id(case.stale_gold_id)
        current_in = current_id in retrieved_ids
        stale_out = stale_id not in retrieved_ids
        cases.append(
            CaseResult(
                case_id=case.case_id,
                passed=current_in and stale_out,
                detail={
                    "current_surfaced": current_in,
                    "stale_suppressed": stale_out,
                    "retrieved": sorted(retrieved_ids),
                },
            )
        )
    correct = sum(1 for c in cases if c.passed)
    score = accuracy(correct, len(cases))
    return MetricResult(
        name="changed_preference_correctness",
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        n=len(cases),
        cases=cases,
    )


async def crossref_precision(
    cross_referencer: CrossReferencer,
    gold_set: GoldSet,
    *,
    threshold: float = 1.0,
) -> MetricResult:
    """Cross-reference precision (FR-RET-6, PRD §9 cross-reference quality).

    For each crossref case, runs ``find_related`` and computes the fraction of
    surfaced connections that are in ``relevant_gold_ids`` — precision over
    recall, since a wrong "this reminds me of…" is worse than a miss. With no
    connections surfaced, precision is 1.0 (no false positives). The headline
    score is the mean precision across cases.
    """

    precisions: list[float] = []
    cases: list[CaseResult] = []
    for case in gold_set.crossref_cases:
        ctx = RetrievalContext(
            project=case.project,
            scopes=[Scope.global_()],
            entities=list(case.entities),
        )
        refs = await cross_referencer.find_related(ctx, max_suggestions=case.max_suggestions)
        surfaced_ids = [r.memory.id for r in refs]
        relevant = {gold_set.runtime_id(g) for g in case.relevant_gold_ids}
        if surfaced_ids:
            hits = sum(1 for mid in surfaced_ids if mid in relevant)
            p = hits / len(surfaced_ids)
        else:
            p = 1.0
        precisions.append(p)
        cases.append(
            CaseResult(
                case_id=case.case_id,
                passed=p >= 1.0,
                detail={
                    "precision": p,
                    "surfaced": surfaced_ids,
                    "relevant": sorted(relevant),
                    "reasons": [r.reason for r in refs],
                },
            )
        )
    score = mean(precisions)
    return MetricResult(
        name="crossref_precision",
        score=score,
        passed=all(c.passed for c in cases) if cases else True,
        threshold=threshold,
        n=len(cases),
        cases=cases,
    )


async def classifier_accuracy(
    extractor: Extractor,
    gold_set: GoldSet,
    *,
    threshold: float = 0.9,
) -> MetricResult:
    """Classifier accuracy: global vs project scope-decision (R1, the gate, §9).

    For each classifier case, runs ``Extractor.classify`` on the bare statement
    and compares the returned controlled :class:`ScopeDecision` against the gold
    ``expected_scope_decision`` (the make-or-break distinction that drives the
    no-leak rule under the category split). When a case sets
    ``expected_cross_ref`` the cross-reference flag must also match. The default
    threshold is high (0.9). Score is the accuracy across cases.
    """

    cases: list[CaseResult] = []
    correct = 0
    for case in gold_set.classifier_cases:
        ctx = RetrievalContext(
            project=case.project,
            scopes=([Scope.project(case.project)] if case.project else [Scope.global_()]),
        )
        classification = await extractor.classify(case.statement, ctx)
        ok = classification.scope_decision == case.expected_scope_decision
        if case.expected_cross_ref is not None:
            ok = ok and (classification.cross_ref_candidate == case.expected_cross_ref)
        correct += int(ok)
        cases.append(
            CaseResult(
                case_id=case.case_id,
                passed=ok,
                detail={
                    "expected_scope_decision": case.expected_scope_decision.value,
                    "predicted_scope_decision": classification.scope_decision.value,
                    "predicted_category": classification.category,
                    "predicted_cross_ref": classification.cross_ref_candidate,
                },
            )
        )
    score = accuracy(correct, len(cases))
    return MetricResult(
        name="classifier_accuracy",
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        n=len(cases),
        cases=cases,
    )


async def retrieval_latency(
    retriever: Retriever,
    gold_set: GoldSet,
    *,
    settings: Settings | None = None,
    repeats: int = 5,
    target_ms: float | None = None,
) -> MetricResult:
    """Retrieval p95 latency vs target (PRD §9 latency SLA, FR-RET-2).

    Runs every injection case's scoped retrieval ``repeats`` times, records
    per-call wall-clock latency (ms), and reports the p95. Passes when p95 is at
    or under ``target_ms`` (defaults to ``settings.retrieval.p95_latency_target_ms``).
    ``score`` is the measured p95 in ms (lower is better), so a separate
    ``passed`` flag carries the SLA verdict.
    """

    if target_ms is None:
        s = settings or Settings()
        target_ms = float(s.retrieval.p95_latency_target_ms)

    samples_ms: list[float] = []
    for _ in range(max(1, repeats)):
        for case in gold_set.injection_cases:
            scopes = [Scope.parse(sc) for sc in case.scopes]
            ctx = RetrievalContext(
                project=case.project,
                scopes=scopes,
                entities=list(case.entities),
                recent_text=case.query,
            )
            start = time.perf_counter()
            await retriever.scoped_retrieve(case.query, ctx, top_k=case.top_k)
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    p95 = percentile(samples_ms, 95.0)
    return MetricResult(
        name="retrieval_p95_latency_ms",
        score=p95,
        passed=p95 <= target_ms,
        threshold=target_ms,
        n=len(samples_ms),
        notes=[
            f"p50={percentile(samples_ms, 50.0):.3f}ms",
            f"max={max(samples_ms) if samples_ms else 0.0:.3f}ms",
        ],
    )


async def no_leak_check(
    storage: StorageBackend,
    gold_set: GoldSet,
    *,
    threshold: float = 1.0,
) -> MetricResult:
    """No-leak check: a project_fact never appears in an unrelated project (PRD §9).

    For each no-leak case, queries the *unrelated* project's scope (global +
    that project) and asserts the project_fact's runtime id is NOT among the
    results. Queries the storage backend directly (the scope boundary is the
    backend's responsibility, FR-STO-3). A case passes when the fact does not
    leak; score is the fraction of cases with no leak.
    """

    cases: list[CaseResult] = []
    for case in gold_set.no_leak_cases:
        unrelated_scopes = [Scope.global_(), Scope.project(case.unrelated_project)]
        results = await storage.scoped_query(
            case.query,
            unrelated_scopes,
            entities=list(case.entities) or None,
            top_k=case.top_k,
        )
        result_ids = {r.memory.id for r in results}
        fact_id = gold_set.runtime_id(case.fact_gold_id)
        leaked = fact_id in result_ids
        cases.append(
            CaseResult(
                case_id=case.case_id,
                passed=not leaked,
                detail={
                    "leaked": leaked,
                    "unrelated_project": case.unrelated_project,
                    "result_ids": sorted(result_ids),
                },
            )
        )
    clean = sum(1 for c in cases if c.passed)
    score = accuracy(clean, len(cases))
    return MetricResult(
        name="no_leak_check",
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        n=len(cases),
        cases=cases,
    )
