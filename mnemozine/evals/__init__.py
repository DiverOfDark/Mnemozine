"""Mnemozine EVAL harness (PRD §9, deliverable #7).

This subpackage builds the success-metrics harness the PRD insists be in place
in Phase 1 and run on every change + on a schedule thereafter:

* :mod:`mnemozine.evals.goldset` — the fixed, operator-labeled gold-set data
  model + loader, with a small committed fixture (``fixtures/gold_set.json``) so
  the whole harness runs **offline against fakes**.
* :mod:`mnemozine.evals.distractors` — the synthetic distractor generator that
  inflates the store at 1x / 10x / 100x to prove *precision stays flat* before a
  large real store exists (and load-tests traversal).
* :mod:`mnemozine.evals.metrics` — the six §9 metric runners + pure calculation
  helpers (injection precision@k, changed-preference correctness, cross-reference
  precision, classifier accuracy, retrieval p95 latency, no-leak check).
* :mod:`mnemozine.evals.bootstrap` — the eval-set bootstrap: auto-propose
  extracted candidates during historical import; the **operator** labels them
  yes/no via a Markdown/CLI review pass (USER-TASK DEPENDENCY).
* :mod:`mnemozine.evals.runner` — orchestration tying it together, including the
  headline precision-stays-flat scaling run.
* :mod:`mnemozine.evals.cli` — the ``mnemozine-eval`` Typer app.

Everything codes against the :mod:`mnemozine.interfaces` Protocols only.
"""

from __future__ import annotations

from mnemozine.evals.distractors import (
    DEFAULT_INFLATION_LEVELS,
    DistractorGenerator,
)
from mnemozine.evals.goldset import (
    DEFAULT_GOLD_SET_PATH,
    ClassifierCase,
    CrossRefCase,
    GoldMemory,
    GoldSet,
    InjectionCase,
    NoLeakCase,
    PreferenceCase,
    load_gold_set,
    save_gold_set,
)
from mnemozine.evals.metrics import (
    CaseResult,
    MetricResult,
    accuracy,
    changed_preference_correctness,
    classifier_accuracy,
    crossref_precision,
    injection_precision,
    mean,
    no_leak_check,
    percentile,
    precision_at_k,
    recall_at_k,
    retrieval_latency,
)
from mnemozine.evals.runner import (
    EvalReport,
    EvalRunner,
    ScalingReport,
    default_inmemory_runner,
)

__all__ = [
    # gold set
    "GoldSet",
    "GoldMemory",
    "InjectionCase",
    "PreferenceCase",
    "CrossRefCase",
    "ClassifierCase",
    "NoLeakCase",
    "load_gold_set",
    "save_gold_set",
    "DEFAULT_GOLD_SET_PATH",
    # distractors
    "DistractorGenerator",
    "DEFAULT_INFLATION_LEVELS",
    # metrics
    "MetricResult",
    "CaseResult",
    "injection_precision",
    "changed_preference_correctness",
    "crossref_precision",
    "classifier_accuracy",
    "retrieval_latency",
    "no_leak_check",
    "precision_at_k",
    "recall_at_k",
    "percentile",
    "mean",
    "accuracy",
    # runner
    "EvalRunner",
    "EvalReport",
    "ScalingReport",
    "default_inmemory_runner",
]
