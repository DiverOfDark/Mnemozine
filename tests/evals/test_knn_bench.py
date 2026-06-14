"""Unit tests for the KNN over-fetch benchmark (FR-RET-2, PRD §9).

These run fully offline (deterministic seeded vectors, no FalkorDB/Ollama) and
assert the benchmark faithfully models the filter-after-KNN starvation that the
``retrieval.knn_overfetch_factor`` / ``_cap`` knobs defend against:

* the default ``factor=10`` holds recall@k flat across 1x/10x/100x for a
  realistically-selective scope (the configured over-fetch returns the true
  in-scope top_k);
* a starved ``factor=1`` collapses recall;
* a tighter scope needs a larger factor (the documented ``factor >= 1/fraction``
  rule), and raising the factor recovers recall.
"""

from __future__ import annotations

from typer.testing import CliRunner

from mnemozine.config import RetrievalSettings
from mnemozine.evals.cli import app
from mnemozine.evals.knn_bench import (
    KnnBenchConfig,
    effective_overfetch_k,
    run_knn_overfetch_bench,
)

runner = CliRunner()


# --------------------------------------------------------------------------- #
# effective_overfetch_k mirrors the backend K = min(top_k*factor, cap) floor
# --------------------------------------------------------------------------- #


def test_effective_overfetch_k_factor() -> None:
    r = RetrievalSettings(knn_overfetch_factor=10, knn_overfetch_cap=512)
    assert effective_overfetch_k(10, r) == 100


def test_effective_overfetch_k_capped() -> None:
    r = RetrievalSettings(knn_overfetch_factor=100, knn_overfetch_cap=512)
    # 10 * 100 = 1000 > cap -> clipped to the cap.
    assert effective_overfetch_k(10, r) == 512


def test_effective_overfetch_k_never_below_top_k() -> None:
    # A pathological tiny cap can never ask for fewer than top_k neighbours.
    r = RetrievalSettings(knn_overfetch_factor=1, knn_overfetch_cap=2)
    assert effective_overfetch_k(10, r) == 10


# --------------------------------------------------------------------------- #
# Benchmark behaviour
# --------------------------------------------------------------------------- #


def test_default_factor_holds_recall_flat() -> None:
    """Default knn_overfetch_factor returns the true top_k at every level."""

    report = run_knn_overfetch_bench(retrieval=RetrievalSettings(), config=KnnBenchConfig(top_k=10))
    assert [r.level for r in report.results] == [1, 10, 100]
    # recall@k = 1.0 flat -> the configured over-fetch is sufficient.
    assert report.passed
    assert report.min_recall == 1.0
    # The store genuinely grows ~100x while selectivity is held constant.
    assert report.results[-1].total_memories > 10 * report.results[0].total_memories


def test_starved_factor_collapses_recall() -> None:
    """factor=1 (no over-fetch) starves the post-KNN scope filter."""

    report = run_knn_overfetch_bench(
        retrieval=RetrievalSettings(knn_overfetch_factor=1, knn_overfetch_cap=512),
        config=KnnBenchConfig(top_k=10),
    )
    assert not report.passed
    # The 100x level recall should be far below 1.0.
    assert report.results[-1].recall_at_k < 0.5


def test_tighter_scope_needs_larger_factor_and_recovers() -> None:
    """A 5%% scope starves factor=10 but recovers once the factor is raised."""

    tight = KnnBenchConfig(top_k=10, in_scope_fraction_target=0.05)

    starved = run_knn_overfetch_bench(
        retrieval=RetrievalSettings(knn_overfetch_factor=10), config=tight
    )
    assert not starved.passed  # 1/0.05 = 20 > 10 -> insufficient

    # factor >= 1/fraction (with headroom) recovers recall to 1.0.
    recovered = run_knn_overfetch_bench(
        retrieval=RetrievalSettings(knn_overfetch_factor=40), config=tight
    )
    assert recovered.passed
    assert recovered.min_recall == 1.0


def test_report_render_carries_tuning_hint_on_fail() -> None:
    report = run_knn_overfetch_bench(
        retrieval=RetrievalSettings(knn_overfetch_factor=1),
        config=KnnBenchConfig(top_k=10),
    )
    text = report.render()
    assert "KNN over-fetch benchmark" in text
    assert "recall@k" in text
    assert "knn_overfetch_factor" in text  # the actionable tuning hint


def test_deterministic_across_runs() -> None:
    a = run_knn_overfetch_bench(retrieval=RetrievalSettings())
    b = run_knn_overfetch_bench(retrieval=RetrievalSettings())
    assert [r.recall_at_k for r in a.results] == [r.recall_at_k for r in b.results]


# --------------------------------------------------------------------------- #
# CLI subcommand
# --------------------------------------------------------------------------- #


def test_cli_knn_bench_passes_at_default() -> None:
    result = runner.invoke(app, ["knn-bench"])
    assert result.exit_code == 0
    assert "KNN over-fetch benchmark" in result.stdout
    assert "PASS" in result.stdout


def test_cli_knn_bench_fails_when_starved() -> None:
    result = runner.invoke(app, ["knn-bench", "--factor", "1"])
    assert result.exit_code == 1
    assert "FAIL" in result.stdout


def test_cli_knn_bench_tighter_scope_then_recover() -> None:
    starved = runner.invoke(app, ["knn-bench", "--in-scope-fraction", "0.05"])
    assert starved.exit_code == 1
    recovered = runner.invoke(app, ["knn-bench", "--in-scope-fraction", "0.05", "--factor", "40"])
    assert recovered.exit_code == 0
