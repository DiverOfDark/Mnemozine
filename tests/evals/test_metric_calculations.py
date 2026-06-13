"""Unit tests for the pure §9 metric calculation helpers (PRD §9).

These test the arithmetic of each metric in isolation, with hand-built inputs
and no I/O — the spec's "unit-test each metric calculation". The async runners
are exercised separately in ``test_metric_runners.py`` and ``test_runner.py``.
"""

from __future__ import annotations

import pytest

from mnemozine.evals.metrics import (
    accuracy,
    mean,
    percentile,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_half_relevant(self) -> None:
        assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 4) == 0.5

    def test_distractor_in_top_k_lowers_precision(self) -> None:
        # The §9 scenario: a distractor surfacing inside top-k drops precision.
        assert precision_at_k(["a", "distractor"], {"a"}, 2) == 0.5

    def test_truncates_to_k(self) -> None:
        # Only the first k count; a relevant item past k does not help.
        assert precision_at_k(["x", "y", "a"], {"a"}, 2) == 0.0

    def test_short_list_denominator(self) -> None:
        # Fewer than k retrieved -> denominator is the retrieved count.
        assert precision_at_k(["a"], {"a"}, 5) == 1.0

    def test_empty_retrieval_no_false_positives(self) -> None:
        assert precision_at_k([], {"a"}, 5) == 1.0

    def test_k_zero(self) -> None:
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestRecallAtK:
    def test_full_recall(self) -> None:
        assert recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0

    def test_partial_recall(self) -> None:
        assert recall_at_k(["a", "x"], {"a", "b"}, 5) == 0.5

    def test_recall_respects_k(self) -> None:
        # b is present but past k -> not recalled.
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5

    def test_empty_relevant_is_vacuous_pass(self) -> None:
        assert recall_at_k(["a"], set(), 5) == 1.0


class TestMean:
    def test_basic(self) -> None:
        assert mean([1.0, 0.0]) == 0.5

    def test_empty_is_vacuous_pass(self) -> None:
        assert mean([]) == 1.0


class TestAccuracy:
    def test_basic(self) -> None:
        assert accuracy(3, 4) == 0.75

    def test_zero_total_is_vacuous_pass(self) -> None:
        assert accuracy(0, 0) == 1.0

    def test_all_correct(self) -> None:
        assert accuracy(5, 5) == 1.0


class TestPercentile:
    def test_p95_interpolates_like_numpy(self) -> None:
        # numpy.percentile(range(1..101), 95) == 95.05 with the default 'linear'
        # method; this helper reproduces it without the numpy dependency.
        values = list(range(1, 101))
        assert percentile([float(v) for v in values], 95.0) == pytest.approx(95.05)

    def test_p50_median_even(self) -> None:
        assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == pytest.approx(2.5)

    def test_p0_is_min_p100_is_max(self) -> None:
        vals = [5.0, 1.0, 9.0, 3.0]
        assert percentile(vals, 0.0) == 1.0
        assert percentile(vals, 100.0) == 9.0

    def test_single_value(self) -> None:
        assert percentile([42.0], 95.0) == 42.0

    def test_empty(self) -> None:
        assert percentile([], 95.0) == 0.0
