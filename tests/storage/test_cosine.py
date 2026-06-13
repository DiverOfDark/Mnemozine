"""Unit tests for the shared cosine helper."""

from __future__ import annotations

import math

from mnemozine.storage.cosine import cosine_similarity


def test_identical_vectors() -> None:
    assert math.isclose(cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)


def test_orthogonal_vectors() -> None:
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)


def test_opposite_vectors() -> None:
    assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)


def test_degenerate_inputs_return_zero() -> None:
    # zero vector, length mismatch, empty -> 0.0 (no signal, never raises)
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([], []) == 0.0
