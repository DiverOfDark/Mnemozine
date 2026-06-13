"""Unit tests for the gold-set data model + committed fixture (PRD §9)."""

from __future__ import annotations

import pytest

from mnemozine.evals.goldset import (
    GoldSet,
    all_gold_ids,
    load_gold_set,
    runtime_ids,
    save_gold_set,
)
from mnemozine.schema.models import MemoryType, Scope, Tier


def test_fixture_loads() -> None:
    gs = load_gold_set()
    assert isinstance(gs, GoldSet)
    assert gs.memories
    # Every metric is represented so the harness runs end-to-end offline.
    assert gs.injection_cases
    assert gs.preference_cases
    assert gs.crossref_cases
    assert gs.classifier_cases
    assert gs.no_leak_cases


def test_fixture_case_ids_reference_real_memories() -> None:
    gs = load_gold_set()
    ids = all_gold_ids(gs)
    for case in gs.injection_cases:
        for g in (*case.should_surface, *case.should_not_surface):
            assert g in ids, f"injection case references unknown gold id {g!r}"
    for case in gs.preference_cases:
        assert case.current_gold_id in ids
        assert case.stale_gold_id in ids
    for case in gs.crossref_cases:
        for g in case.relevant_gold_ids:
            assert g in ids
    for case in gs.no_leak_cases:
        assert case.fact_gold_id in ids


def test_materialize_deterministic_ids() -> None:
    gs = load_gold_set()
    units = gs.materialize_memories()
    # Runtime ids derive from the fixture-stable gold ids (no random uuids).
    assert all(
        u.id == gs.runtime_id(g.gold_id)
        for u, g in zip(units, gs.memories, strict=True)
    )


def test_superseded_memory_is_closed() -> None:
    gs = load_gold_set()
    stale = gs.memory_by_gold_id("pref-rust-errors-stale")
    assert stale.superseded
    unit = stale.to_memory()
    assert not unit.is_active
    assert unit.valid_to is not None


def test_age_days_makes_unit_older() -> None:
    gs = load_gold_set()
    current = gs.memory_by_gold_id("pref-rust-errors-current").to_memory()
    older = gs.memory_by_gold_id("pref-rust-errors-stale").to_memory()
    assert older.valid_from < current.valid_from


def test_project_fact_scope_parsed() -> None:
    gs = load_gold_set()
    fact = gs.memory_by_gold_id("fact-rustcli-tokio").to_memory()
    assert fact.type is MemoryType.PROJECT_FACT
    assert fact.scope == Scope.project("rust-cli")
    assert fact.tier is Tier.HOT


def test_runtime_ids_mapping() -> None:
    gs = load_gold_set()
    rt = runtime_ids(gs, ["pref-rust-errors-current"])
    assert rt == {"gold-pref-rust-errors-current"}


def test_save_and_reload_roundtrip(tmp_path) -> None:  # noqa: ANN001
    gs = load_gold_set()
    out = tmp_path / "gs.json"
    save_gold_set(gs, out)
    reloaded = load_gold_set(out)
    assert reloaded.name == gs.name
    assert len(reloaded.memories) == len(gs.memories)
    assert [m.gold_id for m in reloaded.memories] == [m.gold_id for m in gs.memories]


def test_memory_by_gold_id_missing_raises() -> None:
    gs = load_gold_set()
    with pytest.raises(KeyError):
        gs.memory_by_gold_id("does-not-exist")
