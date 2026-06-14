"""Smoke tests that the shared contracts import and the fakes satisfy them.

These guard the foundation: if a downstream change breaks a Protocol shape or a
schema invariant, this fails immediately rather than deep inside a module.
"""

from __future__ import annotations

import pytest

import mnemozine
from mnemozine.config import Settings
from mnemozine.interfaces import (
    EmbeddingProvider,
    LLMProvider,
    StorageBackend,
    WriteDecision,
)
from mnemozine.schema.events import IngestEvent, content_hash
from mnemozine.schema.models import MemoryUnit, RawChunk, Scope, ScopeDecision, Tier
from tests.conftest import FakeEmbeddingProvider, FakeLLMProvider, InMemoryStorage


def test_package_imports() -> None:
    assert mnemozine.__version__


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(InMemoryStorage(), StorageBackend)
    assert isinstance(FakeEmbeddingProvider(), EmbeddingProvider)
    assert isinstance(FakeLLMProvider(), LLMProvider)


def test_settings_has_tuning_params() -> None:
    s = Settings()
    # §6.6 initial values that the PRD pins explicitly.
    assert s.inject.token_budget == 500
    assert s.crossref.max_suggestions in (1, 2)
    assert s.embedding.model == "bge-m3"


def test_scope_roundtrip() -> None:
    assert Scope.global_().as_str() == "global"
    assert Scope.project("rust-cli").as_str() == "project:rust-cli"
    assert Scope.parse("project:rust-cli").project_id == "rust-cli"
    assert Scope.parse("global").is_global


def test_scope_hierarchical_roundtrip() -> None:
    # A sub-scope round-trips through the canonical string form.
    sub = Scope.project("Mnemozine", "auth")
    assert sub.as_str() == "project:Mnemozine/auth"
    assert Scope.parse("project:Mnemozine/auth").segments == ["Mnemozine", "auth"]
    assert sub.project_id == "Mnemozine"  # the project segment, not the leaf
    assert sub.leaf == "auth"
    # child() is the constructor for going one level deeper.
    assert Scope.project("Mnemozine").child("auth").as_str() == "project:Mnemozine/auth"


def test_scope_ancestors_compose_chain() -> None:
    # ancestors() yields [global, project:P, project:P/sub] (root first, self last).
    chain = [s.as_str() for s in Scope.project("Mnemozine", "auth").ancestors()]
    assert chain == ["global", "project:Mnemozine", "project:Mnemozine/auth"]
    assert [s.as_str() for s in Scope.global_().ancestors()] == ["global"]


def test_scope_no_leak_ancestor_or_self() -> None:
    g = Scope.global_()
    proj = Scope.project("Mnemozine")
    auth = Scope.project("Mnemozine", "auth")
    db = Scope.project("Mnemozine", "db")
    other = Scope.project("Other")

    # A query at auth sees ancestor-or-self: global, project, auth itself.
    assert g.contains(auth)
    assert proj.contains(auth)
    assert auth.contains(auth)
    # Siblings never leak into each other.
    assert not db.contains(auth)
    assert not auth.contains(db)
    # A different project never leaks.
    assert not other.contains(auth)
    # is_descendant_of is the symmetric view.
    assert auth.is_descendant_of(proj)
    assert auth.is_descendant_of(g)
    assert not auth.is_descendant_of(db)


def test_memory_unit_category_split() -> None:
    # The 3-value MemoryType is gone: scope decision is controlled, category free-form.
    m = MemoryUnit(
        content="Prefers thiserror over anyhow.",
        scope=Scope.global_(),
        category="Preference",  # normalized to a lowercased slug
        cross_ref_candidate=False,
    )
    assert m.category == "preference"
    assert m.scope_decision is ScopeDecision.GLOBAL
    proj = MemoryUnit(content="pins tokio 1.38", scope=Scope.project("rust-cli"))
    assert proj.scope_decision is ScopeDecision.PROJECT
    assert proj.category == "fact"  # DEFAULT_CATEGORY


def test_raw_chunk_retains_normalized_input() -> None:
    chunk = RawChunk(
        content_hash="deadbeef",
        content="user:I prefer thiserror.",
        source="claude_code",
        session_id="sess-1",
        scope=Scope.project("rust-cli"),
        project="rust-cli",
        event_count=1,
        memory_ids=["m1"],
    )
    assert chunk.scope.as_str() == "project:rust-cli"
    assert chunk.memory_ids == ["m1"]


def test_content_hash_is_offset_invariant() -> None:
    # FR-ING-5: hashing is on content, not byte/line offset.
    a = content_hash("user:hello world")
    b = content_hash("user:hello world")
    assert a == b
    assert content_hash("user:different") != a


def test_ingest_event_idempotency_key(sample_events: list[IngestEvent]) -> None:
    e = sample_events[0]
    src, sess, h = e.idempotency_key()
    assert src == "claude_code"
    assert sess == "sess-1"
    assert h == e.content_hash()


def test_memory_unit_validity_window(sample_memory: MemoryUnit) -> None:
    # FR-STO-1 / FR-MNT-1: a fresh unit is active; supersede() closes the window.
    assert sample_memory.is_active
    assert sample_memory.valid_to is None
    sample_memory.supersede()
    assert not sample_memory.is_active
    assert sample_memory.valid_to is not None
    assert sample_memory.tier is Tier.HOT  # supersede != archive


@pytest.mark.asyncio
async def test_inmemory_write_decisions(sample_memory: MemoryUnit) -> None:
    # add
    store = InMemoryStorage()
    r1 = await store.upsert_memory(sample_memory)
    assert r1.decision is WriteDecision.ADD

    # reinforce: identical content, same scope/entities
    dup = sample_memory.model_copy(update={"id": "other", "confidence": 0.95})
    r2 = await store.upsert_memory(dup)
    assert r2.decision is WriteDecision.REINFORCE

    # supersede: a contradicting global-decision memory flips the window
    store2 = InMemoryStorage(contradicts=lambda new, existing: True)
    await store2.upsert_memory(sample_memory)
    new_pref = MemoryUnit(
        content="Prefers anyhow over thiserror now.",
        scope=Scope.global_(),
        category="preference",
        entities=["rust", "error-handling"],
        confidence=0.9,
        provenance=sample_memory.provenance,
    )
    r3 = await store2.upsert_memory(new_pref)
    assert r3.decision is WriteDecision.SUPERSEDE
    assert r3.superseded and r3.superseded[0].valid_to is not None


@pytest.mark.asyncio
async def test_inmemory_scoped_query(sample_memory: MemoryUnit) -> None:
    store = InMemoryStorage()
    await store.upsert_memory(sample_memory)
    hits = await store.scoped_query(
        "thiserror error handling", [Scope.global_()], entities=["rust"]
    )
    assert hits and hits[0].memory.id == sample_memory.id
    # archived memories drop off the hot path
    await store.archive(sample_memory.id)
    assert sample_memory.tier is Tier.ARCHIVE
    hits2 = await store.scoped_query("thiserror", [Scope.global_()])
    assert not hits2
