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


@pytest.mark.asyncio
async def test_persist_mentions_on_fake_is_protocol_conformant() -> None:
    # The new StorageBackend.persist_mentions Protocol method is present on the
    # in-memory fake and returns the count of asserted memory->entity edges.
    from mnemozine.schema.models import Entity, MemoryUnit, Provenance

    store = InMemoryStorage()
    await store.upsert_entity(Entity(id="e-rust", canonical_name="rust"))
    await store.upsert_memory(
        MemoryUnit(
            id="m1",
            content="rust note",
            scope=Scope.global_(),
            category="fact",
            entities=["rust"],
            provenance=Provenance(source="claude_code", session_id="s1"),
        )
    )
    assert await store.persist_mentions() == 1
    # Idempotent: a re-run asserts the same edge, count unchanged.
    assert await store.persist_mentions() == 1


@pytest.mark.asyncio
async def test_resolve_or_create_entity_protocol_conformant_on_all_fakes() -> None:
    # The new StorageBackend.resolve_or_create_entity Protocol method is present on
    # all three in-memory fakes (the conftest InMemoryStorage, the contract test's
    # FakeFalkor-backed real backend, and the evals OfflineStorage) and is
    # identity-by-normalized-name: resolving the same name twice returns ONE node.
    from mnemozine.evals._offline_store import OfflineStorage
    from mnemozine.schema.models import Entity
    from mnemozine.storage.backend import GraphitiStorageBackend
    from tests.storage.fake_falkor import FakeGraphitiClient

    falkor_backend = GraphitiStorageBackend(
        client=FakeGraphitiClient(),  # type: ignore[arg-type]
        embeddings=FakeEmbeddingProvider(),
    )
    fakes: list[StorageBackend] = [
        InMemoryStorage(),
        OfflineStorage(),
        falkor_backend,
    ]
    for store in fakes:
        assert isinstance(store, StorageBackend)
        first = await store.resolve_or_create_entity(Entity(canonical_name="rust"))
        # Same normalized name, different case + new alias -> same id, folded alias.
        second = await store.resolve_or_create_entity(
            Entity(canonical_name="Rust", aliases=["rust-lang"])
        )
        assert second.id == first.id
        assert "rust-lang" in second.aliases
        # A different name mints a distinct node.
        other = await store.resolve_or_create_entity(Entity(canonical_name="async"))
        assert other.id != first.id


@pytest.mark.asyncio
async def test_add_memory_mentions_protocol_conformant_on_all_fakes() -> None:
    # The new StorageBackend.add_memory_mentions Protocol method (the inline
    # per-memory mention seam) is present on all three in-memory fakes (the conftest
    # InMemoryStorage, the contract test's FakeFalkor-backed real backend, and the
    # evals OfflineStorage): it id-keyed MERGEs the memory's mention edges and is
    # idempotent (a re-call asserts the same edges, adds none).
    from mnemozine.evals._offline_store import OfflineStorage
    from mnemozine.schema.models import Entity, MemoryUnit, Provenance
    from mnemozine.storage.backend import GraphitiStorageBackend
    from tests.storage.fake_falkor import FakeGraphitiClient

    falkor_backend = GraphitiStorageBackend(
        client=FakeGraphitiClient(),  # type: ignore[arg-type]
        embeddings=FakeEmbeddingProvider(),
    )
    fakes: list[StorageBackend] = [
        InMemoryStorage(),
        OfflineStorage(),
        falkor_backend,
    ]
    for store in fakes:
        assert isinstance(store, StorageBackend)
        await store.upsert_entity(Entity(id="e-rust", canonical_name="rust"))
        await store.upsert_entity(Entity(id="e-tokio", canonical_name="tokio"))
        await store.upsert_memory(
            MemoryUnit(
                id="m1",
                content="rust + tokio",
                scope=Scope.global_(),
                category="fact",
                entities=["rust", "tokio"],
                provenance=Provenance(source="claude_code", session_id="s1"),
            )
        )
        first = await store.add_memory_mentions("m1", ["e-rust", "e-tokio"])
        assert first == 2
        # Idempotent: a re-call re-asserts the same edges, the edge set never grows.
        second = await store.add_memory_mentions("m1", ["e-rust", "e-tokio"])
        assert second == 2
        # The mention edges landed (dict-backed fakes expose .mentions directly;
        # the FakeFalkor-backed real backend exposes them on its driver).
        mentions = getattr(store, "mentions", None)
        if mentions is None:
            mentions = store._client.driver.mentions  # type: ignore[attr-defined]
        assert mentions == {("m1", "e-rust"), ("m1", "e-tokio")}


@pytest.mark.asyncio
async def test_co_mention_methods_on_fake_are_protocol_conformant() -> None:
    # The three new co-mention StorageBackend Protocol methods are present on the
    # in-memory fake: co_mention_pairs (derived from mentions), entity_mention_counts
    # (document frequency), and upsert_co_mention (idempotent weighted edge).
    from mnemozine.schema.models import Entity, MemoryUnit, Provenance

    store = InMemoryStorage()
    await store.upsert_entity(Entity(id="e-rust", canonical_name="rust"))
    await store.upsert_entity(Entity(id="e-tokio", canonical_name="tokio"))
    for mid in ("m1", "m2"):
        await store.upsert_memory(
            MemoryUnit(
                id=mid,
                content=f"{mid} body",
                scope=Scope.global_(),
                category="fact",
                entities=["rust", "tokio"],
                provenance=Provenance(source="claude_code", session_id="s1"),
            )
        )
    await store.persist_mentions()

    assert await store.co_mention_pairs(min_shared=2) == [("e-rust", "e-tokio", 2)]
    assert await store.entity_mention_counts() == {"e-rust": 2, "e-tokio": 2}
    edge = await store.upsert_co_mention("e-rust", "e-tokio", weight=1.0, shared=2)
    assert edge.relation == "co_mentioned"
    # Idempotent upsert: a re-assert keeps a single edge.
    await store.upsert_co_mention("e-rust", "e-tokio", weight=1.0, shared=2)
    assert len(store.co_mentions) == 1


@pytest.mark.asyncio
async def test_relation_norm_methods_on_fake_are_protocol_conformant() -> None:
    # The two new relation-registry StorageBackend Protocol methods are present on
    # the in-memory fake: list_relations (active-edge label counts) and
    # merge_relations (idempotent relabel folding parallel edges).
    from mnemozine.schema.models import Edge, Entity

    store = InMemoryStorage()
    await store.upsert_entity(Entity(id="e-rust", canonical_name="rust"))
    await store.upsert_entity(Entity(id="e-tokio", canonical_name="tokio"))
    await store.upsert_edge(
        Edge(from_entity="e-rust", to_entity="e-tokio", relation="used_in", weight=0.6)
    )

    assert dict(await store.list_relations()) == {"used_in": 1}
    # Relabel 'used_in' -> 'uses'; the active edge now carries the canonical label.
    assert await store.merge_relations("used_in", "uses") == 1
    assert dict(await store.list_relations()) == {"uses": 1}
    # Idempotent: nothing left labelled 'used_in', and same==same is a no-op.
    assert await store.merge_relations("used_in", "uses") == 0
    assert await store.merge_relations("uses", "uses") == 0


@pytest.mark.asyncio
async def test_entity_dedup_job_drives_merge_entities_repointing_all_layers() -> None:
    # EntityDedupJob folds true-duplicate entities via the existing merge_entities
    # path, which repoints the RELATES + MENTIONS + CO_MENTIONS layers onto the
    # survivor on the in-memory fake (the contract's correctness invariant).
    from mnemozine.maintenance.entity_dedup import EntityDedupJob
    from mnemozine.schema.models import Edge, Entity, MemoryUnit, Provenance

    store = InMemoryStorage()
    await store.upsert_entity(
        Entity(id="e-rust", canonical_name="rust", aliases=["rustc"])
    )
    await store.upsert_entity(Entity(id="e-rust2", canonical_name="Rust"))
    await store.upsert_entity(Entity(id="e-tokio", canonical_name="tokio"))
    await store.upsert_edge(
        Edge(from_entity="e-rust2", to_entity="e-tokio", relation="uses", weight=0.7)
    )
    await store.upsert_memory(
        MemoryUnit(
            id="m1",
            content="Rust note",
            scope=Scope.global_(),
            category="fact",
            entities=["Rust"],
            provenance=Provenance(source="claude_code", session_id="s1"),
        )
    )
    await store.persist_mentions()
    await store.upsert_co_mention("e-rust2", "e-tokio", weight=2.0, shared=3)

    settings = Settings()
    settings.graph.entity_dedup_mode = "exact"
    report = await EntityDedupJob(store, settings=settings).run()

    assert report.entities_merged == 1
    # Mentions + co-mention repointed onto the survivor (no orphan edge).
    assert ("m1", "e-rust") in store.mentions
    assert ("e-rust", "e-tokio") in store.co_mentions
    assert "e-rust2" not in store.entities
    # No memory deleted.
    assert "m1" in store.memories
    # Idempotent: a second pass finds no duplicate group and merges 0.
    second = await EntityDedupJob(store, settings=settings).run()
    assert second.entities_merged == 0


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
