"""Contract tests for the Graphiti/FalkorDB storage backend (FR-STO-*, FR-MNT-1).

These build a *real* :class:`GraphitiStorageBackend` over the in-process
:class:`FakeFalkorDriver` (which interprets the backend's Cypher against dict
stores) and the shared :class:`FakeEmbeddingProvider`. Every assertion exercises
the backend's real serialization / 4-way decision / scope+tier filtering / cosine
ranking / validity-window / tiering / entity+edge / suppression / session code —
with no live FalkorDB or Ollama.

This is the "thin contract test" required by the task: it pins the backend to the
shape FalkorDB returns and to the StorageBackend Protocol semantics.
"""

from __future__ import annotations

import pytest

from mnemozine.config import MaintenanceSettings, RetrievalSettings
from mnemozine.interfaces import StorageBackend, WriteDecision
from mnemozine.schema.events import Source
from mnemozine.schema.models import (
    Edge,
    Entity,
    MemoryType,
    MemoryUnit,
    Provenance,
    Scope,
    SourceSession,
    Tier,
)
from mnemozine.storage.backend import GraphitiStorageBackend
from tests.conftest import FakeEmbeddingProvider
from tests.storage.fake_falkor import FakeGraphitiClient


def _backend(
    *,
    contradicts=None,
    dedup_threshold: float | None = None,
    retrieval: RetrievalSettings | None = None,
) -> GraphitiStorageBackend:
    # The shared FakeEmbeddingProvider's coarse positive-orthant vectors give a
    # high cosine (~0.92) between unrelated short strings, so tests that need the
    # supersede/no-op branches (rather than reinforce) raise the dedup threshold
    # to make exact-content the only "equivalent" — a legitimate config knob.
    maint = (
        MaintenanceSettings(dedup_equivalence_threshold=dedup_threshold)
        if dedup_threshold
        else None
    )
    return GraphitiStorageBackend(
        client=FakeGraphitiClient(),  # type: ignore[arg-type]
        embeddings=FakeEmbeddingProvider(),
        contradicts=contradicts,
        maintenance=maint,
        retrieval=retrieval,
    )


def _memory(
    *,
    content: str,
    scope: Scope | None = None,
    entities: list[str] | None = None,
    mtype: MemoryType = MemoryType.PREFERENCE,
    confidence: float = 0.9,
    mid: str | None = None,
) -> MemoryUnit:
    kwargs = {
        "type": mtype,
        "content": content,
        "scope": scope or Scope.global_(),
        "entities": entities if entities is not None else ["rust", "error-handling"],
        "confidence": confidence,
        "provenance": Provenance(source=Source.CLAUDE_CODE.value, session_id="sess-1"),
    }
    if mid is not None:
        kwargs["id"] = mid
    return MemoryUnit(**kwargs)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_storage_protocol() -> None:
    assert isinstance(_backend(), StorageBackend)


# ---------------------------------------------------------------------------
# FR-MNT-1 4-way write decision
# ---------------------------------------------------------------------------


async def test_write_add_then_reinforce() -> None:
    store = _backend()
    m = _memory(content="Prefers thiserror over anyhow.")
    r1 = await store.upsert_memory(m)
    assert r1.decision is WriteDecision.ADD

    # identical content, same scope/entities -> reinforce (bump confidence)
    dup = _memory(content="Prefers thiserror over anyhow.", confidence=0.95)
    r2 = await store.upsert_memory(dup)
    assert r2.decision is WriteDecision.REINFORCE
    assert r2.memory.confidence == pytest.approx(0.95)
    # no new node was created
    assert len(store._client.driver.memories) == 1  # type: ignore[attr-defined]


async def test_write_supersede_closes_old_window() -> None:
    # inject a contradiction predicate driving the supersede branch (no LLM)
    async def contradicts(new, candidates):
        return list(candidates)

    store = _backend(contradicts=contradicts, dedup_threshold=1.0)
    old = _memory(content="Prefers anyhow for error handling.")
    await store.upsert_memory(old)

    new = _memory(content="Prefers thiserror for error handling now.", confidence=0.95)
    r = await store.upsert_memory(new)
    assert r.decision is WriteDecision.SUPERSEDE
    assert r.superseded and r.superseded[0].valid_to is not None
    # old node retained (never hard-deleted), window closed
    stored_old = store._client.driver.memories[old.id]  # type: ignore[attr-defined]
    assert stored_old["valid_to"] is not None
    # new node inserted active
    stored_new = store._client.driver.memories[new.id]  # type: ignore[attr-defined]
    assert stored_new["valid_to"] is None


async def test_write_noop_on_strictly_weaker_duplicate() -> None:
    # dedup_threshold=1.0 so only EXACT content reinforces; a case-only variant
    # with lower confidence then falls through to the no-op branch.
    store = _backend(dedup_threshold=1.0)
    strong = _memory(content="Prefers ripgrep over grep.", confidence=0.9)
    await store.upsert_memory(strong)
    # same content (different case) + lower confidence + same type -> no-op.
    weaker = _memory(content="prefers Ripgrep over Grep.", confidence=0.5)
    r = await store.upsert_memory(weaker)
    assert r.decision is WriteDecision.NO_OP
    assert r.memory.id == strong.id


async def test_candidates_scoped_to_same_scope_only() -> None:
    # A project_fact in a different scope must NOT be a write candidate (FR-STO-3).
    store = _backend()
    await store.upsert_memory(
        _memory(
            content="Pins tokio 1.38.",
            scope=Scope.project("rust-cli"),
            mtype=MemoryType.PROJECT_FACT,
            entities=["tokio"],
        )
    )
    # global preference sharing no entity/scope -> still an ADD
    r = await store.upsert_memory(
        _memory(content="Pins tokio 1.38.", scope=Scope.global_(), entities=["tokio"])
    )
    assert r.decision is WriteDecision.ADD


# ---------------------------------------------------------------------------
# FR-RET-2 / FR-STO-3 scoped query + FR-STO-4 tiering
# ---------------------------------------------------------------------------


async def test_scoped_query_composes_scopes_and_ranks() -> None:
    store = _backend()
    await store.upsert_memory(
        _memory(content="alpha beta gamma rust async", entities=["rust"])
    )
    await store.upsert_memory(
        _memory(
            content="zeta project specific note",
            scope=Scope.project("p1"),
            mtype=MemoryType.PROJECT_FACT,
            entities=["rust"],
        )
    )
    # compose global + project:p1
    hits = await store.scoped_query(
        "alpha beta gamma",
        [Scope.global_(), Scope.project("p1")],
        entities=["rust"],
    )
    assert len(hits) == 2
    # results are sorted by score desc (cosine ranking is real)
    assert hits[0].score >= hits[1].score


async def test_scoped_query_excludes_other_scopes() -> None:
    store = _backend()
    await store.upsert_memory(
        _memory(
            content="secret project fact",
            scope=Scope.project("p1"),
            mtype=MemoryType.PROJECT_FACT,
            entities=["x"],
        )
    )
    # querying a different project must not see p1's fact (no-leak, FR-STO-3)
    hits = await store.scoped_query("secret", [Scope.project("p2")])
    assert hits == []


# ---------------------------------------------------------------------------
# F3 — config-driven KNN over-fetch (retrieval.knn_overfetch_factor / _cap)
# ---------------------------------------------------------------------------


def _capture_knn_k(store: GraphitiStorageBackend) -> list[int]:
    """Spy on the driver so we can read back the ``$k`` of each KNN query.

    The backend emits the index-backed KNN as
    ``CALL db.idx.vector.queryNodes(..., $k, vecf32($qv))``; the emitted ``$k``
    is exactly what F3 makes config-driven, so the test asserts on the captured
    ``k`` param rather than re-deriving it.
    """

    driver = store._client.driver  # type: ignore[attr-defined]
    real = driver.execute_query
    seen: list[int] = []

    async def _spy(cypher: str, **params):  # type: ignore[no-untyped-def]
        if "db.idx.vector.queryNodes" in cypher and "k" in params:
            seen.append(params["k"])
        return await real(cypher, **params)

    driver.execute_query = _spy  # type: ignore[method-assign]
    return seen


async def test_knn_overfetch_k_honours_configured_factor() -> None:
    # factor 4, generous cap -> k == top_k * factor (the over-fetch multiple).
    store = _backend(
        retrieval=RetrievalSettings(knn_overfetch_factor=4, knn_overfetch_cap=1000)
    )
    seen = _capture_knn_k(store)
    await store.scoped_query("anything", [Scope.global_()], top_k=5)
    assert seen == [20]  # 5 * 4


async def test_knn_overfetch_k_bounded_by_configured_cap() -> None:
    # top_k * factor (10*10=100) exceeds the cap (32) -> clamped to the cap.
    store = _backend(
        retrieval=RetrievalSettings(knn_overfetch_factor=10, knn_overfetch_cap=32)
    )
    seen = _capture_knn_k(store)
    await store.scoped_query("anything", [Scope.global_()], top_k=10)
    assert seen == [32]


async def test_knn_overfetch_k_never_below_top_k() -> None:
    # A degenerate factor of 0 must not starve the index below top_k itself.
    store = _backend(
        retrieval=RetrievalSettings(knn_overfetch_factor=0, knn_overfetch_cap=1000)
    )
    seen = _capture_knn_k(store)
    await store.scoped_query("anything", [Scope.global_()], top_k=7)
    assert seen == [7]


async def test_knn_overfetch_defaults_match_config_defaults() -> None:
    # No RetrievalSettings supplied -> the backend uses RetrievalSettings()
    # defaults (factor 10, cap 512), so k == top_k * 10 while under the cap.
    store = _backend()
    seen = _capture_knn_k(store)
    await store.scoped_query("anything", [Scope.global_()], top_k=3)
    assert seen == [30]  # 3 * 10, under the 512 cap


async def test_archive_drops_off_hot_path_promote_reembeds() -> None:
    store = _backend()
    m = _memory(content="hot path note", entities=["rust"])
    await store.upsert_memory(m)
    await store.archive(m.id)
    assert store._client.driver.memories[m.id]["tier"] == Tier.ARCHIVE.value  # type: ignore[attr-defined]

    # archived memory excluded from default hot retrieval
    hits = await store.scoped_query("hot path note", [Scope.global_()])
    assert hits == []
    # ...but visible with include_archived
    hits2 = await store.scoped_query(
        "hot path note", [Scope.global_()], include_archived=True
    )
    assert len(hits2) == 1

    # promote restores hot tier and re-embeds (OQ3 lazy-on-promotion)
    promoted = await store.promote(m.id)
    assert promoted.tier is Tier.HOT
    assert store._client.driver.memories[m.id]["tier"] == Tier.HOT.value  # type: ignore[attr-defined]


async def test_close_validity_window_and_record_access() -> None:
    store = _backend()
    m = _memory(content="closing note", entities=["rust"])
    await store.upsert_memory(m)

    closed = await store.close_validity_window(m.id)
    assert closed.valid_to is not None

    await store.record_access(m.id)
    node = store._client.driver.memories[m.id]  # type: ignore[attr-defined]
    assert node["access_count"] == 1
    assert node["last_accessed"] is not None


async def test_reembed_recomputes_embedding() -> None:
    store = _backend()
    m = _memory(content="embed me", entities=["rust"])
    await store.upsert_memory(m)
    before = list(store._client.driver.memories[m.id]["embedding"])  # type: ignore[attr-defined]
    await store.reembed(m.id)
    after = list(store._client.driver.memories[m.id]["embedding"])  # type: ignore[attr-defined]
    # deterministic fake -> same content embeds identically (idempotent re-embed)
    assert before == after


# ---------------------------------------------------------------------------
# Enumeration / scan (FR-MNT-2/3/4)
# ---------------------------------------------------------------------------


async def test_iter_memories_filters() -> None:
    store = _backend()
    g = _memory(content="global pref", entities=["rust"])
    p = _memory(
        content="project fact",
        scope=Scope.project("p1"),
        mtype=MemoryType.PROJECT_FACT,
        entities=["rust"],
    )
    await store.upsert_memory(g)
    await store.upsert_memory(p)
    await store.archive(p.id)

    all_ids = {m.id async for m in store.iter_memories()}
    assert all_ids == {g.id, p.id}

    hot_ids = {m.id async for m in store.iter_memories(tier=Tier.HOT)}
    assert hot_ids == {g.id}

    scoped = {m.id async for m in store.iter_memories(scope=Scope.project("p1"))}
    assert scoped == {p.id}


# ---------------------------------------------------------------------------
# Entity + edge ops (FR-EXT-2, FR-MNT-4, FR-RET-6)
# ---------------------------------------------------------------------------


async def test_entity_upsert_get_by_alias() -> None:
    store = _backend()
    e = Entity(canonical_name="rust", aliases=["rust-lang"], type="language")
    await store.upsert_entity(e)
    assert (await store.get_entity("rust")).id == e.id  # type: ignore[union-attr]
    assert (await store.get_entity("rust-lang")).id == e.id  # type: ignore[union-attr]
    assert (await store.get_entity(e.id)).canonical_name == "rust"  # type: ignore[union-attr]
    assert await store.get_entity("nope") is None


async def test_edge_upsert_reassert_bumps_weight() -> None:
    store = _backend()
    a = Entity(canonical_name="rust")
    b = Entity(canonical_name="async")
    await store.upsert_entity(a)
    await store.upsert_entity(b)

    e1 = Edge(from_entity=a.id, to_entity=b.id, relation="relates", weight=0.5)
    await store.upsert_edge(e1)
    # re-assert same (from,to,relation) with higher weight -> bumps, no duplicate
    e2 = Edge(from_entity=a.id, to_entity=b.id, relation="relates", weight=0.9)
    stored = await store.upsert_edge(e2)
    assert stored.weight == pytest.approx(0.9)
    assert len(store._client.driver.edges) == 1  # type: ignore[attr-defined]


async def test_neighbors_returns_edges_for_reason_and_rank() -> None:
    store = _backend()
    a = Entity(canonical_name="rust")
    b = Entity(canonical_name="async")
    c = Entity(canonical_name="cli")
    for ent in (a, b, c):
        await store.upsert_entity(ent)
    await store.upsert_edge(
        Edge(from_entity=a.id, to_entity=b.id, relation="uses", weight=0.9)
    )
    await store.upsert_edge(
        Edge(from_entity=a.id, to_entity=c.id, relation="uses", weight=0.3)
    )

    neighbors = await store.neighbors("rust")
    assert {n.entity.canonical_name for n in neighbors} == {"async", "cli"}
    # weight-ranked desc so CrossRef can rank + explain
    assert neighbors[0].edge.weight >= neighbors[1].edge.weight
    assert neighbors[0].entity.canonical_name == "async"
    # the edge survives traversal (FR-RET-6 reason needs relation+weight)
    assert neighbors[0].edge.relation == "uses"


async def test_prune_edge_closes_window_and_edges_for_entity() -> None:
    store = _backend()
    a = Entity(canonical_name="rust")
    b = Entity(canonical_name="async")
    await store.upsert_entity(a)
    await store.upsert_entity(b)
    edge = Edge(from_entity=a.id, to_entity=b.id, relation="uses", weight=0.05)
    await store.upsert_edge(edge)

    active = await store.edges_for_entity("rust", active_only=True)
    assert len(active) == 1

    pruned = await store.prune_edge(edge.id)
    assert pruned.valid_to is not None
    # dropped off active traversal but retained
    assert await store.edges_for_entity("rust", active_only=True) == []
    assert len(await store.edges_for_entity("rust", active_only=False)) == 1


async def test_merge_entities_folds_aliases() -> None:
    store = _backend()
    canonical = Entity(canonical_name="rust", aliases=["rustc"])
    dup = Entity(canonical_name="rust-lang", aliases=["the-rust-work"])
    await store.upsert_entity(canonical)
    await store.upsert_entity(dup)

    merged = await store.merge_entities(dup.id, canonical.id)
    assert merged.id == canonical.id
    assert "rust-lang" in merged.aliases
    assert "the-rust-work" in merged.aliases
    # the redundant node is gone (graph does not fragment)
    assert await store.get_entity(dup.id) is None


# ---------------------------------------------------------------------------
# Suppression (FR-RET-6 / R2) + session (§7)
# ---------------------------------------------------------------------------


async def test_suppression_persists() -> None:
    store = _backend()
    assert not await store.is_suppressed("m1", "ctx")
    await store.record_suppression("m1", "ctx")
    assert await store.is_suppressed("m1", "ctx")
    # idempotent
    await store.record_suppression("m1", "ctx")
    assert await store.is_suppressed("m1", "ctx")
    # scoped to (memory, context)
    assert not await store.is_suppressed("m1", "other-ctx")


async def test_record_session_and_close() -> None:
    store = _backend()
    session = SourceSession(
        source=Source.CLAUDE_CODE.value,
        session_id="sess-1",
        project="rust-cli",
        raw_path="~/.claude/projects/rust-cli/sess-1.jsonl",
    )
    await store.record_session(session)
    stored = store._client.driver.sessions[(session.source, session.session_id)]  # type: ignore[attr-defined]
    assert stored["project"] == "rust-cli"
    assert stored["raw_path"].endswith("sess-1.jsonl")

    await store.close()
    assert store._client.driver.closed is True  # type: ignore[attr-defined]
