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

from datetime import UTC, datetime

import pytest

from mnemozine.config import MaintenanceSettings, RetrievalSettings
from mnemozine.interfaces import StorageBackend, WriteDecision
from mnemozine.migrations import CURRENT_DATA_VERSION, UNSTAMPED_DATA_VERSION
from mnemozine.schema.events import Source
from mnemozine.schema.models import (
    Edge,
    Entity,
    MemoryUnit,
    Provenance,
    RawChunk,
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
    category: str = "preference",
    cross_ref_candidate: bool = False,
    confidence: float = 0.9,
    mid: str | None = None,
) -> MemoryUnit:
    # Core redesign (category split): the old controlled ``type`` is gone — a
    # memory now carries a HIERARCHICAL ``scope`` (global vs project:<...>) for the
    # no-leak decision plus a FREE-FORM ``category`` string and a
    # ``cross_ref_candidate`` flag. A global-scope memory with category
    # 'preference' is the new shape of the old PREFERENCE; a project-scoped memory
    # with category 'project_fact' the old PROJECT_FACT.
    kwargs = {
        "content": content,
        "scope": scope or Scope.global_(),
        "category": category,
        "cross_ref_candidate": cross_ref_candidate,
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
            category="project_fact",
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
            category="project_fact",
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
            category="project_fact",
            entities=["x"],
        )
    )
    # querying a different project must not see p1's fact (no-leak, FR-STO-3)
    hits = await store.scoped_query("secret", [Scope.project("p2")])
    assert hits == []


# ---------------------------------------------------------------------------
# Hierarchical scope: ancestor-composition + no-leak (FR-STO-3, CRITICAL)
# ---------------------------------------------------------------------------


async def test_scoped_query_composes_ancestor_chain() -> None:
    """A query at project:P/auth sees auth + project:P + global ancestors.

    The hierarchical no-leak rule (FR-STO-3): a query at a deep scope retrieves
    every memory whose stored scope is an ancestor-or-self of the query scope.
    We seed one memory at each level of the chain and assert the deep query
    composes all three.
    """

    store = _backend()
    await store.upsert_memory(
        _memory(content="global wide", scope=Scope.global_(), entities=["e"])
    )
    await store.upsert_memory(
        _memory(
            content="project level",
            scope=Scope.project("Mnemozine"),
            category="project_fact",
            entities=["e"],
        )
    )
    await store.upsert_memory(
        _memory(
            content="auth subscope",
            scope=Scope.project("Mnemozine", "auth"),
            category="project_fact",
            entities=["e"],
        )
    )

    # Query at the deepest scope composes its whole ancestor chain.
    hits = await store.scoped_query(
        "global project auth",
        [Scope.project("Mnemozine", "auth")],
        entities=["e"],
    )
    contents = {h.memory.content for h in hits}
    assert contents == {"global wide", "project level", "auth subscope"}


async def test_scoped_query_no_leak_to_sibling_subscope() -> None:
    """A project_fact in project:P/auth NEVER returns for the sibling project:P/db.

    Siblings are not on each other's ancestor chain, so even an exact vector
    match must not leak across them (the headline no-leak guarantee).
    """

    store = _backend()
    await store.upsert_memory(
        _memory(
            content="auth only secret",
            scope=Scope.project("Mnemozine", "auth"),
            category="project_fact",
            entities=["e"],
        )
    )
    # The sibling sub-scope shares the project ancestor but is NOT an ancestor of
    # auth, so it must see nothing of auth's.
    leaked = await store.scoped_query(
        "auth only secret",
        [Scope.project("Mnemozine", "db")],
        entities=["e"],
    )
    assert leaked == []
    # ...but a query AT auth (or deeper) does see it.
    own = await store.scoped_query(
        "auth only secret",
        [Scope.project("Mnemozine", "auth")],
        entities=["e"],
    )
    assert [h.memory.content for h in own] == ["auth only secret"]


async def test_scoped_query_no_leak_parent_to_child() -> None:
    """A query at the parent scope never sees a descendant's memory (no-leak).

    Composition is ancestor-or-SELF only: project:P sees global + project:P, but
    NOT project:P/auth (a child is a descendant, not an ancestor).
    """

    store = _backend()
    await store.upsert_memory(
        _memory(
            content="deep child fact",
            scope=Scope.project("Mnemozine", "auth"),
            category="project_fact",
            entities=["e"],
        )
    )
    hits = await store.scoped_query(
        "deep child fact", [Scope.project("Mnemozine")], entities=["e"]
    )
    assert hits == []


async def test_scoped_query_compose_ancestors_false_matches_exact_only() -> None:
    """compose_ancestors=False matches the exact scope string, no ancestors.

    A maintenance pass that must not widen passes compose_ancestors=False; then a
    query at project:P/auth must NOT pull in the global/project ancestors.
    """

    store = _backend()
    await store.upsert_memory(
        _memory(content="global wide", scope=Scope.global_(), entities=["e"])
    )
    await store.upsert_memory(
        _memory(
            content="auth subscope",
            scope=Scope.project("Mnemozine", "auth"),
            category="project_fact",
            entities=["e"],
        )
    )
    hits = await store.scoped_query(
        "global auth",
        [Scope.project("Mnemozine", "auth")],
        entities=["e"],
        compose_ancestors=False,
    )
    # Only the exact-scope memory; the global ancestor is excluded.
    assert [h.memory.content for h in hits] == ["auth subscope"]


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
        category="project_fact",
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


# ---------------------------------------------------------------------------
# Raw-chunk tier roundtrip (offline re-extraction/reindex; survives R4, §7)
# ---------------------------------------------------------------------------


def _raw_chunk(
    *,
    content_hash: str = "deadbeef",
    content: str = "normalized chunk text",
    scope: Scope | None = None,
    session_id: str = "sess-1",
    source: str = "claude_code",
    memory_ids: list[str] | None = None,
    ingested_at: datetime | None = None,
) -> RawChunk:
    sc = scope or Scope.project("Mnemozine")
    return RawChunk(
        content_hash=content_hash,
        content=content,
        source=source,
        session_id=session_id,
        scope=sc,
        project=sc.project_id or "",
        event_count=3,
        memory_ids=memory_ids if memory_ids is not None else [],
        ingested_at=ingested_at or datetime.now(UTC),
    )


async def test_raw_chunk_roundtrip_persist_and_read() -> None:
    store = _backend()
    chunk = _raw_chunk(memory_ids=["m1", "m2"])
    persisted = await store.persist_raw_chunk(chunk)
    assert persisted.content_hash == "deadbeef"

    read = [c async for c in store.iter_raw_chunks()]
    assert len(read) == 1
    got = read[0]
    # Every load-bearing field survives the flatten/rehydrate roundtrip.
    assert got.content_hash == "deadbeef"
    assert got.content == "normalized chunk text"
    assert got.scope.as_str() == Scope.project("Mnemozine").as_str()
    assert got.project == "Mnemozine"
    assert got.session_id == "sess-1"
    assert got.event_count == 3
    assert got.memory_ids == ["m1", "m2"]


async def test_raw_chunk_persist_is_idempotent_on_content_hash() -> None:
    store = _backend()
    await store.persist_raw_chunk(_raw_chunk(content_hash="h1", memory_ids=["m1"]))
    # Re-persist the SAME content_hash with updated memory_ids -> overwrite, not
    # a duplicate node (FR-ING-5 idempotency / R4 re-extraction safety).
    await store.persist_raw_chunk(
        _raw_chunk(content_hash="h1", memory_ids=["m1", "m2", "m3"])
    )
    chunks = [c async for c in store.iter_raw_chunks()]
    assert len(chunks) == 1
    assert chunks[0].memory_ids == ["m1", "m2", "m3"]


async def test_iter_raw_chunks_filters_exact_scope_no_composition() -> None:
    """iter_raw_chunks matches the EXACT scope (a re-extraction must not widen)."""

    store = _backend()
    await store.persist_raw_chunk(
        _raw_chunk(content_hash="g", scope=Scope.global_())
    )
    await store.persist_raw_chunk(
        _raw_chunk(content_hash="p", scope=Scope.project("Mnemozine"))
    )
    await store.persist_raw_chunk(
        _raw_chunk(content_hash="a", scope=Scope.project("Mnemozine", "auth"))
    )

    # Exact-scope only: the project filter must NOT pull in the global ancestor
    # nor the auth descendant.
    project_only = {
        c.content_hash
        async for c in store.iter_raw_chunks(scope=Scope.project("Mnemozine"))
    }
    assert project_only == {"p"}

    by_session = {
        c.content_hash async for c in store.iter_raw_chunks(session_id="sess-1")
    }
    assert by_session == {"g", "p", "a"}


async def test_re_extract_from_raw_chunks_supersedes_prior_memories() -> None:
    """The re_extract seam supersedes the memories a chunk previously produced.

    Without a text-based extractor entry point the seam still closes the prior
    memories' validity windows (so a reindex never leaves a stale + fresh copy
    both active) and reports the pass.
    """

    store = _backend()
    m = _memory(content="old extracted fact", scope=Scope.project("Mnemozine"))
    await store.upsert_memory(m)
    await store.persist_raw_chunk(
        _raw_chunk(scope=Scope.project("Mnemozine"), memory_ids=[m.id])
    )

    class _NoopExtractor:
        async def extract(self, chunk):  # type: ignore[no-untyped-def]
            return []

        async def classify(self, statement, context):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    report = await store.re_extract_from_raw_chunks(
        _NoopExtractor(),  # type: ignore[arg-type]
        scope=Scope.project("Mnemozine"),
    )
    assert report.re_extracted == 1
    # The chunk's prior memory had its window closed (superseded).
    assert store._client.driver.memories[m.id]["valid_to"] is not None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Reclassify (R1): re-tag scope/category/cross_ref from stored content
# ---------------------------------------------------------------------------


async def test_reclassify_memory_updates_scope_in_place() -> None:
    store = _backend()
    m = _memory(content="was global", scope=Scope.global_(), entities=["rust"])
    await store.upsert_memory(m)

    updated = await store.reclassify_memory(
        m.id, scope=Scope.project("Mnemozine", "auth")
    )
    assert updated.scope.as_str() == Scope.project("Mnemozine", "auth").as_str()
    # Persisted: re-reading sees the new scope, and the no-leak rule now applies —
    # the re-scoped memory is reachable only at/under its new scope.
    stored = store._client.driver.memories[m.id]  # type: ignore[attr-defined]
    assert stored["scope"] == Scope.project("Mnemozine", "auth").as_str()
    reread = await store.get_memory(m.id)
    assert reread is not None and reread.scope.as_str() == updated.scope.as_str()


async def test_reclassify_memory_relabels_category_normalized() -> None:
    store = _backend()
    m = _memory(content="a decision", category="fact", entities=["rust"])
    await store.upsert_memory(m)

    # Mixed-case + whitespace -> normalized to a lowercased/trimmed slug.
    updated = await store.reclassify_memory(m.id, category="  Decision  ")
    assert updated.category == "decision"
    assert store._client.driver.memories[m.id]["category"] == "decision"  # type: ignore[attr-defined]


async def test_reclassify_memory_toggles_cross_ref_only() -> None:
    store = _backend()
    m = _memory(content="seed idea", entities=["rust"], cross_ref_candidate=False)
    await store.upsert_memory(m)

    updated = await store.reclassify_memory(m.id, cross_ref_candidate=True)
    assert updated.cross_ref_candidate is True
    # Untouched fields are preserved (scope/category unchanged).
    assert updated.scope.as_str() == m.scope.as_str()
    assert updated.category == m.category
    assert store._client.driver.memories[m.id]["cross_ref_candidate"] is True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Category registry: list / filter / merge (FR-MNT-2/4, category split)
# ---------------------------------------------------------------------------


async def test_list_categories_counts_active_only() -> None:
    store = _backend()
    await store.upsert_memory(
        _memory(content="pref one", category="preference", entities=["a"])
    )
    await store.upsert_memory(
        _memory(content="pref two", category="preference", entities=["b"])
    )
    g = _memory(content="a gotcha", category="gotcha", entities=["c"])
    await store.upsert_memory(g)
    # Close one preference's window: it must drop out of the active counts.
    closed = _memory(content="stale pref", category="preference", entities=["d"])
    await store.upsert_memory(closed)
    await store.close_validity_window(closed.id)

    counts = dict(await store.list_categories())
    assert counts["preference"] == 2  # the closed one is excluded
    assert counts["gotcha"] == 1


async def test_merge_categories_relabels_and_is_idempotent() -> None:
    store = _backend()
    await store.upsert_memory(
        _memory(content="g1", category="gotcha", entities=["a"])
    )
    await store.upsert_memory(
        _memory(content="g2", category="gotchas", entities=["b"])
    )
    await store.upsert_memory(
        _memory(content="g3", category="gotchas", entities=["c"])
    )

    # Fold the near-duplicate 'gotchas' into the canonical 'gotcha'.
    n = await store.merge_categories("gotchas", "gotcha")
    assert n == 2
    counts = dict(await store.list_categories())
    assert counts.get("gotcha") == 3
    assert "gotchas" not in counts
    # Idempotent: re-running relabels nothing.
    assert await store.merge_categories("gotchas", "gotcha") == 0


# ---------------------------------------------------------------------------
# Data-versioning / in-place migration (mnemozine.migrations)
# ---------------------------------------------------------------------------


async def test_memory_write_stamps_current_data_version() -> None:
    """A memory write persists data_version (default CURRENT) and reads it back."""

    store = _backend()
    m = _memory(content="versioned fact", entities=["rust"])
    await store.upsert_memory(m)

    # Persisted as a scalar prop on the node...
    stored = store._client.driver.memories[m.id]  # type: ignore[attr-defined]
    assert stored["data_version"] == CURRENT_DATA_VERSION
    # ...and read back through the full rehydrate path.
    reread = await store.get_memory(m.id)
    assert reread is not None
    assert reread.data_version == CURRENT_DATA_VERSION


async def test_raw_chunk_write_stamps_current_data_version() -> None:
    """A raw-chunk write persists data_version (default CURRENT) and reads it back."""

    store = _backend()
    await store.persist_raw_chunk(_raw_chunk(content_hash="v1"))

    stored = store._client.driver.raw_chunks["v1"]  # type: ignore[attr-defined]
    assert stored["data_version"] == CURRENT_DATA_VERSION
    read = [c async for c in store.iter_raw_chunks()]
    assert read[0].data_version == CURRENT_DATA_VERSION


async def test_legacy_record_read_back_as_version_zero() -> None:
    """A node written with no data_version prop reads back as 0 (legacy/unstamped)."""

    store = _backend()
    m = _memory(content="legacy fact", entities=["rust"])
    await store.upsert_memory(m)
    # Simulate a pre-feature node: strip the property the migration introduced.
    del store._client.driver.memories[m.id]["data_version"]  # type: ignore[attr-defined]

    reread = await store.get_memory(m.id)
    assert reread is not None
    assert reread.data_version == UNSTAMPED_DATA_VERSION == 0


async def test_min_data_version_empty_store_returns_current() -> None:
    """An empty store has nothing to migrate -> min is CURRENT_DATA_VERSION."""

    store = _backend()
    assert await store.min_data_version() == CURRENT_DATA_VERSION


async def test_min_data_version_mins_over_both_tiers() -> None:
    """min_data_version takes the min across BOTH memories and raw chunks."""

    store = _backend()
    m = _memory(content="a fact", entities=["rust"])
    await store.upsert_memory(m)
    await store.persist_raw_chunk(_raw_chunk(content_hash="c1"))
    # Both freshly written at CURRENT.
    assert await store.min_data_version() == CURRENT_DATA_VERSION

    # A single unstamped/legacy memory drags the whole-store min down to 0.
    del store._client.driver.memories[m.id]["data_version"]  # type: ignore[attr-defined]
    assert await store.min_data_version() == 0


async def test_min_data_version_legacy_chunk_pulls_min_to_zero() -> None:
    """Any unstamped raw chunk makes min_data_version 0 (the chunk tier counts)."""

    store = _backend()
    await store.upsert_memory(_memory(content="ok", entities=["rust"]))
    await store.persist_raw_chunk(_raw_chunk(content_hash="legacy"))
    # Strip the chunk's version prop -> coalesces to 0.
    del store._client.driver.raw_chunks["legacy"]["data_version"]  # type: ignore[attr-defined]
    assert await store.min_data_version() == 0


async def test_iter_memories_below_version_selects_only_stale() -> None:
    """iter_memories_below_version yields exactly the memories under the target."""

    store = _backend()
    stale = _memory(content="stale", entities=["rust"], mid="stale-id")
    current = _memory(content="current", entities=["go"], mid="current-id")
    await store.upsert_memory(stale)
    await store.upsert_memory(current)
    # Demote one to a legacy/unstamped record (coalesces to 0).
    del store._client.driver.memories["stale-id"]["data_version"]  # type: ignore[attr-defined]

    below = [m.id async for m in store.iter_memories_below_version(CURRENT_DATA_VERSION)]
    assert below == ["stale-id"]
    # Nothing is below 0.
    assert [m async for m in store.iter_memories_below_version(0)] == []


async def test_set_data_version_stamps_and_is_idempotent() -> None:
    """set_data_version stamps the given ids, returns the count, and is idempotent."""

    store = _backend()
    a = _memory(content="a", entities=["rust"], mid="a")
    b = _memory(content="b", entities=["go"], mid="b")
    await store.upsert_memory(a)
    await store.upsert_memory(b)
    # Pretend they are legacy/unstamped.
    del store._client.driver.memories["a"]["data_version"]  # type: ignore[attr-defined]
    del store._client.driver.memories["b"]["data_version"]  # type: ignore[attr-defined]

    n = await store.set_data_version(["a", "b"], CURRENT_DATA_VERSION)
    assert n == 2
    assert store._client.driver.memories["a"]["data_version"] == CURRENT_DATA_VERSION  # type: ignore[attr-defined]
    assert store._client.driver.memories["b"]["data_version"] == CURRENT_DATA_VERSION  # type: ignore[attr-defined]
    # After stamping, none remain below the target.
    assert [m async for m in store.iter_memories_below_version(CURRENT_DATA_VERSION)] == []
    # Empty id list is a no-op.
    assert await store.set_data_version([], CURRENT_DATA_VERSION) == 0


async def test_iter_chunks_below_version_and_set_chunk_data_version() -> None:
    """The raw-chunk tier has the symmetric below/set seam (cheap-migration path)."""

    store = _backend()
    await store.persist_raw_chunk(_raw_chunk(content_hash="stale"))
    await store.persist_raw_chunk(_raw_chunk(content_hash="fresh"))
    # Demote one chunk to legacy/unstamped.
    del store._client.driver.raw_chunks["stale"]["data_version"]  # type: ignore[attr-defined]

    below = [c.content_hash async for c in store.iter_chunks_below_version(CURRENT_DATA_VERSION)]
    assert below == ["stale"]

    n = await store.set_chunk_data_version(["stale"], CURRENT_DATA_VERSION)
    assert n == 1
    assert store._client.driver.raw_chunks["stale"]["data_version"] == CURRENT_DATA_VERSION  # type: ignore[attr-defined]
    # Now nothing is below the target, so min reaches CURRENT across both tiers.
    assert [c async for c in store.iter_chunks_below_version(CURRENT_DATA_VERSION)] == []
    assert await store.min_data_version() == CURRENT_DATA_VERSION
    # Empty hash list is a no-op.
    assert await store.set_chunk_data_version([], CURRENT_DATA_VERSION) == 0


async def test_reclassify_memory_bumps_data_version() -> None:
    """reclassify_memory re-stamps the touched memory to CURRENT_DATA_VERSION."""

    store = _backend()
    m = _memory(content="will be reclassified", entities=["rust"], mid="rc")
    await store.upsert_memory(m)
    # Demote to a legacy/unstamped record so the bump is observable.
    del store._client.driver.memories["rc"]["data_version"]  # type: ignore[attr-defined]
    assert (await store.get_memory("rc")).data_version == 0  # type: ignore[union-attr]

    updated = await store.reclassify_memory("rc", category="decision")
    assert updated.data_version == CURRENT_DATA_VERSION
    # Persisted: the stamp survives the write and a fresh read.
    assert store._client.driver.memories["rc"]["data_version"] == CURRENT_DATA_VERSION  # type: ignore[attr-defined]
    reread = await store.get_memory("rc")
    assert reread is not None and reread.data_version == CURRENT_DATA_VERSION
    # A reclassify that changes nothing else still bumps the version.
    del store._client.driver.memories["rc"]["data_version"]  # type: ignore[attr-defined]
    again = await store.reclassify_memory("rc")
    assert again.data_version == CURRENT_DATA_VERSION
    assert store._client.driver.memories["rc"]["data_version"] == CURRENT_DATA_VERSION  # type: ignore[attr-defined]


async def test_re_extract_from_raw_chunks_bumps_chunk_data_version() -> None:
    """re_extract_from_raw_chunks re-stamps each re-processed chunk to CURRENT."""

    store = _backend()
    await store.persist_raw_chunk(
        _raw_chunk(content_hash="rx", scope=Scope.project("Mnemozine"))
    )
    # Demote to a legacy/unstamped chunk.
    del store._client.driver.raw_chunks["rx"]["data_version"]  # type: ignore[attr-defined]

    class _NoopExtractor:
        async def extract(self, chunk):  # type: ignore[no-untyped-def]
            return []

        async def classify(self, statement, context):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    report = await store.re_extract_from_raw_chunks(
        _NoopExtractor(),  # type: ignore[arg-type]
        scope=Scope.project("Mnemozine"),
    )
    assert report.re_extracted == 1
    # The chunk was re-stamped up to the current version.
    assert store._client.driver.raw_chunks["rx"]["data_version"] == CURRENT_DATA_VERSION  # type: ignore[attr-defined]
    reread = [c async for c in store.iter_raw_chunks()]
    assert reread[0].data_version == CURRENT_DATA_VERSION
