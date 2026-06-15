"""Offline (no FalkorDB/Ollama) display-read tests against the in-memory fakes.

These pin the embedding-free WebUI READ contract (the four
:class:`~mnemozine.interfaces.StorageBackend` display methods —
``store_stats`` / ``query_memories`` / ``get_memory_display`` /
``graph_snapshot``) on BOTH packaged in-memory ``StorageBackend`` fakes:

* :class:`mnemozine.evals._offline_store.OfflineStorage` (the packaged offline
  store shipped in the wheel for ``mnemozine-eval --offline``), and
* :class:`tests.conftest.InMemoryStorage` (the richer shared test fake).

Parametrizing every test over both keeps the two fakes behaviourally consistent
with each other and with the FalkorDB backend's Cypher contract test
(``tests/storage/test_backend_contract.py``), so the route layer can build wire
models from either fake or the real backend identically.

The three load-bearing properties (mirroring the task acceptance criteria):

* ``store_stats`` counts MATCH a row-count baseline computed independently from
  the seeded units (an aggregate must never disagree with a brute count);
* ``query_memories`` returns the right PAGE + TOTAL and NO embedding on any item
  (a ``MemoryView`` has no embedding field at all — display reads never carry the
  vector);
* ``graph_snapshot`` respects the node CAP (and the over-fetch ``truncated``
  sentinel does not false-positive at exactly the cap).
"""

from __future__ import annotations

from collections import Counter

import pytest

from mnemozine.evals._offline_store import OfflineStorage
from mnemozine.interfaces import MemoryView, StorageBackend
from mnemozine.schema.models import (
    Edge,
    Entity,
    MemoryUnit,
    Provenance,
    RawChunk,
    Scope,
    Tier,
)
from tests.conftest import InMemoryStorage

# Both in-memory StorageBackend fakes must agree on the display-read contract;
# parametrize every test over both.
FAKES = [OfflineStorage, InMemoryStorage]


def _memory(
    *,
    content: str,
    scope: Scope | None = None,
    entities: list[str] | None = None,
    category: str = "preference",
    cross_ref_candidate: bool = False,
    source: str = "claude_code",
    mid: str | None = None,
) -> MemoryUnit:
    kwargs: dict = {
        "content": content,
        "scope": scope or Scope.global_(),
        "category": category,
        "cross_ref_candidate": cross_ref_candidate,
        "entities": entities if entities is not None else ["rust"],
        "provenance": Provenance(source=source, session_id="sess-1"),
    }
    if mid is not None:
        kwargs["id"] = mid
    return MemoryUnit(**kwargs)


def _raw_chunk(*, content_hash: str, scope: Scope | None = None) -> RawChunk:
    sc = scope or Scope.global_()
    return RawChunk(
        content_hash=content_hash,
        content="normalized chunk text",
        source="claude_code",
        session_id="sess-1",
        scope=sc,
        project=sc.project_id or "",
        memory_ids=[],
    )


async def _seed(store: StorageBackend) -> None:
    """Seed the same small store the FalkorDB contract test uses (one per axis).

    Two entities + one structural edge, four memories spanning category / scope /
    tier / source / active-vs-superseded / cross-ref, plus one raw chunk. The
    distinct contents mean every memory is an independent ADD on both fakes (their
    write decision only reinforces on EXACT same-content within scope+entities).
    """

    await store.upsert_entity(Entity(id="ent-rust", canonical_name="rust", type="language"))
    await store.upsert_entity(Entity(id="ent-tokio", canonical_name="tokio", type="library"))
    await store.upsert_edge(
        Edge(id="e1", from_entity="ent-rust", to_entity="ent-tokio", relation="uses", weight=0.7)
    )
    await store.upsert_memory(
        _memory(content="prefers thiserror", entities=["rust"], mid="m1")
    )
    await store.upsert_memory(
        _memory(content="prefers anyhow", entities=["rust"], mid="m2")
    )
    await store.close_validity_window("m2")  # superseded
    await store.upsert_memory(
        _memory(
            content="pins tokio 1.38",
            scope=Scope.project("rust-cli"),
            entities=["tokio"],
            category="decision",
            source="openai",
            mid="m3",
        )
    )
    await store.upsert_memory(
        _memory(
            content="idea: async cli streaming logs",
            entities=["tokio", "rust"],
            category="idea",
            cross_ref_candidate=True,
            mid="m4",
        )
    )
    await store.archive("m4")
    await store.persist_raw_chunk(_raw_chunk(content_hash="rc1"))


# ---------------------------------------------------------------------------
# Protocol conformance — both fakes implement the full StorageBackend now.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Fake", FAKES)
def test_fake_satisfies_storage_protocol(Fake) -> None:
    assert isinstance(Fake(), StorageBackend)


# ---------------------------------------------------------------------------
# store_stats: aggregates MATCH an independent row-count baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Fake", FAKES)
async def test_store_stats_counts_match_row_count_baseline(Fake) -> None:
    store = Fake()
    await _seed(store)
    stats = await store.store_stats()

    # Independent brute-force baseline straight off the stored units, so the
    # aggregate can never silently disagree with a row count.
    mems = list(store.memories.values())
    assert stats.total_memories == len(mems) == 4
    assert stats.by_category == dict(Counter(m.category for m in mems))
    assert stats.by_category == {"preference": 2, "decision": 1, "idea": 1}
    assert stats.by_scope_decision == dict(
        Counter(m.scope_decision.value for m in mems)
    )
    assert stats.by_scope_decision == {"global": 3, "project": 1}
    assert stats.by_tier == dict(Counter(m.tier.value for m in mems))
    assert stats.by_tier == {"hot": 3, "archive": 1}
    assert stats.by_source == dict(Counter(m.provenance.source for m in mems))
    assert stats.by_source == {"claude_code": 3, "openai": 1}

    active = sum(1 for m in mems if m.is_active)
    assert stats.active_count == active == 3
    assert stats.superseded_count == len(mems) - active == 1
    assert stats.entity_count == len(store.entities) == 2
    assert stats.raw_chunk_count == len(store.raw_chunks) == 1


@pytest.mark.parametrize("Fake", FAKES)
async def test_store_stats_empty_store_is_all_zero(Fake) -> None:
    stats = await Fake().store_stats()
    assert stats.total_memories == 0
    assert stats.active_count == 0
    assert stats.superseded_count == 0
    assert stats.entity_count == 0
    assert stats.raw_chunk_count == 0
    assert stats.by_category == {}
    assert stats.by_scope_decision == {}
    assert stats.by_tier == {}
    assert stats.by_source == {}


# ---------------------------------------------------------------------------
# query_memories: right page + total, and NO embedding on any item
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Fake", FAKES)
async def test_query_memories_items_carry_no_embedding(Fake) -> None:
    store = Fake()
    await _seed(store)
    page = await store.query_memories()
    assert page.total == 4
    assert len(page.items) == 4
    for view in page.items:
        # The display projection is a MemoryView, which structurally has NO
        # embedding/vector field at all — that is the whole point of the contract.
        assert isinstance(view, MemoryView)
        assert not hasattr(view, "embedding")
        assert "embedding" not in MemoryView.__slots__
        # The flattened provenance scalars the wire models need are present.
        assert isinstance(view.source, str)


@pytest.mark.parametrize("Fake", FAKES)
async def test_query_memories_filters(Fake) -> None:
    store = Fake()
    await _seed(store)

    # category (normalized lowercased/trimmed)
    pref = await store.query_memories(category="Preference")
    assert pref.total == 2 and {v.id for v in pref.items} == {"m1", "m2"}
    # active None=both / True=open / False=closed
    assert {v.id for v in (await store.query_memories(active=True)).items} == {
        "m1",
        "m3",
        "m4",
    }
    assert {v.id for v in (await store.query_memories(active=False)).items} == {"m2"}
    # scope is the EXACT stored scope (no ancestor composition)
    scoped = await store.query_memories(scope=Scope.project("rust-cli"))
    assert {v.id for v in scoped.items} == {"m3"}
    # tier
    assert {v.id for v in (await store.query_memories(tier=Tier.ARCHIVE)).items} == {"m4"}
    # source (exact provenance source)
    assert {v.id for v in (await store.query_memories(source="openai")).items} == {"m3"}
    # entity (case-insensitive membership)
    assert {v.id for v in (await store.query_memories(entity="Tokio")).items} == {
        "m3",
        "m4",
    }
    # free-text substring of content (case-insensitive)
    assert {v.id for v in (await store.query_memories(q="TOKIO")).items} == {"m3"}


@pytest.mark.parametrize("Fake", FAKES)
async def test_query_memories_paging_total_is_full_filtered_count(Fake) -> None:
    store = Fake()
    await _seed(store)
    first = await store.query_memories(limit=2, offset=0)
    second = await store.query_memories(limit=2, offset=2)
    third = await store.query_memories(limit=2, offset=4)
    # total is the WHOLE filtered set before paging, not len(items).
    assert first.total == second.total == third.total == 4
    assert len(first.items) == 2
    assert len(second.items) == 2
    assert len(third.items) == 0
    ids = {v.id for v in first.items} | {v.id for v in second.items}
    assert ids == {"m1", "m2", "m3", "m4"}
    # the two pages are disjoint (no overlap / no gap)
    assert {v.id for v in first.items}.isdisjoint({v.id for v in second.items})


@pytest.mark.parametrize("Fake", FAKES)
async def test_get_memory_display_is_embedding_free_view(Fake) -> None:
    store = Fake()
    await _seed(store)
    view = await store.get_memory_display("m1")
    assert view is not None
    assert isinstance(view, MemoryView)
    assert view.id == "m1"
    assert view.content == "prefers thiserror"
    assert view.is_active is True
    assert view.scope_decision.value == "global"
    assert view.source == "claude_code"
    assert not hasattr(view, "embedding")
    # unknown id -> None
    assert await store.get_memory_display("nope") is None


# ---------------------------------------------------------------------------
# graph_snapshot: respects the node cap (+ no false-positive truncation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Fake", FAKES)
async def test_graph_snapshot_full_subgraph(Fake) -> None:
    store = Fake()
    await _seed(store)
    snap = await store.graph_snapshot()
    ent_ids = {n.id for n in snap.nodes if n.kind == "entity"}
    assert ent_ids == {"ent-rust", "ent-tokio"}
    # the archived cross-ref candidate becomes an idea_seed node + mentions edges
    idea_ids = {n.id for n in snap.nodes if n.kind == "idea_seed"}
    assert idea_ids == {"m4"}
    relates = [e for e in snap.edges if e.kind == "relates"]
    assert any(e.source == "ent-rust" and e.target == "ent-tokio" for e in relates)
    mentions = [e for e in snap.edges if e.kind == "mentions"]
    assert mentions and all(e.source == "m4" for e in mentions)
    assert snap.truncated is False
    # per-entity in-scope memory_count is computed in the snapshot
    by_id = {n.id: n for n in snap.nodes}
    assert by_id["ent-tokio"].memory_count >= 1


@pytest.mark.parametrize("Fake", FAKES)
async def test_graph_snapshot_node_cap_truncates(Fake) -> None:
    store = Fake()
    await _seed(store)
    snap = await store.graph_snapshot(node_limit=1)
    assert len([n for n in snap.nodes if n.kind == "entity"]) == 1
    assert snap.truncated is True


@pytest.mark.parametrize("Fake", FAKES)
async def test_graph_snapshot_no_false_positive_at_exact_cap(Fake) -> None:
    store = Fake()
    await _seed(store)  # exactly two entities (rust, tokio)
    snap = await store.graph_snapshot(node_limit=2)
    assert len([n for n in snap.nodes if n.kind == "entity"]) == 2
    assert snap.truncated is False


@pytest.mark.parametrize("Fake", FAKES)
async def test_graph_snapshot_scope_filter_bounds_idea_seeds(Fake) -> None:
    store = Fake()
    await _seed(store)
    # m4 (the idea seed) is global-scoped; filtering to the project scope drops it.
    snap = await store.graph_snapshot(scope=Scope.project("rust-cli"))
    assert {n.id for n in snap.nodes if n.kind == "idea_seed"} == set()
    # but the global idea seed surfaces when scoped to global
    glob = await store.graph_snapshot(scope=Scope.global_())
    assert {n.id for n in glob.nodes if n.kind == "idea_seed"} == {"m4"}
