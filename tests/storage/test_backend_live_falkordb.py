"""Live integration tests for the index-backed vector KNN (FR-RET-2 / FR-STO-2).

Unlike ``test_backend_contract.py`` (which runs against the in-process fake), this
module stands up a **real** :class:`GraphitiStorageBackend` over a **real**
FalkorDB and asserts that :meth:`GraphitiStorageBackend.scoped_query` returns
correct nearest-neighbour ordering *from the FalkorDB vector index* — the headline
PRD claim (Goal-5: the effective search space stays flat as the store grows) that
the fake cannot prove because it cannot execute ``db.idx.vector.queryNodes``.

It is the proof-of-life for the #1 fix: the candidate ranking comes from the
vector index, not an in-process full scan.

Skipped automatically when no FalkorDB is reachable on the configured URL
(default ``redis://localhost:6379``), so the offline suite stays green with no
infra. To run it, bring up FalkorDB, e.g.::

    docker run -d -p 6379:6379 falkordb/falkordb

then ``pytest tests/storage/test_backend_live_falkordb.py``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest

from mnemozine.config import FalkorDBSettings, MaintenanceSettings, Settings
from mnemozine.maintenance.entity_dedup import EntityDedupJob
from mnemozine.schema.events import Source
from mnemozine.schema.models import (
    Edge,
    Entity,
    MemoryUnit,
    Provenance,
    Scope,
)
from mnemozine.storage.backend import GraphitiStorageBackend
from mnemozine.storage.graphiti_client import (
    CO_MENTION_TYPE,
    ENTITY_LABEL,
    ENTITY_NAME_KEY_INDEX,
    MEMORY_LABEL,
    MEMORY_VECTOR_INDEX,
    MENTIONS_TYPE,
    RELATES_TYPE,
    GraphitiClient,
)

pytestmark = pytest.mark.live_falkordb

# bge-m3 is 1024-d in prod; the live test uses a tiny dimensionality with hand-
# picked unit vectors so nearest-neighbour ordering is exactly predictable.
_DIM = 4

# Content -> known unit vector. Cosine sims to "north" ([1,0,0,0]) are:
#   north 1.0  >  northeast 0.8  >  east 0.0 == up 0.0
_VECTORS: dict[str, list[float]] = {
    "north": [1.0, 0.0, 0.0, 0.0],
    "northeast": [0.8, 0.6, 0.0, 0.0],
    "east": [0.0, 1.0, 0.0, 0.0],
    "up": [0.0, 0.0, 1.0, 0.0],
}


class _FixedEmbeddingProvider:
    """Content-keyed embeddings so the live KNN ordering is deterministic."""

    @property
    def dimensions(self) -> int:
        return _DIM

    async def embed(self, text: str) -> list[float]:
        return list(_VECTORS.get(text, [0.0, 0.0, 0.0, 1.0]))

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def _memory(
    content: str,
    *,
    scope: Scope | None = None,
    entities: list[str] | None = None,
    category: str = "preference",
) -> MemoryUnit:
    # Category split: a global-scope memory tagged 'preference' is the new shape
    # of the old PREFERENCE type; scope (not a type field) drives the no-leak rule.
    return MemoryUnit(
        content=content,
        scope=scope or Scope.global_(),
        category=category,
        entities=entities if entities is not None else ["dir"],
        confidence=0.9,
        provenance=Provenance(source=Source.CLAUDE_CODE.value, session_id="sess-live"),
    )


async def _falkordb_reachable(url: str) -> bool:
    try:
        from falkordb.asyncio import FalkorDB
    except ImportError:  # pragma: no cover - dep always present in this repo
        return False
    try:
        db = FalkorDB.from_url(url)
        g = db.select_graph("__mnemozine_ping__")
        await g.query("RETURN 1")
        await db.connection.aclose()
        return True
    except Exception:  # noqa: BLE001 - any connection failure => skip
        return False


@pytest.fixture
async def live_backend() -> GraphitiStorageBackend:
    """A real backend on a throwaway graph; skips if FalkorDB is unreachable.

    A fresh, uniquely-named graph per test means no leftover vector index at the
    wrong dimensionality (the index outlives ``DETACH DELETE`` of the nodes) and
    no cross-test contamination. The dedup threshold is raised to ~1.0 so the
    hand-picked near vectors are stored as distinct ADDs rather than reinforced.
    """

    url = "redis://localhost:6379"
    if not await _falkordb_reachable(url):
        pytest.skip(f"no FalkorDB reachable at {url}")

    graph_name = f"mnemozine_live_{uuid.uuid4().hex[:12]}"
    client = GraphitiClient(
        FalkorDBSettings(url=url, graph_name=graph_name),
        embedding_dimensions=_DIM,
    )
    await client.connect()  # builds the vector index (FR-STO-2)
    backend = GraphitiStorageBackend(
        client=client,
        embeddings=_FixedEmbeddingProvider(),
        maintenance=MaintenanceSettings(dedup_equivalence_threshold=0.999999),
    )
    try:
        yield backend
    finally:
        # Drop the throwaway graph entirely (nodes + index), then close.
        try:
            from falkordb.asyncio import FalkorDB

            db = FalkorDB.from_url(url)
            await db.select_graph(graph_name).delete()
            await db.connection.aclose()
        except Exception:  # noqa: BLE001 - cleanup is best-effort
            pass
        await client.close()


async def test_scoped_query_uses_vector_index_and_orders_by_similarity(
    live_backend: GraphitiStorageBackend,
) -> None:
    """The #1 proof: ranking comes from the FalkorDB vector index, in cosine order."""

    for content in ("north", "northeast", "east", "up"):
        result = await live_backend.upsert_memory(_memory(content))
        # Distinct vectors => every write is an ADD (no accidental reinforce).
        assert result.memory.content == content

    hits = await live_backend.scoped_query("north", [Scope.global_()], top_k=4)
    ordering = [h.memory.content for h in hits]
    assert ordering == ["north", "northeast", "east", "up"], ordering

    # Scores are real cosine similarities recovered from the index distance.
    by_content = {h.memory.content: h.score for h in hits}
    assert by_content["north"] == pytest.approx(1.0, abs=1e-4)
    assert by_content["northeast"] == pytest.approx(0.8, abs=1e-3)
    assert by_content["east"] == pytest.approx(0.0, abs=1e-4)
    # Monotonic non-increasing.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


async def test_scoped_query_top_k_truncates_to_true_nearest(
    live_backend: GraphitiStorageBackend,
) -> None:
    for content in ("north", "northeast", "east", "up"):
        await live_backend.upsert_memory(_memory(content))

    hits = await live_backend.scoped_query("north", [Scope.global_()], top_k=2)
    assert [h.memory.content for h in hits] == ["north", "northeast"]


async def test_scoped_query_scope_isolation_no_cross_project_leak(
    live_backend: GraphitiStorageBackend,
) -> None:
    await live_backend.upsert_memory(_memory("north", scope=Scope.project("p1")))
    # Querying a different project must see nothing (FR-STO-3 no-leak), even though
    # the vector is a perfect match — the index post-filter excludes it.
    leak = await live_backend.scoped_query("north", [Scope.project("p2")], top_k=5)
    assert leak == []
    # ...but the owning project sees it.
    own = await live_backend.scoped_query("north", [Scope.project("p1")], top_k=5)
    assert [h.memory.content for h in own] == ["north"]


async def test_scoped_query_composes_ancestors_against_real_index(
    live_backend: GraphitiStorageBackend,
) -> None:
    """Ancestor-composition runs through the real FalkorDB index post-filter.

    A query at project:proj/auth must compose its ancestor chain (global +
    project:proj + project:proj/auth) in the index ``WHERE m.scope IN $scopes``,
    while a sibling sub-scope (project:proj/db) must NOT leak — proving the
    no-leak rule holds against the live index, not just the fake.
    """

    await live_backend.upsert_memory(_memory("north", scope=Scope.global_()))
    await live_backend.upsert_memory(
        _memory("northeast", scope=Scope.project("proj"))
    )
    await live_backend.upsert_memory(
        _memory("east", scope=Scope.project("proj", "auth"))
    )
    await live_backend.upsert_memory(
        _memory("up", scope=Scope.project("proj", "db"))
    )

    hits = await live_backend.scoped_query(
        "north", [Scope.project("proj", "auth")], top_k=10
    )
    contents = {h.memory.content for h in hits}
    # Composes global + project + auth-self; the db sibling never leaks.
    assert contents == {"north", "northeast", "east"}
    assert "up" not in contents


async def test_scoped_query_entity_filter_via_index(
    live_backend: GraphitiStorageBackend,
) -> None:
    await live_backend.upsert_memory(_memory("north", entities=["dir"]))
    await live_backend.upsert_memory(_memory("northeast", entities=["other"]))

    hits = await live_backend.scoped_query(
        "north", [Scope.global_()], entities=["dir"], top_k=5
    )
    contents = [h.memory.content for h in hits]
    assert "north" in contents
    assert "northeast" not in contents  # filtered: no shared entity


async def test_scoped_query_small_scope_recall_via_starvation_fallback(
    live_backend: GraphitiStorageBackend,
) -> None:
    """FR-RET-2 starvation fallback against the REAL vector index.

    A small ``project:p1`` scope buried under a larger, NEARER out-of-scope corpus
    is starved by FalkorDB's post-KNN scope filter (the nearest neighbours are all
    out-of-scope and post-filtered away). With the over-fetch K bounded tight (so
    the out-of-scope corpus crowds the KNN cut), pure KNN would return []; the
    scope-pre-filtered fallback (gated by the cheap in-scope COUNT) must still
    recall the in-scope memory at the default top_k.
    """

    from mnemozine.config import RetrievalSettings

    # Bound the over-fetch so $k == 2: the two nearest (out-of-scope) neighbours
    # fill the KNN cut and the in-scope match is post-filtered to nothing.
    live_backend._retrieval = RetrievalSettings(  # type: ignore[attr-defined]
        knn_overfetch_factor=1, knn_overfetch_cap=2, scope_scan_max=4000
    )
    # Distinct near vectors for the out-of-scope noise (so they aren't deduped),
    # both strictly nearer to the query "north" ([1,0,0,0]) than the in-scope vec.
    _VECTORS["noise_near_a"] = [1.0, 0.0, 0.0, 0.0]
    _VECTORS["noise_near_b"] = [0.9, 0.4358899, 0.0, 0.0]
    _VECTORS["scope_far"] = [0.0, 1.0, 0.0, 0.0]
    try:
        await live_backend.upsert_memory(
            _memory("noise_near_a", scope=Scope.project("other"))
        )
        await live_backend.upsert_memory(
            _memory("noise_near_b", scope=Scope.project("other"))
        )
        await live_backend.upsert_memory(
            _memory("scope_far", scope=Scope.project("p1"))
        )

        hits = await live_backend.scoped_query(
            "north", [Scope.project("p1")], top_k=5
        )
        # Pure KNN starves to []; the gated fallback recalls the in-scope memory.
        assert [h.memory.content for h in hits] == ["scope_far"]
    finally:
        for key in ("noise_near_a", "noise_near_b", "scope_far"):
            _VECTORS.pop(key, None)


async def test_scoped_query_emits_index_backed_knn_cypher(
    live_backend: GraphitiStorageBackend,
) -> None:
    """Guard that the query path is the index KNN, not the full-scan fallback.

    Inspect the Cypher actually sent to the driver: it must call the FalkorDB
    vector-index procedure over the memory embedding index.
    """

    captured: list[str] = []
    real_execute = live_backend._client.execute_query

    async def _spy(cypher: str, **params: object) -> object:
        captured.append(cypher)
        return await real_execute(cypher, **params)

    live_backend._client.execute_query = _spy  # type: ignore[method-assign]
    await live_backend.upsert_memory(_memory("north"))
    await live_backend.scoped_query("north", [Scope.global_()], top_k=3)

    knn = [c for c in captured if "db.idx.vector.queryNodes" in c]
    assert knn, f"scoped_query did not use the vector index; cypher seen: {captured}"
    assert "vecf32(" in knn[0]
    # The index name the client created is the one being queried (label-based).
    assert MEMORY_VECTOR_INDEX  # name is exported for ops reference


async def test_upsert_edge_create_and_reassert_against_real_falkordb(
    live_backend: GraphitiStorageBackend,
) -> None:
    """Edge CREATE + re-assert run against REAL FalkorDB (relationship write path).

    Regression guard: the in-process fake does not parse Cypher, so a malformed
    relationship-CREATE statement (an unbalanced property-map brace) passed every
    offline contract test yet was rejected by real FalkorDB the moment a real
    extraction emitted a relationship (the FR-EXT/FR-ING 4-way write calls
    ``upsert_edge``). This exercises the create path live, asserts the edge is
    readable back, and that a re-assert bumps the weight rather than erroring or
    duplicating.
    """

    a = Entity(canonical_name="black", type="tool")
    b = Entity(canonical_name="python", type="language")
    await live_backend.upsert_entity(a)
    await live_backend.upsert_entity(b)

    edge = Edge(from_entity=a.id, to_entity=b.id, relation="formats", weight=1.0)
    created = await live_backend.upsert_edge(edge)
    assert created.from_entity == a.id and created.to_entity == b.id

    # The edge is readable back from the real graph (proves the CREATE landed).
    incident = await live_backend.edges_for_entity(a.id)
    assert any(e.relation == "formats" for e in incident), incident

    # Re-asserting the same active relation bumps the weight (no duplicate, no error).
    reassert = await live_backend.upsert_edge(
        Edge(from_entity=a.id, to_entity=b.id, relation="formats", weight=2.5)
    )
    assert reassert.weight == pytest.approx(2.5)
    incident2 = await live_backend.edges_for_entity(a.id)
    assert sum(1 for e in incident2 if e.relation == "formats") == 1, incident2


async def test_legacy_edge_without_from_to_props_against_real_falkordb(
    live_backend: GraphitiStorageBackend,
) -> None:
    """A LEGACY edge (no from/to PROPS) resolves from REAL FalkorDB topology.

    Reproduces the live /api/graph 500 against the real engine: the 2026-06-14
    backfill + merge-rewired edges store ONLY ``{id, relation, weight, valid_from}``
    — no ``from_entity``/``to_entity`` props. We create exactly that shape with raw
    Cypher (the backend's own ``upsert_edge`` always writes the props, so the legacy
    shape has to be created directly), then assert all four read paths resolve the
    endpoints from the relationship topology (``a.id``/``b.id`` and
    ``startNode(r).id``/``endNode(r).id``) WITHOUT mutating the stored edge. Before
    the read-side fix these raised ``KeyError: 'from_entity'``.
    """

    a = Entity(id="ent-legacy-a", canonical_name="legacy-alpha")
    b = Entity(id="ent-legacy-b", canonical_name="legacy-beta")
    await live_backend.upsert_entity(a)
    await live_backend.upsert_entity(b)

    # Create the legacy edge directly: only id/relation/weight/valid_from props.
    await live_backend._client.execute_query(
        f"MATCH (a:{ENTITY_LABEL} {{id: $from}}) "
        f"MATCH (b:{ENTITY_LABEL} {{id: $to}}) "
        f"CREATE (a)-[r:{RELATES_TYPE} {{id: $id, relation: $relation, "
        "weight: $weight, valid_from: $valid_from}]->(b) RETURN r",
        **{
            "from": a.id,
            "to": b.id,
            "id": "legacy-live-1",
            "relation": "relates",
            "weight": 0.4,
            "valid_from": "2026-06-14T00:00:00+00:00",
        },
    )

    # edges_for_entity resolves the endpoints from topology (no KeyError).
    incident = await live_backend.edges_for_entity(a.id)
    legacy = [e for e in incident if e.id == "legacy-live-1"]
    assert legacy, incident
    assert legacy[0].from_entity == a.id and legacy[0].to_entity == b.id

    # neighbors traversal returns the neighbor + a correctly-endpointed edge.
    neighbors = await live_backend.neighbors(a.id)
    nbr = [n for n in neighbors if n.entity.id == b.id]
    assert nbr, neighbors
    assert {nbr[0].edge.from_entity, nbr[0].edge.to_entity} == {a.id, b.id}

    # graph_snapshot's single aggregate edge query resolves it from a.id/b.id.
    snap = await live_backend.graph_snapshot()
    relates = [e for e in snap.edges if e.kind == "relates"]
    assert any(
        {e.source, e.target} == {a.id, b.id} for e in relates
    ), relates

    # prune_edge closes the window and returns it endpointed (no KeyError).
    pruned = await live_backend.prune_edge("legacy-live-1")
    assert pruned.valid_to is not None
    assert {pruned.from_entity, pruned.to_entity} == {a.id, b.id}


async def test_activity_log_append_and_query_against_real_falkordb(
    live_backend: GraphitiStorageBackend,
) -> None:
    """FalkorDBActivityLog append+query run against REAL FalkorDB (Q3 write path).

    Regression guard: the offline activity tests use the InMemoryActivityLog, so
    the FalkorDB node-create Cypher was never exercised. ``CREATE (a:Label $props)``
    with a *parameterized map* is rejected by real FalkorDB ("Encountered unhandled
    type in inlined properties"), which would have broken every WebUI activity
    write the moment the log was enabled. This exercises the real CREATE + SET path
    live, asserts events with NULL-valued fields (a maintenance event has no
    source/session/project) append cleanly, and round-trip through query() with
    their JSON-encoded ref_memory_ids/detail intact.
    """

    from mnemozine.activity.log import FalkorDBActivityLog
    from mnemozine.activity.models import (
        ActivityQuery,
        ingest_event,
        maintenance_event,
    )

    client = live_backend._client
    log = FalkorDBActivityLog(client)

    full = ingest_event(
        source="claude_code",
        session_id="sess-live",
        project="proj-live",
        summary="ingested live chunk",
        ref_memory_ids=["m-live-1", "m-live-2"],
        detail={"chunks": 2},
    )
    # A maintenance event carries NULL session_id/project — the exact shape that
    # the parameterized-map CREATE rejected.
    sparse = maintenance_event(job_name="consolidate", summary="live pass")

    await log.append(full)
    await log.append(sparse)

    events = await log.query(ActivityQuery(limit=10))
    by_summary = {e.summary: e for e in events}
    assert "ingested live chunk" in by_summary
    assert "live pass" in by_summary
    # JSON-encoded list/map fields survive the FalkorDB round-trip.
    assert by_summary["ingested live chunk"].ref_memory_ids == ["m-live-1", "m-live-2"]
    assert by_summary["ingested live chunk"].detail == {"chunks": 2}
    assert by_summary["live pass"].session_id is None
    assert by_summary["live pass"].detail == {"job_name": "consolidate"}

    # ref_memory_id filtering (Python-side over the JSON array) works end-to-end.
    filtered = await log.query(ActivityQuery(ref_memory_id="m-live-1", limit=10))
    assert [e.summary for e in filtered] == ["ingested live chunk"]


# ---------------------------------------------------------------------------
# Display reads (the WebUI READ surface) against REAL FalkorDB.
#
# The four embedding-free display methods (store_stats / query_memories /
# get_memory_display / graph_snapshot) back the slow WebUI endpoints. The offline
# contract tests only exercise them against the in-process fake, which is not a
# real Cypher engine — so the aggregation Cypher, the ``CONTAINS '"source":"..."'``
# provenance-blob filter, the embedding-free view projection map, and the bounded
# graph_snapshot edge/idea-seed scan are unproven against a real engine. These
# tests close that gap (the user-flagged "no live test for the new methods").
# ---------------------------------------------------------------------------


async def _seed_display_live(backend: GraphitiStorageBackend) -> None:
    """Seed a small live store covering every display-read axis (mirror of the fake seed)."""

    await backend.upsert_entity(
        Entity(id="ent-rust", canonical_name="rust", type="language")
    )
    await backend.upsert_entity(
        Entity(id="ent-tokio", canonical_name="tokio", type="library")
    )
    await backend.upsert_edge(
        Edge(
            id="e-live",
            from_entity="ent-rust",
            to_entity="ent-tokio",
            relation="uses",
            weight=0.7,
        )
    )
    # active global preference (claude_code source).
    await backend.upsert_memory(
        _memory("north", scope=Scope.global_(), entities=["rust"], category="preference")
    )
    # project-scoped decision from a DIFFERENT source (openai) — exercises the
    # provenance-blob source filter against the real engine.
    proj = _memory(
        "east",
        scope=Scope.project("rust-cli"),
        entities=["tokio"],
        category="decision",
    )
    proj.provenance = Provenance(source="openai", session_id="sess-2")
    await backend.upsert_memory(proj)
    # archived cross-ref idea seed (tier=archive) for the graph idea-seed node.
    idea = MemoryUnit(
        content="up",
        scope=Scope.global_(),
        category="idea",
        cross_ref_candidate=True,
        entities=["tokio", "rust"],
        confidence=0.7,
        provenance=Provenance(source=Source.CLAUDE_CODE.value, session_id="sess-3"),
    )
    await backend.upsert_memory(idea)


async def test_store_stats_aggregates_against_real_falkordb(
    live_backend: GraphitiStorageBackend,
) -> None:
    """store_stats COUNT/grouping aggregates run against the real engine."""

    await _seed_display_live(live_backend)
    stats = await live_backend.store_stats()
    assert stats.total_memories == 3
    assert stats.by_category == {"preference": 1, "decision": 1, "idea": 1}
    assert stats.by_scope_decision == {"global": 2, "project": 1}
    assert stats.active_count == 3
    assert stats.superseded_count == 0
    # by_source decodes the provenance blob per distinct value — proves the real
    # grouped read + JSON decode, not just the fake's stored-dict shortcut.
    assert stats.by_source == {"claude_code": 2, "openai": 1}
    assert stats.entity_count == 2


async def test_query_memories_filters_and_source_blob_against_real_falkordb(
    live_backend: GraphitiStorageBackend,
) -> None:
    """query_memories pushes filters/paging into real Cypher; source matches the blob.

    The headline regression for the flagged fragility: ``source='openai'`` is
    matched via ``m.provenance CONTAINS '"source":"openai"'`` over the real stored
    ``model_dump_json()`` blob — this proves the compact-JSON dependency holds on a
    live engine, not just in the substring-matching fake.
    """

    await _seed_display_live(live_backend)

    page = await live_backend.query_memories()
    assert page.total == 3 and len(page.items) == 3
    # The view is embedding-free but carries the decoded scalar source.
    assert {v.source for v in page.items} == {"claude_code", "openai"}

    oai = await live_backend.query_memories(source="openai")
    assert oai.total == 1
    assert [v.content for v in oai.items] == ["east"]
    assert oai.items[0].source == "openai"

    # category + scope + entity filters all land in Cypher.
    decision = await live_backend.query_memories(category="decision")
    assert [v.content for v in decision.items] == ["east"]
    scoped = await live_backend.query_memories(scope=Scope.project("rust-cli"))
    assert [v.content for v in scoped.items] == ["east"]
    tok = await live_backend.query_memories(entity="Tokio")
    assert {v.content for v in tok.items} == {"east", "up"}


async def test_query_memories_never_selects_embedding_live(
    live_backend: GraphitiStorageBackend,
) -> None:
    """The list read's Cypher never names the embedding (proven on the real path)."""

    await _seed_display_live(live_backend)
    captured: list[str] = []
    real_execute = live_backend._client.execute_query

    async def _spy(cypher: str, **params: object) -> object:
        captured.append(cypher)
        return await real_execute(cypher, **params)

    live_backend._client.execute_query = _spy  # type: ignore[method-assign]
    await live_backend.query_memories(category="preference", limit=10)
    assert captured, "query_memories ran no Cypher"
    assert all("embedding" not in c for c in captured), captured


async def test_get_memory_display_embedding_free_against_real_falkordb(
    live_backend: GraphitiStorageBackend,
) -> None:
    """get_memory_display keys a unit by id and returns the embedding-free view."""

    await _seed_display_live(live_backend)
    page = await live_backend.query_memories(category="preference")
    mid = page.items[0].id
    view = await live_backend.get_memory_display(mid)
    assert view is not None
    assert view.id == mid
    assert view.content == "north"
    assert view.is_active is True
    assert view.scope_decision.value == "global"
    assert view.source == "claude_code"
    assert await live_backend.get_memory_display("does-not-exist") is None


async def test_graph_snapshot_bounded_against_real_falkordb(
    live_backend: GraphitiStorageBackend,
) -> None:
    """graph_snapshot returns the bounded entity/idea-seed subgraph from real Cypher.

    Proves the single aggregate edge query, the idea-seed/memory_count scan (now
    LIMIT-bounded), and the truncated sentinel against the real engine.
    """

    await _seed_display_live(live_backend)
    snap = await live_backend.graph_snapshot()
    ent_ids = {n.id for n in snap.nodes if n.kind == "entity"}
    assert ent_ids == {"ent-rust", "ent-tokio"}
    idea_ids = {n.id for n in snap.nodes if n.kind == "idea_seed"}
    assert idea_ids  # the archived cross-ref candidate is an idea-seed node
    relates = [e for e in snap.edges if e.kind == "relates"]
    assert any(
        {e.source, e.target} == {"ent-rust", "ent-tokio"} for e in relates
    ), relates
    mentions = [e for e in snap.edges if e.kind == "mentions"]
    assert mentions  # idea-seed -> entity mention edges
    # Exactly node_limit entities (2) must NOT be a false-positive truncation.
    assert snap.truncated is False
    full = await live_backend.graph_snapshot(node_limit=2)
    assert full.truncated is False
    # One fewer than the entity count DOES truncate.
    cut = await live_backend.graph_snapshot(node_limit=1)
    assert len([n for n in cut.nodes if n.kind == "entity"]) == 1
    assert cut.truncated is True

    # --- degree-ordered default selection from real Cypher -------------------
    # Add an ISOLATED (edge-less, degree-0) entity; the two connected entities
    # (rust/tokio, each degree 1 via the RELATES edge) must out-rank it in the
    # degree-ranked default selection, so node_limit=2 keeps the connected pair,
    # never the isolate. Proves the OPTIONAL MATCH/count(r)/ORDER BY deg Cypher.
    await live_backend.upsert_entity(
        Entity(id="ent-isolated", canonical_name="isolated", type="language")
    )
    ranked = await live_backend.graph_snapshot(node_limit=2)
    ranked_ids = {n.id for n in ranked.nodes if n.kind == "entity"}
    assert ranked_ids == {"ent-rust", "ent-tokio"}
    assert "ent-isolated" not in ranked_ids
    # 3 entities now exist but only 2 kept -> truncated reported.
    assert ranked.truncated is True
    # The kept set is connected: the RELATES edge between them is surfaced.
    ranked_relates = [e for e in ranked.edges if e.kind == "relates"]
    assert any(
        {e.source, e.target} == {"ent-rust", "ent-tokio"} for e in ranked_relates
    ), ranked_relates


async def test_resolve_or_create_entity_index_backed_and_case_insensitive(
    live_backend: GraphitiStorageBackend,
) -> None:
    """resolve_or_create_entity reuses the node for a normalized name, case-insensitive.

    Against real FalkorDB: the name_key range index backs the lookup, resolving
    ``Rust`` folds onto the existing ``rust`` node (no duplicate), and the spelling
    + a new alias fold into the survivor's aliases. Idempotent.
    """

    assert ENTITY_NAME_KEY_INDEX  # name is exported for ops reference

    first = await live_backend.resolve_or_create_entity(Entity(canonical_name="rust"))
    second = await live_backend.resolve_or_create_entity(
        Entity(canonical_name="Rust", aliases=["rust-lang"])
    )
    assert second.id == first.id
    assert "Rust" in second.aliases
    assert "rust-lang" in second.aliases

    # Exactly one entity node for the normalized name (index-backed dedup-on-write).
    rows = await live_backend._query(
        f"MATCH (e:{ENTITY_LABEL}) WHERE e.name_key = $k RETURN count(e) AS n",
        k="rust",
    )
    assert int(rows[0][0]) == 1

    # Idempotent: a third resolve of the same name adds no node.
    third = await live_backend.resolve_or_create_entity(Entity(canonical_name="RUST"))
    assert third.id == first.id
    rows2 = await live_backend._query(
        f"MATCH (e:{ENTITY_LABEL}) WHERE e.name_key = $k RETURN count(e) AS n",
        k="rust",
    )
    assert int(rows2[0][0]) == 1


async def test_ensure_entity_name_index_idempotent_on_reconnect(
    live_backend: GraphitiStorageBackend,
) -> None:
    """ensure_entity_name_index swallows the already-indexed error on re-create.

    The fixture already built the index on connect; calling it again (as a second
    connect would) must be a no-op, never raise. backfill_entity_name_keys is also
    re-runnable: a second pass stamps zero nodes.
    """

    # Re-creating an existing index must not raise (already-indexed swallowed).
    await live_backend._client.ensure_entity_name_index()
    await live_backend._client.ensure_entity_name_index()

    # Seed a node WITHOUT name_key (simulating a pre-v2 entity), then backfill.
    await live_backend._query(
        f"CREATE (e:{ENTITY_LABEL} {{id: $id, canonical_name: $cn, aliases: [], "
        "type: null}) RETURN e",
        id="legacy-1",
        cn="Legacy",
    )
    stamped = await live_backend.backfill_entity_name_keys()
    assert stamped >= 1
    rows = await live_backend._query(
        f"MATCH (e:{ENTITY_LABEL} {{id: $id}}) RETURN e.name_key AS k",
        id="legacy-1",
    )
    assert rows[0][0] == "legacy"
    # Re-running the backfill finds nothing unset and stamps zero (idempotent).
    assert await live_backend.backfill_entity_name_keys() == 0


async def test_add_memory_mentions_asserts_edges_against_real_falkordb(
    live_backend: GraphitiStorageBackend,
) -> None:
    """add_memory_mentions id-keyed MERGEs the mention edges against real FalkorDB.

    The inline-mentions seam: a freshly upserted memory's MNEMOZINE_MENTIONS edges
    are asserted by exact node id and are idempotent — a re-call re-asserts the same
    edges (the same count) and creates no parallel duplicate, exactly mirroring
    persist_mentions' write semantics, but driven per-memory at ingest time.
    """

    rust = await live_backend.resolve_or_create_entity(Entity(canonical_name="rust"))
    tokio = await live_backend.resolve_or_create_entity(Entity(canonical_name="tokio"))
    result = await live_backend.upsert_memory(
        _memory("north", entities=["rust", "tokio"])
    )
    memory_id = result.memory.id

    n = await live_backend.add_memory_mentions(memory_id, [rust.id, tokio.id])
    assert n == 2

    async def _edge_count() -> int:
        rows = await live_backend._query(
            f"MATCH (m:{MEMORY_LABEL} {{id: $mid}})"
            f"-[r:{MENTIONS_TYPE}]->(:{ENTITY_LABEL}) RETURN count(r) AS n",
            mid=memory_id,
        )
        return int(rows[0][0])

    assert await _edge_count() == 2
    # Both endpoints are the resolved entity ids (no dangling / wrong-id edge).
    rows = await live_backend._query(
        f"MATCH (m:{MEMORY_LABEL} {{id: $mid}})"
        f"-[:{MENTIONS_TYPE}]->(e:{ENTITY_LABEL}) RETURN e.id AS eid ORDER BY eid",
        mid=memory_id,
    )
    assert sorted(row[0] for row in rows) == sorted([rust.id, tokio.id])

    # Idempotent: re-asserting the same edges creates no parallel duplicate.
    n2 = await live_backend.add_memory_mentions(memory_id, [rust.id, tokio.id])
    assert n2 == 2
    assert await _edge_count() == 2


async def test_exact_dedup_collapses_duplicate_name_nodes_against_real_falkordb(
    live_backend: GraphitiStorageBackend,
) -> None:
    """exact_name_dedup over real FalkorDB: normalized-name dups collapse to one node.

    The one-time catch-up the ``dedup-entities`` CLI runs after the v2 migration,
    proven against a live store: two ``GitHub`` / ``github`` nodes (case drift) with
    DIFFERENT edge sets fold via the real ``merge_entities`` Cypher to a single
    survivor that carries BOTH edge sets (the co-mention to ``rust`` AND the memory
    mention AND the RELATES->``tokio``), no memory is deleted, and the survivor holds
    ``name_key`` so the unique-normalized-name invariant holds. A re-run merges 0
    (FR-MNT-5).
    """

    gh1 = Entity(id="e-gh1", canonical_name="GitHub", aliases=["gh"])
    gh2 = Entity(id="e-gh2", canonical_name="github")
    rust = Entity(id="e-rust", canonical_name="rust")
    tokio = Entity(id="e-tokio", canonical_name="tokio")
    for e in (gh1, gh2, rust, tokio):
        await live_backend.upsert_entity(e)
    # gh1's edge set: co-mention with rust.
    await live_backend.upsert_co_mention("e-gh1", "e-rust", weight=1.0, shared=2)
    # gh2's edge set: a RELATES edge to tokio + a memory mention.
    await live_backend.upsert_edge(
        Edge(from_entity="e-gh2", to_entity="e-tokio", relation="uses", weight=0.6)
    )
    result = await live_backend.upsert_memory(
        _memory("north", entities=["github"])
    )
    memory_id = result.memory.id
    await live_backend.persist_mentions()

    settings = Settings()
    settings.graph.entity_dedup_mode = "exact"
    report = await EntityDedupJob(live_backend, settings=settings).run()
    assert report.entities_merged == 1

    # Exactly one node remains for the normalized name, and it carries name_key.
    rows = await live_backend._query(
        f"MATCH (e:{ENTITY_LABEL}) WHERE e.name_key = $k RETURN e.id AS id",
        k="github",
    )
    assert len(rows) == 1
    survivor_id = rows[0][0]
    dead_id = "e-gh1" if survivor_id == "e-gh2" else "e-gh2"

    # The duplicate node is gone (graph does not fragment).
    gone = await live_backend._query(
        f"MATCH (e:{ENTITY_LABEL} {{id: $id}}) RETURN count(e) AS n", id=dead_id
    )
    assert int(gone[0][0]) == 0

    # BOTH edge sets landed on the survivor — co-mention to rust, RELATES to tokio,
    # and the memory mention — none orphaned onto the dead node.
    co = await live_backend._query(
        f"MATCH (s:{ENTITY_LABEL} {{id: $id}})-[:{CO_MENTION_TYPE}]-(o:{ENTITY_LABEL}) "
        "RETURN o.id AS oid ORDER BY oid",
        id=survivor_id,
    )
    assert "e-rust" in {row[0] for row in co}
    rel = await live_backend._query(
        f"MATCH (s:{ENTITY_LABEL} {{id: $id}})-[:{RELATES_TYPE}]->(o:{ENTITY_LABEL}) "
        "RETURN o.id AS oid",
        id=survivor_id,
    )
    assert "e-tokio" in {row[0] for row in rel}
    men = await live_backend._query(
        f"MATCH (m:{MEMORY_LABEL})-[:{MENTIONS_TYPE}]->(e:{ENTITY_LABEL} {{id: $id}}) "
        "RETURN m.id AS mid",
        id=survivor_id,
    )
    assert memory_id in {row[0] for row in men}

    # No memory was deleted.
    mem_alive = await live_backend._query(
        f"MATCH (m:{MEMORY_LABEL} {{id: $id}}) RETURN count(m) AS n", id=memory_id
    )
    assert int(mem_alive[0][0]) == 1

    # Idempotent: a second pass finds no collision and merges 0 (FR-MNT-5).
    second = await EntityDedupJob(live_backend, settings=settings).run()
    assert second.entities_merged == 0
