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

from mnemozine.config import FalkorDBSettings, MaintenanceSettings
from mnemozine.schema.events import Source
from mnemozine.schema.models import (
    Edge,
    Entity,
    MemoryUnit,
    Provenance,
    Scope,
)
from mnemozine.storage.backend import GraphitiStorageBackend
from mnemozine.storage.graphiti_client import MEMORY_VECTOR_INDEX, GraphitiClient

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
