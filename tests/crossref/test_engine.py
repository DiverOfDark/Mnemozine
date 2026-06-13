"""Unit tests for the cross-reference engine (FR-RET-6 / UC-3), offline.

Exercises the four behaviors called out in the task against the shared fake
storage graph (``InMemoryStorage``) and ``FakeEmbeddingProvider`` — no live
FalkorDB/Ollama/Qwen:

* **traversal ranking** — shared-entity candidates are scored and ordered by
  relevance (overlap + connecting-edge weight);
* **threshold gating** — only connections above
  ``crossref.relevance_threshold`` surface, capped at ``max_suggestions``;
* **reason generation** — every surfaced connection carries a human-readable
  reason naming the shared entities (UC-3 "shares async-runtime, cli-parsing");
* **suppression** — a dismissed suggestion stops resurfacing in that context,
  persisted through the storage backend (R2);

plus the **vector-similarity fallback** when no shared-entity path exists.
"""

from __future__ import annotations

import pytest

from mnemozine.config import Settings
from mnemozine.crossref import CrossReferenceEngine, context_key_for
from mnemozine.interfaces import RetrievalContext
from mnemozine.schema.models import (
    Edge,
    Entity,
    MemoryType,
    MemoryUnit,
    Provenance,
    Scope,
)
from tests.conftest import FakeEmbeddingProvider, InMemoryStorage

# ---------------------------------------------------------------------------
# Builders for seeding the fake storage graph
# ---------------------------------------------------------------------------


def _prov() -> Provenance:
    return Provenance(source="claude_code", session_id="sess-x")


async def _add_entity(store: InMemoryStorage, name: str) -> Entity:
    ent = Entity(canonical_name=name)
    await store.upsert_entity(ent)
    return ent


async def _add_edge(
    store: InMemoryStorage,
    a: Entity,
    b: Entity,
    *,
    relation: str = "relates_to",
    weight: float = 1.0,
) -> Edge:
    edge = Edge(from_entity=a.id, to_entity=b.id, relation=relation, weight=weight)
    return await store.upsert_edge(edge)


async def _add_idea(
    store: InMemoryStorage,
    *,
    content: str,
    entities: list[str],
    scope: Scope | None = None,
    mtype: MemoryType = MemoryType.IDEA_SEED,
    confidence: float = 0.9,
) -> MemoryUnit:
    mem = MemoryUnit(
        type=mtype,
        content=content,
        scope=scope or Scope.global_(),
        entities=entities,
        confidence=confidence,
        provenance=_prov(),
    )
    # Insert directly (bypassing the 4-way write decision, which is for the
    # write path, not cross-ref seeding).
    store.memories[mem.id] = mem
    return mem


def _settings(**crossref_overrides: object) -> Settings:
    s = Settings()
    for k, v in crossref_overrides.items():
        setattr(s.crossref, k, v)
    return s


def _engine(store: InMemoryStorage, **crossref_overrides: object) -> CrossReferenceEngine:
    """Build an engine over ``store`` with crossref settings overrides."""

    return CrossReferenceEngine(
        store, FakeEmbeddingProvider(), _settings(**crossref_overrides)
    )


# ---------------------------------------------------------------------------
# Traversal ranking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_traversal_finds_shared_entity_idea() -> None:
    store = InMemoryStorage()
    # Working context: project D about async runtime + cli parsing.
    idea = await _add_idea(
        store,
        content="idea for project C: an async cli tool sharing async cli concepts",
        entities=["async", "cli"],
    )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["async", "cli"],
        recent_text="working on an async cli tool",
    )
    eng = _engine(store)
    hits = await eng.find_related(ctx)

    assert len(hits) == 1
    assert hits[0].memory.id == idea.id
    assert set(hits[0].shared_entities) == {"async", "cli"}


@pytest.mark.asyncio
async def test_traversal_ranks_more_overlap_first() -> None:
    store = InMemoryStorage()
    strong = await _add_idea(
        store,
        content="async cli idea matching async cli context",
        entities=["async", "cli"],
    )
    weak = await _add_idea(
        store,
        content="async database idea matching async context only",
        entities=["async", "db"],
    )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["async", "cli"],
        recent_text="async cli context idea matching",
    )
    eng = _engine(store, max_suggestions=2)
    hits = await eng.find_related(ctx)

    ids = [h.memory.id for h in hits]
    assert strong.id in ids
    # The two-entity-overlap idea must outrank the one-entity-overlap idea.
    assert ids.index(strong.id) == 0
    if weak.id in ids:
        assert ids.index(strong.id) < ids.index(weak.id)


@pytest.mark.asyncio
async def test_edge_weight_boosts_ranking() -> None:
    """A strong connecting edge raises a candidate's relevance (weight-rank)."""

    store = InMemoryStorage()
    async_ent = await _add_entity(store, "async")
    cli_ent = await _add_entity(store, "cli")
    # An edge from a context entity to a neighbor the candidate also carries.
    await _add_edge(store, async_ent, cli_ent, relation="pairs_with", weight=1.0)

    idea = await _add_idea(
        store,
        content="async idea connected via strong edge",
        entities=["async"],
    )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["async"],
        recent_text="async idea connected strong edge",
    )
    eng = _engine(store, relevance_threshold=0.1)
    hits = await eng.find_related(ctx)
    assert hits
    assert hits[0].memory.id == idea.id


@pytest.mark.asyncio
async def test_no_shared_entity_no_graph_hit() -> None:
    store = InMemoryStorage()
    await _add_idea(
        store,
        content="completely unrelated python data idea",
        entities=["python", "pandas"],
    )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["rust", "async"],
        recent_text="rust async work",
    )
    eng = _engine(store)
    hits = await eng.find_related(ctx)
    assert hits == []


@pytest.mark.asyncio
async def test_preferences_do_not_surface_as_crossref() -> None:
    """Only idea_seed/project nodes surface — not preferences (FR-RET-6)."""

    store = InMemoryStorage()
    await _add_idea(
        store,
        content="prefers async runtime tokio for async work",
        entities=["async"],
        mtype=MemoryType.PREFERENCE,
    )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["async"],
        recent_text="async work tokio prefers runtime",
    )
    eng = _engine(store, relevance_threshold=0.1)
    hits = await eng.find_related(ctx)
    assert hits == []


@pytest.mark.asyncio
async def test_neighborhood_expansion_links_via_edge() -> None:
    """A candidate carrying a *neighbor* entity surfaces via one-hop expansion."""

    store = InMemoryStorage()
    async_ent = await _add_entity(store, "async")
    tokio_ent = await _add_entity(store, "tokio")
    await _add_edge(store, async_ent, tokio_ent, relation="implemented_by", weight=0.9)

    # Candidate shares only "tokio", which is a 1-hop neighbor of context "async".
    idea = await _add_idea(
        store,
        content="tokio runtime idea async related neighbor",
        entities=["tokio"],
    )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["async"],
        recent_text="tokio runtime idea async related neighbor",
    )
    eng = _engine(store, relevance_threshold=0.1)
    hits = await eng.find_related(ctx)
    assert hits
    assert hits[0].memory.id == idea.id
    assert "tokio" in hits[0].shared_entities
    assert "implemented_by" in hits[0].reason


# ---------------------------------------------------------------------------
# Threshold gating + cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_threshold_gates_low_relevance() -> None:
    store = InMemoryStorage()
    # One shared of many candidate entities -> low overlap -> low score.
    await _add_idea(
        store,
        content="async sprawling idea with many many unrelated entities listed",
        entities=["async", "db", "http", "cache", "queue", "auth"],
    )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["async"],
        recent_text="async sprawling idea many entities",
    )
    # High threshold (default 0.8) should suppress the weak connection.
    eng_high = _engine(store, relevance_threshold=0.8)
    assert await eng_high.find_related(ctx) == []

    # Lowering the threshold lets it through -> proves the gate, not absence.
    eng_low = _engine(store, relevance_threshold=0.05)
    assert await eng_low.find_related(ctx)


@pytest.mark.asyncio
async def test_default_threshold_is_high_precision_first() -> None:
    # PRD §6.6: start high (precision over recall).
    assert Settings().crossref.relevance_threshold == 0.8


@pytest.mark.asyncio
async def test_max_suggestions_caps_results() -> None:
    store = InMemoryStorage()
    for i in range(5):
        await _add_idea(
            store,
            content=f"async cli idea number {i} async cli matching",
            entities=["async", "cli"],
        )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["async", "cli"],
        recent_text="async cli idea matching number",
    )
    eng = _engine(store, max_suggestions=2, relevance_threshold=0.1)
    hits = await eng.find_related(ctx)
    assert len(hits) == 2


@pytest.mark.asyncio
async def test_max_suggestions_override_argument() -> None:
    store = InMemoryStorage()
    for i in range(3):
        await _add_idea(
            store,
            content=f"async cli idea {i} async cli matching",
            entities=["async", "cli"],
        )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["async", "cli"],
        recent_text="async cli idea matching",
    )
    eng = _engine(store, max_suggestions=2, relevance_threshold=0.1)
    assert len(await eng.find_related(ctx, max_suggestions=1)) == 1
    assert await eng.find_related(ctx, max_suggestions=0) == []


# ---------------------------------------------------------------------------
# Reason generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_surfaced_connection_has_a_reason() -> None:
    store = InMemoryStorage()
    await _add_idea(
        store,
        content="async cli tool idea async cli parsing",
        entities=["async", "cli"],
    )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["async", "cli"],
        recent_text="async cli tool parsing idea",
    )
    eng = _engine(store, relevance_threshold=0.1)
    hits = await eng.find_related(ctx)
    assert hits
    for h in hits:
        assert h.reason
        assert h.reason.startswith("shares ")


@pytest.mark.asyncio
async def test_reason_names_shared_entities_uc3_shape() -> None:
    # UC-3 example shape: "shares async-runtime, cli-parsing".
    store = InMemoryStorage()
    await _add_idea(
        store,
        content="project C idea about async-runtime cli-parsing concepts",
        entities=["async-runtime", "cli-parsing"],
    )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["async-runtime", "cli-parsing"],
        recent_text="async-runtime cli-parsing concepts project idea",
    )
    eng = _engine(store, relevance_threshold=0.1)
    hits = await eng.find_related(ctx)
    assert hits
    assert "async-runtime" in hits[0].reason
    assert "cli-parsing" in hits[0].reason


# ---------------------------------------------------------------------------
# Suppression / feedback (R2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suppression_stops_resurfacing() -> None:
    store = InMemoryStorage()
    idea = await _add_idea(
        store,
        content="async cli idea async cli to dismiss",
        entities=["async", "cli"],
    )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["async", "cli"],
        recent_text="async cli idea dismiss",
    )
    eng = _engine(store, relevance_threshold=0.1)

    before = await eng.find_related(ctx)
    assert any(h.memory.id == idea.id for h in before)

    # Dismiss it for this context, then it must not resurface here.
    await eng.suppress(idea.id, context_key_for(ctx))
    after = await eng.find_related(ctx)
    assert all(h.memory.id != idea.id for h in after)


@pytest.mark.asyncio
async def test_suppression_persisted_through_storage_backend() -> None:
    # R2: dismissal is owned by the storage backend, not the CrossReferencer,
    # so it survives across CrossReferencer instances/processes.
    store = InMemoryStorage()
    idea = await _add_idea(
        store,
        content="async cli idea async cli persist",
        entities=["async", "cli"],
    )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["async", "cli"],
        recent_text="async cli idea persist",
    )
    key = context_key_for(ctx)

    eng1 = _engine(store, relevance_threshold=0.1)
    await eng1.suppress(idea.id, key)
    assert await store.is_suppressed(idea.id, key)

    # A brand-new engine over the same backend still sees the suppression.
    eng2 = _engine(store, relevance_threshold=0.1)
    after = await eng2.find_related(ctx)
    assert all(h.memory.id != idea.id for h in after)


@pytest.mark.asyncio
async def test_suppression_is_context_scoped() -> None:
    # Dismissing in one context must not suppress the same idea in a different
    # context (different entities -> different context key).
    store = InMemoryStorage()
    idea = await _add_idea(
        store,
        content="async cli graphql idea async cli graphql versatile",
        entities=["async", "cli", "graphql"],
    )
    ctx_a = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["async", "cli"],
        recent_text="async cli idea versatile",
    )
    ctx_b = RetrievalContext(
        project="project-e",
        scopes=[Scope.global_()],
        entities=["graphql"],
        recent_text="graphql idea versatile",
    )
    eng = _engine(store, relevance_threshold=0.1)

    await eng.suppress(idea.id, context_key_for(ctx_a))
    # Suppressed in ctx_a...
    assert all(h.memory.id != idea.id for h in await eng.find_related(ctx_a))
    # ...but still surfaces in the genuinely different ctx_b.
    assert any(h.memory.id == idea.id for h in await eng.find_related(ctx_b))


def test_context_key_stable_and_order_independent() -> None:
    a = RetrievalContext(project="p", entities=["async", "cli"])
    b = RetrievalContext(project="p", entities=["cli", "async"])  # reordered
    c = RetrievalContext(project="p", entities=["async"])
    assert context_key_for(a) == context_key_for(b)
    assert context_key_for(a) != context_key_for(c)


# ---------------------------------------------------------------------------
# Vector-similarity fallback (FR-RET-6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vector_fallback_when_no_shared_entity() -> None:
    """When no shared-entity path exists, a near-identical idea surfaces via vectors."""

    store = InMemoryStorage()
    probe = "build a distributed event sourcing ledger"
    # Candidate shares NO entity with the context but has identical content so
    # the deterministic fake embedding gives cosine 1.0 (>= fallback threshold).
    idea = await _add_idea(
        store,
        content=probe,
        entities=["event-sourcing"],
    )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["cqrs"],  # disjoint from the candidate's entities
        recent_text=probe,
    )
    eng = _engine(store, relevance_threshold=0.6)
    hits = await eng.find_related(ctx)
    assert hits
    assert hits[0].memory.id == idea.id
    assert "semantically similar" in hits[0].reason


@pytest.mark.asyncio
async def test_vector_fallback_respects_its_threshold() -> None:
    store = InMemoryStorage()
    await _add_idea(
        store,
        content="totally different subject matter here",
        entities=["unrelated"],
    )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["cqrs"],
        recent_text="build a distributed event sourcing ledger",
    )
    # Dissimilar content -> cosine below vector_fallback_threshold -> no hit.
    eng = CrossReferenceEngine(
        store,
        FakeEmbeddingProvider(),
        _settings(relevance_threshold=0.1, vector_fallback_threshold=0.99),
    )
    assert await eng.find_related(ctx) == []


@pytest.mark.asyncio
async def test_vector_hit_must_also_clear_relevance_threshold() -> None:
    """A vector hit passes its own fallback gate but is still subject to the final
    ``relevance_threshold`` surfacing gate (the two thresholds compose)."""

    store = InMemoryStorage()
    probe = "build a distributed event sourcing ledger"
    await _add_idea(store, content=probe, entities=["event-sourcing"])
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["cqrs"],
        recent_text=probe,
    )
    # Identical content -> cosine ~1.0 -> clears fallback (0.75). But set the
    # final relevance gate above 1.0 so nothing can surface despite the match.
    eng = CrossReferenceEngine(
        store,
        FakeEmbeddingProvider(),
        _settings(relevance_threshold=1.01, vector_fallback_threshold=0.75),
    )
    assert await eng.find_related(ctx) == []


@pytest.mark.asyncio
async def test_graph_path_preferred_over_vector() -> None:
    """A graph (shared-entity) hit is preferred; vector path doesn't run when it wins."""

    store = InMemoryStorage()
    graph_idea = await _add_idea(
        store,
        content="async cli idea async cli shared entity path",
        entities=["async", "cli"],
    )
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["async", "cli"],
        recent_text="async cli idea shared entity path",
    )
    eng = _engine(store, relevance_threshold=0.1, max_suggestions=2)
    hits = await eng.find_related(ctx)
    assert hits
    # Graph hit carries an explainable shared-entity reason (not the vector label).
    top = hits[0]
    assert top.memory.id == graph_idea.id
    assert top.reason.startswith("shares ")
    assert "semantically similar" not in top.reason


@pytest.mark.asyncio
async def test_no_recent_text_skips_vector_fallback_safely() -> None:
    # No probe text -> vector fallback is a no-op (no crash, just no hits).
    store = InMemoryStorage()
    await _add_idea(store, content="some idea", entities=["unrelated"])
    ctx = RetrievalContext(
        project="project-d",
        scopes=[Scope.global_()],
        entities=["cqrs"],
        recent_text=None,
    )
    eng = _engine(store, relevance_threshold=0.1)
    assert await eng.find_related(ctx) == []


@pytest.mark.asyncio
async def test_engine_satisfies_protocol() -> None:
    from mnemozine.interfaces import CrossReferencer

    eng = CrossReferenceEngine(InMemoryStorage(), FakeEmbeddingProvider(), Settings())
    assert isinstance(eng, CrossReferencer)
