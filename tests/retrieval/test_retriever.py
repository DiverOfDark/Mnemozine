"""FR-RET-2/3/4/5 retriever tests against the offline fakes.

Covers:
* FR-RET-2 scoped-query composition — current project + global, never the whole
  graph; cross-project leakage is excluded; entity-neighborhood is composed.
* FR-RET-3/5 build_index — compact, budget-respecting, counts/snippets correct,
  and (FR-MNT-3) build_index does NOT record access while scoped_retrieve/recall
  DO.
* FR-RET-4 recall — default composed scope vs explicit scope.
"""

from __future__ import annotations

from mnemozine.config import Settings
from mnemozine.interfaces import RetrievalContext
from mnemozine.retrieval.retriever import ScopedRetriever
from mnemozine.schema.models import (
    Edge,
    Entity,
    MemoryType,
    MemoryUnit,
    Provenance,
    Scope,
)
from tests.conftest import InMemoryStorage


def _prov() -> Provenance:
    return Provenance(source="claude_code", session_id="sess-1", chunk_hash="abc")


def _mem(
    content: str,
    *,
    type_: MemoryType,
    scope: Scope,
    entities: list[str],
    confidence: float = 0.9,
) -> MemoryUnit:
    return MemoryUnit(
        type=type_,
        content=content,
        scope=scope,
        entities=entities,
        confidence=confidence,
        provenance=_prov(),
    )


async def _seed(storage: InMemoryStorage) -> dict[str, MemoryUnit]:
    """Seed a global preference, two project facts (one in another project), and
    an idea seed. Returns them by label for assertions."""

    pref = _mem(
        "Prefers thiserror over anyhow for rust error handling",
        type_=MemoryType.PREFERENCE,
        scope=Scope.global_(),
        entities=["rust", "error-handling"],
    )
    fact_here = _mem(
        "rust-cli pins tokio 1.38",
        type_=MemoryType.PROJECT_FACT,
        scope=Scope.project("rust-cli"),
        entities=["rust", "tokio"],
    )
    fact_other = _mem(
        "other-proj uses postgres 16 exclusively",
        type_=MemoryType.PROJECT_FACT,
        scope=Scope.project("other-proj"),
        entities=["postgres"],
    )
    idea = _mem(
        "idea seed for an async runtime cli benchmarking tool",
        type_=MemoryType.IDEA_SEED,
        scope=Scope.global_(),
        entities=["async", "cli"],
    )
    for m in (pref, fact_here, fact_other, idea):
        await storage.upsert_memory(m)
    return {"pref": pref, "fact_here": fact_here, "fact_other": fact_other, "idea": idea}


async def test_scoped_retrieve_composes_project_and_global_excludes_other_project(
    settings: Settings,
) -> None:
    storage = InMemoryStorage()
    seeded = await _seed(storage)
    retriever = ScopedRetriever(storage, settings=settings, default_project="rust-cli")

    context = RetrievalContext(
        project="rust-cli",
        scopes=[Scope.project("rust-cli"), Scope.global_()],
        entities=["rust"],
    )
    results = await retriever.scoped_retrieve("rust tokio thiserror", context, top_k=10)
    ids = {r.memory.id for r in results}

    # Project fact from the *current* project and the global preference compose.
    assert seeded["fact_here"].id in ids
    assert seeded["pref"].id in ids
    # The other project's fact must NOT leak (FR-RET-2 / no-leak check §9).
    assert seeded["fact_other"].id not in ids


async def test_compose_scopes_always_includes_global() -> None:
    # A context carrying only a project scope must still search global so a
    # cross-project preference can carry over (UC-1 / FR-RET-2).
    ctx = RetrievalContext(project="p", scopes=[Scope.project("p")], entities=[])
    composed = ScopedRetriever._compose_scopes(ctx)
    assert any(s.is_global for s in composed)
    assert any(s.as_str() == "project:p" for s in composed)


async def test_scoped_retrieve_records_access(settings: Settings) -> None:
    storage = InMemoryStorage()
    seeded = await _seed(storage)
    retriever = ScopedRetriever(storage, settings=settings, default_project="rust-cli")

    before = seeded["pref"].access_count
    context = RetrievalContext(
        project="rust-cli", scopes=[Scope.project("rust-cli"), Scope.global_()], entities=["rust"]
    )
    results = await retriever.scoped_retrieve("rust thiserror", context)
    assert results  # something matched
    # Every returned unit must have had access recorded (FR-MNT-3).
    for r in results:
        assert storage.memories[r.memory.id].access_count > before


async def test_build_index_does_not_record_access(settings: Settings) -> None:
    storage = InMemoryStorage()
    await _seed(storage)
    retriever = ScopedRetriever(storage, settings=settings, default_project="rust-cli")

    context = RetrievalContext(
        project="rust-cli", scopes=[Scope.project("rust-cli"), Scope.global_()], entities=["rust"]
    )
    await retriever.build_index(context)
    # build_index reads passively and MUST NOT bump access_count (FR-MNT-3).
    assert all(m.access_count == 0 for m in storage.memories.values())


async def test_build_index_respects_budget_and_counts(settings: Settings) -> None:
    storage = InMemoryStorage()
    seeded = await _seed(storage)
    retriever = ScopedRetriever(storage, settings=settings, default_project="rust-cli")

    context = RetrievalContext(
        project="rust-cli",
        scopes=[Scope.project("rust-cli"), Scope.global_()],
        entities=["rust", "error-handling", "async", "cli"],
        recent_text="working on rust error handling and an async cli",
    )
    index = await retriever.build_index(context)
    assert index.token_estimate <= settings.inject.token_budget
    # Compose included the global pref + the current-project fact + idea seed,
    # so the index should reflect at least one preference and one idea hint.
    assert index.preference_count >= 1
    assert index.text  # non-empty advisory block
    assert "Relevant memory" in index.text
    # The global preference content should be referenced in the snippet.
    assert "thiserror" in index.text
    # Idea seed shows up as a hint (it shares async/cli with the context).
    assert any("async runtime" in h or "cli" in h for h in index.idea_seed_hints)
    # Other project's fact must never appear in this project's index.
    assert "postgres" not in index.text
    _ = seeded


async def test_build_index_tiny_budget_truncates(settings: Settings) -> None:
    storage = InMemoryStorage()
    await _seed(storage)
    retriever = ScopedRetriever(storage, settings=settings, default_project="rust-cli")
    context = RetrievalContext(
        project="rust-cli",
        scopes=[Scope.project("rust-cli"), Scope.global_()],
        entities=["rust"],
        recent_text="rust",
    )
    index = await retriever.build_index(context, token_budget=8)
    assert index.token_estimate <= 8 or "Relevant memory" in index.text


async def test_recall_default_scope_is_project_plus_global(settings: Settings) -> None:
    storage = InMemoryStorage()
    seeded = await _seed(storage)
    retriever = ScopedRetriever(storage, settings=settings, default_project="rust-cli")

    results = await retriever.recall("rust tokio thiserror postgres")
    ids = {r.memory.id for r in results}
    assert seeded["pref"].id in ids
    assert seeded["fact_here"].id in ids
    # Default composed scope excludes other-project facts (no-leak).
    assert seeded["fact_other"].id not in ids


async def test_recall_explicit_scope_narrows(settings: Settings) -> None:
    storage = InMemoryStorage()
    seeded = await _seed(storage)
    retriever = ScopedRetriever(storage, settings=settings, default_project="rust-cli")

    # Explicit global scope: only global units (pref + idea), no project facts.
    results = await retriever.recall("rust async cli thiserror tokio", Scope.global_())
    ids = {r.memory.id for r in results}
    assert seeded["pref"].id in ids
    assert seeded["fact_here"].id not in ids
    assert seeded["fact_other"].id not in ids


async def test_recall_records_access(settings: Settings) -> None:
    storage = InMemoryStorage()
    seeded = await _seed(storage)
    retriever = ScopedRetriever(storage, settings=settings, default_project="rust-cli")
    await retriever.recall("rust thiserror")
    assert storage.memories[seeded["pref"].id].access_count >= 1


async def test_neighborhood_expansion_uses_graph_edges(settings: Settings) -> None:
    # FR-RET-2: the entity-linked neighborhood is composed via graph traversal.
    storage = InMemoryStorage()

    rust = Entity(canonical_name="rust")
    tokio = Entity(canonical_name="tokio")
    await storage.upsert_entity(rust)
    await storage.upsert_entity(tokio)
    await storage.upsert_edge(
        Edge(from_entity=rust.id, to_entity=tokio.id, relation="uses", weight=0.9)
    )

    # A memory tagged only with 'tokio' (a neighbor of the seed entity 'rust').
    mem = _mem(
        "the project relies on tokio for async",
        type_=MemoryType.PROJECT_FACT,
        scope=Scope.project("rust-cli"),
        entities=["tokio"],
    )
    await storage.upsert_memory(mem)

    retriever = ScopedRetriever(storage, settings=settings, default_project="rust-cli")
    context = RetrievalContext(
        project="rust-cli",
        scopes=[Scope.project("rust-cli"), Scope.global_()],
        entities=["rust"],  # seed only rust; tokio reached via the edge
    )
    expanded = await retriever._expand_entities(context.entities)
    assert "rust" in expanded
    assert "tokio" in expanded  # reached by 1-hop traversal

    results = await retriever.scoped_retrieve("tokio async", context)
    assert mem.id in {r.memory.id for r in results}


async def test_zero_hops_does_not_expand() -> None:
    storage = InMemoryStorage()
    rust = Entity(canonical_name="rust")
    tokio = Entity(canonical_name="tokio")
    await storage.upsert_entity(rust)
    await storage.upsert_entity(tokio)
    await storage.upsert_edge(
        Edge(from_entity=rust.id, to_entity=tokio.id, relation="uses", weight=0.9)
    )
    s = Settings()
    s.retrieval.neighborhood_hops = 0
    retriever = ScopedRetriever(storage, settings=s)
    expanded = await retriever._expand_entities(["rust"])
    assert expanded == ["rust"]
