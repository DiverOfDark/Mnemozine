"""FR-RET-2/3/4/5 retriever tests against the offline fakes.

Covers:
* FR-RET-2 scoped-query composition — current project + global, never the whole
  graph; cross-project leakage is excluded; entity-neighborhood is composed.
* ANCESTOR-COMPOSITION (core redesign / no-leak) — a query in a sub-scope
  retrieves global + every ancestor + the leaf via the storage
  ancestor-composing ``scoped_query``; siblings never leak.
* FR-RET-3/5 build_index — compact, budget-respecting, counts/snippets correct
  (now split on the controlled ScopeDecision: ``global_count`` / ``project_count``
  and cross-ref hints driven by the ``cross_ref_candidate`` flag), and
  (FR-MNT-3) build_index does NOT record access while scoped_retrieve/recall DO.
* FR-RET-4 recall — default composed scope vs explicit scope.
"""

from __future__ import annotations

from mnemozine.config import Settings
from mnemozine.interfaces import RetrievalContext
from mnemozine.retrieval.retriever import ScopedRetriever
from mnemozine.schema.models import (
    Edge,
    Entity,
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
    scope: Scope,
    entities: list[str],
    category: str = "fact",
    cross_ref_candidate: bool = False,
    confidence: float = 0.9,
) -> MemoryUnit:
    """Build a MemoryUnit on the category-split contract (no legacy ``type``).

    ``scope`` drives the controlled scope decision (global vs project); the
    free-form ``category`` and the ``cross_ref_candidate`` flag carry the
    semantic role the old ``MemoryType`` used to.
    """

    return MemoryUnit(
        content=content,
        scope=scope,
        category=category,
        cross_ref_candidate=cross_ref_candidate,
        entities=entities,
        confidence=confidence,
        provenance=_prov(),
    )


async def _seed(storage: InMemoryStorage) -> dict[str, MemoryUnit]:
    """Seed a global preference, two project facts (one in another project), and
    a cross-ref-candidate idea seed. Returns them by label for assertions."""

    pref = _mem(
        "Prefers thiserror over anyhow for rust error handling",
        scope=Scope.global_(),
        category="preference",
        entities=["rust", "error-handling"],
    )
    fact_here = _mem(
        "rust-cli pins tokio 1.38",
        scope=Scope.project("rust-cli"),
        category="decision",
        entities=["rust", "tokio"],
    )
    fact_other = _mem(
        "other-proj uses postgres 16 exclusively",
        scope=Scope.project("other-proj"),
        category="decision",
        entities=["postgres"],
    )
    idea = _mem(
        "idea seed for an async runtime cli benchmarking tool",
        scope=Scope.global_(),
        category="idea",
        cross_ref_candidate=True,
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


# ---------------------------------------------------------------------------
# ANCESTOR-COMPOSITION (core redesign / no-leak): a sub-module query composes
# its ancestor chain — global + project + leaf — via the storage
# ancestor-composing scoped_query, while siblings never leak.
# ---------------------------------------------------------------------------


async def _seed_hierarchy(storage: InMemoryStorage) -> dict[str, MemoryUnit]:
    """Seed one memory at each level of a project's scope path + a sibling.

    Levels (for project "Mnemozine"):
      global               -> a cross-project preference
      project:Mnemozine    -> a project-wide fact
      project:Mnemozine/auth -> a leaf (sub-module) fact
    plus a sibling project:Mnemozine/db fact that must never leak into auth.
    """

    g = _mem(
        "Prefers structured logging everywhere",
        scope=Scope.global_(),
        category="preference",
        entities=["logging"],
    )
    proj = _mem(
        "Mnemozine targets python 3.13 across all modules",
        scope=Scope.project("Mnemozine"),
        category="decision",
        entities=["python"],
    )
    leaf = _mem(
        "auth module uses argon2 for password hashing",
        scope=Scope.project("Mnemozine", "auth"),
        category="decision",
        entities=["argon2", "auth"],
    )
    sibling = _mem(
        "db module pins postgres 16 for the schema",
        scope=Scope.project("Mnemozine", "db"),
        category="decision",
        entities=["postgres", "db"],
    )
    for m in (g, proj, leaf, sibling):
        await storage.upsert_memory(m)
    return {"global": g, "project": proj, "leaf": leaf, "sibling": sibling}


async def test_submodule_query_composes_ancestor_chain(settings: Settings) -> None:
    """A query in a sub-scope retrieves global + every ancestor + the leaf.

    Core redesign: ``Scope.project("Mnemozine", "auth").ancestors()`` is
    ``[global, project:Mnemozine, project:Mnemozine/auth]``; the storage's
    ancestor-composing ``scoped_query`` matches any stored scope that is an
    ancestor-or-self of the query scope. So a query at the ``auth`` sub-module
    sees the global preference, the project-wide fact AND the leaf fact — but
    never the sibling ``db`` fact.
    """

    storage = InMemoryStorage()
    seeded = await _seed_hierarchy(storage)
    retriever = ScopedRetriever(storage, settings=settings, default_project="Mnemozine")

    # Query scoped at the deepest (leaf) sub-module. The retriever passes the
    # leaf scope through; storage composes its ancestor-or-self chain.
    context = RetrievalContext(
        project="Mnemozine",
        scopes=[Scope.project("Mnemozine", "auth")],
        entities=[],
    )
    results = await retriever.scoped_retrieve(
        "python logging argon2 password postgres schema", context, top_k=10
    )
    ids = {r.memory.id for r in results}

    # Composes the WHOLE ancestor chain: global + project + leaf.
    assert seeded["global"].id in ids
    assert seeded["project"].id in ids
    assert seeded["leaf"].id in ids
    # The sibling sub-module must NEVER leak into a different sub-module's query.
    assert seeded["sibling"].id not in ids


async def test_ancestors_chain_is_root_first_self_last() -> None:
    """Sanity-pin the composed chain the retrieval path relies on (root-first)."""

    chain = Scope.project("Mnemozine", "auth").ancestors()
    assert [s.as_str() for s in chain] == [
        "global",
        "project:Mnemozine",
        "project:Mnemozine/auth",
    ]


async def test_project_query_does_not_see_descendant_submodule(
    settings: Settings,
) -> None:
    """A shallower (project-level) query must NOT pull a descendant leaf's memory.

    Ancestor-composition is one-directional: a query at ``project:Mnemozine``
    composes ``[global, project:Mnemozine]`` and so sees global + project facts
    but not a deeper ``project:Mnemozine/auth`` leaf (which is a *descendant*,
    not an ancestor-or-self).
    """

    storage = InMemoryStorage()
    seeded = await _seed_hierarchy(storage)
    retriever = ScopedRetriever(storage, settings=settings, default_project="Mnemozine")

    context = RetrievalContext(
        project="Mnemozine",
        scopes=[Scope.project("Mnemozine")],
        entities=[],
    )
    results = await retriever.scoped_retrieve("python logging argon2 postgres", context, top_k=10)
    ids = {r.memory.id for r in results}

    assert seeded["global"].id in ids
    assert seeded["project"].id in ids
    # The leaf and sibling sub-modules are descendants -> excluded.
    assert seeded["leaf"].id not in ids
    assert seeded["sibling"].id not in ids


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
        entities=["rust", "error-handling", "async", "cli", "tokio"],
        recent_text="working on rust error handling and an async cli with tokio",
    )
    index = await retriever.build_index(context)
    assert index.token_estimate <= settings.inject.token_budget
    # Compose included the global pref + idea (both global) + the current-project
    # fact. Counts now split on the controlled ScopeDecision rather than the old
    # MemoryType: global_count covers the pref + idea, project_count the fact.
    assert index.global_count >= 1
    assert index.project_count >= 1
    assert index.text  # non-empty advisory block
    assert "Relevant memory" in index.text
    # The global preference content should be referenced in a snippet.
    assert "thiserror" in index.text
    # The idea seed shows up as a cross-ref hint, driven by the cross_ref_candidate
    # FLAG (not the old idea_seed type), because it shares async/cli with the context.
    assert any("async runtime" in h or "cli" in h for h in index.cross_ref_hints)
    # Other project's fact must never appear in this project's index.
    assert "postgres" not in index.text
    _ = seeded


async def test_build_index_cross_ref_hints_driven_by_flag(settings: Settings) -> None:
    """The cross-ref hints come from ``cross_ref_candidate``, not a fixed type.

    Two global memories share the active entities; only the one flagged
    ``cross_ref_candidate`` becomes a cross-ref hint, proving the hint surface is
    driven by the flag rather than the (now-dropped) idea_seed type.
    """

    storage = InMemoryStorage()
    flagged = _mem(
        "async cli idea worth chasing later",
        scope=Scope.global_(),
        category="idea",
        cross_ref_candidate=True,
        entities=["async", "cli"],
    )
    not_flagged = _mem(
        "async cli reference note, plain fact",
        scope=Scope.global_(),
        category="fact",
        cross_ref_candidate=False,
        entities=["async", "cli"],
    )
    for m in (flagged, not_flagged):
        await storage.upsert_memory(m)

    retriever = ScopedRetriever(storage, settings=settings, default_project="svc")
    context = RetrievalContext(
        project="svc",
        scopes=[Scope.project("svc"), Scope.global_()],
        entities=["async", "cli"],
        recent_text="async cli idea note worth chasing reference",
    )
    index = await retriever.build_index(context)

    joined = " ".join(index.cross_ref_hints)
    assert "worth chasing" in joined  # the flagged one is a hint
    assert "plain fact" not in joined  # the unflagged one is not


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
        scope=Scope.project("rust-cli"),
        category="decision",
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
