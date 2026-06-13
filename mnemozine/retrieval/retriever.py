"""``ScopedRetriever`` — the FR-RET-2/3/4/5 retrieval implementation.

Implements the :class:`mnemozine.interfaces.Retriever` Protocol on top of a
:class:`~mnemozine.interfaces.StorageBackend`. It depends *only* on the
``StorageBackend`` Protocol (never a concrete sibling module), so it works
identically against the real Graphiti/FalkorDB backend and the offline
``InMemoryStorage`` test fake.

Responsibilities:

* **scoped_retrieve (FR-RET-2)** — compose the context's scopes (current project
  + global) with the active entity neighborhood, expand that neighborhood by up
  to ``retrieval.neighborhood_hops`` via ``StorageBackend.neighbors`` (so the
  searched subset is *current project + global prefs + entity-linked
  neighborhood*), then delegate the bounded semantic search to
  ``StorageBackend.scoped_query`` — **never** a whole-graph scan. Records access
  on the returned units (FR-MNT-3).
* **build_index (FR-RET-3/5)** — produce the compact, token-budgeted injection
  index (counts + entity tags + idea-seed hints + top-preference snippets only),
  truncated to ``inject.token_budget``. Deliberately does **not** record access
  (its reads are passive/automatic; see FR-MNT-3 note in interfaces).
* **recall (FR-RET-4)** — on-demand full-detail recall across the default
  composed scope (project + global) or an explicit scope. Records access.
"""

from __future__ import annotations

from collections.abc import Sequence

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import (
    InjectionIndex,
    RetrievalContext,
    RetrievedMemory,
    StorageBackend,
)
from mnemozine.retrieval.budget import IndexParts, render_index
from mnemozine.schema.models import MemoryType, Scope


def _truncate_snippet(content: str, *, max_chars: int = 160) -> str:
    """One-line, length-bounded snippet of a memory's content for the index."""

    flat = " ".join(content.split())
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 1].rstrip() + "…"


class ScopedRetriever:
    """Scoped retrieval, injection-index construction, and recall (FR-RET-*).

    Constructed with a ``StorageBackend`` (the only hard dependency) and, when
    supplied, an explicit project scope for the running session so ``recall``
    with ``scope=None`` and ``build_index`` know the default project context.

    Parameters
    ----------
    storage:
        The backend implementing :class:`mnemozine.interfaces.StorageBackend`.
    settings:
        Process settings; ``inject.token_budget``, ``retrieval.neighborhood_hops``
        and ``retrieval.top_k`` are read from here. Defaults to ``get_settings()``.
    default_project:
        Project id for the running session, used to compose the default scope for
        ``recall(scope=None)`` and as the project for a bare ``build_index``
        context. ``None`` -> global-only default.
    """

    def __init__(
        self,
        storage: StorageBackend,
        *,
        settings: Settings | None = None,
        default_project: str | None = None,
    ) -> None:
        self._storage = storage
        self._settings = settings or get_settings()
        self._default_project = default_project

    # -- scope composition (FR-RET-2) --------------------------------------

    def _default_scopes(self) -> list[Scope]:
        """The default composed scope: current project + global (FR-RET-2)."""

        scopes: list[Scope] = []
        if self._default_project:
            scopes.append(Scope.project(self._default_project))
        scopes.append(Scope.global_())
        return scopes

    @staticmethod
    def _compose_scopes(context: RetrievalContext) -> list[Scope]:
        """Compose the scopes to search, always including global (FR-RET-2).

        Uses the context's scopes when present; otherwise falls back to
        ``project:<context.project>`` + global. Global is always included so a
        preference learned elsewhere can carry over (UC-1), and duplicates are
        removed.
        """

        scopes: list[Scope] = list(context.scopes)
        if not scopes:
            if context.project:
                scopes.append(Scope.project(context.project))
        # Always ensure global is part of the composition.
        if not any(s.is_global for s in scopes):
            scopes.append(Scope.global_())
        # De-dup preserving order.
        seen: set[str] = set()
        out: list[Scope] = []
        for s in scopes:
            key = s.as_str()
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    async def _expand_entities(self, entities: Sequence[str]) -> list[str]:
        """Expand active entities into their linked neighborhood (FR-RET-2).

        Walks up to ``retrieval.neighborhood_hops`` hops out from the seed
        entities via ``StorageBackend.neighbors`` (which yields the connecting
        edge so the traversal is real graph traversal, not a guess), bounded by
        ``maintenance.max_node_degree`` per node. The resulting entity set bounds
        the semantic search subset so the effective search space stays roughly
        constant as the store grows (FR-RET-2). Returns the seeds plus neighbors,
        deduped.
        """

        if not entities:
            return []

        hops = max(0, self._settings.retrieval.neighborhood_hops)
        max_degree = self._settings.maintenance.max_node_degree

        seen: set[str] = set()
        ordered: list[str] = []
        for e in entities:
            low = e.lower()
            if low not in seen:
                seen.add(low)
                ordered.append(low)

        frontier = list(ordered)
        for _ in range(hops):
            next_frontier: list[str] = []
            for name in frontier:
                neighbors = await self._storage.neighbors(
                    name, max_degree=max_degree, active_only=True
                )
                for nb in neighbors:
                    cand = nb.entity.canonical_name.lower()
                    if cand not in seen:
                        seen.add(cand)
                        ordered.append(cand)
                        next_frontier.append(cand)
            frontier = next_frontier
            if not frontier:
                break
        return ordered

    # -- FR-RET-2 scoped retrieve ------------------------------------------

    async def scoped_retrieve(
        self, query: str, context: RetrievalContext, *, top_k: int = 10
    ) -> list[RetrievedMemory]:
        """Scoped semantic retrieval for ``context`` (FR-RET-2).

        Composes scopes (project + global), expands the entity neighborhood, then
        delegates the bounded search to ``StorageBackend.scoped_query`` — never a
        graph-wide scan. Records access for the returned units (FR-MNT-3) since
        this is a deliberate read.
        """

        scopes = self._compose_scopes(context)
        neighborhood = await self._expand_entities(context.entities)
        results = await self._storage.scoped_query(
            query,
            scopes,
            entities=neighborhood or None,
            top_k=top_k,
        )
        await self._record_access(results)
        return results

    # -- FR-RET-3 / FR-RET-5 injection index -------------------------------

    async def build_index(
        self, context: RetrievalContext, *, token_budget: int | None = None
    ) -> InjectionIndex:
        """Build the compact, token-budgeted injection index (FR-RET-3/5).

        Pulls scoped candidates (preferences + project facts + idea seeds within
        the composed scope/neighborhood), ranks them, renders a compact index
        (counts + entity tags + 1-line idea-seed hints + top-preference snippets
        only), and truncates to ``token_budget`` (defaults to
        ``inject.token_budget`` = 500) by dropping the lowest-ranked snippets —
        never overflowing.

        Does **not** record access: SessionStart/mid-session injection fires on
        every turn, so counting it would inflate ``access_count`` for every
        memory and corrupt FR-MNT-3 decay ranking (see interfaces note).
        """

        budget = token_budget if token_budget is not None else self._settings.inject.token_budget
        scopes = self._compose_scopes(context)
        neighborhood = await self._expand_entities(context.entities)

        # A wide candidate pull so counts/snippets reflect the neighborhood. The
        # query is the recent text (most specific signal) falling back to the
        # joined entity tags so an empty-text SessionStart still scopes by topic.
        query = context.recent_text or " ".join(context.entities)
        candidates = await self._storage.scoped_query(
            query,
            scopes,
            entities=neighborhood or None,
            top_k=max(self._settings.retrieval.top_k, self._settings.inject.max_preference_snippets)
            * 3,
            include_archived=False,
        )

        preferences = [r for r in candidates if r.memory.type is MemoryType.PREFERENCE]
        project_facts = [r for r in candidates if r.memory.type is MemoryType.PROJECT_FACT]
        idea_seeds = [r for r in candidates if r.memory.type is MemoryType.IDEA_SEED]

        # Top-preference snippets only (FR-RET-3 contract), best-first.
        snippet_cap = self._settings.inject.max_preference_snippets
        pref_snippets = [_truncate_snippet(r.memory.content) for r in preferences[:snippet_cap]]

        # One-line idea-seed hints with their shared entity tags as the reason.
        idea_hints: list[str] = []
        for r in idea_seeds:
            shared = sorted(set(e.lower() for e in r.memory.entities) & set(neighborhood))
            tag = f" (shares {', '.join(shared)})" if shared else ""
            idea_hints.append(_truncate_snippet(r.memory.content, max_chars=80) + tag)

        # Entity tags shown in the summary: the active context entities (seeds),
        # not the full expanded neighborhood, so the summary stays compact.
        entity_tags = list(dict.fromkeys(e.lower() for e in context.entities))[:6]

        parts = IndexParts(
            preference_snippets=pref_snippets,
            idea_seed_hints=idea_hints,
            entity_tags=entity_tags,
            preference_count=len(preferences),
            project_fact_count=len(project_facts),
        )
        text, est = render_index(parts, token_budget=budget)

        return InjectionIndex(
            text=text,
            token_estimate=est,
            preference_count=len(preferences),
            project_fact_count=len(project_facts),
            idea_seed_hints=idea_hints,
            entity_tags=entity_tags,
        )

    # -- FR-RET-4 recall ---------------------------------------------------

    async def recall(
        self, query: str, scope: Scope | None = None, *, top_k: int = 10
    ) -> list[RetrievedMemory]:
        """On-demand full-detail recall (FR-RET-4, UC-4).

        Backs the ``recall(query, scope?)`` MCP tool. When ``scope`` is given,
        searches just that scope; when ``None``, searches the default composed
        scope (current project + global). Records access (deliberate read,
        FR-MNT-3).
        """

        scopes = [scope] if scope is not None else self._default_scopes()
        results = await self._storage.scoped_query(query, scopes, top_k=top_k)
        await self._record_access(results)
        return results

    # -- access recording (FR-MNT-3) --------------------------------------

    async def _record_access(self, results: Sequence[RetrievedMemory]) -> None:
        """Bump access bookkeeping for deliberately-read units (FR-MNT-3)."""

        for r in results:
            await self._storage.record_access(r.memory.id)
