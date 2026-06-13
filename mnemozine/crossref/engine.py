"""The cross-reference engine (FR-RET-6 / UC-3).

Given the current working context, find related ``idea_seed``/project memories
and surface only the explainable, above-threshold ones — each with a
human-readable reason, capped at ``crossref.max_suggestions``, with dismissed
suggestions suppressed so they stop resurfacing (R2).

Two paths, in priority order:

1. **Graph traversal over shared entities (preferred — explainable).** Expand
   the working-context entities one hop via ``StorageBackend.neighbors`` (which
   yields the connecting :class:`~mnemozine.schema.models.Edge`, so relation +
   weight survive for the reason and weight-rank), then gather candidate
   memories scoped to the search subset. Each candidate's score derives from
   how much its entities overlap the (expanded) context plus the strength of the
   connecting edges; the reason is literally the shared entities.

2. **Vector-similarity fallback.** When the graph path surfaces nothing above
   ``crossref.relevance_threshold``, embed the working-context text and compare
   it to candidate idea/project content, gated by the *separate*
   ``crossref.vector_fallback_threshold``. A vector hit carries a clearly
   labelled "semantically similar" reason (no shared entity to cite).

Suppression is persisted **through the storage backend**
(``record_suppression`` / ``is_suppressed``), not in this object, so a dismissal
survives across calls and processes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

from mnemozine.config import Settings, get_settings
from mnemozine.crossref.scoring import (
    build_reason,
    cosine_similarity,
    graph_relevance,
)
from mnemozine.interfaces import (
    CrossReference,
    EmbeddingProvider,
    RetrievalContext,
    RetrievedMemory,
    StorageBackend,
)
from mnemozine.schema.models import (
    Edge,
    MemoryType,
    Scope,
)

# Memory types eligible to surface as a cross-reference (FR-RET-6 / UC-3):
# only candidate ideas and project-scoped facts power "this reminds me of…".
_CROSSREF_TYPES: frozenset[MemoryType] = frozenset(
    {MemoryType.IDEA_SEED, MemoryType.PROJECT_FACT}
)


def context_key_for(context: RetrievalContext) -> str:
    """Derive a stable suppression key for a working context (FR-RET-6 / R2).

    A dismissal is scoped to *the context it was dismissed in* so the same
    suggestion can still surface in a genuinely different context later. We key
    on the project plus the sorted active entities (the things that actually
    drive what surfaces), hashed for compactness and stability. The recent free
    text is intentionally excluded so trivial wording changes don't defeat a
    dismissal.
    """

    project = context.project or ""
    entities = ",".join(sorted({e for e in context.entities if e}))
    raw = f"{project}|{entities}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class CrossReferenceEngine:
    """Concrete :class:`mnemozine.interfaces.CrossReferencer` (FR-RET-6).

    Constructor deps (per INTERFACES.md): the storage backend (graph traversal,
    candidate retrieval, **and** suppression persistence — the backend owns that
    store, R2) and an embedding provider for the vector-similarity fallback.
    ``settings`` supplies the §6.6 tuning knobs (thresholds, caps) so nothing is
    a magic constant.
    """

    def __init__(
        self,
        storage: StorageBackend,
        embeddings: EmbeddingProvider,
        settings: Settings | None = None,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._settings = settings or get_settings()

    # -- FR-RET-6 primary API ---------------------------------------------

    async def find_related(
        self, context: RetrievalContext, *, max_suggestions: int | None = None
    ) -> list[CrossReference]:
        """Find related idea_seed/project nodes for the current context (FR-RET-6).

        Graph traversal over shared entities first (explainable), vector
        fallback second; only above ``crossref.relevance_threshold``, capped at
        ``crossref.max_suggestions``, each with a reason, suppressed items
        excluded.
        """

        cfg = self._settings.crossref
        cap = cfg.max_suggestions if max_suggestions is None else max_suggestions
        if cap <= 0:
            return []

        ctx_key = context_key_for(context)

        # 1) Graph path over shared entities (preferred, explainable).
        graph_hits = await self._graph_candidates(context)

        # 2) Vector fallback only when the graph path found nothing surfacing.
        surfacing = [h for h in graph_hits if h.score >= cfg.relevance_threshold]
        if not surfacing:
            vector_hits = await self._vector_candidates(context, exclude=graph_hits)
            graph_hits = self._merge(graph_hits, vector_hits)

        # 3) Threshold-gate, drop reasonless connections, exclude suppressed.
        ranked = sorted(graph_hits, key=lambda h: h.score, reverse=True)
        out: list[CrossReference] = []
        for hit in ranked:
            if hit.score < cfg.relevance_threshold:
                continue
            if not hit.reason:
                # A connection without a reason must not surface (FR-RET-6).
                continue
            if await self._storage.is_suppressed(hit.memory.id, ctx_key):
                continue
            out.append(hit)
            if len(out) >= cap:
                break
        return out

    async def suppress(self, memory_id: str, context_key: str) -> None:
        """Record a dismissal so the suggestion stops resurfacing (FR-RET-6, R2).

        Delegates to the storage backend, which owns the suppression store, so
        the dismissal survives across calls and process boundaries. ``context_key``
        is the working-context key (see :func:`context_key_for`); callers that
        hold a :class:`RetrievalContext` rather than a key can derive it with
        that helper.
        """

        await self._storage.record_suppression(memory_id, context_key)

    # -- graph traversal path ---------------------------------------------

    async def _graph_candidates(
        self, context: RetrievalContext
    ) -> list[CrossReference]:
        """Collect candidate cross-references via shared-entity graph traversal.

        Expands the context entities one hop (collecting the connecting edges
        per neighbor for the reason/weight), retrieves the candidate memories in
        the composed scope subset, and scores each by entity overlap modulated by
        connecting-edge strength.
        """

        ctx_entities = self._dedupe(context.entities)
        if not ctx_entities:
            return []

        # Expand the entity neighborhood one hop and remember which edges connect
        # each neighbor back to a context entity (for the reason + weight-rank).
        expanded, edges_by_entity = await self._expand_neighborhood(ctx_entities)

        scopes = self._scopes(context)
        candidates = await self._retrieve_candidates(
            context, scopes, expanded
        )

        hits: list[CrossReference] = []
        for cand in candidates:
            mem = cand.memory
            if mem.type not in _CROSSREF_TYPES:
                continue
            shared = self._shared_entities(mem.entities, expanded)
            if not shared:
                continue
            connecting = self._edges_for_shared(shared, edges_by_entity)
            score = graph_relevance(
                shared_entities=shared,
                context_entities=expanded,
                candidate_entities=mem.entities,
                connecting_edges=connecting,
            )
            reason = build_reason(shared, connecting)
            if not reason:
                continue
            hits.append(
                CrossReference(
                    memory=mem,
                    score=score,
                    reason=reason,
                    shared_entities=shared,
                )
            )
        return self._dedupe_by_memory(hits)

    async def _expand_neighborhood(
        self, entities: Sequence[str]
    ) -> tuple[list[str], dict[str, list[Edge]]]:
        """One-hop entity-neighborhood expansion (FR-RET-2 depth, FR-RET-6).

        Returns the expanded entity name set (originals + their active neighbors)
        and a map ``neighbor_name -> connecting edges`` so a candidate that
        carries a neighbor entity can cite the relation that links it back to the
        working context.
        """

        max_degree = self._settings.maintenance.max_node_degree
        hops = max(0, self._settings.retrieval.neighborhood_hops)

        expanded: list[str] = list(entities)
        edges_by_entity: dict[str, list[Edge]] = {}
        seen: set[str] = set(entities)
        frontier: list[str] = list(entities)

        for _ in range(hops):
            next_frontier: list[str] = []
            for name in frontier:
                neighbors = await self._storage.neighbors(
                    name, max_degree=max_degree, active_only=True
                )
                for nb in neighbors:
                    nb_name = nb.entity.canonical_name
                    edges_by_entity.setdefault(nb_name, []).append(nb.edge)
                    if nb_name not in seen:
                        seen.add(nb_name)
                        expanded.append(nb_name)
                        next_frontier.append(nb_name)
            frontier = next_frontier
            if not frontier:
                break
        return expanded, edges_by_entity

    async def _retrieve_candidates(
        self,
        context: RetrievalContext,
        scopes: Sequence[Scope],
        entities: Sequence[str],
    ) -> list[RetrievedMemory]:
        """Pull candidate memories scoped to the search subset (FR-RET-2).

        Uses the backend's scoped, entity-bounded query so the search space stays
        bounded (never a graph-wide scan). The query text is the recent working
        text when present, else a join of the active entities, so the backend's
        semantic/lexical filter has something to match.
        """

        query = context.recent_text or " ".join(entities)
        top_k = max(self._settings.retrieval.top_k, self._settings.crossref.max_suggestions * 5)
        return await self._storage.scoped_query(
            query,
            scopes,
            entities=list(entities),
            top_k=top_k,
        )

    # -- vector fallback path ---------------------------------------------

    async def _vector_candidates(
        self,
        context: RetrievalContext,
        *,
        exclude: Iterable[CrossReference],
    ) -> list[CrossReference]:
        """Vector-similarity fallback (FR-RET-6), gated by its own threshold.

        Only runs when the graph path surfaced nothing. Embeds the working-context
        text and scores candidate idea/project content by cosine similarity,
        keeping only those above ``crossref.vector_fallback_threshold``. The
        resulting reason is explicitly labelled as semantic similarity.
        """

        probe = (context.recent_text or "").strip()
        if not probe:
            # Nothing to embed against -> no vector fallback.
            return []

        cfg = self._settings.crossref
        excluded_ids = {h.memory.id for h in exclude}
        scopes = self._scopes(context)

        # Broad scoped pull (no entity filter) so vector fallback can reach ideas
        # that share no entity with the context — that is its whole purpose.
        candidates = await self._storage.scoped_query(
            probe,
            scopes,
            entities=None,
            top_k=max(self._settings.retrieval.top_k, cfg.max_suggestions * 5),
        )
        cand_memories = [
            c.memory
            for c in candidates
            if c.memory.type in _CROSSREF_TYPES and c.memory.id not in excluded_ids
        ]
        if not cand_memories:
            return []

        probe_vec = await self._embeddings.embed(probe)
        texts = [m.content for m in cand_memories]
        cand_vecs = await self._embeddings.embed_batch(texts)

        hits: list[CrossReference] = []
        for mem, vec in zip(cand_memories, cand_vecs, strict=True):
            sim = cosine_similarity(probe_vec, vec)
            if sim < cfg.vector_fallback_threshold:
                continue
            shared = self._shared_entities(mem.entities, context.entities)
            reason = build_reason(shared, via_vector=True)
            if not reason:
                continue
            hits.append(
                CrossReference(
                    memory=mem,
                    score=sim,
                    reason=reason,
                    shared_entities=shared,
                )
            )
        return hits

    # -- small pure helpers ------------------------------------------------

    def _scopes(self, context: RetrievalContext) -> list[Scope]:
        """The composed scopes to search: context scopes, else project + global."""

        if context.scopes:
            return list(context.scopes)
        scopes: list[Scope] = [Scope.global_()]
        if context.project:
            scopes.append(Scope.project(context.project))
        return scopes

    @staticmethod
    def _dedupe(items: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    @staticmethod
    def _shared_entities(
        candidate_entities: Sequence[str], context_entities: Sequence[str]
    ) -> list[str]:
        """Entities a candidate shares with the (expanded) context, ordered.

        Order follows the context-entity order so the most context-relevant
        shared entity leads the generated reason.
        """

        cand = set(candidate_entities)
        out: list[str] = []
        seen: set[str] = set()
        for e in context_entities:
            if e in cand and e not in seen:
                seen.add(e)
                out.append(e)
        return out

    @staticmethod
    def _edges_for_shared(
        shared: Sequence[str], edges_by_entity: dict[str, list[Edge]]
    ) -> list[Edge]:
        edges: list[Edge] = []
        for name in shared:
            edges.extend(edges_by_entity.get(name, []))
        return edges

    @staticmethod
    def _merge(
        primary: Sequence[CrossReference], extra: Sequence[CrossReference]
    ) -> list[CrossReference]:
        """Merge two candidate lists, preferring the primary entry per memory."""

        by_id: dict[str, CrossReference] = {}
        for hit in primary:
            by_id[hit.memory.id] = hit
        for hit in extra:
            by_id.setdefault(hit.memory.id, hit)
        return list(by_id.values())

    @staticmethod
    def _dedupe_by_memory(hits: Sequence[CrossReference]) -> list[CrossReference]:
        """Keep the highest-scoring cross-reference per memory id."""

        best: dict[str, CrossReference] = {}
        for hit in hits:
            current = best.get(hit.memory.id)
            if current is None or hit.score > current.score:
                best[hit.memory.id] = hit
        return list(best.values())
