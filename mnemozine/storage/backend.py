"""The Graphiti-on-FalkorDB :class:`StorageBackend` implementation (FR-STO-*, FR-MNT-1).

This is the concrete storage core: it persists ``MemoryUnit`` / ``Entity`` /
``Edge`` / ``SourceSession`` into FalkorDB (the single graph+vector store,
FR-STO-2) through the :class:`~mnemozine.storage.graphiti_client.GraphitiClient`
seam, with bge-m3 embeddings (:class:`~mnemozine.interfaces.EmbeddingProvider`)
stored alongside each memory node for semantic search.

What it owns (the §7 model is richer than Graphiti's native node types, so the
backend keeps its own labels — see ``MEMORY_LABEL`` et al.):

* **FR-STO-1 temporal validity windows** — memories carry ``valid_from`` /
  ``valid_to``; supersede/decay *close the window* (``valid_to = now``) and never
  hard-delete (FR-MNT-3 "archive, never hard-delete").
* **FR-STO-3 scope tagging + scope-composing queries** — every memory is tagged
  ``global`` / ``project:<id>`` and :meth:`scoped_query` searches only the
  composed scope subset + entity neighborhood (FR-RET-2), never the whole graph.
* **FR-STO-2 vector embeddings in FalkorDB** — each memory node stores its bge-m3
  vector; semantic search uses the FalkorDB vector index.
* **FR-STO-4 hot vs archive tier** — a ``tier`` property; the default retrieval
  path is hot-only.
* **FR-MNT-1 4-way write decision + FR-MNT-4 entity upsert/merge primitives.**

Cypher vs. the fake
-------------------
The Cypher methods are exercised by a contract test against a small in-process
**fake driver** (``execute_query``-compatible) plus the shared
``InMemoryStorage`` fake, so the whole module is unit-testable with no live
FalkorDB / Ollama. The fake driver speaks just enough of FalkorDB's result shape
(``.result_set`` rows) for the backend's normalization to be verified; the
ranking/cosine/decision logic is real and shared.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from mnemozine.config import MaintenanceSettings, RetrievalSettings, get_settings
from mnemozine.interfaces import (
    EmbeddingProvider,
    MaintenanceReport,
    Neighbor,
    RetrievedMemory,
    WriteDecision,
    WriteResult,
)
from mnemozine.migrations import CURRENT_DATA_VERSION, record_data_version
from mnemozine.schema.models import (
    DEFAULT_CATEGORY,
    Edge,
    Entity,
    MemoryUnit,
    Provenance,
    RawChunk,
    Scope,
    ScopeDecision,
    SourceSession,
    Tier,
)
from mnemozine.storage.cosine import cosine_similarity
from mnemozine.storage.graphiti_client import (
    ENTITY_LABEL,
    MEMORY_LABEL,
    MEMORY_VECTOR_INDEX,
    RAW_CHUNK_LABEL,
    RELATES_TYPE,
    SESSION_LABEL,
    SUPPRESSION_LABEL,
    GraphitiClient,
)

if TYPE_CHECKING:
    from mnemozine.interfaces import Extractor

# A predicate the integration pass wires to the FR-MNT-1 cheap contradiction LLM
# call. It is injected (not hard-wired to an LLM) so the backend stays free of an
# LLM dependency and the supersede branch is deterministically testable. Async so
# the real implementation can make the narrowly-scoped LLM call.
ContradictsFn = Callable[[MemoryUnit, list[MemoryUnit]], Awaitable[list[MemoryUnit]]]

# FR-RET-2 index-backed KNN tuning. ``db.idx.vector.queryNodes`` applies the
# scope/tier/entity WHERE *after* the KNN cut, so we over-fetch K to avoid the
# post-filter being starved by nearer out-of-scope neighbours, bounded by an
# absolute cap so a large top_k can't ask the index for an effectively unbounded
# scan (which would defeat the flat-search-space goal). These are now config-driven
# (``retrieval.knn_overfetch_factor`` / ``retrieval.knn_overfetch_cap``, §6.6
# "config, not constants"); the module constants below are only the fallback
# defaults used when no ``RetrievalSettings`` is supplied.
_KNN_OVERFETCH = 10
_KNN_MAX_K = 512

# Substrings FalkorDB uses when the vector index/procedure is unavailable. Used to
# distinguish "index genuinely absent -> fall back to the bounded scan" from a
# real query error (which must surface).
_MISSING_VECTOR_INDEX_MARKERS = (
    "invalid arguments for procedure 'db.idx.vector.querynodes'",
    "unknown procedure 'db.idx.vector.querynodes'",
    "unknown function 'vecf32'",
    "no such index",
)


def _is_missing_vector_index(exc: Exception) -> bool:
    """True if ``exc`` indicates the FalkorDB vector index/procedure is absent.

    Kept conservative (specific markers) so an unrelated failure is never silently
    swallowed into the slow fallback path.
    """

    msg = str(exc).lower()
    return any(marker in msg for marker in _MISSING_VECTOR_INDEX_MARKERS)


async def _no_contradictions(_new: MemoryUnit, _candidates: list[MemoryUnit]) -> list[MemoryUnit]:
    """Default contradiction predicate: nothing contradicts (add/reinforce only)."""

    return []


def _to_iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _from_iso(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


class GraphitiStorageBackend:
    """:class:`mnemozine.interfaces.StorageBackend` over Graphiti/FalkorDB.

    Composes a :class:`GraphitiClient` (the FalkorDB connection + Cypher seam) and
    an :class:`EmbeddingProvider` (bge-m3). The FR-MNT-1 contradiction check is an
    injected async predicate (``contradicts``) so the backend never imports an LLM
    provider and the supersede branch is deterministically testable; the
    integration pass passes a closure over the cheap, narrowly-scoped LLM call.
    """

    def __init__(
        self,
        client: GraphitiClient,
        embeddings: EmbeddingProvider,
        *,
        contradicts: ContradictsFn | None = None,
        maintenance: MaintenanceSettings | None = None,
        retrieval: RetrievalSettings | None = None,
    ) -> None:
        self._client = client
        self._embeddings = embeddings
        self._contradicts = contradicts or _no_contradictions
        self._maint = maintenance or get_settings().maintenance
        # FR-RET-2 KNN over-fetch tuning (§6.6 config, not constants). Falls back
        # to the module-level defaults when no RetrievalSettings is supplied so
        # existing call sites / fakes keep working.
        self._retrieval = retrieval or RetrievalSettings()

    # -- result normalization -------------------------------------------------

    @staticmethod
    def _rows(result: Any) -> list[list[Any]]:
        """Normalize a FalkorDB driver result into a list of row-lists.

        Two concrete result shapes must be collapsed to a plain ``list[list]`` so
        the rest of the backend is shape-agnostic:

        * **the real Graphiti FalkorDB driver** — ``FalkorDriver.execute_query``
          returns ``(records, header, summary)`` where ``records`` is a list of
          **dicts** keyed by the RETURN aliases (e.g. ``{"m": <Node>}``). We project
          each dict back to a row-list in header order so ``row[0]`` is the first
          returned value (a ``falkordb.Node``/``Edge`` or scalar), matching how the
          backend indexes rows.
        * **the in-process fake / a QueryResult-shaped object** — rows live on
          ``.result_set`` as row-lists already.

        Returning the keys of a dict (which bare ``list(dict)`` would do) was the
        original bug: ``row[0]`` then yielded the alias string ``"m"`` rather than
        the node, so every read path silently broke against real FalkorDB.
        """

        if result is None:
            return []
        header: list[str] | None = None
        if isinstance(result, tuple):
            # (records, header, summary) from the real FalkorDriver.
            records = result[0]
            if len(result) > 1 and result[1]:
                header = [str(h) for h in result[1]]
        else:
            records = getattr(result, "result_set", result)
        if not records:
            return []

        out: list[list[Any]] = []
        for r in records:
            if isinstance(r, dict):
                # Dict record keyed by RETURN alias: project in header order so
                # positional indexing (row[0], row[1]) is stable.
                keys = header if header is not None else list(r.keys())
                out.append([r.get(k) for k in keys])
            else:
                out.append(list(r))
        return out

    @staticmethod
    def _props(value: Any) -> dict[str, Any]:
        """Extract a property mapping from a returned graph value.

        A ``falkordb.Node``/``Edge`` exposes its properties on ``.properties`` and
        is *not* dict-iterable (``dict(node)`` raises ``TypeError``); the fake
        driver returns plain ``dict``s. This normalizes both to a property dict so
        the (de)serialization code is driver-agnostic.
        """

        if isinstance(value, dict):
            return value
        props = getattr(value, "properties", None)
        if isinstance(props, dict):
            return dict(props)
        return dict(value)

    async def _query(self, cypher: str, **params: Any) -> list[list[Any]]:
        return self._rows(await self._client.execute_query(cypher, **params))

    # -- (de)serialization of a MemoryUnit <-> node props ---------------------

    async def _memory_props(self, memory: MemoryUnit, *, embedding: list[float]) -> dict[str, Any]:
        """Flatten a :class:`MemoryUnit` into FalkorDB-storable scalar props.

        FalkorDB stores scalars + arrays; the nested ``Provenance`` is JSON-encoded
        into a single property to avoid a second node where it adds no graph value.

        ``embedding`` is included as a plain ``list[float]`` here, but the *insert
        path wraps it in ``vecf32(...)``* (see :meth:`_insert`) so FalkorDB stores
        it as a typed float32 vector — that typed-vector property is what the
        ``CREATE VECTOR INDEX`` actually indexes, and therefore what
        :meth:`scoped_query`'s index-backed KNN can search. Storing it as a bare
        array would leave the vector index empty.
        """

        return {
            "id": memory.id,
            # Core redesign: the old flat `type` is replaced by the free-form
            # `category` string plus the `cross_ref_candidate` flag; the
            # controlled global/project decision is carried by the scope path
            # (persisted as `scope`), not a separate column.
            "category": memory.category,
            "cross_ref_candidate": bool(memory.cross_ref_candidate),
            "content": memory.content,
            "scope": memory.scope.as_str(),
            "entities": list(memory.entities),
            "confidence": float(memory.confidence),
            "provenance": memory.provenance.model_dump_json(),
            "valid_from": _to_iso(memory.valid_from),
            "valid_to": _to_iso(memory.valid_to),
            "tier": memory.tier.value,
            "last_accessed": _to_iso(memory.last_accessed),
            "access_count": int(memory.access_count),
            # Data-versioning (mnemozine.migrations): stamp the data-model version
            # at write time so min_data_version()/migrations can select on it.
            "data_version": int(memory.data_version),
            "embedding": embedding,
        }

    @staticmethod
    def _row_to_memory(props: dict[str, Any]) -> MemoryUnit:
        """Rebuild a :class:`MemoryUnit` from stored node properties."""

        provenance_raw = props.get("provenance")
        provenance = (
            Provenance.model_validate_json(provenance_raw)
            if provenance_raw
            else Provenance.classify_sentinel()
        )
        return MemoryUnit(
            id=props["id"],
            content=props["content"],
            scope=Scope.parse(props["scope"]),
            # Core redesign: load the free-form `category` + `cross_ref_candidate`
            # flag; fall back to DEFAULT_CATEGORY for legacy nodes written before
            # the category split (which had a `type` column and no `category`).
            category=str(props.get("category") or DEFAULT_CATEGORY),
            cross_ref_candidate=bool(props.get("cross_ref_candidate", False)),
            entities=list(props.get("entities") or []),
            confidence=float(props.get("confidence", 1.0)),
            provenance=provenance,
            valid_from=_from_iso(props.get("valid_from")) or datetime.now(UTC),
            valid_to=_from_iso(props.get("valid_to")),
            tier=Tier(props.get("tier", Tier.HOT.value)),
            last_accessed=_from_iso(props.get("last_accessed")),
            access_count=int(props.get("access_count", 0)),
            # Data-versioning: legacy nodes written before this feature have no
            # `data_version` property; record_data_version() maps that to 0 so a
            # migration always picks them up.
            data_version=record_data_version(props.get("data_version")),
        )

    async def _node_to_memory(self, node: Any) -> MemoryUnit:
        """Convert a returned graph node (Node/dict) into a MemoryUnit."""

        return self._row_to_memory(self._props(node))

    # -- the FR-MNT-1 4-way write decision ------------------------------------

    async def _active_candidates(self, memory: MemoryUnit) -> list[MemoryUnit]:
        """Same-scope, overlapping-entity, active candidates (FR-MNT-1 scope).

        This is the *only* comparison set for a write — never a graph-wide scan
        (FR-MNT-1 / FR-RET-2). Restricted in Cypher to the same ``scope`` string,
        an open validity window, and at least one shared entity, then capped at
        ``maintenance.contradiction_candidate_cap`` so the contradiction LLM call
        downstream stays cheap.
        """

        cypher = (
            f"MATCH (m:{MEMORY_LABEL}) "
            "WHERE m.scope = $scope AND m.valid_to IS NULL "
            "AND any(e IN m.entities WHERE e IN $entities) "
            "RETURN m LIMIT $cap"
        )
        rows = await self._query(
            cypher,
            scope=memory.scope.as_str(),
            entities=list(memory.entities),
            cap=self._maint.contradiction_candidate_cap,
        )
        return [await self._node_to_memory(r[0]) for r in rows]

    async def upsert_memory(self, memory: MemoryUnit) -> WriteResult:
        """Insert ``memory`` via the FR-MNT-1 4-way add/reinforce/supersede/no-op.

        Order matches the InMemory fake so behavior is consistent across backends:

        1. **reinforce** — an active candidate with semantically-equivalent content
           (exact match, or cosine >= ``dedup.equivalence_threshold``) exists: bump
           its confidence + refresh timestamp, write no new node.
        2. **supersede** — the injected ``contradicts`` predicate flags one or more
           global-decision (``ScopeDecision.GLOBAL``) candidates as contradicted:
           close their windows and insert the new unit active (UC-2 / Goal 2).
        3. **no-op** — a strictly stronger/equal duplicate of the same category
           already exists (new confidence < existing, same content): keep existing.
        4. **add** — otherwise insert.
        """

        embedding = await self._embeddings.embed(memory.content)
        candidates = await self._active_candidates(memory)

        # 1. reinforce -------------------------------------------------------
        new_content = memory.content.strip()
        for existing in candidates:
            equivalent = existing.content.strip() == new_content
            if not equivalent:
                ex_vec = await self._embeddings.embed(existing.content)
                equivalent = (
                    cosine_similarity(embedding, ex_vec)
                    >= self._maint.dedup_equivalence_threshold
                )
            if equivalent:
                reinforced = await self._reinforce(existing, memory.confidence)
                return WriteResult(decision=WriteDecision.REINFORCE, memory=reinforced)

        # 2. supersede -------------------------------------------------------
        # Contradiction is a preference-reversal check; in the category-split
        # contract a "preference" is a global-scope memory (ScopeDecision.GLOBAL),
        # so the candidate set is the active global-decision candidates.
        pref_candidates = [
            c for c in candidates if c.scope_decision is ScopeDecision.GLOBAL
        ]
        contradicted = await self._contradicts(memory, pref_candidates) if pref_candidates else []
        if contradicted:
            superseded: list[MemoryUnit] = []
            for old in contradicted:
                superseded.append(await self.close_validity_window(old.id))
            inserted = await self._insert(memory, embedding)
            return WriteResult(
                decision=WriteDecision.SUPERSEDE, memory=inserted, superseded=superseded
            )

        # 3. no-op -----------------------------------------------------------
        for existing in candidates:
            if (
                existing.category == memory.category
                and memory.confidence < existing.confidence
                and existing.content.strip().lower() == new_content.lower()
            ):
                return WriteResult(decision=WriteDecision.NO_OP, memory=existing)

        # 4. add -------------------------------------------------------------
        inserted = await self._insert(memory, embedding)
        return WriteResult(decision=WriteDecision.ADD, memory=inserted)

    async def _insert(self, memory: MemoryUnit, embedding: list[float]) -> MemoryUnit:
        # Two FalkorDB constraints drive the exact shape of this CREATE:
        #
        # 1. ``CREATE (m:Label $props)`` — injecting the property *map literal*
        #    from a parameter — fails with "Encountered unhandled type in inlined
        #    properties" on FalkorDB. The portable form is to create the node, then
        #    ``SET m = $props`` (a param map assignment), which also correctly maps
        #    ``None`` values to absent/null props (so ``m.valid_to IS NULL`` holds).
        # 2. The embedding must land as a typed float32 vector (``vecf32``) for the
        #    vector index to index it; a plain array stays invisible to the index.
        #    So it is split out of $props and set through ``vecf32()``.
        props = await self._memory_props(memory, embedding=embedding)
        props.pop("embedding", None)
        await self._query(
            f"CREATE (m:{MEMORY_LABEL}) SET m = $props "
            "SET m.embedding = vecf32($embedding) RETURN m",
            props=props,
            embedding=embedding,
        )
        return memory

    async def _reinforce(self, existing: MemoryUnit, new_confidence: float) -> MemoryUnit:
        """Bump confidence + refresh access timestamp on an existing unit (FR-MNT-1)."""

        existing.confidence = max(existing.confidence, new_confidence)
        existing.last_accessed = datetime.now(UTC)
        await self._query(
            f"MATCH (m:{MEMORY_LABEL} {{id: $id}}) "
            "SET m.confidence = $confidence, m.last_accessed = $last_accessed RETURN m",
            id=existing.id,
            confidence=existing.confidence,
            last_accessed=_to_iso(existing.last_accessed),
        )
        return existing

    # -- scoped semantic retrieval (FR-RET-2 / FR-STO-3) ----------------------

    @staticmethod
    def _composed_scope_strs(
        scopes: Sequence[Scope], *, compose_ancestors: bool
    ) -> list[str]:
        """Expand the query ``scopes`` into the set of scope strings to match.

        ANCESTOR-COMPOSITION / no-leak (FR-STO-3): with ``compose_ancestors`` each
        query scope is replaced by its ancestor-or-self chain
        (:meth:`Scope.ancestors`), so a memory matches iff its stored scope is an
        ancestor-or-self of *some* query scope. A query at ``project:P/auth`` thus
        sees ``project:P/auth`` + ``project:P`` + ``global`` but never the sibling
        ``project:P/db`` (it is not on the auth chain) and never a descendant. The
        result is de-duplicated while preserving the ``WHERE m.scope IN $scopes``
        membership semantics (order is irrelevant — the index post-filter is a set
        membership test). With ``compose_ancestors=False`` the exact scope strings
        are matched (no widening), used by maintenance passes that must not leak
        ancestors in.
        """

        seen: dict[str, None] = {}
        for s in scopes:
            chain = s.ancestors() if compose_ancestors else [s]
            for anc in chain:
                seen[anc.as_str()] = None
        return list(seen)

    def _scoped_filters(
        self,
        scope_strs: list[str],
        entities: Sequence[str] | None,
        include_archived: bool,
    ) -> tuple[list[str], dict[str, Any]]:
        """Build the shared scope/validity/tier/entity WHERE clauses + params.

        Used by both the index-backed KNN path and the full-scan fallback so the
        two stay behaviourally identical on everything except *how* candidates are
        ranked. The clauses reference ``m`` (the matched/yielded memory node).
        """

        where = ["m.scope IN $scopes", "m.valid_to IS NULL"]
        params: dict[str, Any] = {"scopes": scope_strs}
        if not include_archived:
            where.append("m.tier = $hot")
            params["hot"] = Tier.HOT.value
        if entities:
            where.append("any(e IN m.entities WHERE e IN $entities)")
            params["entities"] = list(entities)
        return where, params

    async def scoped_query(
        self,
        query: str,
        scopes: Sequence[Scope],
        *,
        entities: Sequence[str] | None = None,
        top_k: int = 10,
        include_archived: bool = False,
        compose_ancestors: bool = True,
    ) -> list[RetrievedMemory]:
        """Ancestor-composing semantic search within a scope subset (FR-RET-2/FR-STO-3).

        ANCESTOR-COMPOSITION (the no-leak rule). When ``compose_ancestors`` is
        true (the default) each query scope ``S`` is expanded to ``S.ancestors()``
        (``[global, project:P, project:P/sub, ..., S]``, root-first/self-last) and
        a memory matches iff its stored scope is one of those — i.e. an
        ancestor-or-self of ``S``. So a query at ``project:P/auth`` sees
        ``project:P/auth``, ``project:P`` and ``global`` memories but NEVER a
        sibling like ``project:P/db`` (siblings are not on each other's ancestor
        chain, so they cannot leak) and NEVER a descendant. Pass
        ``compose_ancestors=False`` to match the exact scope strings only (e.g. a
        maintenance pass that must not widen). The composed set is de-duplicated
        (overlapping ancestor chains, e.g. two sub-scopes of the same project,
        collapse the shared ``global``/``project:P`` prefix to one scope string).

        Never a graph-wide in-process scan. The candidate ranking comes from
        **FalkorDB's vector index** (FR-STO-2): a ``db.idx.vector.queryNodes`` KNN
        over the ``MnemozineMemory.embedding`` index returns the nearest neighbours
        by cosine *distance*, and the scope / open-validity-window / hot-tier /
        entity-overlap pre-filters are applied as a ``WHERE`` on the yielded nodes.
        Pushing the nearest-neighbour search into the index — rather than pulling
        every scope+entity row into Python and ranking with in-process cosine — is
        what keeps the effective search space roughly constant as the store grows
        (FR-RET-2 / Goal-5). Active (open window, hot tier) memories only, unless
        ``include_archived``.

        Because ``queryNodes`` filters *after* the KNN cut, we over-fetch ``K``
        (``top_k * _KNN_OVERFETCH``, bounded by ``_KNN_MAX_K``) so the post-filter
        still yields the true ``top_k`` nearest *within* the composed scope rather
        than being starved by nearer out-of-scope neighbours.

        Falls back to a scope-pre-filtered scan + in-process cosine **only** when
        the vector index is genuinely absent (e.g. a freshly-created graph before
        ``ensure_vector_index``), so the path degrades gracefully instead of
        raising.
        """

        scope_strs = self._composed_scope_strs(scopes, compose_ancestors=compose_ancestors)
        if not scope_strs:
            return []
        query_vec = await self._embeddings.embed(query)
        where, params = self._scoped_filters(scope_strs, entities, include_archived)

        # Over-fetch K so the post-KNN scope/tier/entity filter is not starved by
        # nearer out-of-scope neighbours; bound it so a huge top_k can't ask the
        # index for an unbounded scan. Both knobs are config-driven (§6.6).
        overfetch = self._retrieval.knn_overfetch_factor
        cap = self._retrieval.knn_overfetch_cap
        knn_k = min(max(top_k * overfetch, top_k), cap)
        knn_cypher = (
            f"CALL db.idx.vector.queryNodes("
            f"'{MEMORY_LABEL}', 'embedding', $k, vecf32($qv)) "
            "YIELD node AS m, score "
            f"WHERE {' AND '.join(where)} "
            "RETURN m, score ORDER BY score ASC LIMIT $top_k"
        )
        try:
            rows = await self._query(
                knn_cypher, k=knn_k, qv=query_vec, top_k=top_k, **params
            )
        except Exception as exc:  # noqa: BLE001 - narrowed by _is_missing_index
            if not _is_missing_vector_index(exc):
                raise
            return await self._scoped_query_fallback(
                query_vec, where, params, top_k=top_k
            )

        scored: list[RetrievedMemory] = []
        for row in rows:
            mem = self._row_to_memory(self._props(row[0]))
            # The index returns cosine *distance* (0 = identical); convert back to
            # the cosine *similarity* RetrievedMemory.score carries everywhere else.
            distance = float(row[1])
            scored.append(RetrievedMemory(memory=mem, score=1.0 - distance))
        return scored

    async def _scoped_query_fallback(
        self,
        query_vec: list[float],
        where: list[str],
        params: dict[str, Any],
        *,
        top_k: int,
    ) -> list[RetrievedMemory]:
        """Scope-pre-filtered scan + in-process cosine, used only if the vector
        index is absent. Still never a graph-wide scan: the same scope / validity
        / tier / entity WHERE bounds the candidate set before ranking.
        """

        cypher = f"MATCH (m:{MEMORY_LABEL}) WHERE {' AND '.join(where)} RETURN m"
        rows = await self._query(cypher, **params)
        scored: list[RetrievedMemory] = []
        for row in rows:
            props = self._props(row[0])
            mem = self._row_to_memory(props)
            vec = props.get("embedding") or []
            score = cosine_similarity(query_vec, vec)
            scored.append(RetrievedMemory(memory=mem, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    # -- validity window / tiering (FR-STO-1, FR-STO-4, FR-MNT-3) -------------

    async def close_validity_window(
        self, memory_id: str, *, at: Any | None = None
    ) -> MemoryUnit:
        """Close a memory's validity window (FR-MNT-1 supersede; never delete)."""

        ts = at if isinstance(at, datetime) else (datetime.now(UTC) if at is None else at)
        rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL} {{id: $id}}) SET m.valid_to = $valid_to RETURN m",
            id=memory_id,
            valid_to=_to_iso(ts),
        )
        return await self._return_one(rows, memory_id)

    async def archive(self, memory_id: str) -> MemoryUnit:
        """Demote to the archive tier (FR-STO-4 / FR-MNT-3; cold, not deleted)."""

        rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL} {{id: $id}}) SET m.tier = $tier RETURN m",
            id=memory_id,
            tier=Tier.ARCHIVE.value,
        )
        return await self._return_one(rows, memory_id)

    async def promote(self, memory_id: str) -> MemoryUnit:
        """Promote back to hot and lazily re-embed (OQ3 lazy-on-promotion)."""

        rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL} {{id: $id}}) SET m.tier = $tier RETURN m",
            id=memory_id,
            tier=Tier.HOT.value,
        )
        memory = await self._return_one(rows, memory_id)
        # OQ3: archive is re-embedded lazily on promotion.
        return await self.reembed(memory.id)

    async def reembed(self, memory_id: str) -> MemoryUnit:
        """Recompute + store the embedding for one memory (OQ3 re-embed pass)."""

        memory = await self.get_memory(memory_id)
        if memory is None:
            raise KeyError(memory_id)
        embedding = await self._embeddings.embed(memory.content)
        # Re-store as a typed float32 vector (vecf32) so the re-embedded value
        # stays indexed by the FalkorDB vector index (see _insert).
        await self._query(
            f"MATCH (m:{MEMORY_LABEL} {{id: $id}}) SET m.embedding = vecf32($embedding) RETURN m",
            id=memory_id,
            embedding=embedding,
        )
        return memory

    async def record_access(self, memory_id: str) -> None:
        """Bump ``access_count`` / ``last_accessed`` for decay ranking (FR-MNT-3)."""

        await self._query(
            f"MATCH (m:{MEMORY_LABEL} {{id: $id}}) "
            "SET m.access_count = coalesce(m.access_count, 0) + 1, "
            "m.last_accessed = $now RETURN m",
            id=memory_id,
            now=_to_iso(datetime.now(UTC)),
        )

    async def get_memory(self, memory_id: str) -> MemoryUnit | None:
        rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL} {{id: $id}}) RETURN m", id=memory_id
        )
        if not rows:
            return None
        return await self._node_to_memory(rows[0][0])

    async def _return_one(self, rows: list[list[Any]], memory_id: str) -> MemoryUnit:
        if not rows:
            raise KeyError(memory_id)
        return await self._node_to_memory(rows[0][0])

    # -- enumeration / scan (FR-MNT-2/3/4, R5) --------------------------------

    async def iter_memories(
        self,
        *,
        scope: Scope | None = None,
        tier: Any | None = None,
        active_only: bool = False,
        valid_before: datetime | None = None,
        unused_since: datetime | None = None,
    ) -> AsyncIterator[MemoryUnit]:
        """Stream stored memory units for whole-store maintenance passes (FR-MNT-*).

        AND-combines the optional filters in Cypher so the maintenance layer never
        pulls more than it asked for. ``unused_since`` keeps units whose
        ``last_accessed`` is null (never used) or older than the cutoff, matching
        the FR-MNT-3 decay sweep semantics.
        """

        where: list[str] = []
        params: dict[str, Any] = {}
        if scope is not None:
            where.append("m.scope = $scope")
            params["scope"] = scope.as_str()
        if tier is not None:
            tier_val = tier.value if isinstance(tier, Tier) else tier
            where.append("m.tier = $tier")
            params["tier"] = tier_val
        if active_only:
            where.append("m.valid_to IS NULL")
        if valid_before is not None:
            where.append("m.valid_from < $valid_before")
            params["valid_before"] = _to_iso(valid_before)
        if unused_since is not None:
            where.append("(m.last_accessed IS NULL OR m.last_accessed < $unused_since)")
            params["unused_since"] = _to_iso(unused_since)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        rows = await self._query(f"MATCH (m:{MEMORY_LABEL}){clause} RETURN m", **params)
        for row in rows:
            yield await self._node_to_memory(row[0])

    async def iter_entities(self) -> AsyncIterator[Entity]:
        """Stream all entity nodes for entity resolution (FR-MNT-4)."""

        rows = await self._query(f"MATCH (e:{ENTITY_LABEL}) RETURN e")
        for row in rows:
            yield self._row_to_entity(row[0])

    # -- entity ops (FR-EXT-2 / FR-MNT-4) -------------------------------------

    @staticmethod
    def _entity_props(entity: Entity) -> dict[str, Any]:
        return {
            "id": entity.id,
            "canonical_name": entity.canonical_name,
            "aliases": list(entity.aliases),
            "type": entity.type,
        }

    @staticmethod
    def _row_to_entity(node: Any) -> Entity:
        props = GraphitiStorageBackend._props(node)
        return Entity(
            id=props["id"],
            canonical_name=props["canonical_name"],
            aliases=list(props.get("aliases") or []),
            type=props.get("type"),
        )

    async def upsert_entity(self, entity: Entity) -> Entity:
        """Insert or update an entity node, keyed on id (FR-EXT-2)."""

        await self._query(
            f"MERGE (e:{ENTITY_LABEL} {{id: $id}}) "
            "SET e.canonical_name = $canonical_name, e.aliases = $aliases, e.type = $type "
            "RETURN e",
            **self._entity_props(entity),
        )
        return entity

    async def get_entity(self, name_or_id: str) -> Entity | None:
        """Resolve an entity by id, canonical name, or alias (FR-MNT-4)."""

        rows = await self._query(
            f"MATCH (e:{ENTITY_LABEL}) "
            "WHERE e.id = $key OR e.canonical_name = $key OR $key IN e.aliases "
            "RETURN e LIMIT 1",
            key=name_or_id,
        )
        if not rows:
            return None
        return self._row_to_entity(rows[0][0])

    async def merge_entities(self, source_id: str, target_id: str) -> Entity:
        """Merge ``source_id`` into ``target_id`` (entity resolution, FR-MNT-4).

        Repoints the source's edges onto the target, folds the source's canonical
        name + aliases into the target's aliases, and deletes the now-redundant
        source node so the graph does not fragment across duplicate entities.
        """

        source = await self.get_entity(source_id)
        target = await self.get_entity(target_id)
        if source is None or target is None:
            raise KeyError(source_id if source is None else target_id)

        # Repoint edges off the source onto the target (both directions).
        await self._query(
            f"MATCH (s:{ENTITY_LABEL} {{id: $src}})-[r:{RELATES_TYPE}]->(o) "
            f"MATCH (t:{ENTITY_LABEL} {{id: $tgt}}) "
            f"MERGE (t)-[nr:{RELATES_TYPE} {{relation: r.relation}}]->(o) "
            "SET nr.weight = coalesce(nr.weight, r.weight), nr.valid_from = r.valid_from, "
            "nr.valid_to = r.valid_to, nr.id = coalesce(nr.id, r.id) "
            "DELETE r",
            src=source_id,
            tgt=target_id,
        )
        await self._query(
            f"MATCH (o)-[r:{RELATES_TYPE}]->(s:{ENTITY_LABEL} {{id: $src}}) "
            f"MATCH (t:{ENTITY_LABEL} {{id: $tgt}}) "
            f"MERGE (o)-[nr:{RELATES_TYPE} {{relation: r.relation}}]->(t) "
            "SET nr.weight = coalesce(nr.weight, r.weight), nr.valid_from = r.valid_from, "
            "nr.valid_to = r.valid_to, nr.id = coalesce(nr.id, r.id) "
            "DELETE r",
            src=source_id,
            tgt=target_id,
        )

        merged_aliases = sorted({*target.aliases, source.canonical_name, *source.aliases})
        target.aliases = merged_aliases
        await self._query(
            f"MATCH (t:{ENTITY_LABEL} {{id: $tgt}}) SET t.aliases = $aliases RETURN t",
            tgt=target_id,
            aliases=merged_aliases,
        )
        await self._query(
            f"MATCH (s:{ENTITY_LABEL} {{id: $src}}) DELETE s", src=source_id
        )
        return target

    async def neighbors(
        self, entity: str, *, max_degree: int | None = None, active_only: bool = True
    ) -> list[Neighbor]:
        """Return entity-linked neighbors with their connecting edges (FR-RET-2/6).

        Each :class:`Neighbor` keeps the relation+weight so CrossRef can build the
        mandatory explainable ``reason`` and weight-rank, and maintenance can prune
        low-weight edges (FR-MNT-4). Bounded by ``max_degree`` (defaults to
        ``maintenance.max_node_degree``) to keep traversal flat.
        """

        resolved = await self.get_entity(entity)
        if resolved is None:
            return []
        cap = max_degree if max_degree is not None else self._maint.max_node_degree
        active_clause = " AND r.valid_to IS NULL" if active_only else ""
        rows = await self._query(
            f"MATCH (e:{ENTITY_LABEL} {{id: $id}})-[r:{RELATES_TYPE}]-(o:{ENTITY_LABEL}) "
            f"WHERE true{active_clause} "
            "RETURN o, r ORDER BY r.weight DESC LIMIT $cap",
            id=resolved.id,
            cap=cap,
        )
        out: list[Neighbor] = []
        for row in rows:
            other = self._row_to_entity(row[0])
            edge = self._row_to_edge(row[1])
            out.append(Neighbor(entity=other, edge=edge))
        return out

    # -- edge ops (FR-EXT-2 / FR-MNT-4 / FR-RET-6) ----------------------------

    @staticmethod
    def _edge_props(edge: Edge) -> dict[str, Any]:
        return {
            "id": edge.id,
            "from_entity": edge.from_entity,
            "to_entity": edge.to_entity,
            "relation": edge.relation,
            "weight": float(edge.weight),
            "valid_from": _to_iso(edge.valid_from),
            "valid_to": _to_iso(edge.valid_to),
        }

    @staticmethod
    def _row_to_edge(rel: Any) -> Edge:
        props = GraphitiStorageBackend._props(rel)
        return Edge(
            id=props["id"],
            from_entity=props["from_entity"],
            to_entity=props["to_entity"],
            relation=props["relation"],
            weight=float(props.get("weight", 1.0)),
            valid_from=_from_iso(props.get("valid_from")) or datetime.now(UTC),
            valid_to=_from_iso(props.get("valid_to")),
        )

    async def upsert_edge(self, edge: Edge) -> Edge:
        """Insert/update a weighted temporal edge keyed on (from,to,relation) (FR-EXT-2).

        Re-asserting an existing active relation raises its weight rather than
        duplicating the edge; the stored edge is returned (its id may be the
        pre-existing one on a re-assert).
        """

        existing = await self._query(
            f"MATCH (a:{ENTITY_LABEL} {{id: $from}})-[r:{RELATES_TYPE} {{relation: $relation}}]->"
            f"(b:{ENTITY_LABEL} {{id: $to}}) "
            "WHERE r.valid_to IS NULL RETURN r LIMIT 1",
            **{"from": edge.from_entity, "to": edge.to_entity, "relation": edge.relation},
        )
        if existing:
            current = self._row_to_edge(existing[0][0])
            new_weight = max(current.weight, edge.weight)
            await self._query(
                f"MATCH (a:{ENTITY_LABEL} {{id: $from}})-[r:{RELATES_TYPE} {{id: $id}}]->(b) "
                "SET r.weight = $weight RETURN r",
                **{"from": edge.from_entity, "id": current.id, "weight": new_weight},
            )
            current.weight = new_weight
            return current

        await self._query(
            f"MATCH (a:{ENTITY_LABEL} {{id: $from_entity}}) "
            f"MATCH (b:{ENTITY_LABEL} {{id: $to_entity}}) "
            f"CREATE (a)-[r:{RELATES_TYPE} {{id: $id, from_entity: $from_entity, "
            "to_entity: $to_entity, relation: $relation, weight: $weight, "
            "valid_from: $valid_from, valid_to: $valid_to}]->(b) RETURN r",
            **self._edge_props(edge),
        )
        return edge

    async def edges_for_entity(
        self, entity: str, *, active_only: bool = True
    ) -> list[Edge]:
        """Edges incident to ``entity`` (FR-MNT-4 pruning, FR-RET-6 traversal)."""

        resolved = await self.get_entity(entity)
        if resolved is None:
            return []
        active_clause = " WHERE r.valid_to IS NULL" if active_only else ""
        rows = await self._query(
            f"MATCH (e:{ENTITY_LABEL} {{id: $id}})-[r:{RELATES_TYPE}]-(){active_clause} "
            "RETURN r",
            id=resolved.id,
        )
        return [self._row_to_edge(row[0]) for row in rows]

    async def prune_edge(self, edge_id: str, *, at: datetime | None = None) -> Edge:
        """Close a low-weight edge's validity window (FR-MNT-4; retained, not deleted)."""

        ts = at or datetime.now(UTC)
        rows = await self._query(
            f"MATCH ()-[r:{RELATES_TYPE} {{id: $id}}]-() SET r.valid_to = $valid_to RETURN r",
            id=edge_id,
            valid_to=_to_iso(ts),
        )
        if not rows:
            raise KeyError(edge_id)
        return self._row_to_edge(rows[0][0])

    # -- suppression persistence (FR-RET-6 / R2) ------------------------------

    async def record_suppression(self, memory_id: str, context_key: str) -> None:
        """Persist a dismissed cross-reference suggestion (FR-RET-6, R2). Idempotent."""

        await self._query(
            f"MERGE (s:{SUPPRESSION_LABEL} {{memory_id: $memory_id, context_key: $context_key}}) "
            "ON CREATE SET s.suppressed_at = $now RETURN s",
            memory_id=memory_id,
            context_key=context_key,
            now=_to_iso(datetime.now(UTC)),
        )

    async def is_suppressed(self, memory_id: str, context_key: str) -> bool:
        """True if ``(memory_id, context_key)`` was previously suppressed (R2)."""

        rows = await self._query(
            f"MATCH (s:{SUPPRESSION_LABEL} "
            "{memory_id: $memory_id, context_key: $context_key}) RETURN s LIMIT 1",
            memory_id=memory_id,
            context_key=context_key,
        )
        return bool(rows)

    # -- sessions + lifecycle -------------------------------------------------

    async def record_session(self, session: SourceSession) -> None:
        """Persist a source-session record for provenance/archive (§7, FR-STO-4)."""

        await self._query(
            f"MERGE (s:{SESSION_LABEL} {{source: $source, session_id: $session_id}}) "
            "SET s.project = $project, s.started_at = $started_at, "
            "s.ended_at = $ended_at, s.raw_path = $raw_path RETURN s",
            source=session.source,
            session_id=session.session_id,
            project=session.project,
            started_at=_to_iso(session.started_at),
            ended_at=_to_iso(session.ended_at),
            raw_path=session.raw_path,
        )

    # -- raw-chunk tier (offline re-extraction/reindex; survives R4 cleanup) --

    @staticmethod
    def _raw_chunk_props(chunk: RawChunk) -> dict[str, Any]:
        """Flatten a :class:`RawChunk` into FalkorDB-storable scalar props."""

        return {
            "id": chunk.id,
            "content_hash": chunk.content_hash,
            "content": chunk.content,
            "source": chunk.source,
            "session_id": chunk.session_id,
            "scope": chunk.scope.as_str(),
            "project": chunk.project,
            "started_at": _to_iso(chunk.started_at),
            "ended_at": _to_iso(chunk.ended_at),
            "event_count": int(chunk.event_count),
            "raw_path": chunk.raw_path,
            "memory_ids": list(chunk.memory_ids),
            "ingested_at": _to_iso(chunk.ingested_at),
            # Data-versioning (mnemozine.migrations): stamped at write/re-extract.
            "data_version": int(chunk.data_version),
        }

    @staticmethod
    def _row_to_raw_chunk(props: dict[str, Any]) -> RawChunk:
        """Rebuild a :class:`RawChunk` from stored node properties."""

        return RawChunk(
            id=props["id"],
            content_hash=props["content_hash"],
            content=props["content"],
            source=props["source"],
            session_id=props["session_id"],
            scope=Scope.parse(props["scope"]),
            project=props.get("project") or (Scope.parse(props["scope"]).project_id or ""),
            started_at=_from_iso(props.get("started_at")),
            ended_at=_from_iso(props.get("ended_at")),
            event_count=int(props.get("event_count", 0)),
            raw_path=props.get("raw_path"),
            memory_ids=list(props.get("memory_ids") or []),
            ingested_at=_from_iso(props.get("ingested_at")) or datetime.now(UTC),
            # Legacy chunks (pre-feature) have no `data_version`; map to 0.
            data_version=record_data_version(props.get("data_version")),
        )

    async def persist_raw_chunk(self, chunk: RawChunk) -> RawChunk:
        """Persist a :class:`RawChunk` (the raw tier). Idempotent on content_hash.

        Re-persisting the same chunk (same FR-ING-5 ``content_hash``) updates its
        ``memory_ids`` / timestamps rather than duplicating, so offline
        re-extraction over the raw tier stays idempotent (R4 / FR-MNT-5).
        """

        props = self._raw_chunk_props(chunk)
        await self._query(
            f"MERGE (c:{RAW_CHUNK_LABEL} {{content_hash: $content_hash}}) "
            "SET c = $props RETURN c",
            content_hash=chunk.content_hash,
            props=props,
        )
        return chunk

    async def iter_raw_chunks(
        self,
        *,
        scope: Scope | None = None,
        session_id: str | None = None,
        source: str | None = None,
        since: datetime | None = None,
    ) -> AsyncIterator[RawChunk]:
        """Stream stored raw chunks for offline re-extraction/reindex (raw tier).

        Filters (all optional, AND-combined) match exactly (no ancestor
        composition — a re-extraction must not widen scope). Async generator:
        iterate with ``async for c in storage.iter_raw_chunks(...)``.
        """

        where: list[str] = []
        params: dict[str, Any] = {}
        if scope is not None:
            where.append("c.scope = $scope")
            params["scope"] = scope.as_str()
        if session_id is not None:
            where.append("c.session_id = $session_id")
            params["session_id"] = session_id
        if source is not None:
            where.append("c.source = $source")
            params["source"] = source
        if since is not None:
            where.append("c.ingested_at >= $since")
            params["since"] = _to_iso(since)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        rows = await self._query(
            f"MATCH (c:{RAW_CHUNK_LABEL}){clause} RETURN c", **params
        )
        for row in rows:
            yield self._row_to_raw_chunk(self._props(row[0]))

    async def re_extract_from_raw_chunks(
        self,
        extractor: Extractor,
        *,
        scope: Scope | None = None,
        session_id: str | None = None,
        supersede_existing: bool = True,
    ) -> MaintenanceReport:
        """Re-run extraction over the retained raw tier (offline reindex seam).

        Iterates :meth:`iter_raw_chunks` (filtered by ``scope`` / ``session_id``)
        and re-runs ``extractor`` over each chunk's normalized ``content`` to
        produce fresh :class:`MemoryUnit`s, upserting them via the FR-MNT-1 write
        path. When ``supersede_existing`` it first closes the validity windows of
        the memories the chunk previously produced (``RawChunk.memory_ids``) so the
        re-extraction replaces the old units rather than duplicating. Idempotent
        and safe to re-run (FR-MNT-5).

        The :class:`~mnemozine.interfaces.Extractor` Protocol consumes
        ``IngestEvent``s; an ``extractor`` that exposes a text-based re-extraction
        entry point (``extract_text``) is used when present, otherwise the seam
        still supersedes the prior memories so a reindex never leaves a stale and a
        fresh copy both active.
        """

        report = MaintenanceReport(job_name="re_extract")
        chunks = [
            c
            async for c in self.iter_raw_chunks(scope=scope, session_id=session_id)
        ]
        extract_text = getattr(extractor, "extract_text", None)
        for chunk in chunks:
            if supersede_existing:
                for mid in chunk.memory_ids:
                    try:
                        await self.close_validity_window(mid)
                    except KeyError:
                        continue
            if callable(extract_text):
                for memory in await extract_text(chunk.content, scope=chunk.scope):
                    await self.upsert_memory(memory)
            # DATA-VERSION STAMP CONTRACT (mnemozine.migrations): re-stamp the
            # re-processed chunk to the current version so a re-extract migration
            # is idempotent. Fresh memories are stamped via their field default in
            # upsert_memory's write path.
            if record_data_version(chunk.data_version) != CURRENT_DATA_VERSION:
                chunk.data_version = CURRENT_DATA_VERSION
                await self.persist_raw_chunk(chunk)
            report.re_extracted += 1
        return report

    async def reclassify_memory(
        self,
        memory_id: str,
        *,
        scope: Scope | None = None,
        category: str | None = None,
        cross_ref_candidate: bool | None = None,
    ) -> MemoryUnit:
        """Update a stored memory's scope/category/cross-ref WITHOUT raw text (R1).

        Re-derives classification from the already-stored content (does NOT need
        the raw transcript, so it survives the 30-day cleanup). Any subset of
        ``scope`` (re-scope; must obey the hierarchical no-leak rule), ``category``
        (free-form re-label, normalized), or ``cross_ref_candidate`` may be given;
        unset fields are left unchanged. Returns the updated unit.
        """

        memory = await self.get_memory(memory_id)
        if memory is None:
            raise KeyError(memory_id)
        sets: list[str] = []
        params: dict[str, Any] = {"id": memory_id}
        # DATA-VERSION STAMP CONTRACT (mnemozine.migrations): a reclassify always
        # re-stamps the touched memory up to the current version, so the cheap
        # migration path is idempotent (a re-run finds nothing below the version).
        memory.data_version = CURRENT_DATA_VERSION
        sets.append("m.data_version = $data_version")
        params["data_version"] = CURRENT_DATA_VERSION
        if scope is not None:
            memory.scope = scope
            sets.append("m.scope = $scope")
            params["scope"] = scope.as_str()
        if category is not None:
            # Reuse the MemoryUnit normalization (lowercase/trim/default) so the
            # stored category matches what list_categories/merge_categories compare.
            normalized = MemoryUnit(
                content=memory.content, scope=memory.scope, category=category
            ).category
            memory.category = normalized
            sets.append("m.category = $category")
            params["category"] = normalized
        if cross_ref_candidate is not None:
            memory.cross_ref_candidate = cross_ref_candidate
            sets.append("m.cross_ref_candidate = $cross_ref_candidate")
            params["cross_ref_candidate"] = bool(cross_ref_candidate)
        if sets:
            await self._query(
                f"MATCH (m:{MEMORY_LABEL} {{id: $id}}) SET {', '.join(sets)} RETURN m",
                **params,
            )
        return memory

    # -- category registry (emergent-category list/merge, FR-MNT-2/4) ---------

    async def list_categories(self) -> list[tuple[str, int]]:
        """List the free-form categories in use with their active-memory counts."""

        rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL}) WHERE m.valid_to IS NULL "
            "RETURN m.category AS category, count(m) AS n"
        )
        out: list[tuple[str, int]] = []
        for row in rows:
            category = row[0] if row[0] is not None else DEFAULT_CATEGORY
            out.append((str(category), int(row[1])))
        return out

    async def merge_categories(self, source: str, target: str) -> int:
        """Re-label every memory tagged ``source`` to ``target`` (category merge).

        Both sides are normalized (lowercased/trimmed) before matching, mirroring
        :class:`MemoryUnit` category normalization. Idempotent. Returns the count
        of memories re-labeled.
        """

        src = source.strip().lower()
        tgt = target.strip().lower()
        if not tgt:
            tgt = DEFAULT_CATEGORY
        if src == tgt:
            return 0
        rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL}) WHERE m.category = $src "
            "SET m.category = $tgt RETURN count(m) AS n",
            src=src,
            tgt=tgt,
        )
        if not rows:
            return 0
        return int(rows[0][0])

    # -- data-versioning / in-place migration (mnemozine.migrations) ----------

    async def min_data_version(self) -> int:
        """Lowest ``data_version`` across all stored memories AND raw chunks.

        Drives the migration runner / startup hook (compared against
        :data:`~mnemozine.migrations.CURRENT_DATA_VERSION`). Legacy nodes with no
        ``data_version`` property are coalesced to 0 (``coalesce(x.data_version,
        0)``), so any unstamped/version-0 record makes this return 0. An empty
        store returns ``CURRENT_DATA_VERSION`` (nothing to migrate).
        """

        mem_rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL}) "
            "RETURN min(coalesce(m.data_version, 0)) AS v"
        )
        chunk_rows = await self._query(
            f"MATCH (c:{RAW_CHUNK_LABEL}) "
            "RETURN min(coalesce(c.data_version, 0)) AS v"
        )
        versions: list[int] = []
        for rows in (mem_rows, chunk_rows):
            if rows and rows[0] and rows[0][0] is not None:
                versions.append(record_data_version(rows[0][0]))
        if not versions:
            return CURRENT_DATA_VERSION
        return min(versions)

    async def iter_memories_below_version(
        self, version: int
    ) -> AsyncIterator[MemoryUnit]:
        """Stream memory units whose ``data_version`` is below ``version``.

        The selection seam a :class:`~mnemozine.migrations.Migration` uses to find
        the records it must touch (unstamped/legacy nodes count as 0 via
        ``coalesce``). Async generator: iterate, do not ``await``.
        """

        rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL}) "
            "WHERE coalesce(m.data_version, 0) < $version RETURN m",
            version=int(version),
        )
        for row in rows:
            yield self._row_to_memory(self._props(row[0]))

    async def set_data_version(self, ids: Sequence[str], version: int) -> int:
        """Stamp ``data_version = version`` onto the given memory ids in place.

        The explicit stamp a migration uses after migrating a batch of MEMORIES (the
        implicit paths are :meth:`reclassify_memory` /
        :meth:`re_extract_from_raw_chunks`; the raw-chunk analogue is
        :meth:`set_chunk_data_version`). Idempotent; returns the number of records
        updated.
        """

        id_list = list(ids)
        if not id_list:
            return 0
        rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL}) WHERE m.id IN $ids "
            "SET m.data_version = $version RETURN count(m) AS n",
            ids=id_list,
            version=int(version),
        )
        if not rows:
            return 0
        return int(rows[0][0])

    async def iter_chunks_below_version(
        self, version: int
    ) -> AsyncIterator[RawChunk]:
        """Stream raw chunks whose ``data_version`` is below ``version``.

        The raw-chunk analogue of :meth:`iter_memories_below_version`: the selection
        seam a migration uses to find the stale chunks it must re-stamp (legacy
        nodes count as 0 via ``coalesce``). Because :meth:`min_data_version` mins
        over the chunk tier too, even the cheap reclassify path must select and
        stamp these (via :meth:`set_chunk_data_version`). Async generator: iterate,
        do not ``await``.
        """

        rows = await self._query(
            f"MATCH (c:{RAW_CHUNK_LABEL}) "
            "WHERE coalesce(c.data_version, 0) < $version RETURN c",
            version=int(version),
        )
        for row in rows:
            yield self._row_to_raw_chunk(self._props(row[0]))

    async def set_chunk_data_version(
        self, content_hashes: Sequence[str], version: int
    ) -> int:
        """Stamp ``data_version = version`` onto the given raw chunks in place.

        The raw-chunk analogue of :meth:`set_data_version`: the explicit stamp a
        cheap migration uses to advance the chunks it selected via
        :meth:`iter_chunks_below_version` WITHOUT re-extracting them. Chunks are
        keyed by their FR-ING-5 ``content_hash`` (the implicit path is
        :meth:`re_extract_from_raw_chunks`). Idempotent; returns the number of
        chunks updated.
        """

        hash_list = list(content_hashes)
        if not hash_list:
            return 0
        rows = await self._query(
            f"MATCH (c:{RAW_CHUNK_LABEL}) WHERE c.content_hash IN $hashes "
            "SET c.data_version = $version RETURN count(c) AS n",
            hashes=hash_list,
            version=int(version),
        )
        if not rows:
            return 0
        return int(rows[0][0])

    async def close(self) -> None:
        """Close the underlying FalkorDB connection/pool."""

        await self._client.close()


# Re-export the vector index name so the maintenance layer / ops can reference it
# without importing the client module directly.
__all__ = [
    "GraphitiStorageBackend",
    "ContradictsFn",
    "MEMORY_VECTOR_INDEX",
]
