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
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from mnemozine.config import MaintenanceSettings, RetrievalSettings, get_settings
from mnemozine.interfaces import (
    EmbeddingProvider,
    GraphSnapshot,
    GraphSnapshotEdge,
    GraphSnapshotNode,
    MaintenanceReport,
    MemoryPage,
    MemoryView,
    Neighbor,
    RetrievedMemory,
    StoreStats,
    WriteDecision,
    WriteResult,
)
from mnemozine.migrations import CURRENT_DATA_VERSION, record_data_version
from mnemozine.schema.models import (
    DEFAULT_CATEGORY,
    SCOPE_DELIMITER,
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
    CO_MENTION_RELATION,
    CO_MENTION_TYPE,
    ENTITY_LABEL,
    MEMORY_LABEL,
    MEMORY_VECTOR_INDEX,
    MENTIONS_TYPE,
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

# graph_snapshot bounds the per-entity memory_count + idea-seed scan IN CYPHER: a
# popular entity (e.g. 'rust') could otherwise pull every linking memory into
# Python. We cap that scan at a generous multiple of the node cap so a normal
# subgraph is never cut while a pathological slice stays bounded. The
# memory_count/idea-seed view this feeds is itself node-bounded, so bounding the
# linked-memory rows proportionally to the node cap is coherent.
_GRAPH_SNAPSHOT_MEMORY_FACTOR = 25

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

    # -- display-read projection (EMBEDDING-FREE) -----------------------------

    # The exact display fields a MemoryView needs, RETURNed as a Cypher map so the
    # 1024-float embedding is NEVER transferred or parsed for a display read. The
    # nested provenance JSON blob is one scalar property; we decode just the few
    # scalar provenance fields the wire models use rather than the whole vector.
    _VIEW_FIELDS = (
        "id",
        "content",
        "scope",
        "category",
        "cross_ref_candidate",
        "entities",
        "confidence",
        "tier",
        "valid_from",
        "valid_to",
        "last_accessed",
        "access_count",
        "provenance",
    )

    @classmethod
    def _view_projection(cls, var: str = "m") -> str:
        """A Cypher map literal selecting only the display fields of node ``var``.

        Used as the RETURN of every display read so the embedding stays in the
        store. E.g. ``RETURN {id: m.id, content: m.content, ...} AS v``.
        """

        pairs = ", ".join(f"{f}: {var}.{f}" for f in cls._VIEW_FIELDS)
        return "{" + pairs + "}"

    @staticmethod
    def _props_to_view(props: dict[str, Any]) -> MemoryView:
        """Rebuild an embedding-free :class:`MemoryView` from a RETURNed field map."""

        provenance_raw = props.get("provenance")
        provenance = (
            Provenance.model_validate_json(provenance_raw)
            if provenance_raw
            else Provenance.classify_sentinel()
        )
        return MemoryView(
            id=props["id"],
            content=props["content"],
            scope=Scope.parse(props["scope"]),
            category=str(props.get("category") or DEFAULT_CATEGORY),
            cross_ref_candidate=bool(props.get("cross_ref_candidate", False)),
            entities=list(props.get("entities") or []),
            confidence=float(props.get("confidence", 1.0)),
            tier=Tier(props.get("tier", Tier.HOT.value)),
            valid_from=_from_iso(props.get("valid_from")) or datetime.now(UTC),
            valid_to=_from_iso(props.get("valid_to")),
            last_accessed=_from_iso(props.get("last_accessed")),
            access_count=int(props.get("access_count", 0)),
            source=provenance.source,
            session_id=provenance.session_id,
            chunk_hash=provenance.chunk_hash,
            raw_path=provenance.raw_path,
        )

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

        Falls back to a scope-pre-filtered scan + in-process cosine in two cases:
        (1) the vector index is genuinely absent (e.g. a freshly-created graph
        before ``ensure_vector_index``), so the path degrades gracefully instead of
        raising; and (2) the index-backed KNN *starves* — its post-filter yields
        fewer than ``top_k`` rows — AND the in-scope active candidate count is at or
        below ``retrieval.scope_scan_max``, so a small scope buried in a large
        out-of-scope corpus still recalls its matches. Case (2) is gated by a cheap
        embedding-free ``COUNT`` so a *huge* scope never triggers a full embedding
        scan and keeps pure-KNN behaviour.
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

        # STARVATION FALLBACK (FR-RET-2). `queryNodes` post-filters the KNN cut by
        # scope/tier/entity, so a SMALL scope buried in a large out-of-scope corpus
        # can be starved: every nearest neighbour is out-of-scope and filtered away,
        # leaving < top_k rows even though matching in-scope memories exist. When
        # that happens, re-run via the scope-PRE-filtered path (it filters BEFORE
        # ranking and therefore cannot starve) — but only if the in-scope active
        # candidate count is small enough that a full embedding scan is cheap, so a
        # huge scope can never trigger one (the large/normal-scope KNN path is left
        # untouched). The COUNT below transfers NO embeddings.
        if len(scored) >= top_k:
            return scored
        in_scope = await self._count_in_scope(where, params)
        if in_scope <= self._retrieval.scope_scan_max:
            return await self._scoped_query_fallback(
                query_vec, where, params, top_k=top_k
            )
        return scored

    async def _count_in_scope(
        self, where: list[str], params: dict[str, Any]
    ) -> int:
        """Cheap COUNT of active in-scope candidates (NO embeddings transferred).

        Reuses the same scope/validity/tier/entity ``where``/``params`` the KNN
        path built (via :meth:`_scoped_filters`) so the gate counts EXACTLY the
        candidate set the scope-pre-filtered fallback would rank — the gate is one
        index-/scan-cheap aggregate, not an embedding read.
        """

        cypher = (
            f"MATCH (m:{MEMORY_LABEL}) WHERE {' AND '.join(where)} "
            "RETURN count(m) AS n"
        )
        rows = await self._query(cypher, **params)
        return int(rows[0][0]) if rows and rows[0] and rows[0][0] is not None else 0

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

    # -- display reads (WebUI READ surface; EMBEDDING-FREE, Cypher-paged) ------

    async def store_stats(self) -> StoreStats:
        """Aggregate store statistics for the top bar / Dashboard (PRD §4.1).

        A few Cypher aggregation queries (``count`` / grouped ``count``) — never a
        whole-store stream. The embedding is never read: every aggregate is over
        scalar properties. ``by_scope_decision`` keys on whether the stored scope
        string is exactly ``'global'`` (the controlled global/project decision);
        ``by_source`` decodes only the ``source`` field out of each distinct
        provenance JSON blob (one small grouped read, not a per-row parse).
        """

        global_scope = Scope.global_().as_str()
        # 1) memory totals + active/superseded in one pass.
        totals = await self._query(
            f"MATCH (m:{MEMORY_LABEL}) "
            "RETURN count(m) AS total, "
            "sum(CASE WHEN m.valid_to IS NULL THEN 1 ELSE 0 END) AS active"
        )
        total_memories = int(totals[0][0]) if totals and totals[0][0] is not None else 0
        active_count = int(totals[0][1]) if totals and totals[0][1] is not None else 0
        superseded_count = total_memories - active_count

        by_category = await self._count_by(
            f"MATCH (m:{MEMORY_LABEL}) "
            "RETURN coalesce(m.category, $default) AS k, count(m) AS n",
            default=DEFAULT_CATEGORY,
        )
        by_tier = await self._count_by(
            f"MATCH (m:{MEMORY_LABEL}) "
            f"RETURN coalesce(m.tier, $default) AS k, count(m) AS n",
            default=Tier.HOT.value,
        )
        # scope_decision: global iff the stored scope string is exactly 'global'.
        scope_rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL}) "
            "RETURN CASE WHEN m.scope = $global THEN $g ELSE $p END AS k, count(m) AS n",
            **{
                "global": global_scope,
                "g": ScopeDecision.GLOBAL.value,
                "p": ScopeDecision.PROJECT.value,
            },
        )
        by_scope_decision = {
            str(r[0]): int(r[1]) for r in scope_rows if r[0] is not None
        }
        # by_source: group by the distinct provenance JSON blob, decode source once
        # per distinct blob rather than parsing every row.
        src_rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL}) "
            "RETURN m.provenance AS prov, count(m) AS n"
        )
        by_source: dict[str, int] = {}
        for row in src_rows:
            source = self._source_of(row[0])
            by_source[source] = by_source.get(source, 0) + int(row[1])

        entity_rows = await self._query(
            f"MATCH (e:{ENTITY_LABEL}) RETURN count(e) AS n"
        )
        entity_count = (
            int(entity_rows[0][0]) if entity_rows and entity_rows[0][0] is not None else 0
        )
        chunk_rows = await self._query(
            f"MATCH (c:{RAW_CHUNK_LABEL}) RETURN count(c) AS n"
        )
        raw_chunk_count = (
            int(chunk_rows[0][0]) if chunk_rows and chunk_rows[0][0] is not None else 0
        )

        return StoreStats(
            total_memories=total_memories,
            by_category=by_category,
            by_scope_decision=by_scope_decision,
            by_tier=by_tier,
            by_source=by_source,
            active_count=active_count,
            superseded_count=superseded_count,
            entity_count=entity_count,
            raw_chunk_count=raw_chunk_count,
        )

    async def scope_counts(self) -> dict[str, int]:
        """Memory count per exact stored scope (one grouped Cypher count).

        ~A handful of rows (the distinct scopes), never a whole-store stream and
        never the embedding — the scope navigator folds these into its tree.
        """

        return await self._count_by(
            f"MATCH (m:{MEMORY_LABEL}) RETURN m.scope AS k, count(m) AS n"
        )

    async def memory_growth(
        self, *, scope: Scope | None = None, days: int = 14, today: date | None = None
    ) -> list[tuple[str, int]]:
        """Memories created per DAY over the trailing ``days`` window (Dashboard trend).

        A single grouped Cypher count over ``valid_from`` — the canonical creation
        timestamp — never a row stream and never the embedding. ``valid_from`` is
        stored as an ISO-8601 string that sorts lexically, so the window is bounded
        by a ``$since`` ISO lower bound and the day key is its first 10 chars
        (``YYYY-MM-DD``). With a NON-GLOBAL ``scope`` set, the count is filtered to
        that exact scope OR any descendant (a scope-string prefix test using the
        same delimiter ``scope.as_str()`` joins with) so a project rolls up its
        sub-scopes. The GLOBAL root (``scope.segments == []``) is the universal
        ancestor of every scope, so it (like ``scope=None``) counts the WHOLE store
        rather than only the literal ``global`` scope. Returns ``[(day, count)]``
        oldest-first; days with zero memories are absent (the web layer zero-fills
        the gap).
        """

        since = self._growth_since(days, today=today)
        where = ["m.valid_from >= $since"]
        params: dict[str, Any] = {"since": since}
        # Exact-or-descendant roll-up (matches Scope.is_descendant_of and the
        # interface docstring). The GLOBAL ROOT (segments == []) is the universal
        # ancestor of every scope, so it emits NO scope clause and counts the whole
        # store — a string prefix test on "global" would (wrongly) match only the
        # literal 'global' scope and exclude every 'project:*' memory.
        if scope is not None and scope.segments:
            # Non-global scope: the stored scope equals this scope OR starts with
            # this scope's string + the path delimiter (so 'project:P' rolls up
            # 'project:P/auth' but never the unrelated 'project:Pulse'). The prefix
            # delimiter MUST be the same one scope.as_str() joins segments with, so
            # a config-overridden delimiter keeps roll-up matching sub-scopes.
            scope_str = scope.as_str()
            where.append("(m.scope = $scope OR m.scope STARTS WITH $scope_prefix)")
            params["scope"] = scope_str
            params["scope_prefix"] = f"{scope_str}{SCOPE_DELIMITER}"
        rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL}) WHERE {' AND '.join(where)} "
            "RETURN left(toString(m.valid_from), 10) AS day, count(m) AS n "
            "ORDER BY day ASC",
            **params,
        )
        return [(str(r[0]), int(r[1])) for r in rows if r[0] is not None]

    @staticmethod
    def _growth_since(days: int, *, today: date | None = None) -> str:
        """ISO lower bound: midnight UTC ``days - 1`` days before ``today``.

        Spans exactly ``days`` calendar days up to and including ``today`` (the UTC
        anchor day; defaults to ``datetime.now(UTC).date()`` when the caller does
        not pin one). Returned as an ISO string so it compares lexically against the
        stored ``valid_from``.
        """

        span = days if days >= 1 else 1
        anchor = today if today is not None else datetime.now(UTC).date()
        start = anchor - timedelta(days=span - 1)
        return datetime(start.year, start.month, start.day, tzinfo=UTC).isoformat()

    async def _count_by(self, cypher: str, **params: Any) -> dict[str, int]:
        """Run a grouped ``RETURN <key> AS k, count(...) AS n`` into a count map."""

        rows = await self._query(cypher, **params)
        return {str(r[0]): int(r[1]) for r in rows if r[0] is not None}

    @staticmethod
    def _source_of(provenance_raw: Any) -> str:
        """Decode just the ``source`` scalar out of a stored provenance JSON blob."""

        if not provenance_raw:
            return Provenance.classify_sentinel().source
        return Provenance.model_validate_json(provenance_raw).source

    def _memory_filter_clauses(
        self,
        *,
        category: str | None,
        scope: Scope | None,
        tier: Tier | None,
        entity: str | None,
        source: str | None,
        active: bool | None,
        q: str | None,
    ) -> tuple[list[str], dict[str, Any]]:
        """Build the shared WHERE clauses + params for the Memories table filters.

        Shared by :meth:`query_memories`'s page query and its ``COUNT`` so the two
        stay in lock-step. Filtering is pushed entirely into Cypher (no Python
        post-filter). ``category`` is normalized like
        :class:`~mnemozine.schema.models.MemoryUnit` (lowercased/trimmed); ``q``
        and ``entity`` are matched case-insensitively.
        """

        where: list[str] = []
        params: dict[str, Any] = {}
        if category is not None:
            where.append("m.category = $category")
            params["category"] = category.strip().lower()
        if scope is not None:
            where.append("m.scope = $scope")
            params["scope"] = scope.as_str()
        if tier is not None:
            where.append("m.tier = $tier")
            params["tier"] = tier.value
        if source is not None:
            where.append("m.provenance CONTAINS $source_needle")
            # Provenance is a JSON blob; match the source field token precisely so
            # 'openai' does not match a substring of another field's value.
            params["source_needle"] = f'"source":"{source}"'
        if active is not None:
            where.append(
                "m.valid_to IS NULL" if active else "m.valid_to IS NOT NULL"
            )
        if entity is not None:
            where.append(
                "any(e IN m.entities WHERE toLower(e) = $entity)"
            )
            params["entity"] = entity.lower()
        if q:
            where.append("toLower(m.content) CONTAINS $q")
            params["q"] = q.lower()
        return where, params

    async def query_memories(
        self,
        *,
        category: str | None = None,
        scope: Scope | None = None,
        tier: Tier | None = None,
        entity: str | None = None,
        source: str | None = None,
        active: bool | None = None,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> MemoryPage:
        """Filter / order / page the Memories table IN CYPHER (PRD §4.2).

        All filters AND-combine in the WHERE; ordering is newest-first by
        ``valid_from``; paging is ``SKIP $offset LIMIT $limit`` — all in FalkorDB.
        The page query RETURNs only the embedding-free display field map (never the
        node, never the vector); ``total`` is a cheap ``count(m)`` over the same
        filtered set so the caller never re-scans the store.
        """

        where, params = self._memory_filter_clauses(
            category=category,
            scope=scope,
            tier=tier,
            entity=entity,
            source=source,
            active=active,
            q=q,
        )
        clause = f" WHERE {' AND '.join(where)}" if where else ""

        total_rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL}){clause} RETURN count(m) AS n", **params
        )
        total = int(total_rows[0][0]) if total_rows and total_rows[0][0] is not None else 0

        page_rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL}){clause} "
            f"RETURN {self._view_projection('m')} AS v "
            "ORDER BY v.valid_from DESC SKIP $offset LIMIT $limit",
            offset=offset,
            limit=limit,
            **params,
        )
        items = [self._props_to_view(self._props(row[0])) for row in page_rows]
        return MemoryPage(items=items, total=total)

    async def get_memory_display(self, memory_id: str) -> MemoryView | None:
        """Read ONE memory for display, EMBEDDING-FREE (detail / non-vector read).

        RETURNs the display field map only (never the node / its vector). Returns
        ``None`` for an unknown id.
        """

        rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL} {{id: $id}}) "
            f"RETURN {self._view_projection('m')} AS v",
            id=memory_id,
        )
        if not rows:
            return None
        return self._props_to_view(self._props(rows[0][0]))

    async def graph_snapshot(
        self,
        *,
        scope: Scope | None = None,
        entity: str | None = None,
        entity_type: str | None = None,
        include_idea_seeds: bool = True,
        node_limit: int = 200,
    ) -> GraphSnapshot:
        """A bounded entity/idea-seed subgraph for the explorer (PRD §4.4).

        Entity nodes are selected (optionally centered on ``entity``'s one-hop
        neighborhood, optionally filtered by ``entity_type``) and capped at
        ``node_limit`` IN CYPHER (via a ``LIMIT $cap+1`` sentinel over-fetch so
        ``truncated`` reports "more exist" accurately, never a false positive at
        exactly ``node_limit``). Structural edges among the kept entities come from
        a SINGLE aggregate edge query (no per-entity N+1). When
        ``include_idea_seeds``, cross-ref-candidate memories in ``scope`` are added
        as ``idea_seed`` nodes (embedding-free projection) with ``mentions`` edges
        to the entities they reference; per-entity ``memory_count`` is the in-scope
        link count, computed in Cypher and BOUNDED (the scan that feeds it is
        ``LIMIT``-capped so a popular entity cannot pull an unbounded slice).
        """

        # Sentinel over-fetch: ask for one more than the cap so we can tell a
        # full-but-not-truncated page (== node_limit) from a truncated one (> it),
        # then trim back to node_limit. Avoids the boundary false-positive where
        # exactly node_limit entities would otherwise report truncated=True.
        over_cap = node_limit + 1

        # --- entity nodes (bounded, optional center + type filter) -----------
        ent_where: list[str] = []
        ent_params: dict[str, Any] = {"cap": over_cap}
        if entity_type is not None:
            ent_where.append("e.type = $entity_type")
            ent_params["entity_type"] = entity_type

        if entity is not None:
            center = await self.get_entity(entity)
            if center is None:
                return GraphSnapshot(nodes=[], edges=[], truncated=False)
            # Center + its one-hop neighbors in a single traversal (no N+1).
            type_clause = " AND o.type = $entity_type" if entity_type is not None else ""
            ent_params["center"] = center.id
            ent_rows = await self._query(
                f"MATCH (c:{ENTITY_LABEL} {{id: $center}}) "
                f"OPTIONAL MATCH (c)-[:{RELATES_TYPE}]-(o:{ENTITY_LABEL}) "
                f"WHERE o IS NULL OR (true{type_clause}) "
                "WITH collect(DISTINCT c) + collect(DISTINCT o) AS ents "
                "UNWIND ents AS e WITH DISTINCT e WHERE e IS NOT NULL "
                "RETURN e LIMIT $cap",
                **ent_params,
            )
        else:
            # Default (no-center) selection: a DEGREE-RANKED bounded slice over the
            # structural layers (RELATES + CO_MENTIONS) instead of an arbitrary
            # ``LIMIT $cap`` slice, so the snapshot surfaces the real connected
            # structure — the highest-degree entities and the neighbors they share
            # an edge with render as a connected subgraph, not isolated nodes. The
            # degree is a single Cypher-side aggregate (OPTIONAL MATCH + count(r)),
            # so no per-node N+1 and no full edge scan beyond the bounded top slice;
            # the existing RELATES + CO_MENTIONS aggregates below then connect the
            # kept set. Tie-break on ``e.id`` keeps the slice deterministic.
            clause = f" WHERE {' AND '.join(ent_where)}" if ent_where else ""
            ent_rows = await self._query(
                f"MATCH (e:{ENTITY_LABEL}){clause} "
                f"OPTIONAL MATCH (e)-[r:{RELATES_TYPE}|{CO_MENTION_TYPE}]-"
                f"(:{ENTITY_LABEL}) "
                "WITH e, count(r) AS deg "
                "ORDER BY deg DESC, e.id "
                "RETURN e LIMIT $cap",
                **ent_params,
            )

        all_entities = [self._row_to_entity(row[0]) for row in ent_rows]
        # The sentinel row (node_limit+1-th) means more entities exist than the cap.
        truncated = len(all_entities) > node_limit
        entities = all_entities[:node_limit]
        entity_by_id = {e.id: e for e in entities}
        entity_id_by_name = {e.canonical_name.lower(): e.id for e in entities}
        kept_ids = list(entity_by_id)

        # --- per-entity in-scope memory_count + idea-seed memories -----------
        # BOUNDED IN CYPHER: a popular entity (e.g. 'rust') could link a huge slice
        # of the store, so this scan is capped at $mem_cap (a generous multiple of
        # the node cap). The result feeds the node-bounded memory_count + idea-seed
        # view, so bounding the linked-memory rows proportionally is coherent and
        # keeps /api/graph flat as the store grows.
        mem_where = ["any(e IN m.entities WHERE toLower(e) IN $names)"]
        mem_params: dict[str, Any] = {
            "names": list(entity_id_by_name),
            "mem_cap": node_limit * _GRAPH_SNAPSHOT_MEMORY_FACTOR,
        }
        if scope is not None:
            mem_where.append("m.scope = $scope")
            mem_params["scope"] = scope.as_str()
        memory_count: dict[str, int] = dict.fromkeys(kept_ids, 0)
        idea_nodes: list[GraphSnapshotNode] = []
        mentions_edges: list[GraphSnapshotEdge] = []
        if kept_ids:
            mem_rows = await self._query(
                f"MATCH (m:{MEMORY_LABEL}) WHERE {' AND '.join(mem_where)} "
                f"RETURN {self._view_projection('m')} AS v, "
                "coalesce(m.cross_ref_candidate, false) AS seed "
                "LIMIT $mem_cap",
                **mem_params,
            )
            for row in mem_rows:
                view = self._props_to_view(self._props(row[0]))
                linked_ids = {
                    entity_id_by_name[name.lower()]
                    for name in view.entities
                    if name.lower() in entity_id_by_name
                }
                for eid in linked_ids:
                    memory_count[eid] = memory_count.get(eid, 0) + 1
                is_seed = bool(row[1]) if len(row) > 1 else False
                if include_idea_seeds and is_seed and linked_ids:
                    idea_nodes.append(self._idea_seed_node(view))
                    for eid in linked_ids:
                        mentions_edges.append(
                            GraphSnapshotEdge(
                                id=f"mentions:{view.id}:{eid}",
                                source=view.id,
                                target=eid,
                                relation="mentions",
                                weight=1.0,
                                active=view.is_active,
                                kind="mentions",
                            )
                        )

        entity_nodes = [
            GraphSnapshotNode(
                id=e.id,
                label=e.canonical_name,
                kind="entity",
                entity_type=e.type,
                memory_count=memory_count.get(e.id, 0),
            )
            for e in entities
        ]

        # --- structural edges among kept entities (SINGLE aggregate query) ---
        struct_edges: list[GraphSnapshotEdge] = []
        if kept_ids:
            edge_rows = await self._query(
                f"MATCH (a:{ENTITY_LABEL})-[r:{RELATES_TYPE}]->(b:{ENTITY_LABEL}) "
                "WHERE a.id IN $ids AND b.id IN $ids "
                "RETURN a.id AS source, b.id AS target, r",
                ids=kept_ids,
            )
            seen: set[str] = set()
            for row in edge_rows:
                # Endpoints come from the matched topology (a.id/b.id), which is the
                # source of truth even for edges missing the from/to props.
                source, target = row[0], row[1]
                edge = self._row_to_edge(row[2], from_entity=source, to_entity=target)
                if edge.id in seen:
                    continue
                seen.add(edge.id)
                struct_edges.append(
                    GraphSnapshotEdge(
                        id=edge.id,
                        source=source,
                        target=target,
                        relation=edge.relation,
                        weight=edge.weight,
                        active=edge.is_active,
                        kind="relates",
                    )
                )

        # --- co-mention edges among kept entities (SECOND aggregate query) ---
        # The weighted entity-entity co-mention layer (kind='co_mention'), derived
        # from MNEMOZINE_MENTIONS by the CoMentionJob. A single aggregate over the
        # kept ids (no N+1), mirroring the RELATES aggregate above but on the
        # distinct CO_MENTION_TYPE so the two layers stay physically separable.
        co_mention_edges: list[GraphSnapshotEdge] = []
        if kept_ids:
            co_rows = await self._query(
                f"MATCH (a:{ENTITY_LABEL})-[r:{CO_MENTION_TYPE}]->(b:{ENTITY_LABEL}) "
                "WHERE a.id IN $ids AND b.id IN $ids "
                "RETURN a.id AS source, b.id AS target, r",
                ids=kept_ids,
            )
            co_seen: set[str] = set()
            for row in co_rows:
                source, target = row[0], row[1]
                props = self._props(row[2])
                edge_id = props.get("id") or f"comention:{source}:{target}"
                if edge_id in co_seen:
                    continue
                co_seen.add(edge_id)
                co_mention_edges.append(
                    GraphSnapshotEdge(
                        id=edge_id,
                        source=source,
                        target=target,
                        relation=props.get("relation", CO_MENTION_RELATION),
                        weight=float(props.get("weight", 1.0)),
                        active=props.get("valid_to") is None,
                        kind="co_mention",
                    )
                )

        return GraphSnapshot(
            nodes=entity_nodes + idea_nodes,
            edges=struct_edges + co_mention_edges + mentions_edges,
            truncated=truncated,
        )

    @staticmethod
    def _idea_seed_node(view: MemoryView) -> GraphSnapshotNode:
        """Build a compact ``idea_seed`` node from an embedding-free memory view."""

        snippet = " ".join(view.content.split())
        if len(snippet) > 80:
            snippet = snippet[:79].rstrip() + "…"
        return GraphSnapshotNode(
            id=view.id,
            label=snippet,
            kind="idea_seed",
            scope=view.scope.as_str(),
            memory_count=1,
        )

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
        pulls more than it asked for. ``unused_since`` keeps units last *used*
        before the cutoff, anchored on ``valid_from`` (ingestion time) when
        ``last_accessed`` is null — i.e. a never-recalled memory is "unused since
        it was ingested", NOT "unused since the beginning of time". This mirrors
        :func:`mnemozine.maintenance.decay.decay_score`'s recency anchor
        (``last_accessed or valid_from``) so the FR-MNT-3 decay sweep's SELECTION
        filter is consistent with its SCORE: a freshly-ingested, never-recalled
        memory is NOT swept until its creation time itself ages past the cutoff.
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
            # Anchor a null last_accessed on valid_from (ingestion time), matching
            # decay_score's recency anchor. valid_from and last_accessed are both
            # stored as ISO-8601 strings, so the lexical < against $unused_since
            # (also _to_iso) is ordering-preserving (same pattern as memory_growth).
            where.append("coalesce(m.last_accessed, m.valid_from) < $unused_since")
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
        """Insert or update an entity node, keyed on id (FR-EXT-2).

        Identity-by-id: the MERGE key is the entity's ``id``, so this is the
        low-level write used once an id is known (the resolve-by-name decision
        lives in :meth:`resolve_or_create_entity`, which calls through here to
        create). Every write ALSO maintains the storage-only ``name_key =
        toLower(canonical_name)`` property so the
        :data:`~mnemozine.storage.graphiti_client.ENTITY_NAME_KEY_INDEX` invariant
        holds for every node — ``name_key`` is NOT a field on
        :class:`~mnemozine.schema.models.Entity` (storage-only, like a memory's
        ``data_version``) and :meth:`_row_to_entity` never reads it back.
        """

        await self._query(
            f"MERGE (e:{ENTITY_LABEL} {{id: $id}}) "
            "SET e.canonical_name = $canonical_name, e.aliases = $aliases, e.type = $type, "
            "e.name_key = toLower($canonical_name) "
            "RETURN e",
            **self._entity_props(entity),
        )
        return entity

    async def resolve_or_create_entity(self, entity: Entity) -> Entity:
        """Resolve an entity by normalized name, creating it only if absent.

        The identity-by-normalized-name seam (the fix for the duplicate-entity
        leak): looks up the existing entity node whose ``name_key`` equals
        ``toLower(entity.canonical_name)`` — case-insensitive and index-backed via
        :data:`~mnemozine.storage.graphiti_client.ENTITY_NAME_KEY_INDEX`. When a
        node already exists for that normalized name this RETURNs the **stored**
        entity (its id is what ``services._persist`` binds edges to) WITHOUT minting
        a new node, folding ``entity.canonical_name`` / ``entity.aliases`` into the
        survivor's aliases when they differ (reusing the same alias-update write as
        :meth:`merge_entities`). When no such node exists it creates one via the
        id-keyed :meth:`upsert_entity` (which also sets ``name_key``).

        Idempotent (FR-MNT-5): resolving the same normalized name twice returns the
        same id and never increases the node count.
        """

        rows = await self._query(
            f"MATCH (e:{ENTITY_LABEL}) WHERE e.name_key = toLower($canonical_name) "
            "RETURN e LIMIT 1",
            canonical_name=entity.canonical_name,
        )
        if not rows:
            # No node for this normalized name yet: create it id-keyed.
            return await self.upsert_entity(entity)

        stored = self._row_to_entity(rows[0][0])
        # Fold the incoming canonical_name + aliases into the survivor's aliases
        # when they add anything new (a different-cased spelling, or a fresh alias),
        # so a later get_entity by either spelling resolves to the same node. Reuses
        # the merge_entities alias-update write (also re-asserts name_key on the
        # survivor for safety).
        incoming = {entity.canonical_name, *entity.aliases}
        merged_aliases = sorted({*stored.aliases, *incoming} - {stored.canonical_name})
        if merged_aliases != sorted(stored.aliases):
            stored.aliases = merged_aliases
            await self._query(
                f"MATCH (t:{ENTITY_LABEL} {{id: $tgt}}) "
                "SET t.aliases = $aliases, t.name_key = toLower(t.canonical_name) "
                "RETURN t",
                tgt=stored.id,
                aliases=merged_aliases,
            )
        return stored

    async def backfill_entity_name_keys(self) -> int:
        """Backfill ``name_key = toLower(canonical_name)`` on entities missing it.

        The STRUCTURAL half of the v2 entity-identity migration
        (:class:`mnemozine.migrations.entity_name_key.EntityNameKeyMigration`):
        first ensures the
        :data:`~mnemozine.storage.graphiti_client.ENTITY_NAME_KEY_INDEX` range index
        exists (idempotent), then runs one idempotent Cypher SET pass that touches
        ONLY entity nodes whose ``name_key`` is still unset
        (``WHERE e.name_key IS NULL``). Re-runnable — a second pass finds nothing
        unset and updates zero nodes. Returns the number of nodes stamped.
        """

        await self._client.ensure_entity_name_index()
        rows = await self._query(
            f"MATCH (e:{ENTITY_LABEL}) WHERE e.name_key IS NULL "
            "SET e.name_key = toLower(e.canonical_name) RETURN count(e) AS n",
        )
        if not rows:
            return 0
        return int(rows[0][0])

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

        Repoints the source's edges onto the target across ALL three edge types —
        the LLM-extracted ``MNEMOZINE_RELATES`` relations (both directions), the
        ``MNEMOZINE_MENTIONS`` memory->entity edges (incoming), and the weighted
        ``MNEMOZINE_CO_MENTIONS`` entity-entity layer (both directions, self-loops
        dropped) — so no edge type is orphaned when a duplicate entity is folded
        away. Then folds the source's canonical name + aliases into the target's
        aliases, and deletes the now-redundant source node so the graph does not
        fragment across duplicate entities. Every repoint is MERGE-onto-target +
        DELETE-source, so re-running (with the source already gone) is a no-op and
        never creates a duplicate parallel edge. Memories are never deleted.
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

        # Repoint the MENTIONS layer (memory -> entity, only INCOMING to the source
        # entity): every memory that mentioned the duplicate now mentions the
        # survivor. MERGE collapses a memory that mentioned BOTH onto one edge, so
        # no duplicate is created (idempotent — a re-run with the source already
        # gone matches nothing).
        await self._query(
            f"MATCH (m:{MEMORY_LABEL})-[r:{MENTIONS_TYPE}]->(s:{ENTITY_LABEL} {{id: $src}}) "
            f"MATCH (t:{ENTITY_LABEL} {{id: $tgt}}) "
            f"MERGE (m)-[:{MENTIONS_TYPE}]->(t) "
            "DELETE r",
            src=source_id,
            tgt=target_id,
        )

        # Repoint the CO_MENTION layer (weighted entity-entity) onto the survivor in
        # CANONICAL direction (lo.id < hi.id). Co-mention is an UNORDERED relation, so
        # every stored edge keeps from.id < to.id; the repoint MUST re-canonicalize or
        # a survivor.id > neighbor.id edge would survive and the next CoMentionJob run
        # would MERGE the canonical reverse, creating a duplicate parallel edge. The
        # undirected ``(s)-[r]-(o)`` match folds an incident edge from EITHER direction;
        # ``o.id <> $tgt`` drops the would-be self-loop; MERGE onto the canonical pair
        # folds onto the survivor's existing edge (highest weight kept) so no duplicate
        # co-mention edge remains, matching upsert_co_mention's canonical re-assert.
        await self._query(
            f"MATCH (s:{ENTITY_LABEL} {{id: $src}})-[r:{CO_MENTION_TYPE}]-(o:{ENTITY_LABEL}) "
            "WHERE o.id <> $tgt "
            "WITH r, o, "
            "CASE WHEN $tgt < o.id THEN $tgt ELSE o.id END AS lo, "
            "CASE WHEN $tgt < o.id THEN o.id ELSE $tgt END AS hi "
            f"MATCH (lon:{ENTITY_LABEL} {{id: lo}}) "
            f"MATCH (hin:{ENTITY_LABEL} {{id: hi}}) "
            f"MERGE (lon)-[nr:{CO_MENTION_TYPE} {{relation: r.relation}}]->(hin) "
            "SET nr.weight = CASE WHEN nr.weight IS NULL OR r.weight > nr.weight "
            "THEN r.weight ELSE nr.weight END, "
            "nr.shared = coalesce(nr.shared, r.shared), "
            "nr.from_entity = lo, nr.to_entity = hi, "
            "nr.id = coalesce(nr.id, r.id), "
            "nr.valid_from = coalesce(nr.valid_from, r.valid_from), nr.valid_to = NULL "
            "DELETE r",
            src=source_id,
            tgt=target_id,
        )
        # Drop any co-mention edge that became a self-loop (the survivor and the
        # duplicate were directly co-mentioned): a node never co-mentions itself.
        await self._query(
            f"MATCH (s:{ENTITY_LABEL} {{id: $src}})-[r:{CO_MENTION_TYPE}]-"
            f"(t:{ENTITY_LABEL} {{id: $tgt}}) DELETE r",
            src=source_id,
            tgt=target_id,
        )

        merged_aliases = sorted({*target.aliases, source.canonical_name, *source.aliases})
        target.aliases = merged_aliases
        # Re-assert name_key on the survivor alongside the alias fold so the
        # unique-normalized-name invariant (and the ENTITY_NAME_KEY_INDEX) holds
        # post-merge — a folded-away duplicate must not leave the survivor without
        # a name_key (e.g. if it predates the v2 backfill).
        await self._query(
            f"MATCH (t:{ENTITY_LABEL} {{id: $tgt}}) "
            "SET t.aliases = $aliases, t.name_key = toLower(t.canonical_name) RETURN t",
            tgt=target_id,
            aliases=merged_aliases,
        )
        await self._query(
            f"MATCH (s:{ENTITY_LABEL} {{id: $src}}) DELETE s", src=source_id
        )
        return target

    async def persist_mentions(self) -> int:
        """Persist (memory)-[:MNEMOZINE_MENTIONS]->(entity) edges from ``m.entities``.

        The resolution is computed **deterministically in Python** (mirroring the
        ``entity_id_by_name`` map the graph-snapshot path already builds): load
        every entity once into a lowered ``{canonical_name|alias -> id}`` map, then
        stream the memories and resolve each memory's ``m.entities`` names against
        that map into exact ``(memory_id, entity_id)`` pairs. The pairs are then
        asserted by a single ``UNWIND``-of-pairs ``MERGE`` **keyed on the exact node
        ids** — MERGE (never CREATE) on the id-bound endpoint pair, so a re-run
        asserts the same edges and creates nothing new (FR-MNT-5). Returns the
        number of distinct mention edges asserted (created-or-matched).

        Why not a pure Cypher cross-product: an all-entities × all-memories
        ``MATCH ... WHERE any(...)`` over the unindexed ``canonical_name`` is
        non-deterministic at this store's scale on FalkorDB — each run committed a
        drifting subset of the true pairs, so the stored mention count crept upward
        and never reached a stable fixpoint. Resolving in Python over a single
        deterministic entity scan and MERGEing by exact id removes that drift: the
        matched pair set is the complete fixpoint and a re-run is a true no-op.
        """

        # 1) Deterministic lowered name/alias -> entity-id map (one entity scan).
        entity_id_by_key: dict[str, str] = {}
        async for entity in self.iter_entities():
            entity_id_by_key.setdefault(entity.canonical_name.lower(), entity.id)
            for alias in entity.aliases:
                entity_id_by_key.setdefault(alias.lower(), entity.id)

        if not entity_id_by_key:
            return 0

        # 2) Resolve each memory's entity-names to exact (memory_id, entity_id)
        #    pairs, deduped as a set so the same pair is asserted once.
        pairs: set[tuple[str, str]] = set()
        async for memory in self.iter_memories():
            for name in memory.entities:
                eid = entity_id_by_key.get(name.lower())
                if eid is not None:
                    pairs.add((memory.id, eid))

        if not pairs:
            return 0

        # 3) Idempotent id-keyed MERGE of the resolved pairs (single set-based
        #    statement). Endpoints matched by exact id, so re-running creates none.
        rows = await self._query(
            "UNWIND $pairs AS pair "
            f"MATCH (m:{MEMORY_LABEL} {{id: pair[0]}}) "
            f"MATCH (e:{ENTITY_LABEL} {{id: pair[1]}}) "
            f"MERGE (m)-[r:{MENTIONS_TYPE}]->(e) "
            "RETURN count(r) AS n",
            pairs=[[mid, eid] for mid, eid in sorted(pairs)],
        )
        if not rows or rows[0][0] is None:
            return 0
        return int(rows[0][0])

    async def add_memory_mentions(
        self, memory_id: str, entity_ids: Sequence[str]
    ) -> int:
        """Assert ``(memory)-[:MNEMOZINE_MENTIONS]->(entity)`` edges at ingest time.

        The per-memory inline-mentions seam — what connects a freshly ingested
        memory to its entities the instant it lands instead of waiting for the 3 AM
        batch :meth:`persist_mentions`. Reuses that method's id-keyed MERGE write
        path: ``UNWIND`` the already-resolved ``entity_ids`` and MERGE a
        :data:`MENTIONS_TYPE` edge from the memory to each entity **keyed on the
        exact node ids** (never a blind CREATE), so a re-call asserts the same edges
        and creates none (FR-MNT-5). Called by ``services._persist`` after each
        :meth:`upsert_memory` with the ids already built from
        :meth:`resolve_or_create_entity`, so no extra entity reads are needed. The
        batch :meth:`persist_mentions` stays a whole-store backstop, so this is
        purely additive. Returns the number of edges asserted (created-or-matched).
        """

        # Dedup + drop falsy ids so the same edge is asserted once and a memory with
        # no resolved entities is a no-op (empty UNWIND).
        ids = sorted({eid for eid in entity_ids if eid})
        if not ids:
            return 0
        rows = await self._query(
            "UNWIND $entity_ids AS eid "
            f"MATCH (m:{MEMORY_LABEL} {{id: $memory_id}}) "
            f"MATCH (e:{ENTITY_LABEL} {{id: eid}}) "
            f"MERGE (m)-[r:{MENTIONS_TYPE}]->(e) "
            "RETURN count(r) AS n",
            memory_id=memory_id,
            entity_ids=ids,
        )
        if not rows or rows[0][0] is None:
            return 0
        return int(rows[0][0])

    async def co_mention_pairs(
        self, *, min_shared: int = 2
    ) -> list[tuple[str, str, int]]:
        """Entity id pairs co-occurring in >= ``min_shared`` shared memories.

        Read-only: derived from the ``MNEMOZINE_MENTIONS`` layer. A single
        set-based aggregate matches every memory that mentions BOTH endpoints —
        ``(a)<-[:MNEMOZINE_MENTIONS]-(m)-[:MNEMOZINE_MENTIONS]->(b)`` with
        ``a.id < b.id`` so each unordered pair appears once — counts the distinct
        shared memories, and keeps only pairs at or above ``min_shared``. The
        a<b ordering makes the output stable for the :class:`CoMentionJob`'s
        deterministic ranking/cap (FR-MNT-5). No weighting happens here.
        """

        rows = await self._query(
            f"MATCH (a:{ENTITY_LABEL})<-[:{MENTIONS_TYPE}]-(m:{MEMORY_LABEL})"
            f"-[:{MENTIONS_TYPE}]->(b:{ENTITY_LABEL}) "
            "WHERE a.id < b.id "
            "WITH a.id AS aid, b.id AS bid, count(DISTINCT m) AS shared "
            "WHERE shared >= $min_shared "
            "RETURN aid, bid, shared",
            min_shared=int(min_shared),
        )
        out: list[tuple[str, str, int]] = []
        for row in rows:
            if row[0] is None or row[1] is None:
                continue
            out.append((str(row[0]), str(row[1]), int(row[2])))
        return out

    async def entity_mention_counts(self) -> dict[str, int]:
        """``{entity_id: distinct-memory mention count}`` over ``MNEMOZINE_MENTIONS``.

        The document-frequency the :class:`CoMentionJob` needs for the TF-IDF-style
        hub down-weight: a cheap grouped count of the distinct memories mentioning
        each entity.
        """

        rows = await self._query(
            f"MATCH (m:{MEMORY_LABEL})-[:{MENTIONS_TYPE}]->(e:{ENTITY_LABEL}) "
            "RETURN e.id AS eid, count(DISTINCT m) AS df"
        )
        counts: dict[str, int] = {}
        for row in rows:
            if row[0] is None:
                continue
            counts[str(row[0])] = int(row[1])
        return counts

    async def upsert_co_mention(
        self, from_entity: str, to_entity: str, *, weight: float, shared: int
    ) -> Edge:
        """Idempotently MERGE a weighted entity-entity co-mention edge (FR-MNT-5).

        MERGEs on ``(from, to)`` over the distinct ``MNEMOZINE_CO_MENTIONS`` type
        with ``relation = CO_MENTION_RELATION`` and re-asserts ``weight`` + a
        ``shared`` count (SET, not sum) so a re-run is idempotent. Mirrors
        :meth:`upsert_edge` but on the co-mention type so the edges stay separable
        from LLM-extracted ``MNEMOZINE_RELATES``. Returns the stored edge.
        """

        now = datetime.now(UTC)
        # Co-mention is an UNORDERED relation; store it canonically (lo.id < hi.id) so
        # the directed MERGE is direction-stable and a re-run (even after a merge that
        # reversed an endpoint) matches the same edge instead of creating a parallel
        # reversed one. ``co_mention_pairs`` already yields a < b, so this is
        # belt-and-suspenders that also lets the MERGE match a survivor edge that a
        # prior merge_entities repoint produced.
        lo, hi = (
            (from_entity, to_entity)
            if from_entity <= to_entity
            else (to_entity, from_entity)
        )
        edge_id = f"comention:{lo}:{hi}"
        await self._query(
            f"MATCH (a:{ENTITY_LABEL} {{id: $lo}}) "
            f"MATCH (b:{ENTITY_LABEL} {{id: $hi}}) "
            f"MERGE (a)-[r:{CO_MENTION_TYPE} {{relation: $relation}}]->(b) "
            "SET r.weight = $weight, r.shared = $shared, r.from_entity = $lo, "
            "r.to_entity = $hi, r.id = coalesce(r.id, $id), "
            "r.valid_from = coalesce(r.valid_from, $valid_from), r.valid_to = NULL "
            "RETURN r",
            lo=lo,
            hi=hi,
            relation=CO_MENTION_RELATION,
            weight=float(weight),
            shared=int(shared),
            id=edge_id,
            valid_from=_to_iso(now),
        )
        return Edge(
            id=edge_id,
            from_entity=lo,
            to_entity=hi,
            relation=CO_MENTION_RELATION,
            weight=float(weight),
            valid_from=now,
            valid_to=None,
        )

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
            "RETURN o, r, startNode(r).id AS src, endNode(r).id AS dst "
            "ORDER BY r.weight DESC LIMIT $cap",
            id=resolved.id,
            cap=cap,
        )
        out: list[Neighbor] = []
        for row in rows:
            other = self._row_to_entity(row[0])
            # Endpoints from the relationship's start/end nodes (topology), so an
            # edge missing the from/to props still resolves correctly.
            edge = self._row_to_edge(row[1], from_entity=row[2], to_entity=row[3])
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
    def _row_to_edge(
        rel: Any, *, from_entity: str | None = None, to_entity: str | None = None
    ) -> Edge:
        """Rebuild an :class:`Edge` from a stored relationship, endpoint-tolerant.

        The from/to entity ids are REDUNDANT with the graph topology (they are the
        relationship's start/end node ids), so older edges may not carry the
        ``from_entity``/``to_entity`` *properties* (the 2026-06-14 backfill and the
        merge-rewired edges store only ``{id, relation, weight, valid_from}``). To
        avoid a ``KeyError`` on those edges, the caller passes the real endpoints
        from the matched topology (``a.id``/``b.id`` or ``startNode(r).id`` /
        ``endNode(r).id``) and we prefer them over the stored props. This is a
        read-side fix only: no existing edge is mutated.

        ``id`` is likewise tolerant — it is only used for dedup, so when absent we
        synthesize a stable one from the available endpoints + relation.
        """

        props = GraphitiStorageBackend._props(rel)
        frm = from_entity if from_entity is not None else props.get("from_entity", "")
        to = to_entity if to_entity is not None else props.get("to_entity", "")
        relation = props.get("relation", "relates")
        edge_id = props.get("id")
        if not edge_id:
            # Synthesize a stable dedup key from whatever endpoints we have.
            edge_id = f"{frm}:{to}:{relation}"
        return Edge(
            id=edge_id,
            from_entity=frm,
            to_entity=to,
            relation=relation,
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
            # We already know the endpoints from the incoming edge param; pass them
            # so the re-assert read never depends on the stored from/to props.
            current = self._row_to_edge(
                existing[0][0],
                from_entity=edge.from_entity,
                to_entity=edge.to_entity,
            )
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
            "RETURN r, startNode(r).id AS src, endNode(r).id AS dst",
            id=resolved.id,
        )
        # Endpoints from the relationship's start/end nodes (topology) so an edge
        # missing the from/to props still resolves correctly.
        return [
            self._row_to_edge(row[0], from_entity=row[1], to_entity=row[2])
            for row in rows
        ]

    async def prune_edge(self, edge_id: str, *, at: datetime | None = None) -> Edge:
        """Close a low-weight edge's validity window (FR-MNT-4; retained, not deleted)."""

        ts = at or datetime.now(UTC)
        rows = await self._query(
            f"MATCH ()-[r:{RELATES_TYPE} {{id: $id}}]-() SET r.valid_to = $valid_to "
            "RETURN r, startNode(r).id AS src, endNode(r).id AS dst",
            id=edge_id,
            valid_to=_to_iso(ts),
        )
        if not rows:
            raise KeyError(edge_id)
        # Endpoints from the relationship's start/end nodes (topology) so an edge
        # missing the from/to props still resolves correctly.
        return self._row_to_edge(
            rows[0][0], from_entity=rows[0][1], to_entity=rows[0][2]
        )

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

    # -- relation registry (relation-label list/merge, FR-MNT-2/4) ------------

    async def list_relations(self) -> list[tuple[str, int]]:
        """List the in-use ``MNEMOZINE_RELATES`` relation labels with their counts.

        The relation analogue of :meth:`list_categories`: a grouped count over
        active (open-window) ``MNEMOZINE_RELATES`` edges so the relation
        normalization job can enumerate the fragmented label vocabulary. A label
        missing the ``relation`` prop coalesces to the default ``"relates"``.
        """

        rows = await self._query(
            f"MATCH ()-[r:{RELATES_TYPE}]->() WHERE r.valid_to IS NULL "
            "RETURN coalesce(r.relation, 'relates') AS relation, count(r) AS n"
        )
        out: list[tuple[str, int]] = []
        for row in rows:
            relation = row[0] if row[0] is not None else "relates"
            out.append((str(relation), int(row[1])))
        return out

    async def merge_relations(self, source_relation: str, target_relation: str) -> int:
        """Relabel every ``source``-relation edge to ``target`` (relation merge).

        The relation analogue of :meth:`merge_categories` / :meth:`merge_entities`:
        for every active ``(a)-[:MNEMOZINE_RELATES {relation: source}]->(b)`` edge,
        MERGE it onto the ``(a, b, target)`` edge — combining ``weight`` via
        ``max`` (matching :meth:`upsert_edge`'s re-assert) and DELETING the now
        redundant parallel source edge so no duplicate parallel edges remain
        between the same pair + canonical relation. Idempotent: ``source ==
        target`` -> 0, and a re-run over already-canonical labels finds no source
        edges (FR-MNT-5). Returns the number of edges relabelled/merged.
        """

        if source_relation == target_relation:
            return 0
        # Count the source edges first (the relabelled total we report) — the
        # MERGE+DELETE below collapses them onto the target edge per (a, b) pair.
        count_rows = await self._query(
            f"MATCH ()-[r:{RELATES_TYPE} {{relation: $source}}]->() "
            "WHERE r.valid_to IS NULL RETURN count(r) AS n",
            source=source_relation,
        )
        n = int(count_rows[0][0]) if count_rows and count_rows[0][0] is not None else 0
        if n == 0:
            return 0
        # Fold each source-relation edge onto the (a, b, target) edge, taking the
        # max weight (re-assert semantics), then delete the redundant source edge.
        # The target edge keeps the source's id/window when it did not already
        # exist (coalesce) so the canonical edge is never left without provenance.
        await self._query(
            f"MATCH (a:{ENTITY_LABEL})-[r:{RELATES_TYPE} {{relation: $source}}]->"
            f"(b:{ENTITY_LABEL}) "
            "WHERE r.valid_to IS NULL "
            f"MERGE (a)-[nr:{RELATES_TYPE} {{relation: $target}}]->(b) "
            "SET nr.weight = "
            "CASE WHEN nr.weight IS NULL THEN r.weight "
            "ELSE (CASE WHEN nr.weight > r.weight THEN nr.weight ELSE r.weight END) END, "
            "nr.from_entity = coalesce(nr.from_entity, r.from_entity), "
            "nr.to_entity = coalesce(nr.to_entity, r.to_entity), "
            "nr.id = coalesce(nr.id, r.id), "
            "nr.valid_from = coalesce(nr.valid_from, r.valid_from), "
            "nr.valid_to = NULL "
            "DELETE r",
            source=source_relation,
            target=target_relation,
        )
        return n

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
