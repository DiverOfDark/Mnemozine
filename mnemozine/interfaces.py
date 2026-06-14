"""Shared layer contracts for Mnemozine — the single most load-bearing module.

Every module in the system codes against the :class:`typing.Protocol` contracts
defined here, **never** against another module's concrete implementation. This
is what lets the layers be built independently and tested against fakes (see
``tests/conftest.py``) without a live FalkorDB / Ollama / Qwen.

Layer map (mirrors PRD §5):

* :class:`IngestSource`   — FR-ING-* : normalize a source into ``IngestEvent``s.
* :class:`Extractor`      — FR-EXT-* : classify + scope chunks into MemoryUnits.
* :class:`StorageBackend` — FR-STO-*/FR-MNT-1 : the 4-way write + scoped query.
* :class:`Retriever`      — FR-RET-* : scoped retrieve, injection index, recall.
* :class:`CrossReferencer`— FR-RET-6 : serendipitous, explainable connections.
* :class:`MaintenanceJob` — FR-MNT-* : scheduled consolidation/decay/resolution.
* :class:`EmbeddingProvider` — bge-m3/Ollama embeddings (FR-STO-2).
* :class:`LLMProvider`    — extraction/classification/contradiction LLM calls.

Conventions
-----------
* I/O-bound operations (storage, LLM, embedding, MCP, ingest watching) are
  ``async``. Pure/CPU helpers are sync.
* Protocols are ``@runtime_checkable`` so fakes can be ``isinstance``-checked in
  tests, but structural typing is the real contract.
* These Protocols define *signatures and semantics only* — no implementation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from mnemozine.schema.events import IngestEvent
from mnemozine.schema.models import (
    Edge,
    Entity,
    MemoryUnit,
    RawChunk,
    Scope,
    ScopeDecision,
    SourceSession,
)

# ---------------------------------------------------------------------------
# Shared value objects passed across layer boundaries
# ---------------------------------------------------------------------------


class WriteDecision(str, Enum):
    """Outcome of the FR-MNT-1 4-way write decision."""

    ADD = "add"
    REINFORCE = "reinforce"
    SUPERSEDE = "supersede"
    NO_OP = "no-op"


@dataclass(slots=True)
class WriteResult:
    """Result of a :meth:`StorageBackend.upsert_memory` call (FR-MNT-1).

    ``decision`` is which of the four branches fired. ``memory`` is the unit now
    considered current for this content (the freshly-inserted unit for add /
    supersede, the reinforced existing unit for reinforce, or the pre-existing
    stronger unit for no-op). ``superseded`` lists units whose validity windows
    were closed as part of a supersede.
    """

    decision: WriteDecision
    memory: MemoryUnit
    superseded: list[MemoryUnit] = field(default_factory=list)


@dataclass(slots=True)
class RetrievedMemory:
    """A memory unit returned from retrieval, with its relevance score."""

    memory: MemoryUnit
    score: float


@dataclass(slots=True)
class CrossReference:
    """A surfaced serendipitous connection (FR-RET-6).

    Every connection carries a human-readable ``reason`` (e.g. "shares entities:
    async-runtime, cli-parsing"); a connection without a reason must not surface.
    """

    memory: MemoryUnit
    score: float
    reason: str
    shared_entities: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InjectionIndex:
    """The compact, token-budgeted SessionStart/mid-session payload (FR-RET-3/5).

    ``text`` is the final, already-truncated string injected into context (it
    MUST respect ``inject.token_budget``; truncate by dropping lowest-ranked
    items rather than overflowing). ``token_estimate`` is the producer's
    estimate of its size. The structured fields are retained for testing and
    for callers that want to re-render.
    """

    text: str
    token_estimate: int
    # Counts by the controlled scope-decision (replaces the old
    # preference_count / project_fact_count which keyed off MemoryType).
    global_count: int = 0
    project_count: int = 0
    # One-line hints for cross-reference seeds (the old idea_seed_hints, now
    # driven by MemoryUnit.cross_ref_candidate rather than a fixed type).
    cross_ref_hints: list[str] = field(default_factory=list)
    entity_tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalContext:
    """The working context used to scope retrieval/injection (FR-RET-2/3).

    Derived from cwd, ``Cargo.toml``/``package.json``, git remote, and recent
    turns (FR-RET-3). ``scopes`` is the composed set of scopes to search
    (current project + global), and ``entities`` are the active entities used to
    bound the entity-linked neighborhood.
    """

    project: str | None = None
    scopes: list[Scope] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    recent_text: str | None = None


@dataclass(slots=True)
class Classification:
    """A lightweight single-statement classification result (FR-EXT-3, R1).

    Returned by :meth:`Extractor.classify` for the eval/reclassify path. Unlike a
    full :class:`MemoryUnit` it carries no provenance/validity/tier bookkeeping —
    a bare statement being scored against the §9 classifier-accuracy eval set has
    no originating ingest session.

    Core redesign — the old single ``type`` field is split into the new contract:

    * :attr:`scope_decision` — the CONTROLLED ``global`` vs ``project`` decision
      that drives :attr:`scope` and the no-leak rule (FR-EXT-3). This is the
      make-or-break R1 classifier-accuracy metric.
    * :attr:`category` — the FREE-FORM, emergent classifier category string (no
      fixed enum); replaces the semantic role of the old type.
    * :attr:`cross_ref_candidate` — preserves the old ``idea_seed`` flag.

    The scope-decision is also derivable from :attr:`scope` (global vs project),
    but is carried explicitly so the eval set can score the decision directly.
    """

    scope_decision: ScopeDecision
    scope: Scope
    category: str = "fact"
    cross_ref_candidate: bool = False
    entities: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass(slots=True)
class Neighbor:
    """An entity-linked neighbor together with the edge that links it (FR-RET-6).

    :meth:`StorageBackend.neighbors` returns these so traversal can produce the
    mandatory human-readable :attr:`CrossReference.reason` and weight-rank
    connections (FR-RET-6), and so maintenance can see edge weight/relation for
    low-weight pruning (FR-MNT-4). ``entity`` is the neighbor node; ``edge`` is
    the (active) edge connecting it to the queried entity.
    """

    entity: Entity
    edge: Edge


# ---------------------------------------------------------------------------
# Ingestion (FR-ING-*)
# ---------------------------------------------------------------------------


@runtime_checkable
class IngestSource(Protocol):
    """A source that normalizes its conversations into ``IngestEvent``s (FR-ING-1).

    Concrete implementations: Claude Code JSONL watcher (FR-ING-2), the LiteLLM
    OpenAI-format gateway callback (FR-ING-3), and Hermes (FR-ING-4).
    """

    @property
    def source_name(self) -> str:
        """The :class:`~mnemozine.schema.events.Source` value this produces."""
        ...

    def stream(self) -> AsyncIterator[IngestEvent]:
        """Yield normalized events as they arrive (near-real-time, FR-ING-2/R4).

        For watcher-based sources this runs indefinitely, tailing new turns. For
        batch/backlog sources it yields the historical events then completes
        (FR-ING-6 backlog import). Implementations strip ``tool_calls`` per
        FR-ING-7 when ``IngestSettings.strip_tool_calls`` is set.
        """
        ...

    def backfill(
        self, *, since: SourceSession | None = None
    ) -> AsyncIterator[IngestEvent]:
        """Replay historical events for backlog import (FR-ING-6, Phase 1).

        Yields events for already-existing transcripts. Must be safe to re-run:
        downstream de-dups on the FR-ING-5 idempotency key.

        Call convention (matches :meth:`stream`): this is an **async generator**,
        not a coroutine — declared ``def`` (no ``async``) returning an
        ``AsyncIterator``. Iterate it directly with ``async for e in
        source.backfill(...)``; do **not** ``await source.backfill(...)``. An
        implementation must use ``yield`` in its body (which makes the function an
        async generator) rather than ``return``-ing an iterator, so the two call
        styles cannot diverge.
        """
        ...


# ---------------------------------------------------------------------------
# Typed extraction (FR-EXT-*)
# ---------------------------------------------------------------------------


@runtime_checkable
class Extractor(Protocol):
    """Turns a chunk of events into classified, scoped MemoryUnits (FR-EXT-*).

    The classifier's accuracy on the CONTROLLED scope decision
    (:class:`~mnemozine.schema.models.ScopeDecision`: ``global`` vs ``project``)
    is the single biggest driver of system quality (FR-EXT-3, R1); this layer is
    the crux and must be independently testable. The free-form
    :attr:`~mnemozine.schema.models.MemoryUnit.category` it also emits is
    *emergent* (no enum) and is normalized later by the category maintenance job.
    """

    async def extract(self, chunk: Sequence[IngestEvent]) -> list[MemoryUnit]:
        """Extract memory units from one chunk/episode (FR-ING-6 unit).

        Each returned unit is:
        * given a CONTROLLED scope decision
          (:class:`~mnemozine.schema.models.ScopeDecision`) that sets its
          hierarchical :class:`~mnemozine.schema.models.Scope` — ``global`` ->
          the root scope, ``project`` -> ``project:<derived-name>[/<sub>...]``
          (FR-EXT-3, no-leak),
        * tagged with a FREE-FORM emergent ``category`` string (FR-EXT-1) and a
          ``cross_ref_candidate`` flag for cross-reference seeds (FR-RET-6),
        * linked to extracted entities (FR-EXT-2),
        * stamped with a confidence score and provenance back to the source
          session/chunk (FR-EXT-4).

        Returns an empty list when the chunk yields no durable memory.
        """
        ...

    async def classify(
        self, statement: str, context: RetrievalContext
    ) -> Classification:
        """Classify a single candidate statement (eval/reclassify path, R1).

        Exposed independently so the classifier can be measured against the eval
        set (FR-EXT-3, §9 classifier accuracy) and so memories can be
        reclassified after correction (R1).

        Returns a lightweight :class:`Classification` (``scope_decision``,
        ``scope``, ``category``, ``cross_ref_candidate``, ``entities``,
        ``confidence``) rather than a full :class:`MemoryUnit`: a bare statement
        has no originating ingest session, so it cannot carry a real
        :class:`~mnemozine.schema.models.Provenance`. A caller that wants to
        persist the result builds a :class:`MemoryUnit` from this plus real
        provenance. (Note: ``MemoryUnit.provenance`` also has a classify
        sentinel default, so building one for a quick eval still validates.)
        """
        ...


# ---------------------------------------------------------------------------
# Storage (FR-STO-* + the FR-MNT-1 write decision)
# ---------------------------------------------------------------------------


@runtime_checkable
class StorageBackend(Protocol):
    """The temporal knowledge-graph store (Graphiti on FalkorDB, FR-STO-*).

    Holds both the graph and the vector embeddings (FR-STO-2). All methods are
    async (network/LLM-bound through Graphiti).
    """

    async def upsert_memory(self, memory: MemoryUnit) -> WriteResult:
        """Insert ``memory`` via the FR-MNT-1 4-way write decision.

        Compares against existing memories in the **same scope with overlapping
        entities** (never a graph-wide scan) and applies exactly one of:

        * **add**       — no related memory exists -> insert.
        * **reinforce** — a semantically equivalent memory exists -> bump
          confidence, refresh timestamp, no new node.
        * **supersede** — a related memory contradicts the new one -> close the
          old unit's validity window (``valid_to = now``) and insert the new one
          active. Delivers UC-2 / Goal 2.
        * **no-op**     — the new memory is strictly weaker/older than existing.

        Contradiction detection is a single narrowly-scoped cheap LLM call over
        same-scope global-decision candidates sharing >=1 entity. (The candidate
        set is keyed on the exact scope string, not the ancestor chain, so a
        write never contradicts a memory in a different scope.)
        """
        ...

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

        ANCESTOR-COMPOSITION (the no-leak rule): the query searches every
        ancestor-or-self of each scope in ``scopes``. When
        ``compose_ancestors`` is true (the default), each query scope ``S`` is
        expanded to ``S.ancestors()`` (``[global, project:P, project:P/sub,
        ..., S]``) and a memory matches iff its stored scope is one of those —
        i.e. an ancestor-or-self of ``S``. So a query at ``project:P/auth`` sees
        ``project:P/auth``, ``project:P`` and ``global`` memories, but NEVER a
        sibling like ``project:P/db`` and NEVER a descendant — siblings cannot
        leak. Pass ``compose_ancestors=False`` to match the exact scope strings
        only (e.g. a maintenance pass that must not widen).

        Never searches the whole graph: restricts to the composed scope set and,
        when given, the ``entities`` neighborhood, then runs semantic search
        inside that subset so effective search space stays roughly constant as
        the store grows (FR-RET-2). Active (open validity window, hot tier)
        memories only, unless ``include_archived``.
        """
        ...

    async def close_validity_window(
        self, memory_id: str, *, at: Any | None = None
    ) -> MemoryUnit:
        """Close a memory's validity window (FR-MNT-1 supersede, FR-STO-1).

        Sets ``valid_to`` so the unit leaves the hot path while being retained
        (never hard-deleted). ``at`` defaults to now. Returns the updated unit.
        """
        ...

    async def archive(self, memory_id: str) -> MemoryUnit:
        """Demote a memory to the archive tier (FR-STO-4, FR-MNT-3).

        Cold storage: retained for history/cross-reference but excluded from the
        default hot retrieval path. Never a hard delete.
        """
        ...

    async def promote(self, memory_id: str) -> MemoryUnit:
        """Promote an archived memory back to the hot tier (OQ3, FR-MNT-3).

        Inverse of :meth:`archive`: moves a unit from the cold archive tier back
        onto the default hot retrieval path. Per the resolved OQ3, the archive
        tier is *re-embedded lazily on promotion* — an implementation should
        re-run embedding for the unit here (or via :meth:`reembed`) if its
        embedding is stale relative to the current embedding model. Returns the
        updated unit.
        """
        ...

    async def reembed(self, memory_id: str) -> MemoryUnit:
        """Recompute and store the embedding for one memory (OQ3, §10).

        Backs both the resolved-OQ3 maintenance paths: the **full background
        re-embed pass over the hot tier** run on an embedding-model change, and
        the **lazy re-embed on promotion** out of archive (see :meth:`promote`).
        A re-embed maintenance job iterates the relevant tier (see
        :meth:`iter_memories`) and calls this per unit. Idempotent: re-embedding
        an already-current unit is a no-op write. Returns the updated unit.

        OQ3 ``migrate-index`` note: the re-embedding strategy needs no *new*
        StorageBackend method beyond this one. The command compares the
        configured ``embedding.dimensions`` against the live FalkorDB vector
        index dimension; if they differ it recreates the index at the new width
        through the existing ``GraphitiClient.ensure_vector_index`` seam (already
        idempotent against the ``dimensions`` config) and then drives the full
        hot-tier re-embed pass by iterating :meth:`iter_memories` and calling
        :meth:`reembed` per unit. So index migration is covered by
        ``embedding.dimensions`` (config) + :meth:`reembed` + :meth:`iter_memories`
        already in this contract — no contract change required.
        """
        ...

    async def record_access(self, memory_id: str) -> None:
        """Bump ``last_accessed``/``access_count`` for decay ranking (FR-MNT-3)."""
        ...

    # --- enumeration / scan (FR-MNT-2/3/4, R5 audit) ----------------------

    def iter_memories(
        self,
        *,
        scope: Scope | None = None,
        tier: Any | None = None,
        active_only: bool = False,
        valid_before: datetime | None = None,
        unused_since: datetime | None = None,
    ) -> AsyncIterator[MemoryUnit]:
        """Stream stored memory units for whole-store maintenance passes.

        The only enumeration entry point for the Maintenance layer; the scoped
        read paths (:meth:`scoped_query`, :meth:`get_entity`, :meth:`neighbors`)
        all need a query/start key and cannot iterate the store. This powers:

        * FR-MNT-2 consolidation (merge related/duplicate facts),
        * FR-MNT-3 decay/archive sweep (rank by recency+access; demote old,
          never-retrieved units) — use ``unused_since`` to select units whose
          ``last_accessed`` is older than a cutoff (``None`` last_accessed counts
          as never used),
        * FR-MNT-4 entity resolution / re-embed passes (iterate a tier), and
        * the R5 audit (walk everything written).

        Filters (all optional, AND-combined): ``scope`` restricts to one scope;
        ``tier`` to ``hot``/``archive`` (a :class:`~mnemozine.schema.models.Tier`);
        ``active_only`` excludes closed validity windows; ``valid_before`` keeps
        only units with ``valid_from`` before the cutoff; ``unused_since`` keeps
        only units not accessed since the cutoff. This is an **async generator**
        (declared ``def`` returning ``AsyncIterator``): iterate with
        ``async for m in storage.iter_memories(...)`` — do not ``await`` it.
        Implementations should page internally so memory stays bounded.
        """
        ...

    def iter_entities(self) -> AsyncIterator[Entity]:
        """Stream all entity nodes for entity resolution (FR-MNT-4).

        Entity resolution must scan every entity to find duplicates to merge
        (``rust`` / ``rust-lang`` / "the Rust work"); :meth:`get_entity` and
        :meth:`neighbors` only do keyed/local reads. Async generator, same call
        convention as :meth:`iter_memories`.
        """
        ...

    # --- entity operations (FR-EXT-2, FR-MNT-4) ---------------------------

    async def upsert_entity(self, entity: Entity) -> Entity:
        """Insert or update an entity node (FR-EXT-2)."""
        ...

    async def get_entity(self, name_or_id: str) -> Entity | None:
        """Resolve an entity by canonical name, alias, or id (FR-MNT-4)."""
        ...

    async def merge_entities(self, source_id: str, target_id: str) -> Entity:
        """Merge ``source_id`` into ``target_id`` (entity resolution, FR-MNT-4).

        Repoints edges, folds aliases, and removes the now-redundant node so the
        graph does not fragment across ``rust`` / ``rust-lang`` / "the Rust work".
        """
        ...

    async def neighbors(
        self, entity: str, *, max_degree: int | None = None, active_only: bool = True
    ) -> list[Neighbor]:
        """Return entity-linked neighbors **with their edges** (FR-RET-2/6, FR-MNT-4).

        Returns :class:`Neighbor` pairs (neighbor entity + the connecting
        :class:`~mnemozine.schema.models.Edge`) rather than bare entities, so
        callers keep the relation label and weight: CrossRef needs them to build
        the mandatory human-readable :attr:`CrossReference.reason` and to
        weight-rank connections (FR-RET-6), and maintenance needs them for
        low-weight edge pruning (FR-MNT-4). Bounded by ``max_degree`` (defaults
        to ``maintenance.max_node_degree``) to keep traversal cost flat;
        ``active_only`` restricts to edges with an open validity window.
        """
        ...

    # --- edge operations (FR-EXT-2, FR-MNT-4, FR-RET-6) -------------------

    async def upsert_edge(self, edge: Edge) -> Edge:
        """Insert or update a weighted, temporal relationship edge (FR-EXT-2).

        FR-EXT-2 requires extracted relationships to be *written into the graph
        with timestamps*. Upsert keys on ``(from_entity, to_entity, relation)``;
        re-asserting an existing relation should refresh/raise its ``weight``
        rather than duplicate the edge. Returns the stored edge.
        """
        ...

    async def edges_for_entity(
        self, entity: str, *, active_only: bool = True
    ) -> list[Edge]:
        """Return the edges incident to ``entity`` (FR-MNT-4 pruning, FR-RET-6).

        Used by maintenance to find and prune edges below
        ``maintenance.edge_weight_floor`` (FR-MNT-4) and by traversal/explainable
        cross-referencing (FR-RET-6). ``active_only`` restricts to edges with an
        open validity window. ``entity`` may be a canonical name, alias, or id.
        """
        ...

    async def prune_edge(self, edge_id: str, *, at: datetime | None = None) -> Edge:
        """Close a low-weight edge's validity window (FR-MNT-4 pruning).

        Like :meth:`close_validity_window` for memories: sets the edge's
        ``valid_to`` so it drops off traversal while remaining retained for
        history (never hard-deleted). ``at`` defaults to now. Returns the edge.
        """
        ...

    # --- suppression persistence (FR-RET-6 feedback, R2) ------------------

    async def record_suppression(self, memory_id: str, context_key: str) -> None:
        """Persist a dismissed cross-reference suggestion (FR-RET-6, R2).

        Backs :meth:`CrossReferencer.suppress`. Storing the
        ``(memory_id, context_key)`` pair here (rather than in CrossRef's own
        memory) is what makes a dismissal survive across calls/process restarts
        so the suggestion stops resurfacing in that context (R2). Idempotent.

        Ownership note: suppression is persisted by the storage backend via this
        method and :meth:`is_suppressed`; :class:`CrossReferencer` delegates to
        the backend rather than owning its own store, so the two layers do not
        each assume the other persists it.
        """
        ...

    async def is_suppressed(self, memory_id: str, context_key: str) -> bool:
        """True if ``(memory_id, context_key)`` was previously suppressed (R2).

        Read side of :meth:`record_suppression`; :meth:`CrossReferencer.find_related`
        consults this (directly or via the backend) to exclude dismissed items.
        """
        ...

    async def record_session(self, session: SourceSession) -> None:
        """Persist a source-session record for provenance/archive (§7, FR-STO-4)."""
        ...

    # --- raw-chunk tier (offline re-extraction/reindex; survives R4 cleanup) --

    async def persist_raw_chunk(self, chunk: RawChunk) -> RawChunk:
        """Persist a :class:`~mnemozine.schema.models.RawChunk` (the raw tier).

        Stores the normalized (tool-calls-stripped) extraction-input chunk so the
        store can re-extract / reindex offline and survive Claude's 30-day local
        transcript cleanup (R4). Idempotent on ``chunk.content_hash`` (the
        FR-ING-5 key): re-persisting the same chunk updates its ``memory_ids`` /
        timestamps rather than duplicating. Gated by
        ``ingest.raw_retention_enabled`` at the call site (default on). Returns
        the stored chunk.
        """
        ...

    def iter_raw_chunks(
        self,
        *,
        scope: Scope | None = None,
        session_id: str | None = None,
        source: str | None = None,
        since: datetime | None = None,
    ) -> AsyncIterator[RawChunk]:
        """Stream stored raw chunks for offline re-extraction/reindex (raw tier).

        The enumeration entry point for the re-extraction seam (see
        :meth:`re_extract_from_raw_chunks`): a maintenance/offline job iterates
        the raw tier and re-runs a newer extractor/classifier or embedding model
        over each chunk's normalized ``content``. Filters (all optional,
        AND-combined): ``scope`` (exact scope, no ancestor composition — a
        re-extraction must not widen), ``session_id``, ``source``, and ``since``
        (``ingested_at`` cutoff). Async generator (declared ``def`` returning
        ``AsyncIterator``): iterate with ``async for c in
        storage.iter_raw_chunks(...)`` — do not ``await`` it. Pages internally so
        memory stays bounded.
        """
        ...

    async def re_extract_from_raw_chunks(
        self,
        extractor: Extractor,
        *,
        scope: Scope | None = None,
        session_id: str | None = None,
        supersede_existing: bool = True,
    ) -> MaintenanceReport:
        """Re-run extraction over the retained raw tier (offline reindex seam).

        The re-extraction seam: iterates :meth:`iter_raw_chunks` (filtered by
        ``scope`` / ``session_id``), feeds each chunk's normalized ``content``
        back through ``extractor`` to produce fresh :class:`MemoryUnit`s, upserts
        them via the FR-MNT-1 write path, and — when ``supersede_existing`` —
        closes the validity windows of the memories the chunk previously produced
        (``RawChunk.memory_ids``) so the new extraction replaces the old. Returns
        a :class:`MaintenanceReport` summarizing the pass. Idempotent and safe to
        re-run (FR-MNT-5): an unchanged extractor re-produces equivalent units
        that reinforce rather than duplicate.
        """
        ...

    async def reclassify_memory(
        self,
        memory_id: str,
        *,
        scope: Scope | None = None,
        category: str | None = None,
        cross_ref_candidate: bool | None = None,
    ) -> MemoryUnit:
        """Update a stored memory's scope/category/cross-ref WITHOUT raw text (R1).

        Backs the WebUI reclassify / re-scope corrections (WEBUI PRD §3) and the
        category-merge job. Re-derives classification *from the already-stored
        content + provenance* — it does NOT need the raw transcript, so it works
        long after Claude's 30-day cleanup. Any of ``scope`` (re-scope; must obey
        the hierarchical no-leak rule), ``category`` (free-form re-label, will be
        normalized), or ``cross_ref_candidate`` (toggle the seed flag) may be
        given; unset fields are left unchanged. Returns the updated unit. (To
        re-derive from raw input rather than just re-tag, use
        :meth:`re_extract_from_raw_chunks`.)
        """
        ...

    # --- category registry (emergent-category list/merge, FR-MNT-2/4) --------

    async def list_categories(self) -> list[tuple[str, int]]:
        """List the free-form categories in use with their memory counts.

        Powers the category registry view + the merge job: returns
        ``(category, count)`` pairs over active memories so near-duplicate
        emergent categories (e.g. 'gotcha' / 'gotchas') can be surfaced and
        merged. Ordering is unspecified; callers sort as needed.
        """
        ...

    async def merge_categories(self, source: str, target: str) -> int:
        """Re-label every memory tagged ``source`` to ``target`` (category merge).

        The category analogue of :meth:`merge_entities`: consolidates a
        fragmented emergent category into a canonical one. Both are normalized
        (lowercased/trimmed) before matching, matching
        :class:`~mnemozine.schema.models.MemoryUnit` category normalization.
        Idempotent. Returns the number of memories re-labeled. Driven by the
        :class:`CategoryMerger` maintenance job.
        """
        ...

    async def close(self) -> None:
        """Close the underlying connection/pool."""
        ...


# ---------------------------------------------------------------------------
# Retrieval & delivery (FR-RET-*)
# ---------------------------------------------------------------------------


@runtime_checkable
class Retriever(Protocol):
    """Scoped retrieval + injection-index construction + on-demand recall (FR-RET-*).

    Sits above :class:`StorageBackend`, composing scopes and entity neighborhoods
    and enforcing the injection token budget.
    """

    async def scoped_retrieve(
        self, query: str, context: RetrievalContext, *, top_k: int = 10
    ) -> list[RetrievedMemory]:
        """Scoped semantic retrieval for the given context (FR-RET-2).

        Composes ``context.scopes`` (current project + global) with the active
        entity neighborhood, delegates the bounded search to the storage
        backend, and **records access** for the returned units (calls
        ``StorageBackend.record_access``, bumping ``access_count`` /
        ``last_accessed``) so decay ranking reflects real use (FR-MNT-3). Compare
        :meth:`build_index`, which deliberately does NOT record access.
        """
        ...

    async def build_index(
        self, context: RetrievalContext, *, token_budget: int | None = None
    ) -> InjectionIndex:
        """Build the compact, token-budgeted injection index (FR-RET-3/5).

        Produces a compact index (counts + entity tags + 1-line idea-seed hints)
        plus short snippets for the **top-ranked preferences only**, then
        truncates to ``token_budget`` (defaults to ``inject.token_budget`` = 500)
        by dropping lowest-ranked items — never overflowing the budget. The
        result is advisory, clearly delimited background context, not a dump.

        Access semantics: ``build_index`` MUST NOT record access — its reads are
        passive/automatic (every SessionStart and mid-session injection calls
        it), so counting them would inflate ``access_count`` for every memory and
        silently corrupt FR-MNT-3 decay ranking. Only the deliberate read paths
        (:meth:`scoped_retrieve`, :meth:`recall`) record access.
        """
        ...

    async def recall(
        self, query: str, scope: Scope | None = None, *, top_k: int = 10
    ) -> list[RetrievedMemory]:
        """On-demand full-detail recall (FR-RET-4).

        Backs the ``recall(query, scope?)`` MCP tool: when the injected index
        hints at something worth chasing, this returns the consolidated detail
        across sessions (UC-4). ``scope=None`` searches the default composed
        scope (current project + global).

        Access semantics: like :meth:`scoped_retrieve`, ``recall`` is a
        deliberate read and **records access** for the returned units
        (``StorageBackend.record_access``) so decay ranking (FR-MNT-3) reflects
        it. (Contrast :meth:`build_index`, which does not.)
        """
        ...


# ---------------------------------------------------------------------------
# Cross-reference engine (FR-RET-6 / UC-3)
# ---------------------------------------------------------------------------


@runtime_checkable
class CrossReferencer(Protocol):
    """Serendipitous, explainable cross-references (FR-RET-6, UC-3)."""

    async def find_related(
        self, context: RetrievalContext, *, max_suggestions: int | None = None
    ) -> list[CrossReference]:
        """Find related cross-ref-candidate/project nodes for the context (FR-RET-6).

        Primary path is graph traversal over shared entities (explainable),
        using ``StorageBackend.neighbors`` (which now returns edges so the
        relation+weight feed the ``reason`` and weight-rank). Vector similarity
        is a fallback gated by its own ``crossref.vector_fallback_threshold``
        (distinct from ``crossref.relevance_threshold``, which gates final
        surfacing). Only connections above ``crossref.relevance_threshold``
        surface, capped at ``crossref.max_suggestions``, each carrying a
        human-readable ``reason``. Suppressed/dismissed items (see
        :meth:`suppress`) are excluded.
        """
        ...

    async def suppress(self, memory_id: str, context_key: str) -> None:
        """Record that a surfaced suggestion was dismissed (FR-RET-6 feedback).

        ``context_key`` identifies the working context the dismissal applies to,
        so the same suggestion stops resurfacing there (R2).

        Persistence: the dismissal is persisted through the storage backend
        (``StorageBackend.record_suppression`` / ``is_suppressed``) — the backend
        owns the suppression store, not this object — so a dismissal survives
        across calls and process/test boundaries and ``find_related`` can read it
        back. CrossReferencer takes the storage backend as a constructor dep and
        delegates here.
        """
        ...


# ---------------------------------------------------------------------------
# Maintenance (FR-MNT-*)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MaintenanceReport:
    """Summary of one maintenance pass, for the audit log (FR-MNT-5, R5)."""

    job_name: str
    consolidated: int = 0
    entities_merged: int = 0
    categories_merged: int = 0
    archived: int = 0
    edges_pruned: int = 0
    re_extracted: int = 0
    notes: list[str] = field(default_factory=list)


@runtime_checkable
class MaintenanceJob(Protocol):
    """A scheduled, idempotent maintenance pass (FR-MNT-5).

    Concrete jobs: dedup/consolidation (FR-MNT-2), entity resolution (FR-MNT-4),
    decay/archive (FR-MNT-3), category merge (core redesign), and audit (R5).
    Each must be safe to run repeatedly.
    """

    @property
    def name(self) -> str:
        """Stable job name, used in logs/reports and the scheduler."""
        ...

    async def run(self) -> MaintenanceReport:
        """Execute the pass once. Idempotent and safe to re-run (FR-MNT-5)."""
        ...


@runtime_checkable
class CategoryMerger(Protocol):
    """Maintenance job that consolidates emergent free-form categories (FR-MNT-2/4).

    The category analogue of entity resolution: because
    :attr:`~mnemozine.schema.models.MemoryUnit.category` is FREE-FORM (no enum),
    the registry fragments over time ('gotcha' / 'gotchas' / 'pitfall'). This job
    proposes near-duplicate categories — embedding/string similarity above
    ``category.merge_similarity_threshold`` — and folds each cluster into one
    canonical category via :meth:`StorageBackend.merge_categories`. A
    :class:`MaintenanceJob` (has ``name`` / ``run``); broken out as its own
    Protocol so the merge policy is independently testable.
    """

    @property
    def name(self) -> str:
        """Stable job name, used in logs/reports and the scheduler."""
        ...

    async def propose_merges(self) -> list[tuple[str, str]]:
        """Propose ``(source_category, target_category)`` merges (no writes).

        Compares the in-use categories (:meth:`StorageBackend.list_categories`)
        and returns the pairs whose similarity exceeds
        ``category.merge_similarity_threshold``, each oriented source->canonical
        (the higher-count / canonical category is the target). Pure/read-only so
        the proposals can be reviewed (WebUI) before applying.
        """
        ...

    async def run(self) -> MaintenanceReport:
        """Apply the proposed category merges once (idempotent, FR-MNT-5).

        Calls :meth:`StorageBackend.merge_categories` for each proposed pair and
        reports the count in :attr:`MaintenanceReport.notes` /
        :attr:`MaintenanceReport.consolidated`.
        """
        ...


# ---------------------------------------------------------------------------
# Activity log (WEBUI Q3 — append-only observability feed)
# ---------------------------------------------------------------------------


@runtime_checkable
class ActivityLog(Protocol):
    """An append-only record of what the memory layer did (WEBUI PRD §3, Q3).

    Backs the WebUI Logs screen + Dashboard feed: ingestion, the FR-MNT-1 4-way
    write decision, maintenance passes (FR-MNT-*), and injections (FR-RET-3/5).
    Concrete impls live in :mod:`mnemozine.activity` —
    ``NullActivityLog`` (the **default**, a no-op so existing pipeline call sites
    and the test suite are unaffected), ``InMemoryActivityLog``, and the
    persisted ``FalkorDBActivityLog`` (which reuses the storage connection — never
    a new source of truth).

    Pipeline seams do **not** call :meth:`append` directly; they go through the
    safe :func:`mnemozine.activity.emit` helper (null-safe, error-swallowing,
    fire-and-forget) so recording activity can never break a write or a recall.
    The WebUI reads via :meth:`query`.
    """

    @property
    def enabled(self) -> bool:
        """False for the no-op default; True for a real persisted/in-memory log.

        ``emit`` fast-paths on this so a disabled log costs nothing.
        """
        ...

    async def append(self, event: Any) -> None:
        """Append one :class:`~mnemozine.activity.ActivityEvent` (append-only)."""
        ...

    async def query(self, query: Any | None = None) -> list[Any]:
        """Read events matching an :class:`~mnemozine.activity.ActivityQuery`.

        Returns newest-first, paged by the query's ``offset``/``limit``. ``None``
        means "recent events with defaults".
        """
        ...

    async def close(self) -> None:
        """Release any held resources (no-op for the in-memory/null impls)."""
        ...


# ---------------------------------------------------------------------------
# LLM + embedding providers (the pluggable model endpoints)
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Text embedding provider (bge-m3 via Ollama by default, FR-STO-2)."""

    @property
    def dimensions(self) -> int:
        """Embedding vector dimensionality (bge-m3 = 1024)."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Embed a single text into a vector."""
        ...

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many texts. Implementations should batch for throughput (R3)."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Pluggable OpenAI-format LLM for extraction/classification (FR-EXT-*, §5.5).

    Default target is local Qwen via an OpenAI-format ``base_url``; a cloud model
    is a drop-in swap (PRD §3 exception, OQ5). Used for extraction, the cheap
    narrowly-scoped contradiction check (FR-MNT-1), and consolidation (FR-MNT-2).
    """

    @property
    def model(self) -> str:
        """The configured LiteLLM model id (provider/model)."""
        ...

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Return a plain-text completion."""
        ...

    async def complete_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | type[Any],
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Return a structured (JSON) completion conforming to ``schema``.

        ``schema`` may be a JSON-schema dict or a pydantic model class. Used for
        typed extraction and the contradiction decision so outputs are parseable.
        """
        ...
