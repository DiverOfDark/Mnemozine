"""The §7 data model: MemoryUnit, Entity, Edge, SourceSession, RawChunk + enums.

These pydantic models are the durable shapes stored in / retrieved from the
temporal knowledge graph. They are deliberately storage-agnostic: a
``StorageBackend`` implementation maps them onto Graphiti/FalkorDB nodes and
edges, but every other layer (extraction, retrieval, maintenance, MCP) speaks
in these models.

Temporal semantics (FR-STO-1 / FR-MNT-1): facts carry a validity window
``(valid_from, valid_to)``. A superseded fact is *not* deleted — its window is
closed (``valid_to = now``), moving it off the hot retrieval path while keeping
it for history and cross-reference (FR-STO-4, FR-MNT-3).

Core data-model redesign (this module is the shared CONTRACT)
-------------------------------------------------------------
Three breaking shifts replace the old flat scope + 3-value ``MemoryType``:

1. **Hierarchical scope.** :class:`Scope` is now an ordered *path* of segments
   rooted at ``global`` — e.g. ``global``, ``project:Mnemozine``,
   ``project:Mnemozine/auth``. Retrieval composes the ancestor chain
   (:meth:`Scope.ancestors`) and the no-leak rule is ancestor-or-self:
   a query at scope ``S`` retrieves memories whose scope is an ancestor-or-self
   of ``S`` (so siblings never leak — :meth:`Scope.contains`).

2. **Category split.** The old ``MemoryType`` did two jobs; they are split:
   * :class:`ScopeDecision` — the CONTROLLED ``global`` vs ``project`` decision
     the extractor makes; this drives scope + the no-leak rule and stays a fixed
     enum.
   * ``MemoryUnit.category`` — a FREE-FORM, emergent string the classifier emits
     (no fixed enum); replaces the *semantic role* of the old type.
   * ``MemoryUnit.cross_ref_candidate`` — a boolean preserving the old
     ``idea_seed`` cross-reference behavior (FR-RET-6) as a flag, not a type.

3. **Raw-chunk retention.** :class:`RawChunk` is a first-class STORED tier: the
   normalized (tool-calls-stripped) extraction-input chunk, retained so the
   store can re-extract / reindex offline and survive Claude's 30-day local
   cleanup (R4).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

# The single source of truth for the data-model version stamped on every record.
# Imported from the migrations package (which is import-light and does NOT import
# schema at module scope) so there is no import cycle: schema -> migrations only.
from mnemozine.migrations import CURRENT_DATA_VERSION


def _utcnow() -> datetime:
    """Timezone-aware UTC now, used as the default validity-window start."""

    return datetime.now(UTC)


def _new_id() -> str:
    """Generate a fresh node id."""

    return uuid4().hex


# ---------------------------------------------------------------------------
# Scope-decision (the CONTROLLED half of the old MemoryType, FR-EXT-1/3)
# ---------------------------------------------------------------------------


class ScopeDecision(str, Enum):
    """The CONTROLLED scope decision the extractor makes (FR-EXT-1/3).

    This is one of the two jobs the old 3-value ``MemoryType`` enum did, kept as
    a fixed/controlled enum because it drives scope assignment and the no-leak
    rule and therefore must NOT drift:

    * ``global``  — applies everywhere (a durable cross-project preference);
      lives in the root :meth:`Scope.global_` scope.
    * ``project`` — scoped to the derived project path; must not leak across
      projects (lives in a ``project:<name>[/<sub>...]`` scope).

    The *semantic role* of the old type (preference / project_fact / idea_seed)
    is now carried separately by the free-form :attr:`MemoryUnit.category`
    string and the :attr:`MemoryUnit.cross_ref_candidate` flag.
    """

    GLOBAL = "global"
    PROJECT = "project"


class MemoryType(str, Enum):
    """DEPRECATED legacy 3-value classification (kept for migration only).

    Superseded by the :class:`ScopeDecision` controlled enum (scope decision)
    plus the free-form :attr:`MemoryUnit.category` string and the
    :attr:`MemoryUnit.cross_ref_candidate` flag. Retained so historical data and
    downstream call sites can be migrated incrementally; new code MUST NOT branch
    on it. :meth:`scope_decision` and :meth:`is_cross_ref` map a legacy value
    onto the new contract.
    """

    PREFERENCE = "preference"
    PROJECT_FACT = "project_fact"
    IDEA_SEED = "idea_seed"

    @property
    def scope_decision(self) -> ScopeDecision:
        """Map a legacy type onto the new controlled scope decision (FR-EXT-3).

        ``preference`` / ``idea_seed`` -> ``global``; ``project_fact`` ->
        ``project``. (idea_seed historically lived in global scope.)
        """

        if self is MemoryType.PROJECT_FACT:
            return ScopeDecision.PROJECT
        return ScopeDecision.GLOBAL

    @property
    def is_cross_ref(self) -> bool:
        """True for ``idea_seed`` — the legacy cross-reference candidate type."""

        return self is MemoryType.IDEA_SEED

    @property
    def category(self) -> str:
        """A reasonable free-form :attr:`MemoryUnit.category` for a legacy type."""

        return self.value

    @classmethod
    def from_split(
        cls, scope_decision: ScopeDecision, cross_ref_candidate: bool
    ) -> MemoryType:
        """Reverse-map the new contract back onto a legacy :class:`MemoryType`.

        The inverse of :attr:`scope_decision` / :attr:`is_cross_ref`, used only by
        the migration-only bootstrap/eval wire schemas that still carry a legacy
        type. ``cross_ref_candidate`` -> ``idea_seed`` (it always lived in global
        scope historically); otherwise ``project`` -> ``project_fact`` and
        ``global`` -> ``preference``.
        """

        if cross_ref_candidate:
            return cls.IDEA_SEED
        if scope_decision is ScopeDecision.PROJECT:
            return cls.PROJECT_FACT
        return cls.PREFERENCE


class Tier(str, Enum):
    """Storage tier of a memory unit (§7, FR-STO-4, FR-MNT-3).

    * ``hot``     — on the default retrieval path.
    * ``archive`` — cold tier: retained for history/cross-reference but excluded
      from default hot retrieval.
    """

    HOT = "hot"
    ARCHIVE = "archive"


# ---------------------------------------------------------------------------
# Hierarchical scope (FR-EXT-3, FR-STO-3, no-leak; PRD §3/§5.5)
# ---------------------------------------------------------------------------

# Canonical root segment + the prefix of a top-level project segment.
GLOBAL_SCOPE = "global"
_PROJECT_PREFIX = "project:"

# Delimiter between ordered scope segments in the canonical string form. Kept as
# a module default but mirrored by ``ScopeSettings.delimiter`` in config so it is
# a tuning knob, not a hard-coded constant; callers that need the configured
# value pass it to :meth:`Scope.as_str` / :meth:`Scope.parse`.
SCOPE_DELIMITER = "/"


class Scope(BaseModel):
    """A hierarchical retrieval/storage scope path (FR-STO-3, FR-EXT-3, no-leak).

    A scope is an **ordered path of segments** rooted at ``global``:

    * ``global``                       — segments = ``[]`` (the root).
    * ``project:Mnemozine``            — segments = ``["Mnemozine"]``.
    * ``project:Mnemozine/auth``       — segments = ``["Mnemozine", "auth"]``.

    The first segment is the project name; any further segments are optional
    sub-scopes (e.g. a module, or a rolled-up subagent/workflow segment). Use the
    constructors :meth:`global_` / :meth:`project` / :meth:`child` rather than
    building by hand.

    Canonical string form (:meth:`as_str`) is what gets persisted as a tag on a
    :class:`MemoryUnit`; :meth:`parse` is the inverse and accepts the same
    strings the old flat scope produced (``global`` / ``project:<name>``), so it
    stays backward-compatible.

    No-leak rule (CRITICAL, FR-STO-3): a query at scope ``S`` retrieves memories
    whose scope is an **ancestor-or-self** of ``S``. :meth:`ancestors` yields the
    composed ancestor chain ``[global, project:P, project:P/sub, ...]`` ending at
    ``self``; :meth:`contains` / :meth:`is_descendant_of` implement the symmetric
    no-leak check so siblings never leak.
    """

    segments: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered scope segments under the global root. [] = global; "
            "[<project>] = project:<project>; [<project>, <sub>...] = a sub-scope."
        ),
    )

    @field_validator("segments")
    @classmethod
    def _segments_nonempty(cls, v: list[str]) -> list[str]:
        for seg in v:
            if not seg or not seg.strip():
                raise ValueError("Scope segments must each be non-empty")
            if SCOPE_DELIMITER in seg:
                raise ValueError(
                    f"Scope segment {seg!r} must not contain the delimiter "
                    f"{SCOPE_DELIMITER!r}"
                )
        return v

    # -- constructors ---------------------------------------------------------

    @classmethod
    def global_(cls) -> Scope:
        """The cross-project global root scope (where global memories live)."""

        return cls(segments=[])

    @classmethod
    def project(cls, project_id: str, *subsegments: str) -> Scope:
        """A project scope ``project:<project_id>[/<sub>...]`` (FR-EXT-3).

        ``project_id`` is the top-level project segment; any ``subsegments`` are
        appended as ordered sub-scopes (e.g. a module or a rolled-up
        subagent/workflow segment).
        """

        if not project_id or not project_id.strip():
            raise ValueError("project_id must be non-empty for a project scope")
        return cls(segments=[project_id, *subsegments])

    def child(self, segment: str) -> Scope:
        """Return a new scope one level deeper (this scope + ``segment``).

        ``Scope.global_().child("Mnemozine")`` == ``Scope.project("Mnemozine")``;
        ``Scope.project("Mnemozine").child("auth")`` ==
        ``Scope.project("Mnemozine", "auth")``.
        """

        if not segment or not segment.strip():
            raise ValueError("scope child segment must be non-empty")
        return Scope(segments=[*self.segments, segment])

    # -- predicates -----------------------------------------------------------

    @property
    def is_global(self) -> bool:
        """True if this is the global root scope (no segments)."""

        return not self.segments

    @property
    def depth(self) -> int:
        """Number of segments below the global root (0 = global)."""

        return len(self.segments)

    @property
    def project_id(self) -> str | None:
        """The top-level project segment, or ``None`` for the global scope.

        Backward-compatible accessor: the old flat ``Scope`` exposed exactly this
        (``None`` for global, the project name otherwise). For a sub-scope it
        returns the *project* (first) segment, not the leaf.
        """

        return self.segments[0] if self.segments else None

    @property
    def leaf(self) -> str | None:
        """The deepest (last) segment, or ``None`` for the global scope."""

        return self.segments[-1] if self.segments else None

    @property
    def parent(self) -> Scope | None:
        """The immediate parent scope, or ``None`` for the global root."""

        if not self.segments:
            return None
        return Scope(segments=self.segments[:-1])

    def ancestors(self) -> list[Scope]:
        """The composed ancestor-or-self chain, root first, ``self`` last.

        Used by retrieval to COMPOSE the scopes to search (FR-RET-2/FR-STO-3):
        ``Scope.project("Mnemozine", "auth").ancestors()`` ->
        ``[global, project:Mnemozine, project:Mnemozine/auth]``. A query passes
        this list (or the subset of it) as the composed scope so a deeper scope
        always also sees its ancestors (global preferences, project facts) but a
        shallower/sibling scope never sees a descendant's memories.
        """

        chain: list[Scope] = [Scope.global_()]
        for i in range(1, len(self.segments) + 1):
            chain.append(Scope(segments=self.segments[:i]))
        return chain

    def is_descendant_of(self, other: Scope) -> bool:
        """True if ``self`` is ``other`` or lies strictly below it (ancestor-or-self).

        ``other`` is an ancestor-or-self of ``self`` iff ``other``'s segments are
        a prefix of ``self``'s. ``global`` is an ancestor of everything.
        """

        if len(other.segments) > len(self.segments):
            return False
        return self.segments[: len(other.segments)] == other.segments

    def contains(self, other: Scope) -> bool:
        """True if ``self`` is an ancestor-or-self of ``other`` (no-leak check).

        The inverse of :meth:`is_descendant_of`: a memory stored at ``self`` is
        visible to a query at ``other`` exactly when ``self.contains(other)``
        (i.e. ``self`` is an ancestor-or-self of the query scope). Siblings
        contain each other only at their shared prefix, never directly, so they
        never leak.
        """

        return other.is_descendant_of(self)

    # -- (de)serialization ----------------------------------------------------

    def as_str(self, delimiter: str = SCOPE_DELIMITER) -> str:
        """Serialize to the canonical persisted tag form.

        ``global`` / ``project:<project>`` / ``project:<project>/<sub>/...``. The
        ``project:`` prefix is on the whole project path (not each segment), and
        sub-segments are joined by ``delimiter`` (default :data:`SCOPE_DELIMITER`,
        config-overridable via ``ScopeSettings.delimiter``).
        """

        if not self.segments:
            return GLOBAL_SCOPE
        return f"{_PROJECT_PREFIX}{delimiter.join(self.segments)}"

    @classmethod
    def parse(cls, value: str, delimiter: str = SCOPE_DELIMITER) -> Scope:
        """Parse a persisted scope string back into a :class:`Scope`.

        Accepts ``global`` and ``project:<project>[<delimiter><sub>...]``. The
        flat strings the old scope produced (``global`` / ``project:<name>``)
        parse unchanged, so persisted data and existing call sites keep working.
        """

        if value == GLOBAL_SCOPE:
            return cls.global_()
        if value.startswith(_PROJECT_PREFIX):
            path = value[len(_PROJECT_PREFIX) :]
            if not path:
                raise ValueError(f"empty project scope path: {value!r}")
            return cls(segments=path.split(delimiter))
        raise ValueError(f"unrecognized scope string: {value!r}")

    def __hash__(self) -> int:  # allow use in sets/dict keys
        return hash(self.as_str())


# Sentinel source value for provenance produced by the single-statement
# classifier path (Extractor.classify) rather than a real ingest session.
CLASSIFY_SOURCE = "classify"


class Provenance(BaseModel):
    """A link from a memory back to its originating source (FR-EXT-4).

    Every memory records where it came from so it can be audited (R5) and so the
    archive/raw tier can preserve the raw path (§7 Source/Session.raw_path). The
    ``chunk_hash`` links a memory back to the :class:`RawChunk` it was extracted
    from, which is the join key for offline re-extraction / reclassification.
    """

    source: str = Field(description="Originating source, e.g. 'claude_code'.")
    session_id: str = Field(description="Originating session id.")
    chunk_hash: str | None = Field(
        default=None,
        description="Content hash of the chunk this memory was extracted from (FR-ING-5).",
    )
    raw_path: str | None = Field(
        default=None,
        description="Path to the raw transcript on disk/archive (§7).",
    )

    @classmethod
    def classify_sentinel(cls) -> Provenance:
        """Provenance for a unit built by the single-statement classifier path.

        ``Extractor.classify`` (the eval/reclassify path, FR-EXT-3, §9) operates
        on a bare statement with no originating ingest session, so it cannot
        supply a real :class:`Provenance`. This sentinel
        (``source='classify', session_id=''``) lets ``classify`` return a valid
        :class:`MemoryUnit` without inventing a fake session; callers that
        persist the unit should overwrite provenance with the real source.
        """

        return cls(source=CLASSIFY_SOURCE, session_id="")

    @property
    def is_classify_sentinel(self) -> bool:
        """True if this provenance is the :meth:`classify_sentinel` placeholder."""

        return self.source == CLASSIFY_SOURCE and self.session_id == ""


class Entity(BaseModel):
    """A canonical entity node (§7).

    Entities (e.g. ``rust``, ``async``, ``cli``, ``error-handling``) link memory
    units and power graph traversal for scoped retrieval (FR-RET-2) and
    cross-referencing (FR-RET-6). Entity resolution merges duplicates (FR-MNT-4).
    """

    id: str = Field(default_factory=_new_id)
    canonical_name: str = Field(description="The canonical name of the entity.")
    aliases: list[str] = Field(
        default_factory=list,
        description="Known aliases merged into this entity (FR-MNT-4).",
    )
    type: str | None = Field(
        default=None,
        description="Optional entity type/category (e.g. 'language', 'tool').",
    )


class Edge(BaseModel):
    """A weighted, temporal relationship between two entities (§7).

    Edges carry their own validity window so Graphiti's edge-invalidation can
    close them on contradiction (FR-MNT-1 underlying layer). ``weight`` is used
    for low-weight edge pruning (FR-MNT-4, ``maintenance.edge_weight_floor``).
    """

    id: str = Field(default_factory=_new_id)
    from_entity: str = Field(description="Source entity id.")
    to_entity: str = Field(description="Target entity id.")
    relation: str = Field(description="Relation label.")
    weight: float = Field(default=1.0, description="Edge weight (FR-MNT-4 pruning).")
    valid_from: datetime = Field(default_factory=_utcnow)
    valid_to: datetime | None = Field(
        default=None,
        description="None = still valid; a timestamp = closed/invalidated.",
    )

    @property
    def is_active(self) -> bool:
        """True if the edge's validity window is open."""

        return self.valid_to is None


class Suppression(BaseModel):
    """A dismissed cross-reference suggestion (FR-RET-6 feedback, R2).

    When the operator dismisses a surfaced connection, the
    ``(memory_id, context_key)`` pair is recorded so the same suggestion stops
    resurfacing in that working context. Persisted by the storage backend (see
    ``StorageBackend.record_suppression`` / ``is_suppressed``) so a dismissal
    survives across calls/process restarts, not just within one
    :class:`CrossReferencer` instance.
    """

    memory_id: str = Field(description="Id of the dismissed memory/suggestion.")
    context_key: str = Field(
        description="Working-context key the dismissal applies to (FR-RET-6)."
    )
    suppressed_at: datetime = Field(default_factory=_utcnow)


# Default free-form category for a memory whose classifier did not emit one.
DEFAULT_CATEGORY = "fact"


class MemoryUnit(BaseModel):
    """The central distilled memory record (§7).

    A ``MemoryUnit`` is the output of typed extraction (FR-EXT-1..4) and the
    subject of the 4-way write decision (FR-MNT-1). Its validity window encodes
    supersession (UC-2 / Goal 2): a reversed preference has ``valid_to`` set and
    drops off the hot path, while the new value is inserted active.

    Classification is now split (core redesign):

    * :attr:`scope` is the HIERARCHICAL :class:`Scope` path (global or
      ``project:<name>[/<sub>...]``), driven by the controlled
      :class:`ScopeDecision` the extractor makes (FR-EXT-3 / no-leak).
    * :attr:`category` is a FREE-FORM, emergent classifier string (no fixed
      enum) — it replaces the *semantic role* of the old ``MemoryType``.
    * :attr:`cross_ref_candidate` is the boolean preserving the old
      ``idea_seed`` cross-reference behavior (FR-RET-6) as a flag, not a type.
    """

    id: str = Field(default_factory=_new_id)
    content: str = Field(description="The distilled memory statement.")
    scope: Scope = Field(
        description="Hierarchical scope path (FR-EXT-3, FR-STO-3, no-leak)."
    )
    # --- the category split (replaces the old MemoryType) -------------------
    category: str = Field(
        default=DEFAULT_CATEGORY,
        description=(
            "FREE-FORM, emergent classifier category (no fixed enum); replaces "
            "the semantic role of the old MemoryType (e.g. 'preference', "
            "'decision', 'gotcha', 'idea'). Merged/normalized by the category "
            "maintenance job (CategoryMerger)."
        ),
    )
    cross_ref_candidate: bool = Field(
        default=False,
        description=(
            "True if the classifier flags this as a cross-reference seed "
            "(preserves the old idea_seed behavior as a flag, FR-RET-6/UC-3)."
        ),
    )

    entities: list[str] = Field(
        default_factory=list,
        description="Canonical names (or ids) of linked entities (FR-EXT-2).",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence (FR-EXT-4).",
    )
    provenance: Provenance = Field(
        default_factory=Provenance.classify_sentinel,
        description=(
            "Link back to source (FR-EXT-4). Defaults to the classify sentinel "
            "(source='classify', session_id='') so the single-statement "
            "Extractor.classify path can build a valid unit without an ingest "
            "session; extract()/persisted units MUST carry real provenance."
        ),
    )

    # Temporal validity window (FR-STO-1, FR-MNT-1).
    valid_from: datetime = Field(default_factory=_utcnow)
    valid_to: datetime | None = Field(
        default=None,
        description="None = active/current; a timestamp = superseded/closed (UC-2).",
    )

    # Tiering + decay bookkeeping (FR-STO-4, FR-MNT-3).
    tier: Tier = Field(default=Tier.HOT, description="hot or archive (FR-STO-4).")
    last_accessed: datetime | None = Field(
        default=None,
        description="Last retrieval time, for decay ranking (FR-MNT-3).",
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times retrieved, for decay ranking (FR-MNT-3).",
    )

    # Data-versioning (in-place migration framework, mnemozine.migrations).
    data_version: int = Field(
        default=CURRENT_DATA_VERSION,
        ge=0,
        description=(
            "Data-model version this record conforms to; stamped at write time "
            "(defaults to mnemozine.migrations.CURRENT_DATA_VERSION). A migration "
            "only touches records with data_version < its own version and stamps "
            "them up. Records written before this feature read back as 0."
        ),
    )

    @field_validator("content")
    @classmethod
    def _content_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("MemoryUnit.content must be non-empty")
        return v

    @field_validator("category")
    @classmethod
    def _category_normalized(cls, v: str) -> str:
        """Normalize the free-form category to a stable lowercased slug.

        Categories are emergent (no enum) but must compare/merge stably, so they
        are lowercased and whitespace-trimmed at the boundary. An empty category
        falls back to :data:`DEFAULT_CATEGORY` rather than failing validation.
        """

        slug = v.strip().lower()
        return slug or DEFAULT_CATEGORY

    @property
    def is_active(self) -> bool:
        """True if this memory is current (validity window still open)."""

        return self.valid_to is None

    @property
    def scope_decision(self) -> ScopeDecision:
        """The controlled scope decision implied by this unit's scope (FR-EXT-3)."""

        return ScopeDecision.GLOBAL if self.scope.is_global else ScopeDecision.PROJECT

    def supersede(self, at: datetime | None = None) -> None:
        """Close this memory's validity window in place (FR-MNT-1 supersede).

        Sets ``valid_to`` so the unit moves off the hot retrieval path while
        remaining retained for history/cross-reference (never hard-deleted).
        """

        self.valid_to = at or _utcnow()


class RawChunk(BaseModel):
    """The retained extraction-input chunk — a first-class STORED tier (§7, R4).

    A ``RawChunk`` is the *normalized* chunk (tool-calls already stripped per
    FR-ING-7) that extraction consumed, persisted so the store can:

    * **re-extract / reindex offline** — re-run a newer extractor/classifier or
      embedding model over the same input without the original transcript;
    * **survive Claude's 30-day local cleanup (R4)** — the raw transcript on
      disk is transient; this is the durable copy of what was actually ingested.

    It is keyed on :attr:`content_hash` (the FR-ING-5
    ``chunk_content_hash(events)``) so re-ingesting the same chunk de-duplicates,
    and links forward to the :attr:`memory_ids` it produced so a re-extraction
    can supersede/replace exactly those memories.
    """

    id: str = Field(default_factory=_new_id)
    content_hash: str = Field(
        description=(
            "FR-ING-5 chunk_content_hash over the normalized events; the "
            "idempotency/join key for re-extraction."
        ),
    )
    content: str = Field(
        description=(
            "Normalized chunk text fed to extraction (tool_calls already "
            "stripped per FR-ING-7). The durable re-extraction input."
        ),
    )
    source: str = Field(description="Originating source, e.g. 'claude_code' (FR-ING-1).")
    session_id: str = Field(description="Originating session id (FR-ING-2).")
    scope: Scope = Field(
        description="Hierarchical scope this chunk was ingested under (FR-EXT-3)."
    )
    project: str = Field(
        description="Derived project name (the scope's project segment; FR-ING-2).",
    )
    started_at: datetime | None = Field(
        default=None, description="Timestamp of the first event in the chunk."
    )
    ended_at: datetime | None = Field(
        default=None, description="Timestamp of the last event in the chunk."
    )
    event_count: int = Field(
        default=0, ge=0, description="Number of normalized events in the chunk."
    )
    raw_path: str | None = Field(
        default=None,
        description="Path to the originating raw transcript (provenance; may be cleaned up).",
    )
    memory_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Ids of the MemoryUnits this chunk produced; lets a re-extraction "
            "supersede/replace exactly those memories (offline reindex)."
        ),
    )
    ingested_at: datetime = Field(
        default_factory=_utcnow,
        description="When this raw chunk was persisted (retention bookkeeping).",
    )
    # Data-versioning (in-place migration framework, mnemozine.migrations). A
    # chunk is re-stamped to CURRENT_DATA_VERSION on re-extraction; raw chunks
    # written before this feature read back as 0.
    data_version: int = Field(
        default=CURRENT_DATA_VERSION,
        ge=0,
        description=(
            "Data-model version this raw chunk conforms to; stamped at write time "
            "(defaults to mnemozine.migrations.CURRENT_DATA_VERSION). Used by "
            "min_data_version() and the re-extraction migration path. Records "
            "written before this feature read back as 0."
        ),
    )


class SourceSession(BaseModel):
    """A record of one ingested source session (§7 Source/Session).

    Tracks the raw transcript path for the archive tier (FR-STO-4) and the
    session window. Idempotent ingest keys on ``(source, session_id, ...)``
    (FR-ING-5).
    """

    source: str = Field(description="Originating source, e.g. 'claude_code'.")
    session_id: str = Field(description="Session id.")
    project: str = Field(description="Derived/explicit project (FR-ING-2).")
    started_at: datetime | None = Field(default=None)
    ended_at: datetime | None = Field(default=None)
    raw_path: str | None = Field(
        default=None,
        description="Path to the raw transcript (for archive/provenance).",
    )
