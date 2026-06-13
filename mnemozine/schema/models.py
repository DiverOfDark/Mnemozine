"""The §7 data model: MemoryUnit, Entity, Edge, SourceSession + enums.

These pydantic models are the durable shapes stored in / retrieved from the
temporal knowledge graph. They are deliberately storage-agnostic: a
``StorageBackend`` implementation maps them onto Graphiti/FalkorDB nodes and
edges, but every other layer (extraction, retrieval, maintenance, MCP) speaks
in these models.

Temporal semantics (FR-STO-1 / FR-MNT-1): facts carry a validity window
``(valid_from, valid_to)``. A superseded fact is *not* deleted — its window is
closed (``valid_to = now``), moving it off the hot retrieval path while keeping
it for history and cross-reference (FR-STO-4, FR-MNT-3).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    """Timezone-aware UTC now, used as the default validity-window start."""

    return datetime.now(UTC)


def _new_id() -> str:
    """Generate a fresh node id."""

    return uuid4().hex


class MemoryType(str, Enum):
    """The exactly-one classification of a memory unit (FR-EXT-1).

    * ``preference``   — durable cross-project fact about how the operator works;
      lives in *global* scope. E.g. "prefers ``thiserror`` over ``anyhow``".
    * ``project_fact`` — specific to one project, must NOT leak across projects;
      lives in *project* scope. E.g. "project A pins tokio 1.38".
    * ``idea_seed``    — a candidate project/concept; a first-class graph node
      with its own embedding + entities. Powers cross-referencing (FR-RET-6).
    """

    PREFERENCE = "preference"
    PROJECT_FACT = "project_fact"
    IDEA_SEED = "idea_seed"


class Tier(str, Enum):
    """Storage tier of a memory unit (§7, FR-STO-4, FR-MNT-3).

    * ``hot``     — on the default retrieval path.
    * ``archive`` — cold tier: retained for history/cross-reference but excluded
      from default hot retrieval.
    """

    HOT = "hot"
    ARCHIVE = "archive"


# Sentinel literals for the two scope kinds (PRD §3/§5.5: global + project:<id>).
GLOBAL_SCOPE = "global"
_PROJECT_PREFIX = "project:"


class Scope(BaseModel):
    """A retrieval/storage scope (PRD §3, §5.5, FR-STO-3, FR-EXT-3).

    Single-operator only: scopes are ``global`` or ``project:<id>``. There is no
    ``user_id`` partitioning. Use the constructors :meth:`global_` and
    :meth:`project` rather than building by hand.

    The string form (:meth:`as_str`) is what gets persisted as a tag on a
    ``MemoryUnit``; :meth:`parse` is the inverse.
    """

    project_id: str | None = Field(
        default=None,
        description="Project id for a project scope; None means the global scope.",
    )

    @classmethod
    def global_(cls) -> Scope:
        """The cross-project global scope (where preferences live)."""

        return cls(project_id=None)

    @classmethod
    def project(cls, project_id: str) -> Scope:
        """A project-specific scope ``project:<id>`` (where project_facts live)."""

        if not project_id:
            raise ValueError("project_id must be non-empty for a project scope")
        return cls(project_id=project_id)

    @property
    def is_global(self) -> bool:
        """True if this is the global scope."""

        return self.project_id is None

    def as_str(self) -> str:
        """Serialize to the persisted tag form: ``global`` or ``project:<id>``."""

        if self.project_id is None:
            return GLOBAL_SCOPE
        return f"{_PROJECT_PREFIX}{self.project_id}"

    @classmethod
    def parse(cls, value: str) -> Scope:
        """Parse a persisted scope string back into a :class:`Scope`."""

        if value == GLOBAL_SCOPE:
            return cls.global_()
        if value.startswith(_PROJECT_PREFIX):
            return cls.project(value[len(_PROJECT_PREFIX) :])
        raise ValueError(f"unrecognized scope string: {value!r}")

    def __hash__(self) -> int:  # allow use in sets/dict keys
        return hash(self.as_str())


# Sentinel source value for provenance produced by the single-statement
# classifier path (Extractor.classify) rather than a real ingest session.
CLASSIFY_SOURCE = "classify"


class Provenance(BaseModel):
    """A link from a memory back to its originating source (FR-EXT-4).

    Every memory records where it came from so it can be audited (R5) and so the
    archive tier can preserve the raw path (§7 Source/Session.raw_path).
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


class MemoryUnit(BaseModel):
    """The central distilled memory record (§7).

    A ``MemoryUnit`` is the output of typed extraction (FR-EXT-1..4) and the
    subject of the 4-way write decision (FR-MNT-1). Its validity window encodes
    supersession (UC-2 / Goal 2): a reversed preference has ``valid_to`` set and
    drops off the hot path, while the new value is inserted active.
    """

    id: str = Field(default_factory=_new_id)
    type: MemoryType = Field(description="Exactly-one classification (FR-EXT-1).")
    content: str = Field(description="The distilled memory statement.")
    scope: Scope = Field(description="global or project:<id> (FR-EXT-3, FR-STO-3).")
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

    @field_validator("content")
    @classmethod
    def _content_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("MemoryUnit.content must be non-empty")
        return v

    @property
    def is_active(self) -> bool:
        """True if this memory is current (validity window still open)."""

        return self.valid_to is None

    def supersede(self, at: datetime | None = None) -> None:
        """Close this memory's validity window in place (FR-MNT-1 supersede).

        Sets ``valid_to`` so the unit moves off the hot retrieval path while
        remaining retained for history/cross-reference (never hard-deleted).
        """

        self.valid_to = at or _utcnow()


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
