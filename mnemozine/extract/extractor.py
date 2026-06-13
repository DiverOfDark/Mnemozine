"""Typed extraction pipeline — the crux component (FR-EXT-1..4, R1).

:class:`TypedExtractor` implements :class:`mnemozine.interfaces.Extractor`. It
turns a chunk of :class:`~mnemozine.schema.events.IngestEvent`s into classified,
scoped, entity-linked :class:`~mnemozine.schema.models.MemoryUnit`s, and exposes
the single-statement :meth:`classify` path used by the §9 classifier-accuracy
eval and by reclassification (R1).

Design notes (why this is structured the way it is):

* **Pluggable LLM.** The only model dependency is an injected
  :class:`mnemozine.interfaces.LLMProvider`. That makes the whole layer
  unit-testable offline with ``FakeLLMProvider`` — no live Qwen — which is the
  PRD's explicit requirement for the make-or-break component (R1, FR-EXT-3).
* **Scope is derived from type in Python, not trusted from the model.** The
  prompt asks the model for a scope, but :meth:`_scope_for` re-derives it
  deterministically from the chosen ``type`` and the chunk's project id:
  ``preference``/``idea_seed`` -> global, ``project_fact`` -> ``project:<id>``.
  This is what actually enforces FR-EXT-3 (scope set at extraction time, no
  cross-project leak) even when the model returns an inconsistent scope string.
* **Provenance is built from the chunk, never invented by the model** (FR-EXT-4):
  it links back to the source/session/chunk-hash of the originating events.
* **Prompts live in** :mod:`mnemozine.extract.prompts` so they are independently
  evaluable and diffable.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from mnemozine.config import Settings, get_settings
from mnemozine.extract.prompts import (
    CLASSIFY_JSON_SCHEMA,
    CLASSIFY_SYSTEM_PROMPT,
    EXTRACT_JSON_SCHEMA,
    EXTRACT_SYSTEM_PROMPT,
    build_classify_prompt,
    build_extract_prompt,
)
from mnemozine.interfaces import Classification, LLMProvider, RetrievalContext
from mnemozine.schema.events import IngestEvent, chunk_content_hash
from mnemozine.schema.models import (
    Edge,
    Entity,
    MemoryType,
    MemoryUnit,
    Provenance,
    Scope,
)

logger = logging.getLogger(__name__)

# Memory types whose scope is always global (FR-EXT-3). project_fact is the only
# type that takes a project scope.
_GLOBAL_TYPES = frozenset({MemoryType.PREFERENCE, MemoryType.IDEA_SEED})


class ExtractedRelationship:
    """A parsed (subject, relation, object) triple from chunk extraction (FR-EXT-2).

    A light value object (not a schema model) returned alongside the memory units
    so the storage/integration layer can materialize :class:`Entity` +
    :class:`Edge` records with timestamps. :meth:`to_edge` builds the temporal
    edge once the entity ids are known.
    """

    __slots__ = ("subject", "relation", "object")

    def __init__(self, subject: str, relation: str, object: str) -> None:  # noqa: A002
        self.subject = subject
        self.relation = relation
        self.object = object

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"ExtractedRelationship(subject={self.subject!r}, "
            f"relation={self.relation!r}, object={self.object!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtractedRelationship):
            return NotImplemented
        return (
            self.subject == other.subject
            and self.relation == other.relation
            and self.object == other.object
        )

    def to_edge(self, from_entity_id: str, to_entity_id: str) -> Edge:
        """Build a temporal :class:`Edge` for this triple (FR-EXT-2).

        Edge ids of the subject/object entities are resolved by the caller
        (typically via :meth:`StorageBackend.upsert_entity`/``get_entity``); the
        edge is created with an open validity window (``valid_to=None``) and the
        relation label, written into the graph with timestamps per FR-EXT-2.
        """

        return Edge(
            from_entity=from_entity_id,
            to_entity=to_entity_id,
            relation=self.relation,
        )


class ExtractionResult:
    """The full output of :meth:`TypedExtractor.extract_full` (FR-EXT-1/2/4).

    Bundles the classified memory units with the extracted entities and
    relationship triples so the integration/storage pass can write the graph in
    one place. :meth:`Extractor.extract` (the Protocol method) returns only the
    ``memories`` list; ``extract_full`` exposes the entities/relationships too.
    """

    __slots__ = ("memories", "entities", "relationships")

    def __init__(
        self,
        memories: list[MemoryUnit],
        entities: list[Entity],
        relationships: list[ExtractedRelationship],
    ) -> None:
        self.memories = memories
        self.entities = entities
        self.relationships = relationships


def _coerce_confidence(value: Any) -> float:
    """Clamp a model-supplied confidence into [0,1]; default to 0.5 on garbage."""

    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.5
    if c < 0.0:
        return 0.0
    if c > 1.0:
        return 1.0
    return c


def _coerce_entities(value: Any) -> list[str]:
    """Normalize a model-supplied entity list: lowercase, trimmed, de-duped."""

    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        tag = item.strip().lower()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def _parse_type(value: Any) -> MemoryType | None:
    """Parse a model ``type`` string into a :class:`MemoryType` (or None)."""

    if not isinstance(value, str):
        return None
    try:
        return MemoryType(value.strip().lower())
    except ValueError:
        return None


class TypedExtractor:
    """Typed extraction over chunks + single statements (FR-EXT-1..4).

    Implements :class:`mnemozine.interfaces.Extractor` structurally. Construct
    with a :class:`~mnemozine.interfaces.LLMProvider` (the only I/O dependency)
    and optional :class:`~mnemozine.config.Settings`; both default-injectable so
    tests can pass a deterministic fake LLM.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        settings: Settings | None = None,
        min_confidence: float = 0.0,
    ) -> None:
        self._llm = llm
        self._settings = settings or get_settings()
        # Memories below this confidence are dropped from extract() (the chunk
        # path). 0.0 keeps everything; callers/evals can raise it. The
        # single-statement classify() path never drops — it always returns its
        # best Classification so the R1 eval can score it.
        self._min_confidence = min_confidence

    # -- scope derivation (FR-EXT-3) --------------------------------------

    def _scope_for(self, mtype: MemoryType, project: str | None) -> Scope:
        """Derive the scope from the memory type at extraction time (FR-EXT-3).

        This is the load-bearing enforcement of FR-EXT-3: scope is a pure
        function of the (trusted) type and the chunk's project, *not* of the
        model's free-text ``scope`` string. ``preference``/``idea_seed`` are
        always global; ``project_fact`` is scoped to the current project so it
        can never leak into another project. A ``project_fact`` with no project
        id falls back to global rather than crashing, but that should not happen
        for a real chunk (which always carries a project).
        """

        if mtype in _GLOBAL_TYPES:
            return Scope.global_()
        if project:
            return Scope.project(project)
        logger.warning(
            "project_fact with no project id; falling back to global scope"
        )
        return Scope.global_()

    # -- chunk extraction (FR-EXT-1/2/3/4) --------------------------------

    async def extract(self, chunk: Sequence[IngestEvent]) -> list[MemoryUnit]:
        """Extract classified, scoped, provenanced memory units from a chunk.

        See :meth:`extract_full` for the entity/relationship-carrying variant;
        this returns only the memory units, as the
        :class:`~mnemozine.interfaces.Extractor` Protocol requires. Returns an
        empty list when the chunk yields no durable memory.
        """

        result = await self.extract_full(chunk)
        return result.memories

    async def extract_full(self, chunk: Sequence[IngestEvent]) -> ExtractionResult:
        """Extract memories + entities + relationships from one chunk (FR-EXT-1/2/4).

        The richer entry point used by the integration/storage pass, which needs
        the entities and relationship triples to write the graph (FR-EXT-2). The
        Protocol :meth:`extract` delegates here.
        """

        events = list(chunk)
        if not events:
            return ExtractionResult(memories=[], entities=[], relationships=[])

        project = events[0].project
        provenance = self._provenance_for(events)

        prompt = build_extract_prompt(events, project=project)
        raw = await self._llm.complete_json(
            prompt,
            schema=EXTRACT_JSON_SCHEMA,
            system=EXTRACT_SYSTEM_PROMPT,
            temperature=self._settings.extraction.temperature,
        )

        memories = self._parse_memories(raw, project=project, provenance=provenance)
        relationships = self._parse_relationships(raw)
        entities = self._collect_entities(memories, relationships)
        return ExtractionResult(
            memories=memories, entities=entities, relationships=relationships
        )

    def _provenance_for(self, events: Sequence[IngestEvent]) -> Provenance:
        """Build provenance linking memories back to the source chunk (FR-EXT-4)."""

        first = events[0]
        raw_path = first.metadata.get("raw_path")
        return Provenance(
            source=first.source.value,
            session_id=first.session_id,
            chunk_hash=chunk_content_hash(list(events)),
            raw_path=str(raw_path) if isinstance(raw_path, str) else None,
        )

    def _parse_memories(
        self,
        raw: dict[str, Any],
        *,
        project: str,
        provenance: Provenance,
    ) -> list[MemoryUnit]:
        """Parse the model's ``memories`` array into validated MemoryUnits.

        Skips malformed entries (missing/blank content, unrecognized type) and
        memories below ``min_confidence``. Scope is re-derived from type
        (FR-EXT-3), provenance is the chunk's (FR-EXT-4) — neither is taken from
        the model.
        """

        items = raw.get("memories")
        if not isinstance(items, list):
            return []

        out: list[MemoryUnit] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            mtype = _parse_type(item.get("type"))
            if mtype is None:
                continue
            confidence = _coerce_confidence(item.get("confidence"))
            if confidence < self._min_confidence:
                continue
            scope = self._scope_for(mtype, project)
            entities = _coerce_entities(item.get("entities"))
            out.append(
                MemoryUnit(
                    type=mtype,
                    content=content.strip(),
                    scope=scope,
                    entities=entities,
                    confidence=confidence,
                    provenance=provenance,
                )
            )
        return out

    def _parse_relationships(self, raw: dict[str, Any]) -> list[ExtractedRelationship]:
        """Parse the model's ``relationships`` triples (FR-EXT-2)."""

        items = raw.get("relationships")
        if not isinstance(items, list):
            return []
        out: list[ExtractedRelationship] = []
        seen: set[tuple[str, str, str]] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            subject = item.get("subject")
            relation = item.get("relation")
            obj = item.get("object")
            if not (
                isinstance(subject, str)
                and isinstance(relation, str)
                and isinstance(obj, str)
            ):
                continue
            s = subject.strip().lower()
            r = relation.strip().lower()
            o = obj.strip().lower()
            if not (s and r and o):
                continue
            key = (s, r, o)
            if key in seen:
                continue
            seen.add(key)
            out.append(ExtractedRelationship(subject=s, relation=r, object=o))
        return out

    def _collect_entities(
        self,
        memories: Sequence[MemoryUnit],
        relationships: Sequence[ExtractedRelationship],
    ) -> list[Entity]:
        """Build the de-duped set of :class:`Entity` nodes (FR-EXT-2).

        Union of every entity tag mentioned on a memory and every endpoint of a
        relationship triple. Entities are emitted by canonical name with no
        ``type`` (entity typing/resolution is the maintenance layer's job,
        FR-MNT-4); the integration pass upserts them so ids are stable.
        """

        names: list[str] = []
        seen: set[str] = set()
        for m in memories:
            for name in m.entities:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        for rel in relationships:
            for name in (rel.subject, rel.object):
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return [Entity(canonical_name=name) for name in names]

    # -- single-statement classification (FR-EXT-3 eval path, R1) ----------

    async def classify(
        self, statement: str, context: RetrievalContext
    ) -> Classification:
        """Classify one statement into a lightweight :class:`Classification` (R1).

        The independently-testable path the §9 classifier-accuracy metric is
        measured on. Returns ``type``/``scope``/``entities``/``confidence`` with
        no provenance/validity (a bare eval statement has no originating
        session). Scope is re-derived from the chosen type (FR-EXT-3), so a
        ``project_fact`` is scoped to ``context.project``. An unparseable or
        empty model response yields a low-confidence ``idea_seed`` fallback so
        the caller can drop it rather than crash.
        """

        prompt = build_classify_prompt(
            statement,
            project=context.project,
            recent_text=context.recent_text,
        )
        raw = await self._llm.complete_json(
            prompt,
            schema=CLASSIFY_JSON_SCHEMA,
            system=CLASSIFY_SYSTEM_PROMPT,
            temperature=self._settings.extraction.temperature,
        )

        mtype = _parse_type(raw.get("type")) if isinstance(raw, dict) else None
        if mtype is None:
            # Unparseable response: return a droppable low-confidence result
            # rather than raising, so an eval batch keeps going.
            return Classification(
                type=MemoryType.IDEA_SEED,
                scope=Scope.global_(),
                entities=[],
                confidence=0.0,
            )

        scope = self._scope_for(mtype, context.project)
        entities = _coerce_entities(raw.get("entities"))
        confidence = _coerce_confidence(raw.get("confidence"))
        return Classification(
            type=mtype,
            scope=scope,
            entities=entities,
            confidence=confidence,
        )
