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
* **Scope is derived in Python, never trusted from the model (no-leak).** The
  prompt asks the model only for the CONTROLLED
  :class:`~mnemozine.schema.models.ScopeDecision` (``global`` vs ``project``);
  :meth:`_scope_for` then derives the final HIERARCHICAL
  :class:`~mnemozine.schema.models.Scope` deterministically from that decision
  and the chunk's PROVENANCE PROJECT PATH: ``global`` -> the root scope,
  ``project`` -> ``derive_scope_from_transcript`` over the originating transcript
  path (so a deep subagent/workflow transcript ROLLS UP to its parent project —
  no opaque ``project:agent-XXXX``), falling back to ``project:<event.project>``
  when no path is available. This is what actually enforces FR-EXT-3 (scope set
  at extraction time, no cross-project leak) even when the model returns an
  inconsistent scope string.
* **Free-form category + cross-ref flag (category split).** The classifier emits
  a FREE-FORM ``category`` slug (the semantic role; no enum) and a ``cross_ref``
  boolean (the old idea_seed cross-reference behavior as a flag, FR-RET-6), NOT
  the old 3-value ``MemoryType``.
* **Provenance is built from the chunk, never invented by the model** (FR-EXT-4):
  it links back to the source/session/chunk-hash of the originating events.
* **Raw-chunk retention.** :func:`build_raw_chunk` builds the first-class
  :class:`~mnemozine.schema.models.RawChunk` (the normalized extraction input) so
  the orchestrator can persist it to the storage raw tier before/with extraction
  (FR-ING-5 idempotency key, R4 survival).
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
from mnemozine.ingestion.claude_code.parser import derive_scope_from_transcript
from mnemozine.interfaces import Classification, LLMProvider, RetrievalContext
from mnemozine.schema.events import IngestEvent, chunk_content_hash
from mnemozine.schema.models import (
    DEFAULT_CATEGORY,
    Edge,
    Entity,
    MemoryUnit,
    Provenance,
    RawChunk,
    Scope,
    ScopeDecision,
)

logger = logging.getLogger(__name__)


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


def _parse_scope_decision(value: Any) -> ScopeDecision | None:
    """Parse the model's controlled ``scope`` decision into a :class:`ScopeDecision`.

    Accepts the two controlled values ``global`` / ``project`` (case-insensitive).
    Returns ``None`` for anything else so the caller can decide how to degrade —
    the model never supplies a scope *path* we trust, only this decision.
    """

    if not isinstance(value, str):
        return None
    try:
        return ScopeDecision(value.strip().lower())
    except ValueError:
        return None


def _coerce_category(value: Any) -> str:
    """Normalize a model-supplied free-form category to a lowercased slug.

    Empty / non-string values fall back to :data:`DEFAULT_CATEGORY`. (The
    :class:`MemoryUnit` validator normalizes again, but the classify() path
    returns a bare :class:`Classification`, so normalize here too.)
    """

    if not isinstance(value, str):
        return DEFAULT_CATEGORY
    slug = value.strip().lower()
    return slug or DEFAULT_CATEGORY


def _coerce_cross_ref(value: Any) -> bool:
    """Coerce the model's ``cross_ref`` flag to a bool (tolerant of strings)."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def build_raw_chunk(
    events: Sequence[IngestEvent],
    scope: Scope,
    *,
    memory_ids: Sequence[str] | None = None,
) -> RawChunk:
    """Build the retained :class:`RawChunk` (raw tier) for a chunk of events.

    The normalized extraction-input chunk persisted via
    :meth:`StorageBackend.persist_raw_chunk` so the store can re-extract / reindex
    offline and survive Claude's 30-day local cleanup (R4). ``content`` is the
    rendered role-tagged transcript (tool_calls already stripped upstream per
    FR-ING-7); ``content_hash`` is the FR-ING-5
    :func:`~mnemozine.schema.events.chunk_content_hash` (the idempotency/join key,
    matching the chunk hash so a memory's ``provenance.chunk_hash`` joins back to
    its raw chunk). ``scope`` is the derived hierarchical scope; ``project`` is its
    project segment. ``memory_ids`` links forward to the units this chunk produced
    (so a re-extraction can supersede exactly those).
    """

    from mnemozine.extract.prompts.extract import render_chunk

    evs = list(events)
    first = evs[0]
    raw_path = first.metadata.get("raw_path")
    timestamps = [e.timestamp for e in evs if e.timestamp is not None]
    return RawChunk(
        content_hash=chunk_content_hash(evs),
        content=render_chunk(evs),
        source=first.source.value,
        session_id=first.session_id,
        scope=scope,
        project=scope.project_id or first.project,
        started_at=min(timestamps) if timestamps else None,
        ended_at=max(timestamps) if timestamps else None,
        event_count=len(evs),
        raw_path=str(raw_path) if isinstance(raw_path, str) else None,
        memory_ids=list(memory_ids or []),
    )


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

    # -- scope derivation (FR-EXT-3, no-leak) ------------------------------

    def _project_scope_from_events(self, events: Sequence[IngestEvent]) -> Scope:
        """Derive the hierarchical PROJECT scope for a chunk (FR-EXT-3 roll-up).

        Routes through
        :func:`~mnemozine.ingestion.claude_code.parser.derive_scope_from_transcript`
        when the chunk carries an originating transcript ``raw_path`` (so a deep
        subagent/workflow transcript ROLLS UP to its parent project — never an
        opaque ``project:agent-XXXX``), using the event ``cwd`` as the literal
        leaf when present. Falls back to ``project:<event.project>`` when no path
        is available (e.g. gateway/Hermes turns that have no transcript on disk).
        """

        first = events[0]
        raw_path = first.metadata.get("raw_path")
        if isinstance(raw_path, str) and raw_path:
            cwd = first.metadata.get("cwd")
            return derive_scope_from_transcript(
                raw_path,
                self._settings,
                cwd=cwd if isinstance(cwd, str) and cwd else None,
            )
        if first.project:
            return Scope.project(first.project)
        logger.warning("chunk with no project and no raw_path; using global scope")
        return Scope.global_()

    def _scope_for(self, decision: ScopeDecision, project_scope: Scope) -> Scope:
        """Derive the final scope from the CONTROLLED decision (FR-EXT-3, no-leak).

        This is the load-bearing enforcement of FR-EXT-3: the final scope is a
        pure function of the controlled :class:`ScopeDecision` (global vs project)
        and the chunk's *derived* project scope, NOT of the model's free-text
        scope string. ``global`` -> the root scope; ``project`` -> the derived
        hierarchical project scope (already rolled up from the transcript path).
        """

        if decision is ScopeDecision.GLOBAL:
            return Scope.global_()
        if not project_scope.is_global:
            return project_scope
        logger.warning(
            "project-decision memory with no project scope; falling back to global"
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

        project_scope = self._project_scope_from_events(events)
        provenance = self._provenance_for(events)

        prompt = build_extract_prompt(events, project=project_scope.project_id or "unknown")
        raw = await self._llm.complete_json(
            prompt,
            schema=EXTRACT_JSON_SCHEMA,
            system=EXTRACT_SYSTEM_PROMPT,
            temperature=self._settings.extraction.temperature,
        )

        memories = self._parse_memories(
            raw, project_scope=project_scope, provenance=provenance
        )
        relationships = self._parse_relationships(raw)
        entities = self._collect_entities(memories, relationships)
        return ExtractionResult(
            memories=memories, entities=entities, relationships=relationships
        )

    def build_raw_chunk(
        self,
        chunk: Sequence[IngestEvent],
        *,
        memory_ids: Sequence[str] | None = None,
    ) -> RawChunk | None:
        """Build the :class:`RawChunk` for a chunk (raw-tier retention seam).

        A convenience over the module-level :func:`build_raw_chunk` that derives
        the hierarchical scope the same way :meth:`extract_full` does (so the raw
        chunk and the memories it produced share the rolled-up project scope). The
        orchestrator persists the returned chunk via
        :meth:`StorageBackend.persist_raw_chunk` before/with extraction when
        ``ingest.raw_retention_enabled`` is on. Returns ``None`` for an empty chunk.
        """

        events = list(chunk)
        if not events:
            return None
        scope = self._project_scope_from_events(events)
        return build_raw_chunk(events, scope, memory_ids=memory_ids)

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
        project_scope: Scope,
        provenance: Provenance,
    ) -> list[MemoryUnit]:
        """Parse the model's ``memories`` array into validated MemoryUnits.

        Skips malformed entries (missing/blank content, missing/unparseable scope
        decision) and memories below ``min_confidence``. Each entry carries the
        category-split signals: the CONTROLLED ``scope`` decision drives the
        re-derived hierarchical scope (FR-EXT-3, no-leak — never the model's
        string), the free-form ``category`` carries the semantic role, and
        ``cross_ref`` preserves the old idea_seed flag. Provenance is the chunk's
        (FR-EXT-4) — neither scope nor provenance is taken from the model.
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
            decision = _parse_scope_decision(item.get("scope"))
            if decision is None:
                continue
            confidence = _coerce_confidence(item.get("confidence"))
            if confidence < self._min_confidence:
                continue
            scope = self._scope_for(decision, project_scope)
            out.append(
                MemoryUnit(
                    content=content.strip(),
                    scope=scope,
                    category=_coerce_category(item.get("category")),
                    cross_ref_candidate=_coerce_cross_ref(item.get("cross_ref")),
                    entities=_coerce_entities(item.get("entities")),
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
        measured on. Returns the category-split contract
        (``scope_decision``/``scope``/``category``/``cross_ref_candidate``/
        ``entities``/``confidence``) with no provenance/validity (a bare eval
        statement has no originating session). The controlled
        :class:`ScopeDecision` drives the re-derived scope (FR-EXT-3), so a
        project-decision statement is scoped to ``context.project``. An unparseable
        or empty model response yields a low-confidence global fallback so the
        caller can drop it rather than crash.
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

        decision = (
            _parse_scope_decision(raw.get("scope")) if isinstance(raw, dict) else None
        )
        if decision is None:
            # Unparseable response: return a droppable low-confidence result
            # rather than raising, so an eval batch keeps going.
            return Classification(
                scope_decision=ScopeDecision.GLOBAL,
                scope=Scope.global_(),
                category=DEFAULT_CATEGORY,
                cross_ref_candidate=False,
                entities=[],
                confidence=0.0,
            )

        # Derive the final scope from the controlled decision + context project
        # (FR-EXT-3 — never the model's string). A bare classify statement has no
        # transcript path, so the project scope is the flat context project.
        project_scope = (
            Scope.project(context.project)
            if context.project
            else Scope.global_()
        )
        scope = self._scope_for(decision, project_scope)
        return Classification(
            scope_decision=decision,
            scope=scope,
            category=_coerce_category(raw.get("category")),
            cross_ref_candidate=_coerce_cross_ref(raw.get("cross_ref")),
            entities=_coerce_entities(raw.get("entities")),
            confidence=_coerce_confidence(raw.get("confidence")),
        )
