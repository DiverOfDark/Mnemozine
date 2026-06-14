"""Shared test fakes and fixtures.

These let every module unit-test against the :mod:`mnemozine.interfaces`
Protocols **without** a live FalkorDB, Ollama, or Qwen endpoint. The fakes
implement just enough behavior to be useful while staying deterministic:

* :class:`InMemoryStorage`     — dict-backed ``StorageBackend`` with a simple but
  real implementation of the FR-MNT-1 4-way write decision.
* :class:`FakeEmbeddingProvider` — deterministic hash-based pseudo-embeddings.
* :class:`FakeLLMProvider`     — scriptable, deterministic LLM responses.

Plus fixtures providing those fakes, a test :class:`Settings`, and sample
``IngestEvent``s.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from mnemozine.config import Settings
from mnemozine.interfaces import (
    Extractor,
    MaintenanceReport,
    Neighbor,
    RetrievedMemory,
    WriteDecision,
    WriteResult,
)
from mnemozine.schema.events import IngestEvent, Role, Source
from mnemozine.schema.models import (
    Edge,
    Entity,
    MemoryUnit,
    Provenance,
    RawChunk,
    Scope,
    SourceSession,
    Tier,
)

# ---------------------------------------------------------------------------
# Fake embedding provider
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider:
    """Deterministic pseudo-embeddings (no Ollama needed).

    Embeds text into a fixed-dimension vector derived from a hash so the same
    text always yields the same vector and similar tests are reproducible.
    Implements :class:`mnemozine.interfaces.EmbeddingProvider` structurally.
    """

    def __init__(self, dimensions: int = 8) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Spread the digest bytes across the requested dimension count.
        return [digest[i % len(digest)] / 255.0 for i in range(self._dimensions)]

    async def embed(self, text: str) -> list[float]:
        return self._vector(text)

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


# ---------------------------------------------------------------------------
# Fake LLM provider
# ---------------------------------------------------------------------------


class FakeLLMProvider:
    """Scriptable, deterministic LLM (no Qwen needed).

    Two scripting modes (use whichever a test needs):

    * **call-order queues** — ``text_responses`` / ``json_responses`` are popped
      in call order; when a queue is empty a benign default is returned. Fine for
      single-call tests.
    * **per-prompt routing** — pass a ``json_responder`` (and/or
      ``text_responder``) callable ``(prompt, system) -> response``. This is what
      multi-call extraction/contradiction tests need: a test that interleaves
      ``extract()`` and the FR-MNT-1 contradiction call cannot rely on call
      ORDER, so it routes by inspecting the prompt. A responder returning
      ``None`` falls back to the queue. Implements
      :class:`mnemozine.interfaces.LLMProvider` structurally.
    """

    def __init__(
        self,
        *,
        model: str = "fake/qwen",
        text_responses: list[str] | None = None,
        json_responses: list[dict[str, Any]] | None = None,
        text_responder: Callable[[str, str | None], str | None] | None = None,
        json_responder: (
            Callable[[str, str | None], dict[str, Any] | None] | None
        ) = None,
    ) -> None:
        self._model = model
        self.text_responses = list(text_responses or [])
        self.json_responses = list(json_responses or [])
        self.text_responder = text_responder
        self.json_responder = json_responder
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append({"kind": "complete", "prompt": prompt, "system": system})
        if self.text_responder is not None:
            routed = self.text_responder(prompt, system)
            if routed is not None:
                return routed
        if self.text_responses:
            return self.text_responses.pop(0)
        return ""

    async def complete_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | type[Any],
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self.calls.append({"kind": "json", "prompt": prompt, "system": system})
        if self.json_responder is not None:
            routed = self.json_responder(prompt, system)
            if routed is not None:
                return routed
        if self.json_responses:
            return self.json_responses.pop(0)
        return {}


# ---------------------------------------------------------------------------
# In-memory storage backend
# ---------------------------------------------------------------------------


def _overlaps(a: Sequence[str], b: Sequence[str]) -> bool:
    return bool(set(a) & set(b))


class InMemoryStorage:
    """Dict-backed :class:`mnemozine.interfaces.StorageBackend` for tests.

    Implements a real (if naive) FR-MNT-1 4-way write decision so write-path
    tests have something meaningful to assert against:

    * candidates = same scope + overlapping entities + active + type=preference;
    * exact-content match on an active unit -> **reinforce** (bump confidence);
    * a candidate flagged contradictory via ``contradicts`` -> **supersede**;
    * a strictly-lower-confidence duplicate of an existing unit -> **no-op**;
    * otherwise -> **add**.

    ``contradicts`` is an injectable predicate so tests can drive the supersede
    branch without an LLM. By default nothing contradicts.
    """

    def __init__(
        self,
        *,
        contradicts: Any | None = None,
    ) -> None:
        self.memories: dict[str, MemoryUnit] = {}
        self.entities: dict[str, Entity] = {}
        self.edges: dict[str, Edge] = {}
        self.sessions: list[SourceSession] = []
        # Raw-chunk tier (offline re-extraction/reindex), keyed on content_hash.
        self.raw_chunks: dict[str, RawChunk] = {}
        # Suppression store: set of (memory_id, context_key) pairs (FR-RET-6/R2).
        self.suppressions: set[tuple[str, str]] = set()
        # Count of reembed() calls per memory id, so re-embed tests can assert.
        self.reembed_calls: dict[str, int] = {}
        # contradicts(new: MemoryUnit, existing: MemoryUnit) -> bool
        self._contradicts = contradicts or (lambda new, existing: False)

    def _resolve_entity_id(self, name_or_id: str) -> str | None:
        """Resolve a name/alias/id to a stored entity id (or None)."""

        if name_or_id in self.entities:
            return name_or_id
        for e in self.entities.values():
            if e.canonical_name == name_or_id or name_or_id in e.aliases:
                return e.id
        return None

    # --- write decision (FR-MNT-1) ---------------------------------------

    def _candidates(self, memory: MemoryUnit) -> list[MemoryUnit]:
        return [
            m
            for m in self.memories.values()
            if m.is_active
            and m.scope.as_str() == memory.scope.as_str()
            and _overlaps(m.entities, memory.entities)
        ]

    async def upsert_memory(self, memory: MemoryUnit) -> WriteResult:
        candidates = self._candidates(memory)

        # reinforce: an equivalent (same content) active memory exists.
        for existing in candidates:
            if existing.content.strip() == memory.content.strip():
                existing.confidence = max(existing.confidence, memory.confidence)
                existing.last_accessed = datetime.now(UTC)
                return WriteResult(decision=WriteDecision.REINFORCE, memory=existing)

        # supersede: a contradicting active global-decision memory -> close window.
        superseded: list[MemoryUnit] = []
        for existing in candidates:
            if existing.scope.is_global and self._contradicts(memory, existing):
                existing.supersede()
                superseded.append(existing)
        if superseded:
            self.memories[memory.id] = memory
            return WriteResult(
                decision=WriteDecision.SUPERSEDE, memory=memory, superseded=superseded
            )

        # no-op: strictly weaker/older duplicate-ish memory already present.
        for existing in candidates:
            if (
                existing.category == memory.category
                and memory.confidence < existing.confidence
                and existing.content.strip().lower() == memory.content.strip().lower()
            ):
                return WriteResult(decision=WriteDecision.NO_OP, memory=existing)

        # add.
        self.memories[memory.id] = memory
        return WriteResult(decision=WriteDecision.ADD, memory=memory)

    # --- retrieval (FR-RET-2 / FR-STO-3) ---------------------------------

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
        # Ancestor-composition (no-leak): expand each query scope to its
        # ancestor-or-self chain so a memory matches iff its stored scope is an
        # ancestor-or-self of a query scope (siblings never leak).
        if compose_ancestors:
            scope_strs = {
                anc.as_str() for s in scopes for anc in s.ancestors()
            }
        else:
            scope_strs = {s.as_str() for s in scopes}
        q_words = set(query.lower().split())
        results: list[RetrievedMemory] = []
        for m in self.memories.values():
            if m.scope.as_str() not in scope_strs:
                continue
            if not m.is_active:
                continue
            if not include_archived and m.tier is Tier.ARCHIVE:
                continue
            if entities and not _overlaps(m.entities, entities):
                continue
            # Cheap lexical overlap score so tests get a stable ordering.
            m_words = set(m.content.lower().split())
            overlap = len(q_words & m_words)
            score = overlap / (len(q_words) or 1)
            if overlap or not q_words:
                results.append(RetrievedMemory(memory=m, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def close_validity_window(
        self, memory_id: str, *, at: Any | None = None
    ) -> MemoryUnit:
        m = self.memories[memory_id]
        m.supersede(at)
        return m

    async def archive(self, memory_id: str) -> MemoryUnit:
        m = self.memories[memory_id]
        m.tier = Tier.ARCHIVE
        return m

    async def promote(self, memory_id: str) -> MemoryUnit:
        # Inverse of archive; lazy re-embed on promotion (OQ3).
        m = self.memories[memory_id]
        m.tier = Tier.HOT
        await self.reembed(memory_id)
        return m

    async def reembed(self, memory_id: str) -> MemoryUnit:
        # No real embeddings in the fake; just record the call for assertions.
        m = self.memories[memory_id]
        self.reembed_calls[memory_id] = self.reembed_calls.get(memory_id, 0) + 1
        return m

    async def record_access(self, memory_id: str) -> None:
        m = self.memories[memory_id]
        m.access_count += 1
        m.last_accessed = datetime.now(UTC)

    # --- enumeration / scan (FR-MNT-2/3/4, R5) ---------------------------

    async def iter_memories(
        self,
        *,
        scope: Scope | None = None,
        tier: Any | None = None,
        active_only: bool = False,
        valid_before: datetime | None = None,
        unused_since: datetime | None = None,
    ) -> AsyncIterator[MemoryUnit]:
        scope_str = scope.as_str() if scope is not None else None
        for m in list(self.memories.values()):
            if scope_str is not None and m.scope.as_str() != scope_str:
                continue
            if tier is not None and m.tier is not tier:
                continue
            if active_only and not m.is_active:
                continue
            if valid_before is not None and m.valid_from >= valid_before:
                continue
            if unused_since is not None:
                # Never-accessed (None) counts as "unused since forever".
                if m.last_accessed is not None and m.last_accessed >= unused_since:
                    continue
            yield m

    async def iter_entities(self) -> AsyncIterator[Entity]:
        for e in list(self.entities.values()):
            yield e

    # --- entity ops (FR-EXT-2 / FR-MNT-4) --------------------------------

    async def upsert_entity(self, entity: Entity) -> Entity:
        self.entities[entity.id] = entity
        return entity

    async def get_entity(self, name_or_id: str) -> Entity | None:
        if name_or_id in self.entities:
            return self.entities[name_or_id]
        for e in self.entities.values():
            if e.canonical_name == name_or_id or name_or_id in e.aliases:
                return e
        return None

    async def merge_entities(self, source_id: str, target_id: str) -> Entity:
        source = self.entities.pop(source_id)
        target = self.entities[target_id]
        target.aliases = sorted({*target.aliases, source.canonical_name, *source.aliases})
        return target

    async def neighbors(
        self, entity: str, *, max_degree: int | None = None, active_only: bool = True
    ) -> list[Neighbor]:
        eid = self._resolve_entity_id(entity)
        if eid is None:
            return []
        out: list[Neighbor] = []
        for edge in self.edges.values():
            if active_only and not edge.is_active:
                continue
            other_id: str | None = None
            if edge.from_entity == eid:
                other_id = edge.to_entity
            elif edge.to_entity == eid:
                other_id = edge.from_entity
            if other_id is None:
                continue
            other = self.entities.get(other_id)
            if other is None:
                continue
            out.append(Neighbor(entity=other, edge=edge))
        # Sort by edge weight desc so traversal can weight-rank (FR-RET-6).
        out.sort(key=lambda n: n.edge.weight, reverse=True)
        if max_degree is not None:
            out = out[:max_degree]
        return out

    # --- edge ops (FR-EXT-2 / FR-MNT-4 / FR-RET-6) -----------------------

    async def upsert_edge(self, edge: Edge) -> Edge:
        # Key on (from, to, relation): re-asserting bumps weight, no duplicate.
        for existing in self.edges.values():
            if (
                existing.is_active
                and existing.from_entity == edge.from_entity
                and existing.to_entity == edge.to_entity
                and existing.relation == edge.relation
            ):
                existing.weight = max(existing.weight, edge.weight)
                return existing
        self.edges[edge.id] = edge
        return edge

    async def edges_for_entity(
        self, entity: str, *, active_only: bool = True
    ) -> list[Edge]:
        eid = self._resolve_entity_id(entity)
        if eid is None:
            return []
        return [
            e
            for e in self.edges.values()
            if (not active_only or e.is_active)
            and (e.from_entity == eid or e.to_entity == eid)
        ]

    async def prune_edge(self, edge_id: str, *, at: datetime | None = None) -> Edge:
        e = self.edges[edge_id]
        e.valid_to = at or datetime.now(UTC)
        return e

    # --- suppression persistence (FR-RET-6 / R2) -------------------------

    async def record_suppression(self, memory_id: str, context_key: str) -> None:
        self.suppressions.add((memory_id, context_key))

    async def is_suppressed(self, memory_id: str, context_key: str) -> bool:
        return (memory_id, context_key) in self.suppressions

    async def record_session(self, session: SourceSession) -> None:
        self.sessions.append(session)

    # --- raw-chunk tier (offline re-extraction/reindex) ------------------

    async def persist_raw_chunk(self, chunk: RawChunk) -> RawChunk:
        # Idempotent on content_hash (the FR-ING-5 key).
        self.raw_chunks[chunk.content_hash] = chunk
        return chunk

    async def iter_raw_chunks(
        self,
        *,
        scope: Scope | None = None,
        session_id: str | None = None,
        source: str | None = None,
        since: datetime | None = None,
    ) -> AsyncIterator[RawChunk]:
        scope_str = scope.as_str() if scope is not None else None
        for chunk in list(self.raw_chunks.values()):
            if scope_str is not None and chunk.scope.as_str() != scope_str:
                continue
            if session_id is not None and chunk.session_id != session_id:
                continue
            if source is not None and chunk.source != source:
                continue
            if since is not None and chunk.ingested_at < since:
                continue
            yield chunk

    async def re_extract_from_raw_chunks(
        self,
        extractor: Extractor,
        *,
        scope: Scope | None = None,
        session_id: str | None = None,
        supersede_existing: bool = True,
    ) -> MaintenanceReport:
        # The fake does not re-parse chunk.content back into events; it exercises
        # the seam (iterate the raw tier, supersede the prior memories). The real
        # backend re-runs ``extractor`` over the normalized content.
        del extractor
        count = 0
        async for chunk in self.iter_raw_chunks(scope=scope, session_id=session_id):
            if supersede_existing:
                for mid in chunk.memory_ids:
                    if mid in self.memories:
                        self.memories[mid].supersede()
            count += 1
        return MaintenanceReport(job_name="re_extract", re_extracted=count)

    async def reclassify_memory(
        self,
        memory_id: str,
        *,
        scope: Scope | None = None,
        category: str | None = None,
        cross_ref_candidate: bool | None = None,
    ) -> MemoryUnit:
        m = self.memories[memory_id]
        if scope is not None:
            m.scope = scope
        if category is not None:
            # Reuse the model's normalization (lowercase/trim).
            m.category = MemoryUnit(content=m.content, scope=m.scope, category=category).category
        if cross_ref_candidate is not None:
            m.cross_ref_candidate = cross_ref_candidate
        return m

    # --- category registry (emergent-category list/merge) ----------------

    async def list_categories(self) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for m in self.memories.values():
            if m.is_active:
                counts[m.category] = counts.get(m.category, 0) + 1
        return list(counts.items())

    async def merge_categories(self, source: str, target: str) -> int:
        src = source.strip().lower()
        tgt = target.strip().lower()
        n = 0
        for m in self.memories.values():
            if m.category == src:
                m.category = tgt
                n += 1
        return n

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    """A default test :class:`Settings` (no environment dependency)."""

    return Settings()


@pytest.fixture
def fake_embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def fake_llm() -> FakeLLMProvider:
    return FakeLLMProvider()


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def sample_provenance() -> Provenance:
    return Provenance(
        source=Source.CLAUDE_CODE.value,
        session_id="sess-1",
        chunk_hash="deadbeef",
        raw_path="~/.claude/projects/demo/sess-1.jsonl",
    )


@pytest.fixture
def sample_events() -> list[IngestEvent]:
    """A small, realistic chunk of normalized events from one Claude Code session."""

    base = datetime(2026, 6, 13, 10, 0, 0, tzinfo=UTC)
    return [
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="rust-cli",
            session_id="sess-1",
            timestamp=base,
            role=Role.USER,
            content="I prefer thiserror over anyhow for error handling in Rust.",
            metadata={"cwd": "/home/op/rust-cli", "git_remote": "git@github:op/rust-cli"},
        ),
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="rust-cli",
            session_id="sess-1",
            timestamp=base + timedelta(minutes=1),
            role=Role.ASSISTANT,
            content="Understood. I'll use thiserror for error types in this project.",
        ),
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="rust-cli",
            session_id="sess-1",
            timestamp=base + timedelta(minutes=2),
            role=Role.USER,
            content="This project pins tokio 1.38.",
        ),
    ]


@pytest.fixture
def sample_memory(sample_provenance: Provenance) -> MemoryUnit:
    """A sample active global preference memory unit (category split contract)."""

    return MemoryUnit(
        content="Prefers thiserror over anyhow for Rust error handling.",
        scope=Scope.global_(),
        category="preference",
        entities=["rust", "error-handling", "thiserror"],
        confidence=0.9,
        provenance=sample_provenance,
    )
