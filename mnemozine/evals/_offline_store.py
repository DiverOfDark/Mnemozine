"""A packaged, minimal in-memory ``StorageBackend`` for the CLI's offline mode.

The unit tests drive the harness with the shared fake in ``tests/conftest.py``
(``InMemoryStorage``), per the project testing convention. But ``tests/`` is not
part of the installed wheel, so the ``mnemozine-eval`` console script cannot rely
on it being importable after ``pip install``. This module ships a small,
self-contained dict-backed store — just the subset of the ``StorageBackend``
Protocol the §9 metrics exercise — so ``mnemozine-eval --offline`` works against
fakes with no FalkorDB/Ollama/Qwen and no test package present.

It is deliberately a thin re-implementation (not a copy of the conftest fake):
only the methods the EVAL harness calls are real; the rest satisfy the Protocol
shape and are no-ops/raise. Tests still use the richer conftest fake.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import Any

from mnemozine.interfaces import (
    Neighbor,
    RetrievedMemory,
    WriteDecision,
    WriteResult,
)
from mnemozine.schema.models import (
    Edge,
    Entity,
    MemoryUnit,
    Scope,
    SourceSession,
    Tier,
)


def _overlaps(a: Sequence[str], b: Sequence[str]) -> bool:
    return bool(set(a) & set(b))


class OfflineStorage:
    """Dict-backed ``StorageBackend`` covering exactly what the §9 metrics need."""

    def __init__(self) -> None:
        self.memories: dict[str, MemoryUnit] = {}
        self.entities: dict[str, Entity] = {}
        self.edges: dict[str, Edge] = {}
        self.sessions: list[SourceSession] = []
        self.suppressions: set[tuple[str, str]] = set()

    async def upsert_memory(self, memory: MemoryUnit) -> WriteResult:
        # Reinforce on exact-content match within scope+entities; else add.
        for existing in self.memories.values():
            if (
                existing.is_active
                and existing.scope.as_str() == memory.scope.as_str()
                and _overlaps(existing.entities, memory.entities)
                and existing.content.strip() == memory.content.strip()
            ):
                existing.confidence = max(existing.confidence, memory.confidence)
                return WriteResult(decision=WriteDecision.REINFORCE, memory=existing)
        self.memories[memory.id] = memory
        return WriteResult(decision=WriteDecision.ADD, memory=memory)

    async def scoped_query(
        self,
        query: str,
        scopes: Sequence[Scope],
        *,
        entities: Sequence[str] | None = None,
        top_k: int = 10,
        include_archived: bool = False,
    ) -> list[RetrievedMemory]:
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
        m = self.memories[memory_id]
        m.tier = Tier.HOT
        return m

    async def reembed(self, memory_id: str) -> MemoryUnit:
        return self.memories[memory_id]

    async def record_access(self, memory_id: str) -> None:
        m = self.memories[memory_id]
        m.access_count += 1
        m.last_accessed = datetime.now(UTC)

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
            yield m

    async def iter_entities(self) -> AsyncIterator[Entity]:
        for e in list(self.entities.values()):
            yield e

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
        self.entities.pop(source_id, None)
        return self.entities[target_id]

    async def neighbors(
        self, entity: str, *, max_degree: int | None = None, active_only: bool = True
    ) -> list[Neighbor]:
        return []

    async def upsert_edge(self, edge: Edge) -> Edge:
        self.edges[edge.id] = edge
        return edge

    async def edges_for_entity(
        self, entity: str, *, active_only: bool = True
    ) -> list[Edge]:
        return []

    async def prune_edge(self, edge_id: str, *, at: datetime | None = None) -> Edge:
        e = self.edges[edge_id]
        e.valid_to = at or datetime.now(UTC)
        return e

    async def record_suppression(self, memory_id: str, context_key: str) -> None:
        self.suppressions.add((memory_id, context_key))

    async def is_suppressed(self, memory_id: str, context_key: str) -> bool:
        return (memory_id, context_key) in self.suppressions

    async def record_session(self, session: SourceSession) -> None:
        self.sessions.append(session)

    async def close(self) -> None:
        return None
