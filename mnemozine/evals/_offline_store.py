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
from datetime import UTC, date, datetime, timedelta
from typing import Any

from mnemozine.interfaces import (
    Extractor,
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
    Edge,
    Entity,
    MemoryUnit,
    RawChunk,
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
        # Persisted (memory)-[:MNEMOZINE_MENTIONS]->(entity) edges as (memory_id,
        # entity_id) pairs (the §9 eval metrics never exercise the mentions pass,
        # but the field keeps the offline fake Protocol-conformant and lets
        # merge_entities repoint them).
        self.mentions: set[tuple[str, str]] = set()
        # Weighted entity-entity co-mention edges keyed on (from_id, to_id) ->
        # (weight, shared). The §9 eval metrics never exercise the co-mention pass,
        # but the field keeps the offline fake Protocol-conformant and lets
        # merge_entities repoint them.
        self.co_mentions: dict[tuple[str, str], tuple[float, int]] = {}
        self.sessions: list[SourceSession] = []
        self.suppressions: set[tuple[str, str]] = set()
        self.raw_chunks: dict[str, RawChunk] = {}

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
        compose_ancestors: bool = True,
    ) -> list[RetrievedMemory]:
        # Ancestor-composition / no-leak (FR-STO-3): expand each query scope to
        # its ancestor-or-self chain when composing (the default), else match the
        # exact scope strings only. Siblings/descendants are never on an ancestor
        # chain, so they can never leak.
        if compose_ancestors:
            scope_strs = {anc.as_str() for s in scopes for anc in s.ancestors()}
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
            m_words = set(m.content.lower().split())
            overlap = len(q_words & m_words)
            score = overlap / (len(q_words) or 1)
            if overlap or not q_words:
                results.append(RetrievedMemory(memory=m, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def close_validity_window(self, memory_id: str, *, at: Any | None = None) -> MemoryUnit:
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

    # --- display reads (WebUI READ surface; EMBEDDING-FREE) ------------------
    #
    # The four embedding-free display reads, mirrored from the conftest
    # ``InMemoryStorage`` fake so the two in-memory ``StorageBackend`` fakes stay
    # behaviourally consistent with each other and with the FalkorDB backend's
    # Cypher contract. They never touch an embedding (the dict store does not even
    # keep one) and project onto the lightweight :class:`MemoryView`.

    @staticmethod
    def _to_view(m: MemoryUnit) -> MemoryView:
        """Project a stored unit onto the embedding-free display view."""

        return MemoryView(
            id=m.id,
            content=m.content,
            scope=m.scope,
            category=m.category,
            cross_ref_candidate=m.cross_ref_candidate,
            entities=list(m.entities),
            confidence=m.confidence,
            tier=m.tier,
            valid_from=m.valid_from,
            valid_to=m.valid_to,
            last_accessed=m.last_accessed,
            access_count=m.access_count,
            source=m.provenance.source,
            session_id=m.provenance.session_id,
            chunk_hash=m.provenance.chunk_hash,
            raw_path=m.provenance.raw_path,
        )

    async def store_stats(self) -> StoreStats:
        by_category: dict[str, int] = {}
        by_scope_decision: dict[str, int] = {}
        by_tier: dict[str, int] = {}
        by_source: dict[str, int] = {}
        active = 0
        for m in self.memories.values():
            by_category[m.category] = by_category.get(m.category, 0) + 1
            decision = m.scope_decision.value
            by_scope_decision[decision] = by_scope_decision.get(decision, 0) + 1
            by_tier[m.tier.value] = by_tier.get(m.tier.value, 0) + 1
            src = m.provenance.source
            by_source[src] = by_source.get(src, 0) + 1
            if m.is_active:
                active += 1
        total = len(self.memories)
        return StoreStats(
            total_memories=total,
            by_category=by_category,
            by_scope_decision=by_scope_decision,
            by_tier=by_tier,
            by_source=by_source,
            active_count=active,
            superseded_count=total - active,
            entity_count=len(self.entities),
            raw_chunk_count=len(self.raw_chunks),
        )

    async def scope_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.memories.values():
            s = m.scope.as_str()
            counts[s] = counts.get(s, 0) + 1
        return counts

    async def memory_growth(
        self, *, scope: Scope | None = None, days: int = 14, today: date | None = None
    ) -> list[tuple[str, int]]:
        span = days if days >= 1 else 1
        anchor = today if today is not None else datetime.now(UTC).date()
        start = anchor - timedelta(days=span - 1)
        counts: dict[str, int] = {}
        for m in self.memories.values():
            if m.valid_from.date() < start:
                continue
            # Exact-or-descendant roll-up; the global root (segments == []) is the
            # universal ancestor, so scope=global counts the whole store. This
            # mirrors GraphitiStorageBackend.memory_growth exactly.
            if scope is not None and not m.scope.is_descendant_of(scope):
                continue
            day = m.valid_from.date().isoformat()
            counts[day] = counts.get(day, 0) + 1
        return sorted(counts.items())

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
        scope_str = scope.as_str() if scope is not None else None
        cat = category.strip().lower() if category is not None else None
        ent = entity.lower() if entity is not None else None
        ql = q.lower() if q else None
        matched = [
            m
            for m in self.memories.values()
            if (cat is None or m.category == cat)
            and (scope_str is None or m.scope.as_str() == scope_str)
            and (tier is None or m.tier is tier)
            and (source is None or m.provenance.source == source)
            and (active is None or m.is_active is active)
            and (ent is None or ent in {e.lower() for e in m.entities})
            and (ql is None or ql in m.content.lower())
        ]
        matched.sort(key=lambda m: m.valid_from, reverse=True)
        total = len(matched)
        page = matched[offset : offset + limit]
        return MemoryPage(items=[self._to_view(m) for m in page], total=total)

    async def get_memory_display(self, memory_id: str) -> MemoryView | None:
        m = self.memories.get(memory_id)
        return self._to_view(m) if m is not None else None

    async def graph_snapshot(
        self,
        *,
        scope: Scope | None = None,
        entity: str | None = None,
        entity_type: str | None = None,
        include_idea_seeds: bool = True,
        node_limit: int = 200,
    ) -> GraphSnapshot:
        scope_str = scope.as_str() if scope is not None else None
        entities = [
            e
            for e in self.entities.values()
            if entity_type is None or (e.type or "") == entity_type
        ]
        if entity is not None:
            center = await self.get_entity(entity)
            if center is None:
                return GraphSnapshot(nodes=[], edges=[], truncated=False)
            keep = {center.id}
            for nb in await self.neighbors(center.canonical_name, active_only=False):
                keep.add(nb.entity.id)
            entities = [e for e in entities if e.id in keep]
        # Over-fetch sentinel: truncated iff the selected set exceeds node_limit
        # (mirrors the backend's LIMIT $cap+1 over-fetch), so exactly node_limit
        # entities is NOT a false-positive truncation.
        truncated = len(entities) > node_limit
        entities = entities[:node_limit]

        entity_by_id = {e.id: e for e in entities}
        entity_id_by_name = {e.canonical_name.lower(): e.id for e in entities}

        memory_count: dict[str, int] = dict.fromkeys(entity_by_id, 0)
        idea_nodes: list[GraphSnapshotNode] = []
        mentions_edges: list[GraphSnapshotEdge] = []
        for m in self.memories.values():
            if scope_str is not None and m.scope.as_str() != scope_str:
                continue
            linked = {
                entity_id_by_name[name.lower()]
                for name in m.entities
                if name.lower() in entity_id_by_name
            }
            for eid in linked:
                memory_count[eid] = memory_count.get(eid, 0) + 1
            if include_idea_seeds and m.cross_ref_candidate and linked:
                snippet = " ".join(m.content.split())
                if len(snippet) > 80:
                    snippet = snippet[:79].rstrip() + "…"
                idea_nodes.append(
                    GraphSnapshotNode(
                        id=m.id,
                        label=snippet,
                        kind="idea_seed",
                        scope=m.scope.as_str(),
                        memory_count=1,
                    )
                )
                for eid in linked:
                    mentions_edges.append(
                        GraphSnapshotEdge(
                            id=f"mentions:{m.id}:{eid}",
                            source=m.id,
                            target=eid,
                            relation="mentions",
                            weight=1.0,
                            active=m.is_active,
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

        struct_edges: list[GraphSnapshotEdge] = []
        seen: set[str] = set()
        for edge in self.edges.values():
            if edge.id in seen:
                continue
            if edge.from_entity in entity_by_id and edge.to_entity in entity_by_id:
                seen.add(edge.id)
                struct_edges.append(
                    GraphSnapshotEdge(
                        id=edge.id,
                        source=edge.from_entity,
                        target=edge.to_entity,
                        relation=edge.relation,
                        weight=edge.weight,
                        active=edge.is_active,
                        kind="relates",
                    )
                )

        # Weighted entity-entity co-mention layer (kind='co_mention') among kept ids.
        co_mention_edges: list[GraphSnapshotEdge] = []
        for (a, b), (weight, _shared) in self.co_mentions.items():
            if a in entity_by_id and b in entity_by_id:
                co_mention_edges.append(
                    GraphSnapshotEdge(
                        id=f"comention:{a}:{b}",
                        source=a,
                        target=b,
                        relation="co_mentioned",
                        weight=weight,
                        active=True,
                        kind="co_mention",
                    )
                )

        return GraphSnapshot(
            nodes=entity_nodes + idea_nodes,
            edges=struct_edges + co_mention_edges + mentions_edges,
            truncated=truncated,
        )

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
        # Repoint mention edges off the source onto the target (set semantics).
        self.mentions = {
            (mid, target_id if eid == source_id else eid)
            for (mid, eid) in self.mentions
        }
        # Repoint co-mention edges off the source onto the target (both directions);
        # drop self-loops, keep the higher-weight edge on a collision so the merge
        # is idempotent and never duplicates.
        repointed_co: dict[tuple[str, str], tuple[float, int]] = {}
        for (a, b), (weight, shared) in self.co_mentions.items():
            na = target_id if a == source_id else a
            nb = target_id if b == source_id else b
            if na == nb:
                continue
            # Canonicalize (lo < hi): co-mention is unordered, so a reversing
            # repoint folds onto the survivor's edge, never a parallel duplicate.
            lo, hi = (na, nb) if na <= nb else (nb, na)
            prev = repointed_co.get((lo, hi))
            if prev is None or weight > prev[0]:
                repointed_co[(lo, hi)] = (weight, shared)
        self.co_mentions = repointed_co
        return self.entities[target_id]

    async def persist_mentions(self) -> int:
        """MERGE (memory)-[:MNEMOZINE_MENTIONS]->(entity) from each m.entities name.

        Thin offline mirror of the backend: resolves each mention name to an
        entity id by case-folded canonical-name/alias match and asserts the
        (memory_id, entity_id) pair into the mentions set (idempotent). Returns
        the number asserted this pass.
        """

        name_to_id: dict[str, str] = {}
        for e in self.entities.values():
            name_to_id[e.canonical_name.lower()] = e.id
            for alias in e.aliases:
                name_to_id.setdefault(alias.lower(), e.id)
        asserted: set[tuple[str, str]] = set()
        for m in self.memories.values():
            for name in m.entities:
                eid = name_to_id.get(name.lower())
                if eid is not None:
                    asserted.add((m.id, eid))
        self.mentions |= asserted
        return len(asserted)

    async def co_mention_pairs(
        self, *, min_shared: int = 2
    ) -> list[tuple[str, str, int]]:
        """Entity id pairs co-occurring in >= ``min_shared`` shared memories.

        Thin offline mirror: groups the mentions set by memory, counts each
        entity pair's distinct shared memories, returns ``(a, b, shared)`` with
        ``a < b`` for pairs at or above ``min_shared``.
        """

        by_memory: dict[str, set[str]] = {}
        for mid, eid in self.mentions:
            by_memory.setdefault(mid, set()).add(eid)
        pair_counts: dict[tuple[str, str], int] = {}
        for eids in by_memory.values():
            ids = sorted(eids)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pair_counts[(ids[i], ids[j])] = pair_counts.get((ids[i], ids[j]), 0) + 1
        return [(a, b, n) for (a, b), n in pair_counts.items() if n >= min_shared]

    async def entity_mention_counts(self) -> dict[str, int]:
        """``{entity_id: distinct-memory mention count}`` over the mentions set."""

        counts: dict[str, int] = {}
        for _mid, eid in self.mentions:
            counts[eid] = counts.get(eid, 0) + 1
        return counts

    async def upsert_co_mention(
        self, from_entity: str, to_entity: str, *, weight: float, shared: int
    ) -> Edge:
        """Idempotently MERGE a weighted entity-entity co-mention edge (SET, not sum)."""

        lo, hi = (
            (from_entity, to_entity)
            if from_entity <= to_entity
            else (to_entity, from_entity)
        )
        self.co_mentions[(lo, hi)] = (float(weight), int(shared))
        return Edge(
            id=f"comention:{lo}:{hi}",
            from_entity=lo,
            to_entity=hi,
            relation="co_mentioned",
            weight=float(weight),
        )

    async def neighbors(
        self, entity: str, *, max_degree: int | None = None, active_only: bool = True
    ) -> list[Neighbor]:
        return []

    async def upsert_edge(self, edge: Edge) -> Edge:
        self.edges[edge.id] = edge
        return edge

    async def edges_for_entity(self, entity: str, *, active_only: bool = True) -> list[Edge]:
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

    # --- raw tier (R4) -------------------------------------------------------

    async def persist_raw_chunk(self, chunk: RawChunk) -> RawChunk:
        # Idempotent on content_hash (the FR-ING-5 key): re-persisting the same
        # chunk updates its memory_ids rather than duplicating.
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
        for c in list(self.raw_chunks.values()):
            if scope_str is not None and c.scope.as_str() != scope_str:
                continue
            if session_id is not None and c.session_id != session_id:
                continue
            if source is not None and c.source != source:
                continue
            if since is not None and c.ingested_at < since:
                continue
            yield c

    async def re_extract_from_raw_chunks(
        self,
        extractor: Extractor,
        *,
        scope: Scope | None = None,
        session_id: str | None = None,
        supersede_existing: bool = True,
    ) -> MaintenanceReport:
        # Minimal offline seam: the §9 metrics do not exercise re-extraction, so
        # this satisfies the Protocol shape with a pass over the raw tier that
        # exercises the supersede + version-stamp behaviour without re-parsing
        # chunk.content back into events.
        del extractor
        report = MaintenanceReport(job_name="re_extract_from_raw_chunks")
        async for chunk in self.iter_raw_chunks(scope=scope, session_id=session_id):
            if supersede_existing:
                for mid in chunk.memory_ids:
                    m = self.memories.get(mid)
                    if m is not None:
                        m.supersede()
            # DATA-VERSION STAMP CONTRACT (mnemozine.migrations): re-stamp the
            # re-processed chunk to the current version so a re-extract migration
            # is idempotent (a re-run finds it already at CURRENT_DATA_VERSION).
            chunk.data_version = CURRENT_DATA_VERSION
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
        m = self.memories[memory_id]
        if scope is not None:
            m.scope = scope
        if category is not None:
            m.category = category.strip().lower() or DEFAULT_CATEGORY
        if cross_ref_candidate is not None:
            m.cross_ref_candidate = cross_ref_candidate
        # DATA-VERSION STAMP CONTRACT (mnemozine.migrations): a reclassify always
        # re-stamps the touched memory up to the current version (the cheap
        # migration path relies on this implicit stamp).
        m.data_version = CURRENT_DATA_VERSION
        return m

    async def list_categories(self) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for m in self.memories.values():
            if not m.is_active:
                continue
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

    # --- relation registry (relation-label list/merge, FR-MNT-2/4) -----------

    async def list_relations(self) -> list[tuple[str, int]]:
        """``(relation_label, active_edge_count)`` over active relation edges.

        Thin offline mirror of the backend's relation registry: a grouped count
        over the active edges (the §9 eval metrics never exercise the relation
        normalization pass, so this stays minimal but Protocol-conformant).
        """

        counts: dict[str, int] = {}
        for e in self.edges.values():
            if not e.is_active:
                continue
            counts[e.relation] = counts.get(e.relation, 0) + 1
        return list(counts.items())

    async def merge_relations(
        self, source_relation: str, target_relation: str
    ) -> int:
        """Relabel every active ``source``-relation edge to ``target`` (idempotent).

        Folds each source-relation edge onto the ``(from, to, target)`` edge
        (max weight, drop the redundant parallel source edge). ``source ==
        target`` -> 0. Returns the relabelled count.
        """

        if source_relation == target_relation:
            return 0
        n = 0
        for edge in list(self.edges.values()):
            if not edge.is_active or edge.relation != source_relation:
                continue
            n += 1
            target_edge = next(
                (
                    e
                    for e in self.edges.values()
                    if e.is_active
                    and e.from_entity == edge.from_entity
                    and e.to_entity == edge.to_entity
                    and e.relation == target_relation
                ),
                None,
            )
            if target_edge is None:
                edge.relation = target_relation
            else:
                target_edge.weight = max(target_edge.weight, edge.weight)
                del self.edges[edge.id]
        return n

    # --- data-versioning / in-place migration (mnemozine.migrations) ---------

    async def min_data_version(self) -> int:
        versions = [
            record_data_version(m.data_version) for m in self.memories.values()
        ]
        versions += [
            record_data_version(c.data_version) for c in self.raw_chunks.values()
        ]
        if not versions:
            return CURRENT_DATA_VERSION
        return min(versions)

    async def iter_memories_below_version(
        self, version: int
    ) -> AsyncIterator[MemoryUnit]:
        for m in list(self.memories.values()):
            if record_data_version(m.data_version) < version:
                yield m

    async def set_data_version(self, ids: Sequence[str], version: int) -> int:
        n = 0
        for mid in ids:
            m = self.memories.get(mid)
            if m is not None:
                m.data_version = version
                n += 1
        return n

    async def iter_chunks_below_version(
        self, version: int
    ) -> AsyncIterator[RawChunk]:
        for c in list(self.raw_chunks.values()):
            if record_data_version(c.data_version) < version:
                yield c

    async def set_chunk_data_version(
        self, content_hashes: Sequence[str], version: int
    ) -> int:
        n = 0
        for h in content_hashes:
            c = self.raw_chunks.get(h)
            if c is not None:
                c.data_version = version
                n += 1
        return n

    async def close(self) -> None:
        return None
