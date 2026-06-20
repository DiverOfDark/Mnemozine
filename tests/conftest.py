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
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from mnemozine.config import Settings
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
        # Persisted (memory)-[:MNEMOZINE_MENTIONS]->(entity) edges as a set of
        # (memory_id, entity_id) pairs — a set so the MERGE is idempotent (a
        # re-assert is a no-op) and merge_entities can repoint them in place.
        self.mentions: set[tuple[str, str]] = set()
        # Weighted entity-entity co-mention edges, keyed on (from_id, to_id) so the
        # MERGE is idempotent (a re-assert overwrites weight, never duplicates) and
        # merge_entities can repoint them in place. Value is (weight, shared).
        self.co_mentions: dict[tuple[str, str], tuple[float, int]] = {}
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

    # --- display reads (WebUI READ surface; EMBEDDING-FREE) --------------

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
        else:
            # Default (no-center) selection: degree-rank by incident structural
            # degree (RELATES edges + CO_MENTIONS) descending, tie-break on id,
            # mirroring the backend's degree-ranked top slice so the snapshot
            # surfaces the connected structure, not an arbitrary slice.
            degree: dict[str, int] = dict.fromkeys((e.id for e in entities), 0)
            for edge in self.edges.values():
                if edge.from_entity in degree:
                    degree[edge.from_entity] += 1
                if edge.to_entity in degree:
                    degree[edge.to_entity] += 1
            for a, b in self.co_mentions:
                if a in degree:
                    degree[a] += 1
                if b in degree:
                    degree[b] += 1
            entities.sort(key=lambda e: (-degree[e.id], e.id))
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

        # Weighted entity-entity co-mention layer (kind='co_mention') among kept
        # entities — the second aggregate edge layer the backend surfaces.
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
                # Anchor a never-accessed (None last_accessed) memory on valid_from
                # (ingestion time), matching the backend's coalesce + decay_score's
                # recency anchor: a never-recalled memory is "unused since it was
                # ingested", NOT "unused since forever". A freshly-ingested,
                # never-recalled memory is therefore NOT selected until its creation
                # time itself ages past the cutoff.
                anchor = m.last_accessed or m.valid_from
                if anchor >= unused_since:
                    continue
            yield m

    async def iter_entities(self) -> AsyncIterator[Entity]:
        for e in list(self.entities.values()):
            yield e

    # --- entity ops (FR-EXT-2 / FR-MNT-4) --------------------------------

    async def upsert_entity(self, entity: Entity) -> Entity:
        self.entities[entity.id] = entity
        return entity

    async def resolve_or_create_entity(self, entity: Entity) -> Entity:
        """Identity-by-normalized-name: reuse the node for ``toLower(name)``.

        Mirrors the backend seam — scan for a stored entity whose
        ``canonical_name.lower()`` matches the incoming one; if found, return that
        existing entity (folding the incoming canonical_name/aliases into its
        aliases when they add something new) WITHOUT creating a duplicate; else
        create. Idempotent: resolving the same name twice returns the same id and
        never grows ``self.entities``.
        """

        key = entity.canonical_name.lower()
        for stored in self.entities.values():
            if stored.canonical_name.lower() == key:
                incoming = {entity.canonical_name, *entity.aliases}
                merged = sorted({*stored.aliases, *incoming} - {stored.canonical_name})
                if merged != sorted(stored.aliases):
                    stored.aliases = merged
                return stored
        return await self.upsert_entity(entity)

    async def backfill_entity_name_keys(self) -> int:
        """No-op: this dict-backed fake resolves by ``canonical_name.lower()``.

        There is no stored ``name_key`` to backfill (resolution lowercases the
        canonical name directly), so the v2 migration's structural pass is a no-op
        here; the tier-stamp half still advances the version floor. Returns 0.
        """

        return 0

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
        # Repoint mention edges off the source onto the target (set semantics so a
        # memory that mentioned BOTH collapses to a single edge, no duplicate).
        repointed = {
            (mid, target_id if eid == source_id else eid)
            for (mid, eid) in self.mentions
        }
        self.mentions = repointed
        # Repoint co-mention edges off the source onto the target (both directions).
        # A self-loop (target<->source) is dropped; a collision keeps the
        # higher-weight edge so the merge stays idempotent and never duplicates.
        repointed_co: dict[tuple[str, str], tuple[float, int]] = {}
        for (a, b), (weight, shared) in self.co_mentions.items():
            na = target_id if a == source_id else a
            nb = target_id if b == source_id else b
            if na == nb:
                continue
            # Co-mention is unordered: re-canonicalize (lo < hi) so a repoint that
            # would reverse an endpoint folds onto the survivor's canonical edge
            # instead of leaving a parallel reversed duplicate.
            lo, hi = (na, nb) if na <= nb else (nb, na)
            prev = repointed_co.get((lo, hi))
            if prev is None or weight > prev[0]:
                repointed_co[(lo, hi)] = (weight, shared)
        self.co_mentions = repointed_co
        return target

    async def persist_mentions(self) -> int:
        """MERGE (memory)-[:MNEMOZINE_MENTIONS]->(entity) from each m.entities name.

        Resolves each mention name to an entity by case-folded canonical-name or
        alias match (mirroring :meth:`get_entity`) and asserts the
        (memory_id, entity_id) pair into the mentions set. Set semantics make the
        whole pass idempotent: a re-run re-asserts the same pairs and adds none.
        Returns the number of mention edges asserted (the size of the resolved set
        for this pass).
        """

        # Build a lower-cased name -> entity-id resolution table once.
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

    async def add_memory_mentions(
        self, memory_id: str, entity_ids: Sequence[str]
    ) -> int:
        """Inline per-memory MNEMOZINE_MENTIONS seam (assert at ingest time).

        Mirrors the backend's id-keyed MERGE: add ``(memory_id, eid)`` to the
        mentions set for each resolved entity id (set semantics so a re-call
        re-asserts the same edges and adds none). Only ids of stored entities are
        asserted, matching the backend's id-bound MATCH. Returns the number of edges
        asserted.
        """

        if memory_id not in self.memories:
            return 0
        asserted = {
            (memory_id, eid) for eid in entity_ids if eid in self.entities
        }
        self.mentions |= asserted
        return len(asserted)

    async def co_mention_pairs(
        self, *, min_shared: int = 2
    ) -> list[tuple[str, str, int]]:
        """Entity id pairs co-occurring in >= ``min_shared`` shared memories.

        Derived from the mentions set: group the mention edges by memory, take
        every entity pair a memory mentions, count the distinct shared memories,
        and return ``(a, b, shared)`` with ``a < b`` for pairs >= ``min_shared``.
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
        return [
            (a, b, n) for (a, b), n in pair_counts.items() if n >= min_shared
        ]

    async def entity_mention_counts(self) -> dict[str, int]:
        """``{entity_id: distinct-memory mention count}`` over the mentions set."""

        counts: dict[str, int] = {}
        seen: set[tuple[str, str]] = set()
        for mid, eid in self.mentions:
            if (mid, eid) in seen:
                continue
            seen.add((mid, eid))
            counts[eid] = counts.get(eid, 0) + 1
        return counts

    async def upsert_co_mention(
        self, from_entity: str, to_entity: str, *, weight: float, shared: int
    ) -> Edge:
        """Idempotently MERGE a weighted entity-entity co-mention edge.

        Keyed on (from, to): a re-assert overwrites weight/shared (SET, not sum) so
        the pass is idempotent and never duplicates. Returns the stored edge.
        """

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
            # DATA-VERSION STAMP CONTRACT: re-stamp the re-processed chunk to the
            # current version so a re-extract migration is idempotent.
            chunk.data_version = CURRENT_DATA_VERSION
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
        # DATA-VERSION STAMP CONTRACT: a reclassify re-stamps to the current
        # version (the cheap migration path relies on this implicit stamp).
        m.data_version = CURRENT_DATA_VERSION
        return m

    # --- data-versioning / in-place migration (mnemozine.migrations) -----

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

    # --- relation registry (relation-label list/merge, FR-MNT-2/4) -------

    async def list_relations(self) -> list[tuple[str, int]]:
        """``(relation_label, active_edge_count)`` over active relation edges.

        The relation analogue of :meth:`list_categories`, grouped over the active
        edges (only the LLM-extracted relation edges carry a normalizable label;
        the co-mention/mention layers live in their own stores).
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
        """Relabel every ``source``-relation edge to ``target`` (relation merge).

        For each active edge with ``relation == source``, fold it onto the
        ``(from, to, target)`` edge — combining ``weight`` via ``max`` and
        dropping the redundant parallel source edge so no duplicate parallel edges
        remain. Idempotent (``source == target`` -> 0). Returns the relabelled
        count.
        """

        if source_relation == target_relation:
            return 0
        n = 0
        for edge in list(self.edges.values()):
            if not edge.is_active or edge.relation != source_relation:
                continue
            n += 1
            # Find an existing target edge between the same endpoints to MERGE onto.
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
                # No parallel target edge: relabel the source edge in place.
                edge.relation = target_relation
            else:
                # Parallel target edge exists: combine weight (max), drop source.
                target_edge.weight = max(target_edge.weight, edge.weight)
                del self.edges[edge.id]
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
