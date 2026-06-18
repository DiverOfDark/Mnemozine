"""Integration-layer glue services that compose the layer Protocols.

These are the small pieces of wiring the integration pass must supply that do not
belong inside any single module (they cross layer boundaries) yet must not be
baked into a module to keep the Protocols clean:

* :func:`make_contradiction_fn` — the FR-MNT-1 contradiction predicate
  (:data:`~mnemozine.storage.ContradictsFn`) the storage backend takes injected.
  It wraps a :class:`~mnemozine.interfaces.LLMProvider` in the single
  narrowly-scoped cheap LLM call; the backend pre-filters candidates (same scope,
  type=preference, shared entity, capped) so this closure just asks which the new
  memory reverses. Keeps the storage layer free of any LLM import.
* :class:`MnemozineIngestService` — the chunk -> extract_full -> store pipeline
  shared by the ingest watcher loop and the Stop/PreCompact hooks
  (:class:`~mnemozine.ingestion.claude_code.hooks.runtime.IngestService`). It
  turns a chunk of :class:`~mnemozine.schema.events.IngestEvent`s into persisted
  MemoryUnits + Entity nodes + relationship Edges (FR-EXT-2), idempotent on the
  FR-ING-5 content hash.

Everything here depends only on the :mod:`mnemozine.interfaces` Protocols and the
public module surfaces, never on another layer's internals.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from mnemozine.config import Settings, get_settings
from mnemozine.extract.extractor import ExtractionResult, TypedExtractor
from mnemozine.extract.raw_tier import extract_with_raw_retention
from mnemozine.ingestion.claude_code.chunker import Chunk, chunk_events
from mnemozine.ingestion.claude_code.parser import read_transcript
from mnemozine.interfaces import LLMProvider, StorageBackend
from mnemozine.maintenance.decision import (
    CONTRADICTION_SCHEMA,
    CONTRADICTION_SYSTEM,
    build_contradiction_prompt,
    parse_contradiction,
)
from mnemozine.schema.events import IngestEvent
from mnemozine.schema.models import Entity, MemoryUnit
from mnemozine.storage.backend import ContradictsFn

logger = logging.getLogger(__name__)


def make_contradiction_fn(llm: LLMProvider) -> ContradictsFn:
    """Build the FR-MNT-1 contradiction predicate over an ``LLMProvider``.

    The returned async predicate matches :data:`mnemozine.storage.ContradictsFn`:
    ``(new, candidates) -> list[MemoryUnit]``. The storage backend has already
    narrowed ``candidates`` to ``type=preference`` units in the same scope sharing
    >=1 entity (capped at ``maintenance.contradiction_candidate_cap``); this asks
    the LLM, per candidate, whether ``new`` reverses it and returns the contradicted
    subset. Reuses the prompt/schema/parsing already defined and tested in
    :mod:`mnemozine.maintenance.decision` so the contradiction logic lives in one
    place.

    A per-candidate LLM error defaults to "no contradiction" (the safe direction —
    a missed reversal leaves both units active and the older one decays naturally,
    whereas a false positive wrongly closes a still-valid preference window).
    """

    async def contradicts(
        new: MemoryUnit, candidates: list[MemoryUnit]
    ) -> list[MemoryUnit]:
        contradicted: list[MemoryUnit] = []
        for candidate in candidates:
            prompt = build_contradiction_prompt(new, candidate)
            try:
                raw = await llm.complete_json(
                    prompt,
                    schema=CONTRADICTION_SCHEMA,
                    system=CONTRADICTION_SYSTEM,
                )
            except Exception:  # noqa: BLE001 - an LLM error must not break the write
                logger.exception("contradiction check failed; treating as non-contradiction")
                continue
            if parse_contradiction(raw):
                contradicted.append(candidate)
        return contradicted

    return contradicts


class MnemozineIngestService:
    """Chunk -> extract -> store pipeline (FR-EXT-2 graph write, FR-ING-5/6).

    Implements the :class:`~mnemozine.ingestion.claude_code.hooks.runtime.IngestService`
    Protocol (``flush_session``) and exposes :meth:`ingest_chunk` for the watcher
    loop. It takes the concrete :class:`~mnemozine.extract.TypedExtractor` (not the
    bare :class:`~mnemozine.interfaces.Extractor` Protocol) because it needs the
    richer :meth:`TypedExtractor.extract_full` so relationship edges and entity
    nodes are written, not just MemoryUnits (see the extraction integration note):
    for each chunk it upserts entities to obtain stable ids, upserts each memory
    via the FR-MNT-1 4-way write, and writes the relationship edges.

    Idempotency (FR-ING-5): a chunk whose content hash was already ingested is
    skipped; the in-process ``_seen`` set is the fast path and the per-event /
    per-chunk de-dup already done upstream by the source/accumulator/chunker is the
    backstop, so re-flushing a session the watcher already tailed is a no-op.
    """

    def __init__(
        self,
        storage: StorageBackend,
        extractor: TypedExtractor,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._storage = storage
        self._extractor = extractor
        self._settings = settings or get_settings()
        self._seen: set[str] = set()

    async def ingest_chunk(self, chunk: Chunk) -> bool:
        """Extract + persist one chunk. Returns True if it was newly ingested."""

        if chunk.content_hash in self._seen:
            return False
        self._seen.add(chunk.content_hash)
        # Raw-tier retention (FR-ING-5 / R4): persist the normalized extraction
        # input as a first-class RawChunk before/with extraction so the store can
        # re-extract/reindex offline and survive Claude's 30-day local cleanup.
        # Gated by ingest.raw_retention_enabled (default on), idempotent on the
        # content hash, and error-swallowing — a drop-in for extract_full.
        result = await extract_with_raw_retention(
            self._extractor, self._storage, chunk.events, settings=self._settings
        )
        await self._persist(result)
        return True

    async def _persist(self, result: ExtractionResult) -> None:
        """Write entities, memories and relationship edges from one extraction."""

        # Resolve entities first to get stable ids for edge resolution (FR-EXT-2).
        # resolve_or_create_entity (NOT upsert_entity) is identity-by-normalized-name:
        # an extracted entity REUSES the existing node for toLower(canonical_name)
        # instead of minting a fresh node per extraction, so the graph stops
        # fragmenting across duplicate entity nodes for the same name.
        entity_ids: dict[str, str] = {}
        for entity in result.entities:
            stored = await self._storage.resolve_or_create_entity(entity)
            entity_ids[entity.canonical_name] = stored.id
            entity_ids[stored.id] = stored.id

        for memory in result.memories:
            await self._storage.upsert_memory(memory)
            # Inline-mentions seam: connect the memory to its entities the instant
            # it lands instead of waiting for the batch MentionsJob. Resolve the
            # memory's entity-name list through the SAME identity-by-normalized-name
            # seam (reusing the entity_ids map already built above, so no extra
            # reads), then idempotently MERGE the MNEMOZINE_MENTIONS edges by id.
            # The batch persist_mentions stays a whole-store backstop, so this is
            # purely additive.
            mention_ids = [
                eid
                for name in memory.entities
                if (eid := await self._resolve_entity_id(name, entity_ids)) is not None
            ]
            if mention_ids:
                await self._storage.add_memory_mentions(memory.id, mention_ids)

        for rel in result.relationships:
            from_id = await self._resolve_entity_id(rel.subject, entity_ids)
            to_id = await self._resolve_entity_id(rel.object, entity_ids)
            if from_id is None or to_id is None:
                logger.debug(
                    "skipping relationship %r->%r: unresolved entity", rel.subject, rel.object
                )
                continue
            await self._storage.upsert_edge(rel.to_edge(from_id, to_id))

    async def _resolve_entity_id(
        self, name: str, entity_ids: dict[str, str]
    ) -> str | None:
        """Resolve an entity name to a stable id, creating the node if needed.

        Goes through the SAME identity-by-normalized-name seam
        (:meth:`~mnemozine.interfaces.StorageBackend.resolve_or_create_entity`) as
        the extracted-entities loop, so a relationship subject/object folds onto the
        existing node for ``toLower(name)`` instead of creating a parallel duplicate
        — both resolution paths now share one seam.
        """

        if name in entity_ids:
            return entity_ids[name]
        resolved = await self._storage.resolve_or_create_entity(
            Entity(canonical_name=name)
        )
        entity_ids[name] = resolved.id
        return resolved.id

    async def ingest_events(self, events: Sequence[IngestEvent]) -> int:
        """Chunk a finite event stream and ingest each chunk. Returns chunk count."""

        count = 0
        for chunk in chunk_events(list(events), self._settings.ingest):
            if await self.ingest_chunk(chunk):
                count += 1
        return count

    async def flush_session(
        self, *, session_id: str, transcript_path: str | None, project: str | None
    ) -> int:
        """Flush one Claude Code session's transcript into the pipeline (FR-ING-6).

        Parses the named transcript, chunks it, and ingests every not-yet-seen
        chunk. Idempotent on the FR-ING-5 content hash, so flushing a session the
        watcher already tailed submits nothing. Returns the number of chunks newly
        submitted. A missing/unreadable transcript is a no-op (a hook must never
        break the user's session).
        """

        if not transcript_path:
            return 0
        try:
            events = read_transcript(
                transcript_path,
                strip_tool_calls=self._settings.ingest.strip_tool_calls,
            )
        except OSError:
            logger.warning("flush_session: cannot read transcript %s", transcript_path)
            return 0
        return await self.ingest_events(events)


__all__ = ["make_contradiction_fn", "MnemozineIngestService"]
