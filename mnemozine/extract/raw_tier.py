"""Raw-tier retention seam: persist the extraction-input chunk with extraction.

Core data-model redesign (FR-ING-5 / R4): every ingested chunk's *normalized*
extraction input is retained as a first-class
:class:`~mnemozine.schema.models.RawChunk` (the raw tier) via
:meth:`StorageBackend.persist_raw_chunk`, before/with extraction, so the store
can re-extract / reindex offline and survive Claude's 30-day local transcript
cleanup (R4).

This is the small orchestration seam the ingest pipeline calls instead of
``extractor.extract_full`` directly when ``ingest.raw_retention_enabled`` is on
(the default): it persists the raw chunk first (so the input is durable even if
extraction crashes), runs extraction, then re-persists the chunk with the
``memory_ids`` of the units it produced (idempotent on the FR-ING-5 content hash,
so the second persist updates the same node) — giving offline re-extraction the
forward link from a raw chunk to exactly the memories it produced.

Kept out of :class:`~mnemozine.extract.TypedExtractor` deliberately: the
extractor must not depend on a storage backend (it stays unit-testable with only
a fake LLM). This function takes both as plain Protocol handles.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from mnemozine.config import Settings, get_settings
from mnemozine.extract.extractor import ExtractionResult, TypedExtractor
from mnemozine.interfaces import StorageBackend
from mnemozine.schema.events import IngestEvent
from mnemozine.schema.models import RawChunk

logger = logging.getLogger(__name__)


async def extract_with_raw_retention(
    extractor: TypedExtractor,
    storage: StorageBackend,
    events: Sequence[IngestEvent],
    *,
    settings: Settings | None = None,
) -> ExtractionResult:
    """Extract a chunk, retaining its normalized input as a :class:`RawChunk`.

    The raw-tier-aware replacement for a bare ``extractor.extract_full(events)``
    call in the ingest pipeline. When ``ingest.raw_retention_enabled`` is on (the
    default):

    1. persist the :class:`RawChunk` (the normalized extraction input) FIRST, so
       the durable copy survives even if extraction then fails;
    2. run extraction (``extractor.extract_full``);
    3. re-persist the same chunk (idempotent on its FR-ING-5 ``content_hash``) now
       carrying the ``memory_ids`` of the units produced, so offline
       re-extraction can supersede exactly those memories.

    A raw-chunk persistence error is logged and swallowed — raw retention is a
    durability nicety and must never break the live extraction/ingest path.
    Returns the :class:`ExtractionResult` from extraction unchanged. With
    retention disabled (or an empty chunk) this is exactly ``extract_full``.
    """

    resolved = settings or get_settings()
    evs = list(events)
    if not evs:
        return await extractor.extract_full(evs)

    if not resolved.ingest.raw_retention_enabled:
        return await extractor.extract_full(evs)

    raw_chunk = extractor.build_raw_chunk(evs)
    # Persist the input first (durable before extraction), best-effort.
    if raw_chunk is not None:
        await _persist_raw_chunk(storage, raw_chunk)

    result = await extractor.extract_full(evs)

    # Re-persist with the produced memory ids (idempotent on content_hash) so the
    # raw chunk links forward to exactly the memories it produced (offline reindex).
    if raw_chunk is not None and result.memories:
        raw_chunk.memory_ids = [m.id for m in result.memories]
        await _persist_raw_chunk(storage, raw_chunk)

    return result


async def _persist_raw_chunk(storage: StorageBackend, chunk: RawChunk) -> None:
    """Persist a raw chunk, swallowing errors (durability must not break ingest)."""

    try:
        await storage.persist_raw_chunk(chunk)
    except Exception:  # noqa: BLE001 - raw retention must never break ingest
        logger.exception(
            "failed to persist raw chunk %s (session %s); continuing",
            chunk.content_hash,
            chunk.session_id,
        )


__all__ = ["extract_with_raw_retention"]
