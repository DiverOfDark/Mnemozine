"""Service-layer tests for the ingest persist path (identity-by-name).

Drives :meth:`MnemozineIngestService._persist` directly with synthetic
:class:`~mnemozine.extract.extractor.ExtractionResult`s (no LLM / FalkorDB) to pin
the entity-identity fix: the persist path resolves an extracted entity to the
EXISTING node for its normalized name instead of minting a duplicate, and
relationship subject/object resolution folds onto the same node.
"""

from __future__ import annotations

import pytest

from mnemozine.config import Settings
from mnemozine.extract.extractor import ExtractedRelationship, ExtractionResult
from mnemozine.schema.models import Entity, MemoryUnit, Provenance, Scope
from mnemozine.services import MnemozineIngestService
from tests.conftest import InMemoryStorage


def _service(storage: InMemoryStorage) -> MnemozineIngestService:
    # The extractor is never invoked (we call _persist directly), so None-typed is
    # fine for these persist-path tests.
    return MnemozineIngestService(storage, None, settings=Settings())  # type: ignore[arg-type]


def _memory(content: str, *, mid: str, entities: list[str]) -> MemoryUnit:
    return MemoryUnit(
        id=mid,
        content=content,
        scope=Scope.global_(),
        category="fact",
        entities=entities,
        provenance=Provenance(source="claude_code", session_id="s1"),
    )


@pytest.mark.asyncio
async def test_two_chunks_same_entity_name_yield_one_node() -> None:
    """Ingesting two chunks that mention the same entity name -> ONE entity node.

    Regression for the duplicate-entity leak: the persist path used to call
    upsert_entity (id-keyed) for every extracted entity, minting a fresh node per
    extraction. It now resolves by normalized name, so the second chunk's ``Rust``
    folds onto the first chunk's ``rust`` node.
    """

    storage = InMemoryStorage()
    service = _service(storage)

    # Chunk 1: extracts the entity "rust".
    await service._persist(
        ExtractionResult(
            memories=[_memory("a", mid="m1", entities=["rust"])],
            entities=[Entity(canonical_name="rust")],
            relationships=[],
        )
    )
    # Chunk 2: extracts the SAME entity under a different-cased spelling.
    await service._persist(
        ExtractionResult(
            memories=[_memory("b", mid="m2", entities=["Rust"])],
            entities=[Entity(canonical_name="Rust")],
            relationships=[],
        )
    )

    # Exactly one entity node for the normalized name "rust".
    assert len(storage.entities) == 1
    survivor = next(iter(storage.entities.values()))
    assert survivor.canonical_name.lower() == "rust"
    # The differing-cased spelling folded into aliases.
    assert "Rust" in survivor.aliases


@pytest.mark.asyncio
async def test_relationship_endpoints_resolve_onto_existing_node() -> None:
    """A relationship subject/object folds onto the existing entity node.

    The extracted-entities loop and relationship resolution now share the one
    resolve_or_create_entity seam, so a relationship naming ``rust`` (already an
    extracted entity) binds its edge to that SAME node — no parallel duplicate.
    """

    storage = InMemoryStorage()
    service = _service(storage)

    await service._persist(
        ExtractionResult(
            memories=[_memory("a", mid="m1", entities=["rust", "tokio"])],
            entities=[
                Entity(canonical_name="rust"),
                Entity(canonical_name="tokio"),
            ],
            relationships=[
                # Subject is an already-extracted entity; object is a NEW name only
                # seen via the relationship (must resolve-or-create through the same
                # seam, not a parallel id-keyed mint).
                ExtractedRelationship(subject="rust", relation="depends_on", object="tokio"),
                ExtractedRelationship(subject="rust", relation="relates", object="async"),
            ],
        )
    )

    # rust, tokio, async -> exactly three nodes (no duplicate rust/tokio).
    assert len(storage.entities) == 3
    name_to_id = {e.canonical_name.lower(): e.id for e in storage.entities.values()}
    assert set(name_to_id) == {"rust", "tokio", "async"}

    # Both edges bound to the SAME rust node (the existing one), and the tokio
    # endpoint is the existing tokio node — no parallel duplicate was created.
    edges = list(storage.edges.values())
    assert len(edges) == 2
    rust_id = name_to_id["rust"]
    tokio_id = name_to_id["tokio"]
    async_id = name_to_id["async"]
    endpoints = {(e.from_entity, e.to_entity) for e in edges}
    assert (rust_id, tokio_id) in endpoints
    assert (rust_id, async_id) in endpoints


@pytest.mark.asyncio
async def test_persist_writes_inline_mention_edges_without_batch_job() -> None:
    """A freshly persisted memory has its MNEMOZINE_MENTIONS edges immediately.

    The inline-mentions seam: ``_persist`` MERGEs the memory's mention edges to its
    resolved entities right after the upsert, so the memory is connected the instant
    it lands — WITHOUT running the batch ``persist_mentions`` job. Reuses the same
    identity-by-name resolution as the entity loop, so the edges bind to the stored
    entity ids.
    """

    storage = InMemoryStorage()
    service = _service(storage)

    await service._persist(
        ExtractionResult(
            memories=[_memory("a", mid="m1", entities=["rust", "tokio"])],
            entities=[
                Entity(canonical_name="rust"),
                Entity(canonical_name="tokio"),
            ],
            relationships=[],
        )
    )

    # Mention edges exist WITHOUT calling storage.persist_mentions().
    name_to_id = {e.canonical_name.lower(): e.id for e in storage.entities.values()}
    assert storage.mentions == {
        ("m1", name_to_id["rust"]),
        ("m1", name_to_id["tokio"]),
    }


@pytest.mark.asyncio
async def test_persist_inline_mentions_resolve_cased_name_onto_one_node() -> None:
    """Inline mentions fold a different-cased mention name onto the existing node.

    The memory's ``entities`` list carries a different-cased spelling than the
    extracted entity; both resolve through the one identity-by-name seam, so the
    mention edge binds to the SINGLE node for that normalized name (no parallel
    duplicate, no dangling mention).
    """

    storage = InMemoryStorage()
    service = _service(storage)

    await service._persist(
        ExtractionResult(
            # Memory mentions "Rust" (cased) while the extracted entity is "rust".
            memories=[_memory("a", mid="m1", entities=["Rust"])],
            entities=[Entity(canonical_name="rust")],
            relationships=[],
        )
    )

    assert len(storage.entities) == 1
    rust_id = next(iter(storage.entities.values())).id
    assert storage.mentions == {("m1", rust_id)}


@pytest.mark.asyncio
async def test_persist_inline_mentions_are_idempotent() -> None:
    """Re-persisting the same extraction re-asserts the same mention edges, adds none."""

    storage = InMemoryStorage()
    service = _service(storage)

    result = ExtractionResult(
        memories=[_memory("a", mid="m1", entities=["rust"])],
        entities=[Entity(canonical_name="rust")],
        relationships=[],
    )
    await service._persist(result)
    after_first = set(storage.mentions)
    await service._persist(result)

    # Set-keyed MERGE: the re-persist did not grow the mention edge set.
    assert storage.mentions == after_first
    rust_id = next(iter(storage.entities.values())).id
    assert storage.mentions == {("m1", rust_id)}
