"""Offline unit tests for the typed extraction layer (FR-EXT-1..4, R1).

Everything here runs against the deterministic ``FakeLLMProvider`` from
``tests/conftest.py`` — no live Qwen/FalkorDB/Ollama. The tests pin the
make-or-break behaviors the PRD calls out:

* classification into exactly one MemoryType (FR-EXT-1),
* scope set at extraction time, derived from type, no cross-project leak (FR-EXT-3),
* entity + relationship extraction (FR-EXT-2),
* confidence + provenance back to the source session/chunk (FR-EXT-4),
* the independently-testable single-statement ``classify`` path (R1).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from mnemozine.config import Settings
from mnemozine.extract import ExtractedRelationship, TypedExtractor
from mnemozine.interfaces import Extractor
from mnemozine.schema.events import IngestEvent, Role, Source, chunk_content_hash
from mnemozine.schema.models import MemoryType, Provenance, Scope
from tests.conftest import FakeLLMProvider

# ---------------------------------------------------------------------------
# Canned model responses (what a well-behaved Qwen would return for the
# sample_events chunk: a global preference + a project_fact + a relationship).
# ---------------------------------------------------------------------------

GOOD_EXTRACT_RESPONSE: dict[str, Any] = {
    "memories": [
        {
            "content": "Prefers thiserror over anyhow for Rust error handling.",
            "type": "preference",
            "scope": "global",
            "entities": ["rust", "error-handling", "thiserror"],
            "confidence": 0.9,
        },
        {
            "content": "This project pins tokio 1.38.",
            "type": "project_fact",
            # NOTE: model returns a *wrong* scope on purpose to prove Python
            # re-derives scope from type and does not trust this string.
            "scope": "global",
            "entities": ["tokio", "async"],
            "confidence": 0.8,
        },
    ],
    "relationships": [
        {"subject": "rust-cli", "relation": "pins", "object": "tokio"},
    ],
}


def make_extractor(
    *,
    json_responses: list[dict[str, Any]] | None = None,
    json_responder: Any | None = None,
    min_confidence: float = 0.0,
) -> tuple[TypedExtractor, FakeLLMProvider]:
    llm = FakeLLMProvider(json_responses=json_responses, json_responder=json_responder)
    extractor = TypedExtractor(
        llm, settings=Settings(), min_confidence=min_confidence
    )
    return extractor, llm


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_typed_extractor_satisfies_protocol() -> None:
    extractor, _ = make_extractor()
    assert isinstance(extractor, Extractor)


# ---------------------------------------------------------------------------
# FR-EXT-1: classification into exactly one MemoryType
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_classifies_each_unit(
    sample_events: list[IngestEvent],
) -> None:
    extractor, _ = make_extractor(json_responses=[GOOD_EXTRACT_RESPONSE])
    memories = await extractor.extract(sample_events)

    assert len(memories) == 2
    types = {m.content: m.type for m in memories}
    assert types["Prefers thiserror over anyhow for Rust error handling."] is (
        MemoryType.PREFERENCE
    )
    assert types["This project pins tokio 1.38."] is MemoryType.PROJECT_FACT
    # Each unit carries exactly one type from the allowed enum.
    for m in memories:
        assert m.type in set(MemoryType)


@pytest.mark.asyncio
async def test_idea_seed_is_extracted_as_its_own_unit() -> None:
    """FR-EXT-1: an idea_seed becomes its own memory unit (-> own node/embedding)."""

    events = [
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="rust-cli",
            session_id="sess-9",
            timestamp=datetime(2026, 6, 13, tzinfo=UTC),
            role=Role.USER,
            content="Idea: a CLI that diffs two SQL schemas and emits a migration.",
        )
    ]
    response = {
        "memories": [
            {
                "content": "Idea: a CLI that diffs two SQL schemas and emits a migration.",
                "type": "idea_seed",
                "scope": "global",
                "entities": ["cli", "sql", "migration"],
                "confidence": 0.7,
            }
        ],
        "relationships": [],
    }
    extractor, _ = make_extractor(json_responses=[response])
    memories = await extractor.extract(events)

    assert len(memories) == 1
    seed = memories[0]
    assert seed.type is MemoryType.IDEA_SEED
    # idea_seed lives in global scope and keeps its entities (for cross-ref).
    assert seed.scope.is_global
    assert "sql" in seed.entities


# ---------------------------------------------------------------------------
# FR-EXT-3: scope set at extraction time, derived from type, no leak
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scope_is_derived_from_type_not_model_string(
    sample_events: list[IngestEvent],
) -> None:
    """FR-EXT-3: scope comes from type+project, NOT the model's scope string."""

    extractor, _ = make_extractor(json_responses=[GOOD_EXTRACT_RESPONSE])
    memories = await extractor.extract(sample_events)

    by_type = {m.type: m for m in memories}
    # preference -> global, regardless of what the model said.
    assert by_type[MemoryType.PREFERENCE].scope.as_str() == "global"
    # project_fact -> project:<chunk project>, even though the model wrongly said
    # "global". This is the no-leak guarantee.
    pf = by_type[MemoryType.PROJECT_FACT]
    assert pf.scope.as_str() == "project:rust-cli"
    assert not pf.scope.is_global


@pytest.mark.asyncio
async def test_project_fact_scoped_to_chunk_project() -> None:
    """FR-EXT-3 no-leak: the project scope is the chunk's project, period."""

    events = [
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="payments-svc",
            session_id="sess-2",
            timestamp=datetime(2026, 6, 13, tzinfo=UTC),
            role=Role.USER,
            content="The auth service runs on port 8081 in this repo.",
        )
    ]
    response = {
        "memories": [
            {
                "content": "The auth service runs on port 8081 in this repo.",
                "type": "project_fact",
                "scope": "project:some-other-project",  # model hallucinated
                "entities": ["auth", "port"],
                "confidence": 0.85,
            }
        ],
        "relationships": [],
    }
    extractor, _ = make_extractor(json_responses=[response])
    memories = await extractor.extract(events)

    assert memories[0].scope == Scope.project("payments-svc")


# ---------------------------------------------------------------------------
# FR-EXT-2: entity & relationship extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entities_extracted_and_normalized() -> None:
    events = [
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="rust-cli",
            session_id="sess-3",
            timestamp=datetime(2026, 6, 13, tzinfo=UTC),
            role=Role.USER,
            content="I prefer thiserror.",
        )
    ]
    response = {
        "memories": [
            {
                "content": "Prefers thiserror.",
                "type": "preference",
                "scope": "global",
                # mixed case + dupes + blank should normalize/de-dupe.
                "entities": ["Rust", "rust", "Error-Handling", "", "thiserror"],
                "confidence": 0.9,
            }
        ],
        "relationships": [],
    }
    extractor, _ = make_extractor(json_responses=[response])
    memories = await extractor.extract(events)

    assert memories[0].entities == ["rust", "error-handling", "thiserror"]


@pytest.mark.asyncio
async def test_relationships_and_entities_collected(
    sample_events: list[IngestEvent],
) -> None:
    extractor, _ = make_extractor(json_responses=[GOOD_EXTRACT_RESPONSE])
    result = await extractor.extract_full(sample_events)

    # The relationship triple is parsed (FR-EXT-2).
    assert ExtractedRelationship("rust-cli", "pins", "tokio") in result.relationships

    # Collected entities are the union of memory tags + relationship endpoints,
    # de-duped, each a first-class Entity node.
    names = {e.canonical_name for e in result.entities}
    assert {"rust", "error-handling", "thiserror", "tokio", "async", "rust-cli"} <= names
    # No duplicate Entity nodes for the same canonical name.
    assert len(names) == len(result.entities)


@pytest.mark.asyncio
async def test_relationship_to_edge_is_temporal() -> None:
    rel = ExtractedRelationship("rust-cli", "pins", "tokio")
    edge = rel.to_edge("id-a", "id-b")
    assert edge.from_entity == "id-a"
    assert edge.to_entity == "id-b"
    assert edge.relation == "pins"
    assert edge.is_active  # FR-EXT-2: written with an open validity window
    assert edge.valid_from is not None


# ---------------------------------------------------------------------------
# FR-EXT-4: confidence + provenance back to the source session/chunk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provenance_links_back_to_source(
    sample_events: list[IngestEvent],
) -> None:
    extractor, _ = make_extractor(json_responses=[GOOD_EXTRACT_RESPONSE])
    memories = await extractor.extract(sample_events)

    expected_hash = chunk_content_hash(sample_events)
    for m in memories:
        prov = m.provenance
        assert isinstance(prov, Provenance)
        assert not prov.is_classify_sentinel  # real provenance, not the sentinel
        assert prov.source == Source.CLAUDE_CODE.value
        assert prov.session_id == "sess-1"
        assert prov.chunk_hash == expected_hash


@pytest.mark.asyncio
async def test_provenance_carries_raw_path_from_metadata() -> None:
    events = [
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="rust-cli",
            session_id="sess-7",
            timestamp=datetime(2026, 6, 13, tzinfo=UTC),
            role=Role.USER,
            content="I prefer thiserror.",
            metadata={"raw_path": "~/.claude/projects/rust-cli/sess-7.jsonl"},
        )
    ]
    response = {
        "memories": [
            {
                "content": "Prefers thiserror.",
                "type": "preference",
                "scope": "global",
                "entities": ["rust"],
                "confidence": 0.9,
            }
        ],
        "relationships": [],
    }
    extractor, _ = make_extractor(json_responses=[response])
    memories = await extractor.extract(events)
    assert memories[0].provenance.raw_path == (
        "~/.claude/projects/rust-cli/sess-7.jsonl"
    )


@pytest.mark.asyncio
async def test_confidence_clamped_into_range() -> None:
    events = [
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="p",
            session_id="s",
            timestamp=datetime(2026, 6, 13, tzinfo=UTC),
            role=Role.USER,
            content="x",
        )
    ]
    response = {
        "memories": [
            {
                "content": "A.",
                "type": "preference",
                "scope": "global",
                "entities": [],
                "confidence": 1.7,  # out of range high
            },
            {
                "content": "B.",
                "type": "preference",
                "scope": "global",
                "entities": [],
                "confidence": -0.4,  # out of range low
            },
            {
                "content": "C.",
                "type": "preference",
                "scope": "global",
                "entities": [],
                "confidence": "garbage",  # unparseable -> 0.5 default
            },
        ],
        "relationships": [],
    }
    extractor, _ = make_extractor(json_responses=[response])
    memories = await extractor.extract(events)
    by_content = {m.content: m.confidence for m in memories}
    assert by_content["A."] == 1.0
    assert by_content["B."] == 0.0
    assert by_content["C."] == 0.5


# ---------------------------------------------------------------------------
# Robustness: empty chunks, no durable memory, malformed model output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_chunk_makes_no_llm_call() -> None:
    extractor, llm = make_extractor(json_responses=[GOOD_EXTRACT_RESPONSE])
    memories = await extractor.extract([])
    assert memories == []
    # No LLM call wasted on an empty chunk.
    assert llm.calls == []


@pytest.mark.asyncio
async def test_no_durable_memory_returns_empty(
    sample_events: list[IngestEvent],
) -> None:
    extractor, _ = make_extractor(
        json_responses=[{"memories": [], "relationships": []}]
    )
    assert await extractor.extract(sample_events) == []


@pytest.mark.asyncio
async def test_malformed_entries_are_skipped(
    sample_events: list[IngestEvent],
) -> None:
    response = {
        "memories": [
            {"content": "  ", "type": "preference", "scope": "global",
             "entities": [], "confidence": 0.9},  # blank content -> skip
            {"content": "Valid.", "type": "not-a-type", "scope": "global",
             "entities": [], "confidence": 0.9},  # bad type -> skip
            "totally-not-a-dict",  # wrong shape -> skip
            {"content": "Kept.", "type": "preference", "scope": "global",
             "entities": ["x"], "confidence": 0.9},  # the only good one
        ],
        "relationships": ["bad", {"subject": "a"}],  # malformed rels -> skipped
    }
    extractor, _ = make_extractor(json_responses=[response])
    result = await extractor.extract_full(sample_events)
    assert [m.content for m in result.memories] == ["Kept."]
    assert result.relationships == []


@pytest.mark.asyncio
async def test_min_confidence_drops_weak_memories(
    sample_events: list[IngestEvent],
) -> None:
    extractor, _ = make_extractor(
        json_responses=[GOOD_EXTRACT_RESPONSE], min_confidence=0.85
    )
    memories = await extractor.extract(sample_events)
    # Only the 0.9 preference survives; the 0.8 project_fact is dropped.
    assert [m.content for m in memories] == [
        "Prefers thiserror over anyhow for Rust error handling."
    ]


@pytest.mark.asyncio
async def test_extract_uses_chunk_project_in_prompt() -> None:
    """The chunk's project id must reach the model prompt (FR-EXT-3 scoping)."""

    events = [
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="my-special-project",
            session_id="s",
            timestamp=datetime(2026, 6, 13, tzinfo=UTC),
            role=Role.USER,
            content="hello",
        )
    ]
    extractor, llm = make_extractor(
        json_responses=[{"memories": [], "relationships": []}]
    )
    await extractor.extract(events)
    assert llm.calls and llm.calls[0]["kind"] == "json"
    assert "my-special-project" in llm.calls[0]["prompt"]
