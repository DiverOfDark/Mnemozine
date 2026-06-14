"""Offline unit tests for the typed extraction layer (FR-EXT-1..4, R1).

Everything here runs against the deterministic ``FakeLLMProvider`` from
``tests/conftest.py`` — no live Qwen/FalkorDB/Ollama. The tests pin the
make-or-break behaviors the PRD calls out, on the CATEGORY-SPLIT contract (core
data-model redesign): the classifier emits per memory unit

* a CONTROLLED ``scope`` decision (``global`` vs ``project``) — FR-EXT-1/3 — from
  which the final hierarchical Scope is derived in Python (no-leak, FR-EXT-3),
* a FREE-FORM ``category`` slug (no enum) and a ``cross_ref`` boolean,

plus entity + relationship extraction (FR-EXT-2), confidence + provenance back to
the source session/chunk (FR-EXT-4), the raw-chunk retention seam, and the
single-statement ``classify`` path (R1).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from mnemozine.config import Settings
from mnemozine.extract import (
    ExtractedRelationship,
    TypedExtractor,
    build_raw_chunk,
    extract_with_raw_retention,
)
from mnemozine.interfaces import Extractor
from mnemozine.schema.events import IngestEvent, Role, Source, chunk_content_hash
from mnemozine.schema.models import Provenance, RawChunk, Scope, ScopeDecision
from tests.conftest import FakeLLMProvider, InMemoryStorage

# ---------------------------------------------------------------------------
# Canned model responses (what a well-behaved Qwen would return for the
# sample_events chunk: a global preference + a project fact + a relationship).
# The model emits the category-split signals: scope decision, free-form category,
# cross_ref flag — NOT the old 3-type enum.
# ---------------------------------------------------------------------------

GOOD_EXTRACT_RESPONSE: dict[str, Any] = {
    "memories": [
        {
            "content": "Prefers thiserror over anyhow for Rust error handling.",
            "scope": "global",
            "category": "preference",
            "cross_ref": False,
            "entities": ["rust", "error-handling", "thiserror"],
            "confidence": 0.9,
        },
        {
            "content": "This project pins tokio 1.38.",
            "scope": "project",
            "category": "decision",
            "cross_ref": False,
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
# FR-EXT-1: the category split — controlled scope decision + free-form category
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_emits_category_split_signals(
    sample_events: list[IngestEvent],
) -> None:
    extractor, _ = make_extractor(json_responses=[GOOD_EXTRACT_RESPONSE])
    memories = await extractor.extract(sample_events)

    assert len(memories) == 2
    by_content = {m.content: m for m in memories}

    pref = by_content["Prefers thiserror over anyhow for Rust error handling."]
    assert pref.scope_decision is ScopeDecision.GLOBAL
    assert pref.category == "preference"  # FREE-FORM string, not an enum
    assert pref.cross_ref_candidate is False

    fact = by_content["This project pins tokio 1.38."]
    assert fact.scope_decision is ScopeDecision.PROJECT
    assert fact.category == "decision"  # emergent free-form category
    # No `type` attribute survives the redesign.
    assert not hasattr(pref, "type")


@pytest.mark.asyncio
async def test_cross_ref_flag_preserves_idea_seed_behavior() -> None:
    """An idea memory is flagged cross_ref_candidate (the old idea_seed behavior)."""

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
                "scope": "global",
                "category": "idea",
                "cross_ref": True,
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
    assert seed.cross_ref_candidate is True
    assert seed.category == "idea"
    # idea/cross-ref seeds live in global scope and keep entities (for cross-ref).
    assert seed.scope.is_global
    assert "sql" in seed.entities


@pytest.mark.asyncio
async def test_free_form_category_is_normalized() -> None:
    """A free-form category is lowercased/trimmed (no enum constraint)."""

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
                "content": "Uses tabs not spaces.",
                "scope": "global",
                "category": "  Coding-Style  ",  # arbitrary slug, mixed case
                "cross_ref": False,
                "entities": [],
                "confidence": 0.9,
            },
            {
                "content": "Default category fallback.",
                "scope": "global",
                "category": "",  # blank -> DEFAULT_CATEGORY
                "cross_ref": False,
                "entities": [],
                "confidence": 0.9,
            },
        ],
        "relationships": [],
    }
    extractor, _ = make_extractor(json_responses=[response])
    memories = await extractor.extract(events)
    by_content = {m.content: m.category for m in memories}
    assert by_content["Uses tabs not spaces."] == "coding-style"
    assert by_content["Default category fallback."] == "fact"  # DEFAULT_CATEGORY


# ---------------------------------------------------------------------------
# FR-EXT-3: scope derived in Python from the decision + project, never the model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scope_is_derived_from_decision_not_model_string(
    sample_events: list[IngestEvent],
) -> None:
    """FR-EXT-3: scope comes from the controlled decision + project, not the LLM."""

    extractor, _ = make_extractor(json_responses=[GOOD_EXTRACT_RESPONSE])
    memories = await extractor.extract(sample_events)

    by_dec = {m.scope_decision: m for m in memories}
    # global decision -> global scope.
    assert by_dec[ScopeDecision.GLOBAL].scope.as_str() == "global"
    # project decision -> project:<chunk project>.
    pf = by_dec[ScopeDecision.PROJECT]
    assert pf.scope.as_str() == "project:rust-cli"
    assert not pf.scope.is_global


@pytest.mark.asyncio
async def test_project_decision_scoped_to_chunk_project() -> None:
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
                "scope": "project",
                "category": "fact",
                "cross_ref": False,
                "entities": ["auth", "port"],
                "confidence": 0.85,
            }
        ],
        "relationships": [],
    }
    extractor, _ = make_extractor(json_responses=[response])
    memories = await extractor.extract(events)

    assert memories[0].scope == Scope.project("payments-svc")


@pytest.mark.asyncio
async def test_project_scope_rolls_up_subagent_transcript() -> None:
    """FR-EXT-3 roll-up: a deep subagent chunk scopes to its parent project.

    When the chunk's events carry a subagent/workflow transcript ``raw_path``, the
    derived project scope is the rolled-up parent project (never project:agent-XXXX
    and never the flat event.project leaf).
    """

    raw_path = (
        "/home/u/.claude/projects/-var-home-u-Projects-Mnemozine/sess-1/"
        "subagents/workflows/wf_abc/agent-DEAD.jsonl"
    )
    events = [
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="agent-DEAD",  # the misleading flat leaf the old bug used
            session_id="sess-1",
            timestamp=datetime(2026, 6, 13, tzinfo=UTC),
            role=Role.USER,
            content="This project uses Postgres for the job queue.",
            metadata={"raw_path": raw_path, "cwd": "/var/home/u/Projects/Mnemozine"},
        )
    ]
    response = {
        "memories": [
            {
                "content": "This project uses Postgres for the job queue.",
                "scope": "project",
                "category": "decision",
                "cross_ref": False,
                "entities": ["postgres"],
                "confidence": 0.9,
            }
        ],
        "relationships": [],
    }
    extractor, _ = make_extractor(json_responses=[response])
    memories = await extractor.extract(events)
    # Rolls up to the parent project, not the opaque agent id.
    assert memories[0].scope == Scope.project("Mnemozine")
    assert "agent-DEAD" not in memories[0].scope.as_str()


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
                "scope": "global",
                "category": "preference",
                "cross_ref": False,
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
                "scope": "global",
                "category": "preference",
                "cross_ref": False,
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
                "scope": "global",
                "category": "fact",
                "cross_ref": False,
                "entities": [],
                "confidence": 1.7,  # out of range high
            },
            {
                "content": "B.",
                "scope": "global",
                "category": "fact",
                "cross_ref": False,
                "entities": [],
                "confidence": -0.4,  # out of range low
            },
            {
                "content": "C.",
                "scope": "global",
                "category": "fact",
                "cross_ref": False,
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
            {"content": "  ", "scope": "global", "category": "fact",
             "cross_ref": False, "entities": [], "confidence": 0.9},  # blank -> skip
            {"content": "Valid.", "scope": "not-a-decision", "category": "fact",
             "cross_ref": False, "entities": [], "confidence": 0.9},  # bad scope -> skip
            "totally-not-a-dict",  # wrong shape -> skip
            {"content": "Kept.", "scope": "global", "category": "fact",
             "cross_ref": False, "entities": ["x"], "confidence": 0.9},  # the only good one
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
    # Only the 0.9 global preference survives; the 0.8 project fact is dropped.
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


# ---------------------------------------------------------------------------
# Raw-chunk retention (the raw tier: offline re-extraction/reindex, R4)
# ---------------------------------------------------------------------------


def test_build_raw_chunk_captures_normalized_input(
    sample_events: list[IngestEvent],
) -> None:
    scope = Scope.project("rust-cli")
    chunk = build_raw_chunk(sample_events, scope)
    assert isinstance(chunk, RawChunk)
    # content_hash matches the FR-ING-5 chunk hash so it joins provenance.chunk_hash.
    assert chunk.content_hash == chunk_content_hash(sample_events)
    # content is the normalized, role-tagged transcript (tool_calls stripped upstream).
    assert "user: I prefer thiserror" in chunk.content
    assert chunk.scope == scope
    assert chunk.project == "rust-cli"
    assert chunk.source == "claude_code"
    assert chunk.session_id == "sess-1"
    assert chunk.event_count == 3
    assert chunk.started_at is not None and chunk.ended_at is not None
    assert chunk.started_at <= chunk.ended_at


@pytest.mark.asyncio
async def test_extract_with_raw_retention_persists_chunk(
    sample_events: list[IngestEvent],
) -> None:
    """The raw-tier seam persists the input chunk and links the produced memories."""

    storage = InMemoryStorage()
    extractor, _ = make_extractor(json_responses=[GOOD_EXTRACT_RESPONSE])

    result = await extract_with_raw_retention(
        extractor, storage, sample_events, settings=Settings()
    )

    # Extraction still returns the same units.
    assert len(result.memories) == 2
    # Exactly one raw chunk was persisted, keyed on the chunk content hash.
    expected_hash = chunk_content_hash(sample_events)
    assert list(storage.raw_chunks.keys()) == [expected_hash]
    stored = storage.raw_chunks[expected_hash]
    # It links forward to exactly the memories it produced (offline reindex join).
    assert set(stored.memory_ids) == {m.id for m in result.memories}
    # The retained scope is the derived project scope (sample events -> rust-cli).
    assert stored.scope == Scope.project("rust-cli")


@pytest.mark.asyncio
async def test_extract_with_raw_retention_disabled_persists_nothing(
    sample_events: list[IngestEvent],
) -> None:
    storage = InMemoryStorage()
    settings = Settings()
    settings.ingest.raw_retention_enabled = False
    extractor, _ = make_extractor(json_responses=[GOOD_EXTRACT_RESPONSE])

    result = await extract_with_raw_retention(
        extractor, storage, sample_events, settings=settings
    )
    assert len(result.memories) == 2
    # Retention off -> no raw chunk persisted, extraction unaffected.
    assert storage.raw_chunks == {}


@pytest.mark.asyncio
async def test_extract_with_raw_retention_empty_chunk_is_noop() -> None:
    storage = InMemoryStorage()
    extractor, llm = make_extractor(json_responses=[GOOD_EXTRACT_RESPONSE])
    result = await extract_with_raw_retention(
        extractor, storage, [], settings=Settings()
    )
    assert result.memories == []
    assert storage.raw_chunks == {}
    assert llm.calls == []  # no extraction call on an empty chunk
