"""Tests for the independently-evaluable extraction prompts (FR-EXT-1/2/3).

The PRD requires the prompts to be kept in a ``prompts/`` module so they are
independently evaluable (R1). These tests assert the prompts are well-formed,
embed the shared rubric, and faithfully carry the project id + statement the
model needs to assign scope at extraction time.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mnemozine.extract.prompts import (
    CLASSIFY_JSON_SCHEMA,
    CLASSIFY_SYSTEM_PROMPT,
    EXTRACT_JSON_SCHEMA,
    EXTRACT_SYSTEM_PROMPT,
    build_classify_prompt,
    build_extract_prompt,
)
from mnemozine.extract.prompts.extract import render_chunk
from mnemozine.extract.prompts.taxonomy import ALLOWED_TYPES, TAXONOMY_RUBRIC
from mnemozine.schema.events import IngestEvent, Role, Source


def test_allowed_types_match_schema_enums() -> None:
    # Prompt taxonomy and JSON schema must agree on the type set (FR-EXT-1).
    assert set(ALLOWED_TYPES) == {"preference", "project_fact", "idea_seed"}
    classify_enum = CLASSIFY_JSON_SCHEMA["properties"]["type"]["enum"]  # type: ignore[index]
    assert set(classify_enum) == set(ALLOWED_TYPES)
    mem_props = EXTRACT_JSON_SCHEMA["properties"]["memories"]["items"]["properties"]  # type: ignore[index]
    assert set(mem_props["type"]["enum"]) == set(ALLOWED_TYPES)


def test_both_prompts_embed_the_shared_rubric() -> None:
    # Single source of truth for the make-or-break definition (FR-EXT-3).
    snippet = "preference vs project_fact"
    assert snippet in TAXONOMY_RUBRIC
    assert snippet in CLASSIFY_SYSTEM_PROMPT
    assert snippet in EXTRACT_SYSTEM_PROMPT


def test_system_prompts_state_scope_rule() -> None:
    # FR-EXT-3: scope decided at extraction time, derived from type.
    for prompt in (CLASSIFY_SYSTEM_PROMPT, EXTRACT_SYSTEM_PROMPT):
        assert "global" in prompt
        assert "project:<project_id>" in prompt


def test_build_classify_prompt_carries_project_and_statement() -> None:
    prompt = build_classify_prompt(
        "This project pins tokio 1.38.",
        project="rust-cli",
        recent_text="earlier we discussed async runtimes",
    )
    assert "rust-cli" in prompt
    assert "This project pins tokio 1.38." in prompt
    assert "async runtimes" in prompt  # recent_text included as advisory context
    assert "project:rust-cli" in prompt


def test_build_classify_prompt_without_project() -> None:
    prompt = build_classify_prompt("I prefer thiserror.")
    assert "unknown" in prompt  # graceful fallback project id
    assert "I prefer thiserror." in prompt


def test_build_extract_prompt_renders_transcript_and_project() -> None:
    events = [
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="rust-cli",
            session_id="s",
            timestamp=datetime(2026, 6, 13, tzinfo=UTC),
            role=Role.USER,
            content="I prefer thiserror.",
        ),
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="rust-cli",
            session_id="s",
            timestamp=datetime(2026, 6, 13, tzinfo=UTC),
            role=Role.ASSISTANT,
            content="Got it.",
        ),
    ]
    prompt = build_extract_prompt(events, project="rust-cli")
    assert "rust-cli" in prompt
    assert "user: I prefer thiserror." in prompt
    assert "assistant: Got it." in prompt


def test_render_chunk_skips_blank_turns() -> None:
    events = [
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="p",
            session_id="s",
            timestamp=datetime(2026, 6, 13, tzinfo=UTC),
            role=Role.USER,
            content="   ",  # blank -> dropped
        ),
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="p",
            session_id="s",
            timestamp=datetime(2026, 6, 13, tzinfo=UTC),
            role=Role.USER,
            content="real content",
        ),
    ]
    rendered = render_chunk(events)
    assert rendered == "user: real content"
