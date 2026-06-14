"""Tests for the independently-evaluable extraction prompts (FR-EXT-1/2/3).

The PRD requires the prompts to be kept in a ``prompts/`` module so they are
independently evaluable (R1). These tests assert the prompts are well-formed,
embed the shared rubric, encode the CATEGORY-SPLIT contract (controlled scope
decision + free-form category + cross_ref flag — core data-model redesign), and
faithfully carry the project id + statement the model needs to assign scope at
extraction time.
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
from mnemozine.extract.prompts.taxonomy import (
    ALLOWED_SCOPE_DECISIONS,
    TAXONOMY_RUBRIC,
)
from mnemozine.schema.events import IngestEvent, Role, Source


def test_allowed_scope_decisions_match_schema_enums() -> None:
    # Prompt taxonomy and JSON schema must agree on the controlled scope decision
    # set — the two-value global/project enum (FR-EXT-3), NOT the old 3-type enum.
    assert set(ALLOWED_SCOPE_DECISIONS) == {"global", "project"}
    classify_enum = CLASSIFY_JSON_SCHEMA["properties"]["scope"]["enum"]  # type: ignore[index]
    assert set(classify_enum) == set(ALLOWED_SCOPE_DECISIONS)
    mem_props = EXTRACT_JSON_SCHEMA["properties"]["memories"]["items"]["properties"]  # type: ignore[index]
    assert set(mem_props["scope"]["enum"]) == set(ALLOWED_SCOPE_DECISIONS)


def test_category_is_free_form_not_enum_constrained() -> None:
    # The category split: category is a FREE-FORM string with NO enum constraint
    # in either schema (emergent categories converge via the merge job, FR-MNT-2/4).
    classify_cat = CLASSIFY_JSON_SCHEMA["properties"]["category"]  # type: ignore[index]
    assert classify_cat["type"] == "string"
    assert "enum" not in classify_cat
    mem_props = EXTRACT_JSON_SCHEMA["properties"]["memories"]["items"]["properties"]  # type: ignore[index]
    assert mem_props["category"]["type"] == "string"
    assert "enum" not in mem_props["category"]
    # cross_ref is a boolean flag in both schemas (the old idea_seed behavior).
    assert mem_props["cross_ref"]["type"] == "boolean"
    assert CLASSIFY_JSON_SCHEMA["properties"]["cross_ref"]["type"] == "boolean"  # type: ignore[index]


def test_both_prompts_embed_the_shared_rubric() -> None:
    # Single source of truth for the make-or-break definition (FR-EXT-3): the
    # global-vs-project scope disambiguation.
    snippet = "global vs project"
    assert snippet in TAXONOMY_RUBRIC
    assert snippet in CLASSIFY_SYSTEM_PROMPT
    assert snippet in EXTRACT_SYSTEM_PROMPT


def test_system_prompts_state_the_controlled_scope_decision() -> None:
    # FR-EXT-3: scope decided at extraction time as a controlled two-value decision.
    for prompt in (CLASSIFY_SYSTEM_PROMPT, EXTRACT_SYSTEM_PROMPT):
        assert '"global"' in prompt
        assert '"project"' in prompt
        # The prompt must instruct the model NOT to emit a scope path (no-leak:
        # Python derives the hierarchical scope from the decision + project).
        assert "scope path" in prompt


def test_build_classify_prompt_carries_project_and_statement() -> None:
    prompt = build_classify_prompt(
        "This project pins tokio 1.38.",
        project="rust-cli",
        recent_text="earlier we discussed async runtimes",
    )
    assert "rust-cli" in prompt
    assert "This project pins tokio 1.38." in prompt
    assert "async runtimes" in prompt  # recent_text included as advisory context


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
