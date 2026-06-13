"""Offline tests for the single-statement classify path (FR-EXT-3, §9, R1).

``Extractor.classify`` is the independently-testable classifier the §9
classifier-accuracy metric is measured on. These tests pin its contract: it
returns a lightweight :class:`Classification` (no provenance/validity), derives
scope from type, and degrades gracefully on bad model output — all offline with
``FakeLLMProvider``.
"""

from __future__ import annotations

from typing import Any

import pytest

from mnemozine.config import Settings
from mnemozine.extract import TypedExtractor
from mnemozine.interfaces import Classification, RetrievalContext
from mnemozine.schema.models import MemoryType, Scope
from tests.conftest import FakeLLMProvider


def make_extractor(
    *, json_responder: Any | None = None, json_responses: list[dict[str, Any]] | None = None
) -> tuple[TypedExtractor, FakeLLMProvider]:
    llm = FakeLLMProvider(json_responder=json_responder, json_responses=json_responses)
    return TypedExtractor(llm, settings=Settings()), llm


@pytest.mark.asyncio
async def test_classify_returns_lightweight_classification() -> None:
    response = {
        "type": "preference",
        "scope": "global",
        "entities": ["rust", "error-handling"],
        "confidence": 0.92,
    }
    extractor, _ = make_extractor(json_responses=[response])
    ctx = RetrievalContext(project="rust-cli")
    result = await extractor.classify(
        "I prefer thiserror over anyhow.", ctx
    )

    assert isinstance(result, Classification)
    assert result.type is MemoryType.PREFERENCE
    assert result.scope.is_global
    assert result.entities == ["rust", "error-handling"]
    assert result.confidence == 0.92


@pytest.mark.asyncio
async def test_classify_project_fact_scoped_to_context_project() -> None:
    """FR-EXT-3: a project_fact classification is scoped to context.project."""

    response = {
        "type": "project_fact",
        "scope": "global",  # model is wrong; Python re-derives from type.
        "entities": ["tokio"],
        "confidence": 0.8,
    }
    extractor, _ = make_extractor(json_responses=[response])
    ctx = RetrievalContext(project="rust-cli")
    result = await extractor.classify("This project pins tokio 1.38.", ctx)

    assert result.type is MemoryType.PROJECT_FACT
    assert result.scope == Scope.project("rust-cli")


@pytest.mark.asyncio
async def test_classify_idea_seed_is_global() -> None:
    response = {
        "type": "idea_seed",
        "scope": "global",
        "entities": ["cli", "sql"],
        "confidence": 0.6,
    }
    extractor, _ = make_extractor(json_responses=[response])
    result = await extractor.classify(
        "Idea: a CLI that diffs SQL schemas.", RetrievalContext(project="x")
    )
    assert result.type is MemoryType.IDEA_SEED
    assert result.scope.is_global


@pytest.mark.asyncio
async def test_classify_unparseable_response_is_droppable() -> None:
    """A bad/empty model response yields a confidence-0 result, never a crash."""

    extractor, _ = make_extractor(json_responses=[{}])  # empty object
    result = await extractor.classify("???", RetrievalContext(project="x"))
    assert result.confidence == 0.0  # droppable
    assert isinstance(result, Classification)


@pytest.mark.asyncio
async def test_classify_passes_project_and_context_to_prompt() -> None:
    captured: dict[str, str] = {}

    def responder(prompt: str, system: str | None) -> dict[str, Any]:
        captured["prompt"] = prompt
        return {
            "type": "project_fact",
            "scope": "project:proj-x",
            "entities": ["a"],
            "confidence": 0.7,
        }

    extractor, _ = make_extractor(json_responder=responder)
    ctx = RetrievalContext(project="proj-x", recent_text="we were discussing tokio")
    await extractor.classify("pins tokio 1.38", ctx)

    assert "proj-x" in captured["prompt"]
    assert "tokio" in captured["prompt"]  # recent_text reached the prompt


@pytest.mark.asyncio
async def test_classify_routing_is_deterministic_by_prompt() -> None:
    """FR-EXT-3/R1: classify is deterministic offline via per-prompt routing."""

    def responder(prompt: str, system: str | None) -> dict[str, Any] | None:
        if "thiserror" in prompt:
            return {
                "type": "preference",
                "scope": "global",
                "entities": ["rust"],
                "confidence": 0.9,
            }
        if "tokio 1.38" in prompt:
            return {
                "type": "project_fact",
                "scope": "project:rust-cli",
                "entities": ["tokio"],
                "confidence": 0.8,
            }
        return None

    extractor, _ = make_extractor(json_responder=responder)
    ctx = RetrievalContext(project="rust-cli")

    pref = await extractor.classify("I prefer thiserror.", ctx)
    fact = await extractor.classify("This project pins tokio 1.38.", ctx)

    assert pref.type is MemoryType.PREFERENCE and pref.scope.is_global
    assert fact.type is MemoryType.PROJECT_FACT
    assert fact.scope == Scope.project("rust-cli")
