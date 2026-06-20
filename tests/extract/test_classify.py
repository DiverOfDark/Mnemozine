"""Offline tests for the single-statement classify path (FR-EXT-3, §9, R1).

``Extractor.classify`` is the independently-testable classifier the §9
classifier-accuracy metric is measured on. These tests pin its CATEGORY-SPLIT
contract (core data-model redesign): it returns a lightweight
:class:`Classification` carrying the CONTROLLED ``scope_decision`` (``global`` vs
``project``), the derived hierarchical ``scope``, a FREE-FORM ``category`` slug,
and a ``cross_ref_candidate`` flag — no provenance/validity — deriving scope in
Python from the decision (never the model's string) and degrading gracefully on
bad model output. All offline with ``FakeLLMProvider``.
"""

from __future__ import annotations

from typing import Any

import pytest

from mnemozine.config import Settings
from mnemozine.extract import TypedExtractor
from mnemozine.interfaces import Classification, RetrievalContext
from mnemozine.schema.models import Scope, ScopeDecision
from tests.conftest import FakeLLMProvider


def make_extractor(
    *, json_responder: Any | None = None, json_responses: list[dict[str, Any]] | None = None
) -> tuple[TypedExtractor, FakeLLMProvider]:
    llm = FakeLLMProvider(json_responder=json_responder, json_responses=json_responses)
    return TypedExtractor(llm, settings=Settings()), llm


@pytest.mark.asyncio
async def test_classify_returns_category_split_classification() -> None:
    response = {
        "scope": "global",
        "category": "preference",
        "cross_ref": False,
        "entities": ["rust", "error-handling"],
        "confidence": 0.92,
    }
    extractor, _ = make_extractor(json_responses=[response])
    ctx = RetrievalContext(project="rust-cli")
    result = await extractor.classify("I prefer thiserror over anyhow.", ctx)

    assert isinstance(result, Classification)
    assert result.scope_decision is ScopeDecision.GLOBAL
    assert result.scope.is_global
    assert result.category == "preference"  # FREE-FORM, not an enum
    assert result.cross_ref_candidate is False
    assert result.entities == ["rust", "error-handling"]
    assert result.confidence == 0.92


@pytest.mark.asyncio
async def test_classify_project_decision_scoped_to_context_project() -> None:
    """FR-EXT-3: a project-decision classification is scoped to context.project."""

    response = {
        "scope": "project",
        "category": "decision",
        "cross_ref": False,
        "entities": ["tokio"],
        "confidence": 0.8,
    }
    extractor, _ = make_extractor(json_responses=[response])
    ctx = RetrievalContext(project="rust-cli")
    result = await extractor.classify("This project pins tokio 1.38.", ctx)

    assert result.scope_decision is ScopeDecision.PROJECT
    assert result.scope == Scope.project("rust-cli")


@pytest.mark.asyncio
async def test_classify_cross_ref_idea_is_global() -> None:
    response = {
        "scope": "global",
        "category": "idea",
        "cross_ref": True,
        "entities": ["cli", "sql"],
        "confidence": 0.6,
    }
    extractor, _ = make_extractor(json_responses=[response])
    result = await extractor.classify(
        "Idea: a CLI that diffs SQL schemas.", RetrievalContext(project="x")
    )
    assert result.scope_decision is ScopeDecision.GLOBAL
    assert result.scope.is_global
    assert result.cross_ref_candidate is True
    assert result.category == "idea"


@pytest.mark.asyncio
async def test_classify_never_trusts_model_scope_string() -> None:
    """Even if the model emits a bogus scope path, Python derives from the decision."""

    response = {
        # A well-behaved model emits only the decision, but assert that a stray
        # path-shaped value in the decision field is rejected (not trusted).
        "scope": "project:some-other-project",
        "category": "fact",
        "cross_ref": False,
        "entities": ["a"],
        "confidence": 0.7,
    }
    extractor, _ = make_extractor(json_responses=[response])
    result = await extractor.classify("x", RetrievalContext(project="rust-cli"))
    # Unparseable decision -> droppable global fallback, never the leaked project.
    assert result.confidence == 0.0
    assert result.scope.is_global


@pytest.mark.asyncio
async def test_classify_unparseable_response_is_droppable() -> None:
    """A bad/empty model response yields a confidence-0 result, never a crash."""

    extractor, _ = make_extractor(json_responses=[{}])  # empty object
    result = await extractor.classify("???", RetrievalContext(project="x"))
    assert result.confidence == 0.0  # droppable
    assert result.scope_decision is ScopeDecision.GLOBAL
    assert isinstance(result, Classification)


@pytest.mark.asyncio
async def test_classify_passes_project_and_context_to_prompt() -> None:
    captured: dict[str, str] = {}

    def responder(prompt: str, system: str | None) -> dict[str, Any]:
        captured["prompt"] = prompt
        return {
            "scope": "project",
            "category": "decision",
            "cross_ref": False,
            "entities": ["a"],
            "confidence": 0.7,
        }

    extractor, _ = make_extractor(json_responder=responder)
    ctx = RetrievalContext(project="proj-x", recent_text="we were discussing tokio")
    await extractor.classify("pins tokio 1.38", ctx)

    assert "proj-x" in captured["prompt"]
    assert "tokio" in captured["prompt"]  # recent_text reached the prompt


@pytest.mark.asyncio
async def test_classify_codebase_specific_statement_is_project() -> None:
    """classifier_prompt: a statement about THIS codebase classifies scope='project'.

    The model (fake) keys off the codebase-specific wording the tightened rubric
    asks it to scope "project"; Python then derives project:<context.project>.
    """

    def responder(prompt: str, system: str | None) -> dict[str, Any]:
        # The tightened rubric must reach the model as the system prompt.
        assert system is not None and "project" in system.lower()
        return {
            "scope": "project",
            "category": "fact",
            "cross_ref": False,
            "entities": ["mcp", "recall"],
            "confidence": 0.85,
        }

    extractor, _ = make_extractor(json_responder=responder)
    ctx = RetrievalContext(project="mnemozine")
    result = await extractor.classify(
        "The MCP server exposes recall(query, scope?) for memory lookups.", ctx
    )

    assert result.scope_decision is ScopeDecision.PROJECT
    assert result.scope == Scope.project("mnemozine")


@pytest.mark.asyncio
async def test_classify_cross_project_preference_is_global() -> None:
    """classifier_prompt: a genuinely cross-project preference classifies scope='global'."""

    def responder(prompt: str, system: str | None) -> dict[str, Any]:
        return {
            "scope": "global",
            "category": "preference",
            "cross_ref": False,
            "entities": ["rust", "error-handling"],
            "confidence": 0.9,
        }

    extractor, _ = make_extractor(json_responder=responder)
    ctx = RetrievalContext(project="mnemozine")
    result = await extractor.classify(
        "Prefers thiserror over anyhow for Rust error handling.", ctx
    )

    assert result.scope_decision is ScopeDecision.GLOBAL
    assert result.scope.is_global


@pytest.mark.asyncio
async def test_classify_routing_is_deterministic_by_prompt() -> None:
    """FR-EXT-3/R1: classify is deterministic offline via per-prompt routing."""

    def responder(prompt: str, system: str | None) -> dict[str, Any] | None:
        if "thiserror" in prompt:
            return {
                "scope": "global",
                "category": "preference",
                "cross_ref": False,
                "entities": ["rust"],
                "confidence": 0.9,
            }
        if "tokio 1.38" in prompt:
            return {
                "scope": "project",
                "category": "decision",
                "cross_ref": False,
                "entities": ["tokio"],
                "confidence": 0.8,
            }
        return None

    extractor, _ = make_extractor(json_responder=responder)
    ctx = RetrievalContext(project="rust-cli")

    pref = await extractor.classify("I prefer thiserror.", ctx)
    fact = await extractor.classify("This project pins tokio 1.38.", ctx)

    assert pref.scope_decision is ScopeDecision.GLOBAL and pref.scope.is_global
    assert fact.scope_decision is ScopeDecision.PROJECT
    assert fact.scope == Scope.project("rust-cli")
