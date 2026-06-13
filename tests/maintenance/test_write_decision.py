"""FR-MNT-1 4-way write decision tests — add / reinforce / SUPERSEDE / no-op.

The supersede branch (closing the old validity window, delivering UC-2 / Goal 2)
is the headline case and is exercised hardest. All offline: the contradiction
LLM call is driven by a scripted ``FakeLLMProvider`` routed by prompt.
"""

from __future__ import annotations

import pytest

from mnemozine.config import Settings
from mnemozine.interfaces import WriteDecision
from mnemozine.maintenance.decision import (
    WriteDecider,
    WriteDecisionConfig,
    build_contradiction_prompt,
    cosine_similarity,
)
from mnemozine.schema.models import MemoryType, MemoryUnit, Provenance, Scope
from tests.conftest import FakeEmbeddingProvider, FakeLLMProvider, InMemoryStorage


def _pref(content: str, *, entities: list[str], confidence: float = 0.9) -> MemoryUnit:
    return MemoryUnit(
        type=MemoryType.PREFERENCE,
        content=content,
        scope=Scope.global_(),
        entities=entities,
        confidence=confidence,
        provenance=Provenance(source="claude_code", session_id="s1"),
    )


def _project_fact(content: str, *, project: str, entities: list[str]) -> MemoryUnit:
    return MemoryUnit(
        type=MemoryType.PROJECT_FACT,
        content=content,
        scope=Scope.project(project),
        entities=entities,
        confidence=0.9,
        provenance=Provenance(source="claude_code", session_id="s1"),
    )


def _never_contradicts(prompt: str, system: str | None) -> dict | None:
    return {"contradicts": False, "reason": "unrelated"}


def _always_contradicts(prompt: str, system: str | None) -> dict | None:
    return {"contradicts": True, "reason": "reversal"}


@pytest.fixture
def settings() -> Settings:
    return Settings()


# --- add ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_when_no_related_memory(settings: Settings) -> None:
    storage = InMemoryStorage()
    llm = FakeLLMProvider(json_responder=_never_contradicts)
    decider = WriteDecider(storage, llm, embeddings=FakeEmbeddingProvider(), settings=settings)

    mem = _pref("Prefers thiserror for Rust errors.", entities=["rust", "error-handling"])
    result = await decider.decide(mem)

    assert result.decision is WriteDecision.ADD
    assert result.memory.id == mem.id
    assert storage.memories[mem.id] is mem
    # No contradiction call needed when there are no candidates.
    assert llm.calls == []


# --- reinforce ------------------------------------------------------------


@pytest.mark.asyncio
async def test_reinforce_on_equivalent_content(settings: Settings) -> None:
    storage = InMemoryStorage()
    llm = FakeLLMProvider(json_responder=_never_contradicts)
    decider = WriteDecider(storage, llm, embeddings=FakeEmbeddingProvider(), settings=settings)

    first = _pref("Prefers thiserror for Rust errors.", entities=["rust"], confidence=0.7)
    await decider.decide(first)

    again = _pref("Prefers thiserror for Rust errors.", entities=["rust"], confidence=0.95)
    result = await decider.decide(again)

    assert result.decision is WriteDecision.REINFORCE
    # Reinforce bumps confidence on the EXISTING unit; no new node.
    assert result.memory.id == first.id
    assert result.memory.confidence == pytest.approx(0.95)
    assert again.id not in storage.memories
    # Reinforce must NOT run the contradiction LLM call.
    assert llm.calls == []


@pytest.mark.asyncio
async def test_reinforce_on_embedding_equivalence(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the embedding-similarity reinforce path (non-identical content).
    storage = InMemoryStorage()
    llm = FakeLLMProvider(json_responder=_never_contradicts)
    embeddings = FakeEmbeddingProvider()
    # Low threshold so the hash-based fake vectors clear it.
    cfg = WriteDecisionConfig(equivalence_threshold=0.0, contradiction_candidate_cap=5)
    decider = WriteDecider(storage, llm, embeddings=embeddings, config=cfg)

    first = _pref("Prefers thiserror.", entities=["rust"], confidence=0.6)
    await decider.decide(first)
    similar = _pref("Likes thiserror best.", entities=["rust"], confidence=0.9)
    result = await decider.decide(similar)

    assert result.decision is WriteDecision.REINFORCE
    assert result.memory.id == first.id
    assert result.memory.confidence == pytest.approx(0.9)


# --- supersede (the headline: closes the old validity window) -------------


@pytest.mark.asyncio
async def test_supersede_closes_old_validity_window(settings: Settings) -> None:
    storage = InMemoryStorage()
    llm = FakeLLMProvider(json_responder=_always_contradicts)
    decider = WriteDecider(storage, llm, embeddings=FakeEmbeddingProvider(), settings=settings)

    old = _pref("Prefers anyhow over thiserror.", entities=["rust", "error-handling"])
    await decider.decide(old)
    assert old.is_active

    new = _pref("Prefers thiserror over anyhow now.", entities=["rust", "error-handling"])
    result = await decider.decide(new)

    assert result.decision is WriteDecision.SUPERSEDE
    # The NEW unit is now the current memory and is active.
    assert result.memory.id == new.id
    assert result.memory.is_active
    assert storage.memories[new.id] is new
    # The OLD unit's validity window is CLOSED (UC-2 / Goal 2) but retained.
    assert result.superseded and result.superseded[0].id == old.id
    assert old.valid_to is not None
    assert not old.is_active
    # It is retained, never hard-deleted.
    assert old.id in storage.memories
    # Exactly one contradiction LLM call (cheap, narrow).
    assert sum(1 for c in llm.calls if c["kind"] == "json") == 1


@pytest.mark.asyncio
async def test_superseded_unit_leaves_hot_query_path(settings: Settings) -> None:
    storage = InMemoryStorage()
    llm = FakeLLMProvider(json_responder=_always_contradicts)
    decider = WriteDecider(storage, llm, embeddings=FakeEmbeddingProvider(), settings=settings)

    old = _pref("Prefers anyhow over thiserror.", entities=["rust", "error-handling"])
    await decider.decide(old)
    new = _pref("Prefers thiserror over anyhow now.", entities=["rust", "error-handling"])
    await decider.decide(new)

    # Closed window => not surfaced on the hot path; only the new value is active.
    hits = await storage.scoped_query("thiserror anyhow", [Scope.global_()], entities=["rust"])
    ids = {h.memory.id for h in hits}
    assert new.id in ids
    assert old.id not in ids


@pytest.mark.asyncio
async def test_supersede_only_considers_preference_candidates(settings: Settings) -> None:
    # A contradicting project_fact must NOT be superseded (FR-MNT-1: pref-level).
    storage = InMemoryStorage()
    llm = FakeLLMProvider(json_responder=_always_contradicts)
    decider = WriteDecider(storage, llm, embeddings=FakeEmbeddingProvider(), settings=settings)

    fact = _project_fact("Project pins tokio 1.38.", project="p", entities=["tokio"])
    await decider.decide(fact)

    new_fact = _project_fact("Project pins tokio 1.40.", project="p", entities=["tokio"])
    result = await decider.decide(new_fact)

    # No preference candidate => no contradiction call => plain add, fact retained.
    assert result.decision is WriteDecision.ADD
    assert fact.is_active
    assert sum(1 for c in llm.calls if c["kind"] == "json") == 0


@pytest.mark.asyncio
async def test_no_contradiction_call_for_non_preference_new(settings: Settings) -> None:
    storage = InMemoryStorage()
    llm = FakeLLMProvider(json_responder=_always_contradicts)
    decider = WriteDecider(storage, llm, embeddings=FakeEmbeddingProvider(), settings=settings)
    # Seed a preference, then write an idea_seed sharing the entity.
    await decider.decide(_pref("Prefers thiserror.", entities=["rust"]))
    seed = MemoryUnit(
        type=MemoryType.IDEA_SEED,
        content="Idea: a rust-based memory CLI.",
        scope=Scope.global_(),
        entities=["rust"],
        confidence=0.8,
        provenance=Provenance(source="claude_code", session_id="s1"),
    )
    result = await decider.decide(seed)
    assert result.decision is WriteDecision.ADD
    # New unit is not a preference => contradiction path is skipped entirely.
    assert sum(1 for c in llm.calls if c["kind"] == "json") == 0


# --- no-op ----------------------------------------------------------------


def test_strictly_weaker_helper_picks_lower_confidence_duplicate() -> None:
    # The no-op target: same type + same normalized content + lower confidence.
    new = _pref("Prefers thiserror.", entities=["rust"], confidence=0.4)
    strong = _pref("PREFERS THISERROR.", entities=["rust"], confidence=0.9)
    target = WriteDecider._strictly_weaker(new, [strong])
    assert target is strong
    # Equal/higher confidence is NOT strictly weaker.
    assert WriteDecider._strictly_weaker(strong, [new]) is None


@pytest.mark.asyncio
async def test_reinforce_takes_precedence_over_no_op(settings: Settings) -> None:
    # An equivalent (exact-content) candidate reinforces; the no-op branch only
    # fires when reinforce did not (e.g. embeddings disabled AND case differs in
    # a way reinforce ignores). Here exact normalized match -> reinforce wins.
    storage = InMemoryStorage()
    llm = FakeLLMProvider(json_responder=_never_contradicts)
    cfg = WriteDecisionConfig(equivalence_threshold=2.0, contradiction_candidate_cap=5)
    decider = WriteDecider(storage, llm, embeddings=None, config=cfg)

    await decider.decide(_pref("Prefers thiserror.", entities=["rust"], confidence=0.9))
    weaker = _pref("prefers thiserror.", entities=["rust"], confidence=0.5)
    result = await decider.decide(weaker)
    assert result.decision is WriteDecision.REINFORCE
    assert weaker.id not in storage.memories


@pytest.mark.asyncio
async def test_candidates_are_scope_bounded(settings: Settings) -> None:
    # A contradicting preference in a DIFFERENT scope must not be superseded.
    storage = InMemoryStorage()
    llm = FakeLLMProvider(json_responder=_always_contradicts)
    decider = WriteDecider(storage, llm, embeddings=FakeEmbeddingProvider(), settings=settings)

    # Project-scoped preference (unusual but valid for the bound test).
    other_scope = MemoryUnit(
        type=MemoryType.PREFERENCE,
        content="Prefers anyhow.",
        scope=Scope.project("other"),
        entities=["rust"],
        confidence=0.9,
        provenance=Provenance(source="claude_code", session_id="s1"),
    )
    await decider.decide(other_scope)

    new_global = _pref("Prefers thiserror.", entities=["rust"])
    result = await decider.decide(new_global)

    # Different scope => not a candidate => add, and no contradiction call fired.
    assert result.decision is WriteDecision.ADD
    assert other_scope.is_active
    assert sum(1 for c in llm.calls if c["kind"] == "json") == 0


# --- supporting pure helpers ---------------------------------------------


def test_cosine_similarity_basics() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_contradiction_prompt_includes_both_and_shared_entities() -> None:
    new = _pref("Prefers thiserror.", entities=["rust", "error-handling"])
    old = _pref("Prefers anyhow.", entities=["rust", "cli"])
    prompt = build_contradiction_prompt(new, old)
    assert "thiserror" in prompt
    assert "anyhow" in prompt
    assert "rust" in prompt  # the shared entity surfaces in the prompt


@pytest.mark.asyncio
async def test_contradiction_candidate_cap_limits_llm_calls() -> None:
    # Many contradicting preferences; cap must bound the number of LLM calls.
    storage = InMemoryStorage()
    call_count = {"n": 0}

    def responder(prompt: str, system: str | None) -> dict | None:
        call_count["n"] += 1
        return {"contradicts": False}  # never supersede so all candidates are tried

    llm = FakeLLMProvider(json_responder=responder)
    cfg = WriteDecisionConfig(equivalence_threshold=2.0, contradiction_candidate_cap=2)
    decider = WriteDecider(storage, llm, embeddings=None, config=cfg)

    for i in range(5):
        await decider.decide(_pref(f"Pref variant {i}.", entities=["rust"]))
    call_count["n"] = 0
    await decider.decide(_pref("A brand new distinct preference.", entities=["rust"]))
    # At most `contradiction_candidate_cap` (2) preference candidates examined.
    assert call_count["n"] <= 2
