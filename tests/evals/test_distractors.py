"""Unit tests for the synthetic distractor generator (PRD §9 synthetic scaling)."""

from __future__ import annotations

from mnemozine.evals.distractors import (
    DEFAULT_INFLATION_LEVELS,
    DistractorGenerator,
    gold_blocklist,
)
from mnemozine.evals.goldset import load_gold_set
from mnemozine.schema.models import MemoryUnit
from tests.conftest import FakeLLMProvider, InMemoryStorage


def test_default_inflation_levels() -> None:
    # PRD §9: 1x / 10x / 100x.
    assert DEFAULT_INFLATION_LEVELS == (1, 10, 100)


async def test_generate_exact_count_deterministic() -> None:
    gold = load_gold_set()
    gen = DistractorGenerator(gold, seed=7)
    a = await gen.generate(50)
    b = await DistractorGenerator(gold, seed=7).generate(50)
    assert len(a) == 50
    assert all(isinstance(u, MemoryUnit) for u in a)
    # Deterministic given the seed: same ids + contents.
    assert [u.id for u in a] == [u.id for u in b]
    assert [u.content for u in a] == [u.content for u in b]


async def test_generated_ids_are_unique() -> None:
    gold = load_gold_set()
    units = await DistractorGenerator(gold, seed=3).generate(200)
    assert len({u.id for u in units}) == len(units)


async def test_distractors_avoid_gold_blocklist() -> None:
    gold = load_gold_set()
    block = gold_blocklist(gold)
    # Sanity: the blocklist contains the gold fixture's key tokens.
    assert "rust" in block
    assert "tokio" in block
    units = await DistractorGenerator(gold, seed=9).generate(300)
    for u in units:
        toks = {e.lower() for e in u.entities}
        if u.scope.project_id:
            toks.add(u.scope.project_id.lower())
        assert not (toks & block), f"distractor {u.content!r} collides with gold"


async def test_inflate_store_adds_multiplier_times_gold() -> None:
    gold = load_gold_set()
    store = InMemoryStorage()
    gen = DistractorGenerator(gold, seed=11)
    inserted = await gen.inflate_store(store, multiplier=10)
    assert inserted == 10 * len(gold.memories)
    # Every distractor must have actually landed in the store (distinct content).
    assert len(store.memories) == inserted


async def test_zero_multiplier_inserts_nothing() -> None:
    gold = load_gold_set()
    store = InMemoryStorage()
    gen = DistractorGenerator(gold, seed=1)
    inserted = await gen.inflate_store(store, multiplier=0)
    assert inserted == 0
    assert store.memories == {}


async def test_llm_backed_path_with_fake_then_template_topup() -> None:
    gold = load_gold_set()
    # FakeLLM returns two clean items; the rest is topped up from templates.
    fake = FakeLLMProvider(
        json_responses=[
            {
                "items": [
                    {
                        "type": "preference",
                        "content": "Prefers a tidy zsh prompt.",
                        "entities": ["zsh", "shell"],
                        "project": None,
                    },
                    {
                        "type": "project_fact",
                        "content": "The blog uses hugo.",
                        "entities": ["hugo", "blog"],
                        "project": "blog",
                    },
                ]
            }
        ]
    )
    gen = DistractorGenerator(gold, llm=fake, seed=5)
    units = await gen.generate(10)
    assert len(units) == 10
    contents = {u.content for u in units}
    assert "Prefers a tidy zsh prompt." in contents


async def test_llm_path_drops_colliding_items() -> None:
    gold = load_gold_set()
    # An LLM item that mentions a gold token (rust) must be filtered out.
    fake = FakeLLMProvider(
        json_responses=[
            {
                "items": [
                    {
                        "type": "preference",
                        "content": "Prefers rust everywhere.",
                        "entities": ["rust"],  # collides with gold blocklist
                        "project": None,
                    }
                ]
            }
        ]
    )
    gen = DistractorGenerator(gold, llm=fake, seed=2)
    units = await gen.generate(3)
    assert all("rust" not in [e.lower() for e in u.entities] for u in units)


async def test_llm_malformed_response_falls_back_to_templates() -> None:
    gold = load_gold_set()
    fake = FakeLLMProvider(json_responses=[{"garbage": True}])
    gen = DistractorGenerator(gold, llm=fake, seed=4)
    units = await gen.generate(8)
    # No crash; fully topped up from the template bank.
    assert len(units) == 8
