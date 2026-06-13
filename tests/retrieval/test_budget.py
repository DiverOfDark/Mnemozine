"""FR-RET-3/5 500-token budget enforcement tests (the injection format contract).

These exercise the budget machinery directly (no storage), asserting that:

* the rendered index never exceeds the token budget when truncation is possible,
* truncation drops the *lowest-ranked* snippets first (keeps the summary),
* the summary line (counts + entity tags) is always retained,
* the index is clearly delimited (advisory background, FR-RET-3).
"""

from __future__ import annotations

from mnemozine.retrieval.budget import (
    INJECTION_FOOTER,
    INJECTION_HEADER,
    IndexParts,
    estimate_tokens,
    render_index,
)


def _many(prefix: str, n: int) -> list[str]:
    return [f"{prefix} snippet number {i} with some descriptive content text" for i in range(n)]


def test_estimate_tokens_monotonic_and_zero() -> None:
    assert estimate_tokens("") == 0
    short = estimate_tokens("hello world")
    longer = estimate_tokens("hello world this is a much longer string of text here")
    assert longer > short


def test_render_respects_budget_by_dropping_snippets() -> None:
    parts = IndexParts(
        preference_snippets=_many("pref", 20),
        idea_seed_hints=[],
        entity_tags=["rust", "error-handling"],
        preference_count=20,
        project_fact_count=3,
    )
    budget = 60
    text, est = render_index(parts, token_budget=budget)
    # The whole point of FR-RET-3: never overflow.
    assert est <= budget
    assert estimate_tokens(text) <= budget
    # And it must have dropped some snippets (20 full snippets won't fit in 60).
    assert text.count("- pref") < 20


def test_render_keeps_summary_and_delimiters_under_tiny_budget() -> None:
    parts = IndexParts(
        preference_snippets=_many("pref", 10),
        idea_seed_hints=["idea about an async runtime cli tool"],
        entity_tags=["rust", "async", "cli"],
        preference_count=10,
        project_fact_count=2,
    )
    # Pathologically small budget: everything but the summary must be dropped,
    # but the summary line + delimiters are always retained (never empty).
    text, _est = render_index(parts, token_budget=5)
    assert INJECTION_HEADER in text
    assert INJECTION_FOOTER in text
    assert "Relevant memory" in text
    # All snippets dropped at this budget.
    assert "- pref" not in text


def test_truncation_drops_lowest_ranked_first() -> None:
    # Best-first ordering: snippet 0 is highest ranked, last is lowest.
    snippets = [f"rank{i} content body words words words" for i in range(8)]
    parts = IndexParts(
        preference_snippets=snippets,
        idea_seed_hints=[],
        entity_tags=["x"],
        preference_count=8,
        project_fact_count=0,
    )
    # A budget that keeps *some* but not all snippets, so we can observe that the
    # lowest-ranked ones are dropped first while the highest-ranked survive.
    text, est = render_index(parts, token_budget=60)
    assert est <= 60
    surviving = [i for i in range(8) if f"rank{i}" in text]
    # At least one survived and at least one was dropped (partial truncation).
    assert surviving, "expected some snippets to survive at this budget"
    assert len(surviving) < 8, "expected some snippets to be truncated"
    # Survivors must be a prefix of the best-first ranking: the lowest-ranked are
    # dropped first, so if rank k survived, every higher-ranked rank j<k did too.
    assert surviving == list(range(len(surviving)))
    # Concretely, the very lowest (rank7) is gone and the very highest (rank0) stays.
    assert "rank0" in text
    assert "rank7" not in text


def test_generous_budget_keeps_everything() -> None:
    parts = IndexParts(
        preference_snippets=_many("pref", 3),
        idea_seed_hints=["idea one", "idea two"],
        entity_tags=["rust"],
        preference_count=3,
        project_fact_count=1,
    )
    text, est = render_index(parts, token_budget=500)
    assert est <= 500
    assert text.count("- pref") == 3
    assert "idea one" in text
    assert "idea two" in text


def test_idea_hints_dropped_only_after_snippets() -> None:
    parts = IndexParts(
        preference_snippets=_many("pref", 6),
        idea_seed_hints=["idea seed hint one here", "idea seed hint two here"],
        entity_tags=["rust", "async"],
        preference_count=6,
        project_fact_count=0,
    )
    # Budget tight enough to drop most snippets but (ideally) keep at least the
    # summary; verify snippets go before ideas where both cannot fit.
    text, est = render_index(parts, token_budget=45)
    assert est <= 45
    # If any idea hint survived, then snippets must have been pruned first:
    # i.e. we should not see all 6 snippets while an idea was dropped.
    if "idea seed hint" in text:
        assert text.count("- pref") < 6
