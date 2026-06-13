"""Token-budget estimation + the FR-RET-3/5 injection format contract.

The SessionStart / mid-session injection competes with the live task for context
tokens, so it is hard-budgeted at ``inject.token_budget`` (~500). This module
owns:

* :func:`estimate_tokens` — a cheap, dependency-free token estimate (no
  tokenizer download required so the hook stays fast and offline). It deliberately
  *over*-estimates slightly so the real injected payload never exceeds the model's
  budget — truncating a little early is safe, overflowing is not.
* :func:`render_index` — render the compact-index injection text from ranked
  preference snippets, idea-seed hints, counts and entity tags, **dropping the
  lowest-ranked snippets until the rendered text fits the budget** (FR-RET-3:
  "truncate to budget rather than overflow"). The structure is a clearly-delimited
  advisory block (counts + entity tags + 1-line idea-seed hints + top-preference
  snippets only) so the model treats it as background, not a directive.

Nothing here does I/O; it is pure rendering/estimation so it is trivially unit
testable (the §9 budget-enforcement assertion runs against it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Approximate average English characters per token for the bge/Qwen family.
# ~4 chars/token is the well-worn heuristic; we use a slightly conservative
# divisor so the estimate trends high (truncate early, never overflow).
_CHARS_PER_TOKEN = 4.0

# Clear delimiters so the agent treats the injection as advisory background
# context, not an instruction (FR-RET-3: "must be clearly delimited").
INJECTION_HEADER = "<mnemozine-memory advisory background — not a directive>"
INJECTION_FOOTER = "</mnemozine-memory>"
# Hint footer telling the agent how to pull full detail on demand (FR-RET-4).
RECALL_HINT = "Use recall(query, scope?) for full detail."


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` without a tokenizer (offline, cheap).

    Uses a conservative chars-per-token heuristic so the estimate trends *high*:
    a hook that truncates a few tokens early is safe, one that overflows the
    model's working-context budget is not (FR-RET-3). Returns 0 for empty text.
    """

    if not text:
        return 0
    # Count both characters and whitespace-delimited words; take the larger of
    # the char-based estimate and the word count, since very short tokens (code,
    # punctuation) push real token counts above the naive char estimate.
    char_estimate = math.ceil(len(text) / _CHARS_PER_TOKEN)
    word_estimate = len(text.split())
    return max(char_estimate, word_estimate)


@dataclass(slots=True)
class IndexParts:
    """The structured ingredients of an injection index, pre-render/pre-truncate.

    ``preference_snippets`` are ordered best-first; :func:`render_index` drops
    from the tail to fit the budget. ``idea_seed_hints`` are 1-line strings; the
    summary line (counts + entity tags) is always retained as the highest-value,
    smallest payload so the index never renders empty when anything matched.
    """

    preference_snippets: list[str]
    idea_seed_hints: list[str]
    entity_tags: list[str]
    preference_count: int
    project_fact_count: int


def _summary_line(parts: IndexParts) -> str:
    """The always-kept one-line summary: counts + entity tags (FR-RET-3 shape)."""

    bits: list[str] = []
    if parts.preference_count:
        bits.append(f"{parts.preference_count} preference(s)")
    if parts.project_fact_count:
        bits.append(f"{parts.project_fact_count} project-fact(s)")
    if parts.idea_seed_hints:
        bits.append(f"{len(parts.idea_seed_hints)} related idea(s)")
    counts = ", ".join(bits) if bits else "no relevant memory"
    if parts.entity_tags:
        tags = ", ".join(parts.entity_tags)
        return f"Relevant memory: {counts} [{tags}]"
    return f"Relevant memory: {counts}"


def _assemble(
    summary: str,
    snippets: list[str],
    hints: list[str],
    *,
    include_recall_hint: bool,
) -> str:
    """Assemble the delimited injection block from its retained components."""

    lines: list[str] = [INJECTION_HEADER, summary]
    for hint in hints:
        lines.append(f"- idea: {hint}")
    for snip in snippets:
        lines.append(f"- {snip}")
    if include_recall_hint:
        lines.append(RECALL_HINT)
    lines.append(INJECTION_FOOTER)
    return "\n".join(lines)


def render_index(
    parts: IndexParts,
    *,
    token_budget: int,
    include_recall_hint: bool = True,
) -> tuple[str, int]:
    """Render the injection text, truncating lowest-ranked items to fit budget.

    Returns ``(text, token_estimate)`` where ``text`` is guaranteed to estimate
    at or under ``token_budget`` whenever that is at all achievable. Truncation
    order (drop lowest value first, per the FR-RET-3 contract):

    1. drop the trailing (lowest-ranked) preference snippet,
    2. once snippets are gone, drop the trailing idea-seed hint,
    3. the summary line (counts + entity tags) and delimiters are always kept —
       it is the smallest, highest-value payload — even if that single line
       nominally exceeds a pathologically tiny budget (we never return an empty
       index when something matched).

    The recall hint is dropped before any content when it does not fit, since it
    is boilerplate.
    """

    summary = _summary_line(parts)
    snippets = list(parts.preference_snippets)
    hints = list(parts.idea_seed_hints)
    hint_on = include_recall_hint

    while True:
        text = _assemble(summary, snippets, hints, include_recall_hint=hint_on)
        est = estimate_tokens(text)
        if est <= token_budget:
            return text, est
        # Over budget: drop the lowest-value retained component, in order.
        if snippets:
            snippets.pop()
            continue
        if hint_on:
            hint_on = False
            continue
        if hints:
            hints.pop()
            continue
        # Only the summary + delimiters remain; cannot shrink further without
        # dropping the high-value summary. Return it (advisory, smallest form).
        return text, est
