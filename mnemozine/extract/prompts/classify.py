"""Single-statement classification prompt (FR-EXT-3, the R1 eval path).

Backs :meth:`mnemozine.interfaces.Extractor.classify`: given a bare statement
and a :class:`~mnemozine.interfaces.RetrievalContext`, the model returns the
category-split signals — the CONTROLLED
:class:`~mnemozine.schema.models.ScopeDecision` (``global`` vs ``project``), a
FREE-FORM ``category`` slug, a ``cross_ref`` boolean, topic entities, and a
confidence. This is the path the §9 classifier-accuracy metric is measured on,
so the prompt is deliberately small, deterministic, and embeds the shared
:data:`~mnemozine.extract.prompts.taxonomy.TAXONOMY_RUBRIC`.

The final hierarchical :class:`~mnemozine.schema.models.Scope` is NEVER taken
from the model — it is derived in Python from the returned ``scope`` decision
plus the provenance/context project (no-leak enforcement). The model only emits
the two-value decision.
"""

from __future__ import annotations

from mnemozine.extract.prompts.taxonomy import (
    ALLOWED_SCOPE_DECISIONS,
    SUGGESTED_CATEGORIES,
    TAXONOMY_RUBRIC,
)

CLASSIFY_SYSTEM_PROMPT = f"""\
You are the memory classifier for a personal AI-memory system. You receive one
statement and decide whether it is a durable memory and, if so, classify it.

{TAXONOMY_RUBRIC}

Respond with a SINGLE JSON object (no prose, no code fence) of the form:
  {{"scope": <one of {list(ALLOWED_SCOPE_DECISIONS)}>,
    "category": <a short lowercase slug, e.g. one of {list(SUGGESTED_CATEGORIES)}>,
    "cross_ref": <true|false>,
    "entities": [<lowercase-hyphenated tags>],
    "confidence": <float 0..1>}}

Emit just the two-value "scope" decision ("global" or "project") — never a scope
path; the system derives the exact scope from the project the statement came from.
If the statement is not a durable memory, still pick the single best scope but set
"confidence" low (<= 0.2) so the caller can drop it.
"""

# JSON schema handed to LLMProvider.complete_json so the structured output is
# parseable. `scope` is the CONTROLLED two-value decision enum; `category` is a
# free-form string (NOT constrained to an enum — emergent categories converge via
# the maintenance merge job, not a fixed schema).
CLASSIFY_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "scope": {
            "type": "string",
            "enum": list(ALLOWED_SCOPE_DECISIONS),
            "description": "The controlled scope decision: 'global' or 'project'.",
        },
        "category": {
            "type": "string",
            "description": "Free-form lowercase category slug (emergent, no enum).",
        },
        "cross_ref": {
            "type": "boolean",
            "description": "True if this is a cross-reference seed / idea.",
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "1-6 lowercase, hyphenated topic tags.",
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["scope", "category", "cross_ref", "entities", "confidence"],
    "additionalProperties": False,
}


def build_classify_prompt(
    statement: str,
    *,
    project: str | None = None,
    recent_text: str | None = None,
) -> str:
    """Build the user prompt for single-statement classification (FR-EXT-3).

    ``project`` is the current project id (from
    :attr:`~mnemozine.interfaces.RetrievalContext.project`); it is shown so the
    model can judge whether the statement is specific to THIS project (scope
    "project") or a cross-project truth (scope "global"). The model does not build
    a scope path — Python derives the final hierarchical scope from the returned
    decision + this project. ``recent_text`` is optional surrounding context that
    helps disambiguate global-vs-project without being part of the statement being
    scored.
    """

    lines: list[str] = []
    project_id = project if project else "unknown"
    lines.append(f"Current project id: {project_id}")
    if recent_text:
        # Surrounding context is advisory only — it must not become the thing
        # being classified, so it is clearly labelled.
        lines.append("Surrounding context (advisory, do NOT classify this):")
        lines.append(recent_text.strip())
    lines.append("")
    lines.append("Statement to classify:")
    lines.append(statement.strip())
    lines.append("")
    lines.append(
        'Return the JSON object. Use scope "project" only if the statement is '
        f'specific to project "{project_id}"; otherwise use "global".'
    )
    return "\n".join(lines)
