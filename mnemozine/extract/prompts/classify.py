"""Single-statement classification prompt (FR-EXT-3, the R1 eval path).

Backs :meth:`mnemozine.interfaces.Extractor.classify`: given a bare statement
and a :class:`~mnemozine.interfaces.RetrievalContext`, the model returns one
:class:`~mnemozine.schema.models.MemoryType`, the scope it implies, topic
entities, and a confidence. This is the path the §9 classifier-accuracy metric
is measured on, so the prompt is deliberately small, deterministic, and embeds
the shared :data:`~mnemozine.extract.prompts.taxonomy.TAXONOMY_RUBRIC`.
"""

from __future__ import annotations

from mnemozine.extract.prompts.taxonomy import ALLOWED_TYPES, TAXONOMY_RUBRIC

CLASSIFY_SYSTEM_PROMPT = f"""\
You are the memory classifier for a personal AI-memory system. You receive one
statement and decide whether it is a durable memory and, if so, of which type.

{TAXONOMY_RUBRIC}

Respond with a SINGLE JSON object (no prose, no code fence) of the form:
  {{"type": <one of {list(ALLOWED_TYPES)}>,
    "scope": "global" | "project:<project_id>",
    "entities": [<lowercase-hyphenated tags>],
    "confidence": <float 0..1>}}

If the statement is not a durable memory of any type, still pick the single best
type but set "confidence" low (<= 0.2) so the caller can drop it.
"""

# JSON schema handed to LLMProvider.complete_json so the structured output is
# parseable. Kept permissive on `scope` (a string) because the project id is
# dynamic; the type is constrained to the allowed enum.
CLASSIFY_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": list(ALLOWED_TYPES)},
        "scope": {
            "type": "string",
            "description": "'global' or 'project:<project_id>'.",
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "1-6 lowercase, hyphenated topic tags.",
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["type", "scope", "entities", "confidence"],
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
    :attr:`~mnemozine.interfaces.RetrievalContext.project`); the model needs it
    to build a ``project:<id>`` scope for a ``project_fact``. ``recent_text`` is
    optional surrounding context that helps disambiguate
    preference-vs-project_fact without being part of the statement being scored.
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
        "Return the JSON object. For a project_fact use scope "
        f'"project:{project_id}".'
    )
    return "\n".join(lines)
