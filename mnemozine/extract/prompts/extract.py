"""Chunk/episode extraction prompt (FR-EXT-1/2/3/4).

Backs :meth:`mnemozine.interfaces.Extractor.extract`: given a whole chunk of
:class:`~mnemozine.schema.events.IngestEvent`s (one Graphiti episode, FR-ING-6),
the model returns a list of memory objects. Each object carries the category-split
signals — the CONTROLLED ``scope`` decision (``global`` vs ``project``, FR-EXT-3),
a FREE-FORM ``category`` slug (FR-EXT-1), and a ``cross_ref`` flag (FR-RET-6) —
plus entity tags and a confidence (FR-EXT-4). The final hierarchical
:class:`~mnemozine.schema.models.Scope` is derived in Python from the decision +
the chunk's provenance project, never trusted from the model (no-leak).

The relationships the model returns become weighted, temporal
:class:`~mnemozine.schema.models.Edge`s in the graph (FR-EXT-2): a triple of
``(subject_entity, relation, object_entity)``.
"""

from __future__ import annotations

from collections.abc import Sequence

from mnemozine.extract.prompts.taxonomy import (
    ALLOWED_SCOPE_DECISIONS,
    SUGGESTED_CATEGORIES,
    TAXONOMY_RUBRIC,
)
from mnemozine.schema.events import IngestEvent

EXTRACT_SYSTEM_PROMPT = f"""\
You are the memory extractor for a personal AI-memory system. You read one chunk
of a conversation transcript (a session excerpt) and distill the DURABLE
memories worth remembering long-term. Most lines in a transcript are NOT durable
memories — be selective. Emit nothing for a chunk that has no durable memory.

{TAXONOMY_RUBRIC}

RELATIONSHIPS (FR-EXT-2): in addition to per-memory entity tags, list any clear
relationships between entities as triples. Use them sparingly and only when
clearly stated. Each triple is
  {{"subject": <entity>, "relation": <short-verb-phrase>, "object": <entity>}}
e.g. {{"subject": "project-a", "relation": "pins", "object": "tokio"}} or
     {{"subject": "rust", "relation": "uses", "object": "error-handling"}}.

Respond with a SINGLE JSON object (no prose, no code fence):
  {{"memories": [
     {{"content": <one concise sentence stating the memory in the third person>,
       "scope": <one of {list(ALLOWED_SCOPE_DECISIONS)}>,
       "category": <a short lowercase slug, e.g. one of {list(SUGGESTED_CATEGORIES)}>,
       "cross_ref": <true|false>,
       "entities": [<lowercase-hyphenated tags>],
       "confidence": <float 0..1>}}
   ],
   "relationships": [
     {{"subject": <entity>, "relation": <verb>, "object": <entity>}}
   ]}}

Rewrite each memory's content as a standalone third-person statement (e.g.
"Prefers thiserror over anyhow for Rust error handling."), not a quote of the
turn. Emit just the two-value "scope" decision ("global" or "project") — never a
scope path; the system derives the exact scope from the project id given below.
Return {{"memories": [], "relationships": []}} if nothing is durable.
"""

# JSON schema for LLMProvider.complete_json. The model returns memories +
# relationships; the hierarchical scope, provenance and validity are stamped by
# Python afterwards. `scope` is the CONTROLLED two-value decision; `category` is a
# free-form string (not enum-constrained — emergent).
EXTRACT_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "scope": {
                        "type": "string",
                        "enum": list(ALLOWED_SCOPE_DECISIONS),
                    },
                    "category": {"type": "string"},
                    "cross_ref": {"type": "boolean"},
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": [
                    "content",
                    "scope",
                    "category",
                    "cross_ref",
                    "entities",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "relation", "object"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["memories", "relationships"],
    "additionalProperties": False,
}


def render_chunk(events: Sequence[IngestEvent]) -> str:
    """Render a chunk of events as a compact role-tagged transcript for the model.

    Mirrors :meth:`IngestEvent.normalized_content` in spirit (role-prefixed,
    trimmed) but keeps each turn on its own line so the model can attribute a
    memory to who said it. ``tool`` turns are dropped — FR-ING-7 already strips
    ``tool_calls`` upstream, and tool chatter carries little durable memory.
    """

    lines: list[str] = []
    for e in events:
        text = e.content.strip()
        if not text:
            continue
        lines.append(f"{e.role.value}: {text}")
    return "\n".join(lines)


def build_extract_prompt(events: Sequence[IngestEvent], *, project: str) -> str:
    """Build the user prompt for chunk extraction (FR-EXT-1..4).

    ``project`` is the chunk's project id — shown so the model can decide whether
    a memory is specific to THIS project (scope "project") or cross-project (scope
    "global"). The final hierarchical scope is derived in Python from the decision
    + this project at extraction time (FR-EXT-3) so a project memory can never
    leak into another project.
    """

    transcript = render_chunk(events)
    return (
        f"Current project id: {project}\n\n"
        "Conversation chunk:\n"
        f"{transcript}\n\n"
        "Extract the durable memories and relationships as the JSON object. "
        'Use scope "project" only for memories specific to this project '
        f'("{project}"); use "global" otherwise.'
    )
