"""Chunk/episode extraction prompt (FR-EXT-1/2/3/4).

Backs :meth:`mnemozine.interfaces.Extractor.extract`: given a whole chunk of
:class:`~mnemozine.schema.events.IngestEvent`s (one Graphiti episode, FR-ING-6),
the model returns a list of memory objects. Each object is classified into one
:class:`~mnemozine.schema.models.MemoryType` (FR-EXT-1), scoped at extraction
time (FR-EXT-3), entity- and relationship-linked (FR-EXT-2), and carries a
confidence (FR-EXT-4 — provenance is attached by the Python orchestration from
the chunk's source session, not invented by the model).

The relationships the model returns become weighted, temporal
:class:`~mnemozine.schema.models.Edge`s in the graph (FR-EXT-2): a triple of
``(subject_entity, relation, object_entity)``.
"""

from __future__ import annotations

from collections.abc import Sequence

from mnemozine.extract.prompts.taxonomy import ALLOWED_TYPES, TAXONOMY_RUBRIC
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
       "type": <one of {list(ALLOWED_TYPES)}>,
       "scope": "global" | "project:<project_id>",
       "entities": [<lowercase-hyphenated tags>],
       "confidence": <float 0..1>}}
   ],
   "relationships": [
     {{"subject": <entity>, "relation": <verb>, "object": <entity>}}
   ]}}

Rewrite each memory's content as a standalone third-person statement (e.g.
"Prefers thiserror over anyhow for Rust error handling."), not a quote of the
turn. Use scope "global" for preference/idea_seed and
"project:<project_id>" for project_fact, using the project id given below.
Return {{"memories": [], "relationships": []}} if nothing is durable.
"""

# JSON schema for LLMProvider.complete_json. The model returns memories +
# relationships; provenance/validity are stamped by Python afterwards.
EXTRACT_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "type": {"type": "string", "enum": list(ALLOWED_TYPES)},
                    "scope": {"type": "string"},
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
                "required": ["content", "type", "scope", "entities", "confidence"],
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

    ``project`` is the chunk's project id — the scope base for any
    ``project_fact`` extracted, decided here at extraction time (FR-EXT-3) so a
    project fact can never leak into another project.
    """

    transcript = render_chunk(events)
    return (
        f"Current project id: {project}\n\n"
        "Conversation chunk:\n"
        f"{transcript}\n\n"
        "Extract the durable memories and relationships as the JSON object. "
        f'For any project_fact use scope "project:{project}".'
    )
