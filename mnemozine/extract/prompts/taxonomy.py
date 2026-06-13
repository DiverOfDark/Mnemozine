"""The shared classification rubric for the extraction prompts (FR-EXT-1/3).

This is the single source of truth for *what each memory type means* and *how
scope is assigned at extraction time*. Both the chunk-extraction prompt and the
single-statement classify prompt embed this text, so the definition the model
sees is identical on both paths — the §9 classifier-accuracy eval (measured on
:meth:`Extractor.classify`) therefore exercises the same rubric the production
:meth:`Extractor.extract` path uses (FR-EXT-3, R1).

Keeping the rubric here (rather than inlined twice) is deliberate: the PRD flags
the ``preference`` vs ``project_fact`` boundary as the make-or-break decision, so
its wording must be edited in exactly one place.
"""

from __future__ import annotations

# The three-way memory taxonomy (FR-EXT-1) and the scope rule (FR-EXT-3).
# Scope is decided AT EXTRACTION TIME, not retrieval time, and is fully
# determined by the type: preference/idea_seed -> global, project_fact ->
# project:<id>. The model is told this so its type choice and scope choice
# cannot diverge.
TAXONOMY_RUBRIC = """\
You classify durable memories about a single software operator from their AI
chat transcripts. Each memory you emit MUST be exactly one of these three types,
and the type fixes the scope:

1. preference  (scope = global)
   A DURABLE, CROSS-PROJECT fact about HOW the operator likes to work: tools,
   libraries, styles, conventions, workflows they favor or reject. It is true
   regardless of which project they are in.
   Examples:
     - "Prefers thiserror over anyhow for Rust error handling."
     - "Likes small, frequently-rebased pull requests."
     - "Always wants type hints on public Python functions."
   A preference answers "what do I, the operator, generally prefer?"

2. project_fact  (scope = project:<project_id>)
   A fact SPECIFIC TO ONE PROJECT that must NOT leak into other projects:
   pinned versions, this repo's layout, this service's endpoints, a decision
   made for this codebase only.
   Examples:
     - "This project pins tokio 1.38."
     - "The auth service runs on port 8081 in this repo."
     - "We decided to use Postgres for project-A's job queue."
   A project_fact answers "what is true about THIS project specifically?"

3. idea_seed  (scope = global)
   A candidate PROJECT or CONCEPT the operator floated, brainstormed, or wished
   existed — not yet a committed project. These become first-class nodes used to
   surface serendipitous cross-references later, so capture the gist.
   Examples:
     - "Idea: a CLI that diffs two SQL schemas and emits a migration."
     - "Thinking about a local-first note app with CRDT sync."
     - "Maybe build an LLM-judge harness for prompt regression tests."
   An idea_seed answers "what might I build or explore?"

CRITICAL DISAMBIGUATION — preference vs project_fact (the most important call):
  - If the statement would still be true and useful in a DIFFERENT project on
    the same topic, it is a preference (global).
  - If the statement is only true because of THIS specific project/repo
    (a pinned version, a local decision, this repo's structure), it is a
    project_fact (project scope) and must be scoped to the current project so it
    never leaks elsewhere.
  - When genuinely torn, prefer project_fact + a LOWER confidence: a wrongly-
    global preference leaks across every project (the worst failure mode),
    whereas a wrongly-scoped project_fact only fails to propagate.

WHAT NOT TO EMIT (return nothing for these):
  - Transient task state, greetings, acknowledgements, or the assistant's own
    chatter ("Understood, I'll use thiserror").
  - One-off questions, debugging steps, or tool/command output.
  - Anything you are not reasonably confident is a DURABLE memory.

SCOPE RULE (decided now, at extraction time — never deferred to retrieval):
  - preference  -> "global"
  - idea_seed   -> "global"
  - project_fact -> "project:<project_id>" using the provided project id.

ENTITIES: for every memory, extract 1-6 lowercase, hyphenated topic tags that a
future query could match on (e.g. "rust", "error-handling", "async", "cli",
"tokio", "postgres"). Use short canonical names, not sentences.

CONFIDENCE: a float in [0,1] for how sure you are this is a real, durable memory
of the chosen type. Lower it when the statement is ambiguous or weakly stated.
"""

# The exact set of allowed type strings, kept next to the rubric so prompt and
# parser agree. Mirrors mnemozine.schema.models.MemoryType values.
ALLOWED_TYPES: tuple[str, ...] = ("preference", "project_fact", "idea_seed")
