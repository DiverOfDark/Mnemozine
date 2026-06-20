"""The shared classification rubric for the extraction prompts (FR-EXT-1/3).

This is the single source of truth for *the controlled scope decision*, *the
free-form emergent category*, and *the cross-reference flag* the classifier emits.
Both the chunk-extraction prompt and the single-statement classify prompt embed
this text, so the definition the model sees is identical on both paths — the §9
classifier-accuracy eval (measured on :meth:`Extractor.classify`) therefore
exercises the same rubric the production :meth:`Extractor.extract` path uses
(FR-EXT-3, R1).

Core data-model redesign (the category split)
---------------------------------------------
The old 3-value ``MemoryType`` enum (``preference`` / ``project_fact`` /
``idea_seed``) did two jobs; the classifier now emits THREE separate signals:

1. ``scope`` — the CONTROLLED :class:`~mnemozine.schema.models.ScopeDecision`
   (``global`` vs ``project``). This is the make-or-break R1 decision that drives
   the hierarchical scope + the no-leak rule and STAYS a fixed two-value enum.
2. ``category`` — a FREE-FORM, emergent string (no enum) naming the *semantic
   role* of the memory (e.g. ``preference``, ``decision``, ``gotcha``, ``idea``).
   It is normalized to a lowercased slug and merged by the category-maintenance
   job; the model is encouraged to reuse common categories but is NOT constrained.
3. ``cross_ref`` — a boolean flag preserving the old ``idea_seed`` behavior
   (FR-RET-6): true when the memory is a candidate/idea/brainstorm worth
   surfacing as a serendipitous cross-reference later.

Keeping the rubric here (rather than inlined twice) is deliberate: the PRD flags
the ``global`` vs ``project`` scope decision as the make-or-break call, so its
wording must be edited in exactly one place.
"""

from __future__ import annotations

# The two-value CONTROLLED scope decision (FR-EXT-3). Mirrors
# ``mnemozine.schema.models.ScopeDecision`` values. Scope is decided AT
# EXTRACTION TIME, not retrieval time, and the final hierarchical Scope is
# derived in Python from this decision + the provenance project path — the model
# never supplies a scope string we trust (no-leak enforcement).
ALLOWED_SCOPE_DECISIONS: tuple[str, ...] = ("global", "project")

# A short, non-binding menu of common free-form categories. The classifier MAY
# emit any lowercase slug; these are suggestions so emergent categories converge
# rather than fragment uncontrollably. NOT an enforced enum.
SUGGESTED_CATEGORIES: tuple[str, ...] = (
    "preference",
    "decision",
    "fact",
    "gotcha",
    "convention",
    "idea",
)

TAXONOMY_RUBRIC = """\
You classify durable memories about a single software operator from their AI
chat transcripts. For EACH durable memory you emit THREE separate signals — they
are independent, do not collapse them into one label:

1. scope  (CONTROLLED — exactly one of "global" or "project")
   This is the most important call. It decides where the memory lives and whether
   it can ever leak across projects. The DEFAULT is "project": choose "global"
   ONLY when the memory is a genuinely cross-project truth about the operator.
   - "global"  — RESERVED for genuinely CROSS-PROJECT operator truths that remain
     true in ANY project: their personal preferences, conventions, rules, and
     coding/working styles (tools, libraries, patterns they favor or reject), OR
     a candidate idea/concept/project they floated that is not tied to one repo.
     If you can imagine the SAME statement applying unchanged in an unrelated
     codebase, it is "global".
       e.g. "Prefers thiserror over anyhow for Rust error handling."
            "Likes small, frequently-rebased pull requests."
            "Always runs the formatter before committing."
            "Idea: a CLI that diffs two SQL schemas and emits a migration."
   - "project" — anything SPECIFIC TO THIS CODEBASE/PROJECT that must NOT leak
     into other projects. This is BROAD and is the DEFAULT. It covers, for THIS
     repo only:
       * this repo's CODE and ARCHITECTURE (modules, classes, functions, data
         model, how components fit together);
       * its FILE/MODULE LAYOUT and where things live;
       * its BUILD/RUN process, commands, tooling, and CI for this project;
       * its ENDPOINTS, ports, service names, and external interfaces;
       * PINNED VERSIONS and dependencies chosen for this codebase;
       * BUGS, GOTCHAS, quirks, and workarounds in this repo;
       * DECISIONS made for this codebase (design/trade-off choices specific here).
       e.g. "This project pins tokio 1.38."
            "The auth service runs on port 8081 in this repo."
            "The MCP server exposes recall(query, scope?) for memory lookups."
            "A BASELINE migration moves the schema to version 1 on first run."
            "Embeddings are stored on MnemozineMemory nodes in FalkorDB."
            "Run `make build` before the integration tests in this repo."

   CRITICAL DISAMBIGUATION — global vs project (the make-or-break decision):
     - If the statement is about THIS codebase/project — its code, architecture,
       build, layout, endpoints, pinned versions, bugs, or a decision made for it
       — scope it "project", even if the topic (e.g. "tokio", "migrations") is
       generic. It is the SPECIFICITY TO THIS REPO, not the topic, that decides.
     - Scope "global" ONLY for a cross-project operator preference, convention,
       rule, style, or floated idea that would read the SAME in an unrelated repo.
     - Bidirectional check:
         "The MCP server exposes recall(query, scope?) ..."  -> project
         "A BASELINE migration to version 1 ..."             -> project
         "Prefers thiserror over anyhow"                     -> global
         "Likes small rebased PRs"                           -> global
     - When genuinely torn, prefer "project" + a LOWER confidence: a wrongly-
       global memory leaks across every project (the worst failure mode), whereas
       a wrongly-scoped project memory only fails to propagate.

   NOTE: you do NOT write a scope path. Just say "global" or "project"; the system
   derives the exact hierarchical scope from the project the memory came from.

2. category  (FREE-FORM — a short lowercase slug naming the memory's role)
   Name what KIND of memory this is, independent of its scope. Reuse a common
   slug when one fits — e.g. one of: preference, decision, fact, gotcha,
   convention, idea — but you MAY coin a new short slug if none fits. Use a
   single lowercase word or hyphenated phrase (e.g. "error-handling-style"),
   never a sentence.

3. cross_ref  (boolean — is this a cross-reference / idea seed?)
   Set true when the memory is a candidate PROJECT or CONCEPT the operator
   floated, brainstormed, or wished existed — something to surface later as a
   serendipitous connection (the old "idea_seed" behavior). These are usually
   "global" scope with category "idea". Set false for ordinary preferences and
   facts.
     true  e.g. "Idea: a local-first note app with CRDT sync."
                "Maybe build an LLM-judge harness for prompt regression tests."
     false e.g. "Prefers thiserror over anyhow."
                "This project pins tokio 1.38."

WHAT NOT TO EMIT (return nothing for these):
  - Transient task state, greetings, acknowledgements, or the assistant's own
    chatter ("Understood, I'll use thiserror").
  - One-off questions, debugging steps, or tool/command output.
  - Anything you are not reasonably confident is a DURABLE memory.

ENTITIES: for every memory, extract 1-6 lowercase, hyphenated topic tags that a
future query could match on (e.g. "rust", "error-handling", "async", "cli",
"tokio", "postgres"). Use short canonical names, not sentences.

CONFIDENCE: a float in [0,1] for how sure you are this is a real, durable memory.
Lower it when the statement is ambiguous, weakly stated, or you are torn on scope.
"""
