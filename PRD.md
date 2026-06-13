# PRD: Unified Conversational Memory Layer

**Status:** Ready for implementation
**Owner:** (you)
**Implementer:** Claude Code
**Last updated:** 2026-06-13 (rev 2 — stack decisions resolved)

---

## 1. Summary

Build a self-hosted memory layer that ingests conversations from all of my AI tools (Claude Code, OpenAI-based agents, and Hermes), distills them into a temporal knowledge graph, and exposes that memory to every agent through a single MCP server. The system must surface relevant memories proactively at the start of each session and on demand mid-session, so that knowledge and preferences learned in one project/session carry over to others, and so that latent connections between projects and ideas get drawn automatically.

The defining constraint of this system is that it **consolidates rather than accumulates**: as data grows, retrieval precision must stay flat, not degrade.

---

## 2. Goals

1. **Cross-project preference propagation.** A durable preference expressed in one project (e.g. a Rust error-handling preference) is available in any other project on the same topic.
2. **Preferences evolve.** When I change my mind, the system returns the *current* truth, not a pile of contradictory historical statements.
3. **Serendipitous cross-reference.** While I work on one thing, the system can surface a related idea or project I worked on elsewhere, with an explainable reason for the connection.
4. **Single shared memory across all agents.** Claude Code, OpenAI agents, and Hermes all read from and write to the same store via one interface.
5. **Stays manageable at scale.** Retrieval precision and latency remain stable as the corpus grows over months/years.

---

## 3. Non-Goals

- Not a general-purpose data warehouse or analytics engine.
- Not a replacement for version control, documentation, or a project management tool.
- Not a multi-tenant SaaS. **Single-operator only** — scopes are `global` + `project:<id>`, with no `user_id` partitioning. (Team support is explicitly out of scope; revisit only if a team materializes.)
- No cloud-only *infrastructure* dependencies. Storage, retrieval, ingestion, and serving are fully self-hostable. **Exception:** the extraction/embedding LLM endpoints are pluggable via an OpenAI-format base_url and MAY point at a cloud model if the operator chooses on cost grounds — but the system MUST run end-to-end against local models (Qwen for extraction, bge-m3 for embeddings) with no cloud dependency.
- Not building a new agent runtime — this is a memory layer that existing agents call.

---

## 4. Users & Core Use Cases

Single primary user (the operator), interacting through multiple agent surfaces.

| # | Use case | Trigger | Expected behavior |
|---|----------|---------|-------------------|
| UC-1 | Preference carries over | Start a session in a new Rust project | Injected context includes my current Rust preferences learned elsewhere |
| UC-2 | Changed preference respected | I previously preferred lib X, later switched to Y | Retrieval returns Y as current; X is retained as history but not surfaced as active |
| UC-3 | Idea resurfaces | Working on project D, which shares concepts with an earlier idea for project C | System surfaces "project C may be relevant" with the reason (shared entities) |
| UC-4 | On-demand recall | Agent asks "what did we decide about auth?" | `recall()` returns the consolidated decision across all sessions |

---

## 5. Architecture Overview

Five layers, top to bottom:

```
[ Conversation sources ]
  Claude Code (JSONL transcripts)   OpenAI agents   Hermes
            |                            |             |
            v                            v             v
[ 1. Ingestion ]  -- normalize to a common event schema --
            |
            v
[ 2. Typed Extraction ]  -- classify: preference / project-fact / idea-seed --
            |
            v
[ 3. Storage ]  -- Graphiti temporal knowledge graph on FalkorDB (graph + vectors) --
            |
            v
[ 4. Retrieval & Delivery ]  -- MCP server + Claude Code hooks --
            |
            v
[ 5. Maintenance ]  -- dedup, consolidation, decay, entity resolution, evals (scheduled) --
```

**Stack decision:** Graphiti (Zep's open-source temporal knowledge graph engine) is the storage core. It is the only option that is simultaneously self-hostable, graph-native (powers cross-referencing), and temporal (handles changing preferences via validity windows). Graphiti provides the engine; this project builds the ingestion, extraction, MCP serving, and maintenance layers around it.

**Graph backend: FalkorDB.** Graphiti requires a Cypher-speaking graph store; FalkorDB (Redis-based) is chosen over Neo4j for a lighter homelab footprint and a permissive license. FalkorDB holds **both the graph and the vector embeddings** — there is no separate Postgres in the hot path. (Earlier drafts mentioned Postgres; it is dropped to avoid a second source of truth. If a relational sidecar is ever needed for operational metadata, add it explicitly then.)

Suggested implementation language: **Python** (Graphiti is Python-native). MCP server may be Python or TypeScript.

### 5.5 Concrete Stack & Decisions

| Concern | Decision |
|---------|----------|
| Graph + vector backend | **FalkorDB** (single store; no Postgres) |
| Temporal KG engine | **Graphiti** — pin a specific released version in `pyproject.toml`; upgrade deliberately, never floating |
| Extraction LLM | Pluggable **OpenAI-format base_url**. Dev/default target: **local Qwen** (e.g. Qwen2.5) self-hosted. Cloud is a drop-in swap via the same env var, chosen later on projected cost. |
| Embedding model | **bge-m3 via Ollama**, self-hosted. Embeddings are the highest-volume LLM cost (every memory + every query) and stay local for stability and cost control. |
| Ingestion unit | **Chunk/session level**, not per-message (matches Graphiti episodes; avoids per-message LLM storms — see FR-ING-6). |
| Scope model | **Single-user.** Scopes = `global` + `project:<id>`. No `user_id`. |
| Idempotency key | `(source, session_id, content-hash)` — **hash-on-content**, resume-safe. |
| SessionStart injection budget | **~500 token hard cap** — compact index + short snippets for top preferences; full detail via `recall()`. |
| Language | Python (engine + ingestion + maintenance); MCP server Python or TypeScript. |
| Secrets in ingest | `tool_calls` are **stripped on ingest**. Credentials that survive in plain content are acceptable — deployment is a secured, fully-local homelab environment (see FR-ING-7). |

---

## 6. Functional Requirements

### 6.1 Ingestion Layer

**FR-ING-1 — Common event schema.** All sources normalize into one schema before storage:

```json
{
  "source": "claude_code | openai | hermes",
  "project": "string (derived or explicit)",
  "session_id": "string",
  "timestamp": "ISO-8601",
  "role": "user | assistant | tool",
  "content": "string",
  "tool_calls": [ ... optional ... ],
  "metadata": { ... source-specific ... }
}
```

**FR-ING-2 — Claude Code ingestion.** Watch Claude Code transcripts, stored as JSONL at `~/.claude/projects/<project>/<session-id>.jsonl` (one JSON object per line, append-only). The `<project>` path is derived from the working directory and should map to the `project` field.
- Support an override via `CLAUDE_CONFIG_DIR` if transcripts are relocated.
- **Critical:** these local transcripts are deleted after 30 days by default (`cleanupPeriodDays`). The ingester must run on a schedule frequent enough that nothing is lost (recommend: tail in near-real-time via a watcher; do not rely on batch-only runs). Optionally bump `cleanupPeriodDays` as a safety net.
- Prefer a `Stop` / `PreCompact` hook to flush a session to the ingester at end-of-session and before compaction, in addition to the directory watcher.

**FR-ING-3 — OpenAI-format ingestion.** Scope: **agents the operator controls and can repoint** at a memory gateway via an OpenAI-format `base_url`. A thin logging gateway/proxy sits in front of the model endpoint, captures turns, and emits events in the common schema. Explicit non-capability: third-party apps that cannot be reconfigured to use the gateway base_url (e.g. ChatGPT desktop, Cursor) are **not** captured by this path. If such a surface ever needs capturing, it requires a separate adapter (out of current scope).

**FR-ING-4 — Hermes ingestion.** Hermes = the Nous Research Hermes agent (`https://hermes-agent.nousresearch.com/`), self-hosted on a homelab VM. Because it is self-owned, instrument the VM deployment to emit events directly into the common schema (preferred), or front its OpenAI-compatible endpoint with the same FR-ING-3 logging gateway if direct instrumentation is impractical.

**FR-ING-5 — Idempotent ingest.** Re-ingesting the same transcript (e.g. after a crash, or after a Claude Code session resume/rewind that rewrites line offsets) must not create duplicates. Key on **`(source, session_id, content-hash)`** — hash the normalized message content, **not** the byte/line offset, so resumed/edited sessions de-duplicate correctly.

**FR-ING-6 — Chunk-level ingestion unit.** The unit of extraction is a **chunk/session**, not an individual message. Accumulate messages into a chunk (per session, bounded by size or session-end) and submit the chunk as one Graphiti episode. This matches Graphiti's episode model and avoids a per-message LLM storm (see R3). The Claude Code `Stop`/`PreCompact` hooks flush the current chunk at session end and before compaction.

**FR-ING-7 — Secret hygiene on ingest.** Strip `tool_calls` (and tool results) from events before storage — they are the highest-density source of raw credentials, file dumps, and command output, and add little durable memory value. Credentials that remain inside ordinary message content are accepted as-is: the entire deployment (FalkorDB, archive tier, LLM endpoints, MCP server) runs in a secured, fully-local homelab environment with no external egress of memory content. No additional content-level secret scrubbing is required for v1.

### 6.2 Typed Extraction Layer (the crux — highest-care component)

**FR-EXT-1 — Memory classification.** Each extracted memory unit MUST be classified into exactly one type:
- **`preference`** — durable, cross-project fact about how I like to work (→ global scope). E.g. "prefers `thiserror` over `anyhow`".
- **`project_fact`** — specific to one project, must NOT leak across projects (→ project scope). E.g. "project A pins tokio 1.38".
- **`idea_seed`** — a candidate project or concept I floated (→ first-class graph node with its own embedding + extracted entities). Powers cross-referencing.

**FR-EXT-2 — Entity & relationship extraction.** Each memory is linked to extracted entities (e.g. `rust`, `async`, `cli`, `error-handling`) and relationships, written into the graph with timestamps.

**FR-EXT-3 — Scoping is set at extraction time, not retrieval time.** The classifier's accuracy on `preference` vs `project_fact` is the single biggest driver of system quality. It must be independently testable (see evals).

**FR-EXT-4 — Confidence & provenance.** Every memory records a confidence score and a provenance link back to the source session/message.

### 6.3 Storage Layer

**FR-STO-1 — Temporal knowledge graph.** Use Graphiti. Facts are stored with validity windows so superseded facts are marked invalid (closed window) rather than deleted.
**FR-STO-2 — Backend.** **FalkorDB** is the single store — it holds the temporal graph and the vector embeddings (bge-m3) for semantic search over memory units and idea-seeds. No separate relational store in v1.
**FR-STO-3 — Scopes.** Memories are tagged with scope (`global`, `project:<id>`) and entities. Retrieval composes scopes.
**FR-STO-4 — Archive tier.** Superseded/decayed memories and raw transcripts move to a cold archive tier — retained (for history and cross-reference) but excluded from the default hot retrieval path.

### 6.4 Retrieval & Delivery Layer

**FR-RET-1 — Single MCP server.** Expose memory to all agents through one MCP server. Must be callable from Claude Code and from OpenAI/Hermes agents.

**FR-RET-2 — Scoped retrieval.** Never search the whole graph. A query is scoped to: current project + global preferences + entity-linked neighborhood, then semantic search within that subset. Effective search space stays roughly constant regardless of total store size.

**FR-RET-3 — Proactive index injection (SessionStart hook).** On Claude Code `SessionStart`, detect context (cwd, `Cargo.toml`/`package.json`, git remote, recent turns), derive active entities, and inject a **compact index** — not a memory dump. Example shape: "Relevant: 3 preferences (rust/error-handling), 1 possibly-related idea (project C — shares async-runtime, cli-parsing)."

**Injection format contract (applies to FR-RET-3 and FR-RET-5).** Injected memory consumes tokens in the live working context and competes with the actual task, so it is hard-budgeted:
- **Hard cap: ~500 tokens** for the SessionStart injection. The hook MUST truncate to budget (drop lowest-ranked items) rather than overflow.
- Structure: a compact index (counts + entity tags + 1-line idea-seed hints) **plus** short content snippets for the top-ranked preferences only. Anything beyond that is pulled on demand via `recall()`.
- The injection is advisory context for the agent, not a directive; it must be clearly delimited so the model treats it as background.
- Mid-session injections (FR-RET-5) draw from the same 500-token budget envelope and should be finer-grained and even smaller per-injection.

**FR-RET-4 — On-demand recall tool.** Expose a `recall(query, scope?)` MCP tool so an agent can pull full detail when the index hints at something worth chasing. Keeps injected context small.

**FR-RET-5 — Mid-session injection (UserPromptSubmit hook).** As the conversation moves into new topics, inject finer-grained memories relevant to the specific prompt.

**FR-RET-6 — Cross-reference engine (UC-3).** Given current working context, find related `idea_seed`/project nodes via graph traversal over shared entities (preferred — explainable) with a vector-similarity fallback. Surface only above a relevance threshold. Each surfaced connection must carry a human-readable reason. Include a suppression/feedback mechanism so dismissed suggestions stop resurfacing.

### 6.5 Maintenance Layer (scheduled job — as important as ingestion)

**FR-MNT-1 — Dedup, reinforce, and supersede on write (4-way decision).** On each write, compare against existing memories in the **same scope with overlapping entities** and decide one of:
- **add** — no related memory exists → insert.
- **reinforce** — a semantically equivalent memory exists → bump confidence, refresh timestamp, no new node.
- **supersede** — a related memory that *contradicts* the new one exists (e.g. "prefers `anyhow`" → "prefers `thiserror`") → **close the old memory's validity window** (`valid_to = now`, moves it off the hot path) and insert the new one as active. This is the mechanism that delivers UC-2 / Goal 2; the temporal model alone does not detect reversals.
- **no-op** — new memory is strictly weaker/older than what exists.

Contradiction detection is a **single cheap LLM call scoped narrowly** to candidate memories of `type=preference` in the same scope sharing ≥1 entity — never a graph-wide scan. Graphiti's native edge-invalidation handles contradiction at the entity-edge level underneath this; FR-MNT-1 adds the MemoryUnit-level preference-reversal decision on top.

**FR-MNT-2 — Tiered consolidation.** Maintain resolution tiers: raw transcript → extracted fact → consolidated theme. Retrieval operates on distilled tiers. A periodic pass merges related facts into higher-level units.

**FR-MNT-3 — Decay & expiry.** Rank memories by recency + access frequency. Old, never-retrieved memories sink and eventually move to the archive tier. Superseded facts (closed validity window) leave the hot path automatically via the temporal model. **Archive, never hard-delete** by default.

**FR-MNT-4 — Entity resolution.** Periodically merge duplicate entities (`rust` / `rust-lang` / "the Rust work") so the graph does not fragment. Prune low-weight edges and cap node degree to keep traversal bounded.

**FR-MNT-5 — Scheduled execution.** Maintenance runs on a cron-like schedule (consolidate, resolve, decay, audit). Must be idempotent and safe to run repeatedly.

### 6.6 Tuning Parameters (set & calibrate during Phase 1)

These are deliberately unset in the spec — they must be empirically calibrated against the eval set, not guessed. Track them as config, not constants.

| Parameter | Governs | Initial guess |
|-----------|---------|---------------|
| `retrieval.p95_latency_target` | §9 latency SLA | set baseline in Phase 1 |
| `crossref.relevance_threshold` | FR-RET-6 surface/suppress cutoff | start high (precision over recall) |
| `crossref.max_suggestions` | FR-RET-6 cap per context | 1–2 |
| `inject.token_budget` | FR-RET-3/5 injection cap | 500 |
| `maintenance.edge_weight_floor` | FR-MNT-4 low-weight edge pruning | calibrate |
| `maintenance.max_node_degree` | FR-MNT-4 traversal-bound cap | calibrate |
| `decay.half_life` | FR-MNT-3 recency ranking | calibrate |
| `decay.archive_after` | FR-MNT-3 hot→archive demotion | e.g. unused N days |
| `dedup.equivalence_threshold` | FR-MNT-1 reinforce-vs-add | calibrate |
| `chunk.max_size` | FR-ING-6 episode size | calibrate vs Qwen context |

---

## 7. Data Model (illustrative)

- **MemoryUnit**: `id, type(preference|project_fact|idea_seed), content, scope, entities[], confidence, provenance, valid_from, valid_to (nullable), tier(hot|archive), last_accessed, access_count`
- **Entity**: `id, canonical_name, aliases[], type`
- **Edge**: `from_entity, to_entity, relation, weight, valid_from, valid_to`
- **Source/Session**: `source, session_id, project, started_at, ended_at, raw_path`

---

## 8. Phasing

All five goals are in scope from day one — the architecture holds all of them. Phasing is **build order to de-risk**, not scope reduction.

**Phase 0 — Foundation.** Repo scaffolding, Graphiti + Postgres up self-hosted, common event schema, MCP server skeleton with a stub `recall()`.

**Phase 1 — Preference propagation (UC-1, UC-2, UC-4).** Build the full vertical slice for the easiest path: Claude Code ingestion → typed extraction → store → SessionStart index injection → `recall()`. This exercises the entire pipeline end-to-end and validates the extraction classifier. Temporal validity windows deliver UC-2.

**Phase 2 — All sources.** Add OpenAI and Hermes ingestion via logging proxy.

**Phase 3 — Cross-reference (UC-3).** Build the cross-reference engine on top of the now-working graph (it reuses all existing plumbing — it's a ranking/precision layer, not new infra). Add suppression/feedback.

**Phase 4 — Maintenance hardening.** Consolidation, entity resolution, decay, scheduled job, eval harness running on a schedule.

---

## 9. Success Metrics & Evaluation

**Build a fixed eval set early (during Phase 1) and run it on every change and on a schedule thereafter.**

**Eval-set construction (bootstrap — USER-TASK DEPENDENCY).** The eval set encodes *your* preferences across *your* projects, so only the operator can label it. Approach: during the Phase-1 historical backlog import, the pipeline auto-proposes extracted candidates; the operator labels them yes/no in a quick CLI/markdown review pass. Target ~40 high-quality cases, ≈2–3 hrs of operator time. **This is an operator deliverable, not a Claude Code deliverable**, and it gates §9 — schedule it explicitly in Phase 1.

**Testing "precision stays flat at 10×" before a large store exists (synthetic scaling).** Build a **synthetic distractor generator** (Phase-1 deliverable): it inflates the store with LLM-produced plausible-but-irrelevant memories (fake projects, fake cross-domain preferences) at 1×, 10×, 100× volume around the fixed gold set. The precision eval runs at each inflation level and asserts no decline. This makes the headline metric testable in week 1 and doubles as FalkorDB traversal load-testing.

- **Precision (primary).** For a held-out set of "should surface / should not surface" cases, measure precision of injected context. Target: precision does NOT decline as the store grows 10x (verified via the synthetic generator above).
- **Preference correctness.** For changed-preference cases, the current value is returned and the stale value is not surfaced as active.
- **Cross-reference quality.** Of proactively surfaced connections, fraction judged relevant (precision over recall — a wrong "this reminds me of…" is worse than a miss).
- **Classifier accuracy.** Independent accuracy of `preference` vs `project_fact` classification (this gates everything else).
- **Latency.** Retrieval p95 stays under target (set during Phase 1) as the store grows.
- **No-leak check.** `project_fact`s never appear in unrelated projects.

---

## 10. Risks & Open Questions

- **R1 — Extraction classifier quality.** The make-or-break component. Mitigation: dedicated eval, human-in-the-loop correction early, ability to reclassify.
- **R2 — Cross-reference noise.** Low precision makes the feature worse than nothing. Mitigation: high threshold, explainable reasons, suppression of dismissed items.
- **R3 — Per-write LLM cost/latency.** Fact extraction calls an LLM per write; batch the historical backlog import, then go incremental. Use a low-latency model for extraction.
- **R4 — Claude Code 30-day transcript cleanup.** Loss risk if the ingester lags. Mitigation: near-real-time watcher + end-of-session hook flush.
- **R5 — Memory drift / poisoning.** Accumulated subtly-wrong memories can bias agent behavior before any single entry looks wrong. Mitigation: periodic audit in the maintenance job; provenance on every memory.
**Resolved (rev 2):**
- ~~OQ1 — Hermes self-owned vs third-party~~ → **Self-owned**, Nous Hermes on a homelab VM; instrument directly (FR-ING-4).
- ~~OQ2 — single user or small team~~ → **Single-user**, no `user_id` partitioning (§3, §5.5).
- ~~OQ3 — embedding model~~ → **bge-m3 via Ollama** (§5.5). Re-embedding strategy: on a model change, run a full background re-embed pass over the hot tier as a maintenance job; archive tier re-embedded lazily on promotion.

**Still open:**
- **OQ4 — Graphiti version pin.** Confirm the exact released version to pin and that it supports FalkorDB as a backend at that version before Phase 0 `docker-compose up`.
- **OQ5 — Cloud-vs-local extraction LLM (final).** Deferred by design: build against local Qwen; make the cloud-vs-local call once projected per-write cost/latency is measured on real backlog volume.

---

## 11. Deliverables

1. Self-hostable deployment (docker-compose) for FalkorDB + Graphiti + MCP server, with Ollama (bge-m3) and a Qwen/OpenAI-format LLM endpoint as configured services.
2. Ingestion services for Claude Code (watcher + hooks), OpenAI-format logging gateway, Hermes adapter; plus the synthetic distractor generator for eval scaling.
3. Typed extraction pipeline with the classifier.
4. MCP server exposing `recall()` and supporting hook-driven injection.
5. Claude Code hook scripts: `SessionStart`, `UserPromptSubmit`, `Stop`/`PreCompact`.
6. Scheduled maintenance job (consolidation, entity resolution, decay, audit).
7. Eval harness + initial eval set.
8. README covering setup, configuration (env vars incl. `CLAUDE_CONFIG_DIR`, `cleanupPeriodDays`, extraction LLM `base_url`/model, embedding endpoint, FalkorDB connection, tuning-parameter overrides per §6.6), and operations.
