# INTERFACES.md — Mnemozine shared contracts

This is the **authoritative contract** every module is built against. Module
agents code to the Protocols in `mnemozine/interfaces.py` and the schema in
`mnemozine/schema/`, **never** to each other's concrete code.

> **The rule:** a module agent only writes inside its own subpackage. It imports
> from `mnemozine.interfaces`, `mnemozine.schema`, and `mnemozine.config`, and
> from nothing else owned by another module. If you need a contract changed,
> request it from the foundation/architect pass — do not fork the shape.

---

## Package layout & who-writes-where

| Path | Owner (layer) | PRD refs |
|------|---------------|----------|
| `mnemozine/config.py` | foundation | §6.6, §5.5 |
| `mnemozine/schema/events.py` | foundation | FR-ING-1, FR-ING-5 |
| `mnemozine/schema/models.py` | foundation | §7 |
| `mnemozine/interfaces.py` | foundation | §5 (all layers) |
| `mnemozine/app.py` | integration pass | §5 wiring, Deliverables |
| `mnemozine/ingestion/**` | **Ingestion agent** | FR-ING-* |
| `mnemozine/extract/**` | **Extraction agent** | FR-EXT-* |
| `mnemozine/storage/**` | **Storage agent** | FR-STO-*, FR-MNT-1 |
| `mnemozine/retrieval/**` | **Retrieval agent** | FR-RET-* |
| `mnemozine/crossref/**` | **Cross-ref agent** | FR-RET-6 |
| `mnemozine/maintenance/**` | **Maintenance agent** | FR-MNT-* |
| `mnemozine/providers/**` | **Providers agent** | EmbeddingProvider, LLMProvider |
| `mnemozine/mcp/**` | **MCP/serving agent** | FR-RET-1, FR-RET-4 |
| `tests/conftest.py` | foundation (shared fakes) | testability |

Each layer subpackage must expose a concrete class implementing the relevant
Protocol(s) below; the integration pass wires them in `mnemozine/app.py`
(`Container.build_*`).

---

## Schema (the nouns)

### `IngestEvent` — FR-ING-1 common event schema (`schema/events.py`)

Fields: `source` (`Source` enum: `claude_code|openai|hermes`), `project`,
`session_id`, `timestamp`, `role` (`Role`: `user|assistant|tool`), `content`,
`tool_calls?`, `metadata`.

Helpers:
- `event.normalized_content()` → role-prefixed, trimmed text used for hashing.
- `event.content_hash()` / `content_hash(str)` → SHA-256 hex (FR-ING-5,
  hash-on-content so resume/rewind is idempotent).
- `event.idempotency_key()` / `idempotency_key(source, session_id, content)` →
  `(source, session_id, content-hash)` (FR-ING-5).
- `chunk_content_hash(events)` → stable hash for a whole chunk/episode (FR-ING-6).

### §7 data model (`schema/models.py`)

- **`MemoryUnit`**: `id, type, content, scope, entities[], confidence,
  provenance, valid_from, valid_to?, tier, last_accessed?, access_count`.
  - `.is_active` → `valid_to is None`.
  - `.supersede(at=None)` → close the validity window in place (FR-MNT-1).
- **`Entity`**: `id, canonical_name, aliases[], type?`.
- **`Edge`**: `id, from_entity, to_entity, relation, weight, valid_from,
  valid_to?` (`.is_active`).
- **`SourceSession`**: `source, session_id, project, started_at?, ended_at?,
  raw_path?`.
- **`Provenance`**: `source, session_id, chunk_hash?, raw_path?` (FR-EXT-4).
  `MemoryUnit.provenance` now **defaults** to the classify sentinel
  (`Provenance.classify_sentinel()` → `source='classify', session_id=''`) so the
  single-statement classifier path can build a valid unit without an ingest
  session; `extract()`/persisted units MUST overwrite it with real provenance.
  `.is_classify_sentinel` flags the placeholder.
- **`Suppression`**: `memory_id, context_key, suppressed_at` — a dismissed
  cross-reference suggestion (FR-RET-6/R2), persisted by the storage backend.

Enums / helpers:
- **`MemoryType`**: `preference | project_fact | idea_seed` (FR-EXT-1).
- **`Tier`**: `hot | archive` (FR-STO-4).
- **`Scope`**: single-operator scopes. Build with `Scope.global_()` or
  `Scope.project("<id>")`. `.as_str()` → `"global"` / `"project:<id>"`;
  `Scope.parse(s)` is the inverse; `.is_global`.

---

## Layer Protocols (the verbs) — `mnemozine/interfaces.py`

I/O-bound methods are `async`. All Protocols are `@runtime_checkable`.

### Shared value objects
- `WriteDecision` — `add | reinforce | supersede | no-op` (FR-MNT-1).
- `WriteResult(decision, memory, superseded[])`.
- `RetrievedMemory(memory, score)`.
- `CrossReference(memory, score, reason, shared_entities[])` — `reason` is
  mandatory (FR-RET-6).
- `Classification(type, scope, entities[], confidence)` — lightweight result of
  `Extractor.classify` (no provenance/validity; FR-EXT-3 eval path, R1).
- `Neighbor(entity, edge)` — an entity-linked neighbor **with the connecting
  edge** (so relation+weight survive traversal; FR-RET-6, FR-MNT-4). Returned by
  `StorageBackend.neighbors`.
- `InjectionIndex(text, token_estimate, preference_count, project_fact_count,
  idea_seed_hints[], entity_tags[])` — `text` is already truncated to budget.
- `RetrievalContext(project?, scopes[], entities[], recent_text?)`.
- `MaintenanceReport(job_name, consolidated, entities_merged, archived,
  edges_pruned, notes[])`.

### `IngestSource` — FR-ING-*
```python
source_name: str  # property
def stream() -> AsyncIterator[IngestEvent]                       # near-real-time tail (FR-ING-2/R4)
def backfill(*, since: SourceSession | None) -> AsyncIterator[IngestEvent]  # backlog (FR-ING-6)
```
Both `stream` and `backfill` are **async generators** — declared `def` (no
`async`) returning `AsyncIterator`, body uses `yield`. Iterate with
`async for e in source.backfill(...)`; do **not** `await` them. Strips
`tool_calls` per FR-ING-7 when `settings.ingest.strip_tool_calls`.

### `Extractor` — FR-EXT-* (the crux)
```python
async def extract(chunk: Sequence[IngestEvent]) -> list[MemoryUnit]
async def classify(statement: str, context: RetrievalContext) -> Classification
```
Each `MemoryUnit` returned by `extract` is classified to exactly one
`MemoryType` (FR-EXT-1), scoped **at extraction time** (FR-EXT-3), entity-linked
(FR-EXT-2), and stamped with confidence + provenance (FR-EXT-4). `classify` is
the independently-testable single-statement path for evals/reclassification; it
returns a lightweight **`Classification`** (no provenance/validity) because a
bare eval statement has no originating session — the §9 classifier-accuracy
metric (R1) is measured on this. (Build a `MemoryUnit` from it + real provenance
to persist.)

### `StorageBackend` — FR-STO-* + FR-MNT-1
```python
async def upsert_memory(memory) -> WriteResult            # 4-way add/reinforce/supersede/no-op
async def scoped_query(query, scopes, *, entities=None, top_k=10, include_archived=False) -> list[RetrievedMemory]
async def close_validity_window(memory_id, *, at=None) -> MemoryUnit
async def archive(memory_id) -> MemoryUnit                # hot -> archive (demote)
async def promote(memory_id) -> MemoryUnit                # archive -> hot (lazy re-embed, OQ3)
async def reembed(memory_id) -> MemoryUnit                # recompute embedding (OQ3 re-embed pass / lazy promote)
async def record_access(memory_id) -> None

# enumeration / scan — the ONLY whole-store iteration (FR-MNT-2/3/4, R5 audit)
def iter_memories(*, scope=None, tier=None, active_only=False,
                  valid_before=None, unused_since=None) -> AsyncIterator[MemoryUnit]
def iter_entities() -> AsyncIterator[Entity]

# entity ops (FR-EXT-2, FR-MNT-4)
async def upsert_entity(entity) -> Entity
async def get_entity(name_or_id) -> Entity | None
async def merge_entities(source_id, target_id) -> Entity
async def neighbors(entity, *, max_degree=None, active_only=True) -> list[Neighbor]  # entity + edge

# edge ops (FR-EXT-2 write, FR-MNT-4 prune, FR-RET-6 traverse)
async def upsert_edge(edge) -> Edge                       # key (from,to,relation); re-assert bumps weight
async def edges_for_entity(entity, *, active_only=True) -> list[Edge]
async def prune_edge(edge_id, *, at=None) -> Edge         # close edge validity window (FR-MNT-4)

# suppression persistence (FR-RET-6 feedback, R2) — backend owns the store
async def record_suppression(memory_id, context_key) -> None
async def is_suppressed(memory_id, context_key) -> bool

async def record_session(session) -> None
async def close() -> None
```
`scoped_query` MUST restrict to the composed scopes + entity neighborhood before
semantic search (FR-RET-2) — never a graph-wide scan. `upsert_memory`'s
contradiction check is a single narrowly-scoped cheap LLM call over
`type=preference` candidates in the same scope sharing ≥1 entity, capped at
`maintenance.contradiction_candidate_cap` (FR-MNT-1).

`iter_memories` / `iter_entities` are **async generators** (the only way to
iterate the whole store — every other read needs a query or start key). They
back FR-MNT-2 consolidation, the FR-MNT-3 decay/archive sweep (use
`unused_since` to find old never-retrieved units; `None` `last_accessed` =
never used), FR-MNT-4 entity resolution + the OQ3 re-embed pass, and the R5
audit. `neighbors` returns `Neighbor(entity, edge)` so traversal keeps the
relation+weight needed for the explainable `reason` (FR-RET-6) and edge pruning
(FR-MNT-4). Suppression is persisted **by the backend**, not by
`CrossReferencer`, so dismissals survive across calls/processes (R2).

### `Retriever` — FR-RET-*
```python
async def scoped_retrieve(query, context, *, top_k=10) -> list[RetrievedMemory]   # FR-RET-2
async def build_index(context, *, token_budget=None) -> InjectionIndex            # FR-RET-3/5
async def recall(query, scope=None, *, top_k=10) -> list[RetrievedMemory]         # FR-RET-4 (UC-4)
```
`build_index` truncates to `settings.inject.token_budget` (500) by dropping
lowest-ranked items — never overflow. Index = counts + entity tags + idea-seed
hints + top-preference snippets only.

**Access recording (FR-MNT-3):** `scoped_retrieve` and `recall` are deliberate
reads and **record access** (`StorageBackend.record_access`, bumping
`access_count`/`last_accessed`). `build_index` does **NOT** — its reads are
passive/automatic (fires on every SessionStart + mid-session), so counting them
would inflate every memory's `access_count` and corrupt decay ranking.

### `CrossReferencer` — FR-RET-6 / UC-3
```python
async def find_related(context, *, max_suggestions=None) -> list[CrossReference]
async def suppress(memory_id, context_key) -> None
```
Graph traversal over shared entities first (explainable, via
`StorageBackend.neighbors` which now yields edges for the `reason`/weight-rank);
vector fallback gated by `crossref.vector_fallback_threshold` (distinct from
`relevance_threshold`). Only above `crossref.relevance_threshold`, capped at
`crossref.max_suggestions`, each with a human-readable `reason`. `suppress`
persists the dismissal **through the storage backend**
(`record_suppression`/`is_suppressed`) — the backend owns the store, so
CrossReferencer takes it as a constructor dep — so suppressed items stop
resurfacing across calls/processes (R2).

### `MaintenanceJob` — FR-MNT-*
```python
name: str  # property
async def run() -> MaintenanceReport
```
Idempotent, safe to re-run (FR-MNT-5). Concrete jobs: consolidation (FR-MNT-2),
entity resolution (FR-MNT-4), decay/archive (FR-MNT-3), audit (R5).

### `EmbeddingProvider` — bge-m3/Ollama (FR-STO-2)
```python
dimensions: int  # property
async def embed(text) -> list[float]
async def embed_batch(texts) -> list[list[float]]
```

### `LLMProvider` — pluggable OpenAI-format LLM (FR-EXT-*, §5.5)
```python
model: str  # property
async def complete(prompt, *, system=None, temperature=0.0, max_tokens=None) -> str
async def complete_json(prompt, *, schema, system=None, temperature=0.0) -> dict
```
Default target local Qwen; cloud is a drop-in swap (OQ5). Backs extraction, the
FR-MNT-1 contradiction check, and consolidation.

---

## Config — `mnemozine/config.py`

`Settings` (env prefix `MNEMOZINE_`, nested `__`). Subsections: `falkordb`,
`extraction`, `embedding`, `inject`, `crossref`, `maintenance`, `ingest`,
`retrieval`, plus `mcp_host/mcp_port`, `log_level`. Every §6.6 tuning parameter
is a field with the PRD's initial value (e.g. `inject.token_budget=500`,
`crossref.relevance_threshold=0.8`, `crossref.max_suggestions=2`). Knobs added
so module code doesn't reach for magic numbers (§6.6 "config, not constants"):
- `crossref.vector_fallback_threshold` (0.75) — FR-RET-6 vector fallback gate,
  distinct from `relevance_threshold`.
- `retrieval.neighborhood_hops` (1) — FR-RET-2 entity-neighborhood traversal
  depth (distinct from `maintenance.max_node_degree` per-node fan-out cap).
- `maintenance.contradiction_candidate_cap` (5) — FR-MNT-1 cap on
  `type=preference` candidates fed to the cheap contradiction LLM call.

Use `get_settings()` for the process-wide cached instance. Full env list:
`.env.example`.

---

## Testing without live infra — `tests/conftest.py`

Fakes that satisfy the Protocols structurally, so any module can unit-test with
no FalkorDB/Ollama/Qwen:

- `InMemoryStorage` — dict-backed `StorageBackend` with a real FR-MNT-1 4-way
  write decision (inject a `contradicts(new, existing)` predicate to drive the
  supersede branch). Now also backs the full contract: `iter_memories` /
  `iter_entities` enumeration, an edge store (`upsert_edge`, `edges_for_entity`,
  `prune_edge`, edge-aware `neighbors` returning `Neighbor`), a suppression set
  (`record_suppression` / `is_suppressed`), and `promote` / `reembed` (the
  latter records `reembed_calls[id]` for assertions).
- `FakeEmbeddingProvider` — deterministic hash-based vectors.
- `FakeLLMProvider` — scriptable: call-order `text_responses` /
  `json_responses` queues **or** per-prompt routing via `text_responder` /
  `json_responder(prompt, system) -> response|None` callables (route by prompt
  so interleaved `extract()` + FR-MNT-1 contradiction calls are deterministic
  offline); records `.calls`.

Fixtures: `settings`, `fake_embeddings`, `fake_llm`, `storage`,
`sample_provenance`, `sample_events` (a Claude Code chunk), `sample_memory`.
