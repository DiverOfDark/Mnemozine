# Mnemozine WebUI — Backend API Contract

**Status:** Phase-1 foundation (typed stubs; OpenAPI complete). Phase-2 fills the
route bodies against the real backend without changing these signatures.
**Source of truth:** the live OpenAPI at `GET /openapi.json` (Swagger UI at
`/docs`). This document is the human-readable mirror; the pydantic models in
`mnemozine/web/schemas.py` ARE the wire contract.

The API is mounted under `/api`. The SPA is served from `/` (static) by the same
FastAPI app (single image). Auth, CORS, and exposure are described at the end.

---

## Endpoints

Every list endpoint returns a `page` envelope (`{ total, limit, offset }`). All
response/request schema names below resolve to a model in
`mnemozine/web/schemas.py` (Activity events mirror `mnemozine/activity/models.py`).

### Health & stats (Dashboard / top bar — PRD §4.1)

| Method | Path | Query params | Response schema |
|--------|------|--------------|-----------------|
| GET | `/api/health` | — | `HealthResponse` |
| GET | `/api/stats` | — | `StoreStatsResponse` |

`HealthResponse` carries `components: ComponentHealth[]` (falkordb / ollama /
llm) and `activity_log_enabled`. `StoreStatsResponse` carries `by_type`,
`by_tier`, `by_source` count maps + active/superseded/entity counts.

### Memories (table + detail — PRD §4.2 / §4.3)

| Method | Path | Query params | Body | Response schema |
|--------|------|--------------|------|-----------------|
| GET | `/api/memories` | `type, scope, tier, entity, source, active, q, limit, offset` | — | `MemoryListResponse` |
| GET | `/api/memories/{memory_id}` | — | — | `MemoryDetail` |
| PATCH | `/api/memories/{memory_id}` | — | `MemoryPatchRequest` | `MutationResponse` |

- `active` is tri-state: `true` = active only, `false` = superseded only, omitted = both.
- `MemoryListResponse` = `{ items: MemoryListItem[], page: Page }`.
- `MemoryDetail` carries `validity: ValidityWindow`, `provenance: Provenance`,
  and the supersession chain `supersedes` / `superseded_by`
  (`SupersessionLink[]`) — the signature feature, first-class.
- `MemoryPatchRequest` (HITL, PRD §4.3, R1/R5) accepts only `type` (reclassify),
  `scope` (re-scope), `tier` (archive=`archive` / restore=`hot`). **Content is not
  editable** (PRD §7). A patch setting nothing → 422.

### Graph explorer (PRD §4.4)

| Method | Path | Query params | Response schema |
|--------|------|--------------|-----------------|
| GET | `/api/graph` | `scope, entity, entity_type, depth, include_crossrefs, limit` | `GraphResponse` |

`GraphResponse` = `{ nodes: GraphNode[], edges: GraphEdge[], truncated }`.
`GraphEdge` carries Cytoscape-style `source`/`target`, `weight`, `active`, and —
for cross-reference overlays — `is_crossref` + the mandatory human-readable
`reason` (FR-RET-6).

### Recall playground (PRD §4.5)

| Method | Path | Body | Response schema |
|--------|------|------|-----------------|
| POST | `/api/recall` | `RecallRequest` | `RecallResponse` |

`RecallRequest` = `{ query, scope?, top_k, include_index_preview }`.
`RecallResponse` = `{ query, scope?, results: ScoredMemory[], index_preview?:
InjectionIndexPreview }`. `ScoredMemory` = `{ memory: MemoryListItem, score, why? }`.
`InjectionIndexPreview` is the ~500-token SessionStart index that would be
injected (`text`, `token_estimate`, `token_budget`, counts, hints, entity tags) —
the precision-debugging payload (FR-RET-3).

### Cross-references (PRD §4.4 / §4.7)

| Method | Path | Query params | Body | Response schema |
|--------|------|--------------|------|-----------------|
| GET | `/api/crossrefs` | `project, entity, include_suppressed, limit, offset` | — | `CrossRefResponse` |
| POST | `/api/crossrefs/{memory_id}/suppress` | — | `SuppressRequest` | `MutationResponse` |

`CrossRefItem` always carries a non-empty `reason` (FR-RET-6), `shared_entities`,
and `suppressed` + `context_key`. Suppress takes `{ context_key }` (R2).

### Activity / Logs (PRD §4.6)

| Method | Path | Query params | Response schema |
|--------|------|--------------|-----------------|
| GET | `/api/activity` | `kind, source, session_id, project, ref_memory_id, since, until, limit, offset` | `ActivityResponse` |

`kind` is repeatable (`?kind=ingest&kind=maintenance`) over
`ingest | extract_decision | maintenance | injection`. `ActivityEventOut` mirrors
`ActivityEvent`: `{ id, kind, source?, summary, ref_memory_ids[], session_id?,
project?, detail{}, ts }`. **Live data appears only when the persisted activity
log is enabled** (`MNEMOZINE_WEB__ENABLE_ACTIVITY_LOG=1`); otherwise the feed is
empty by design (the default `NullActivityLog`).

### Maintenance / Ops (PRD §4.7)

| Method | Path | Query params | Response schema |
|--------|------|--------------|-----------------|
| GET | `/api/maintenance` | — | `MaintenanceStatusResponse` |
| POST | `/api/maintenance/{job}/run` | — | `MaintenanceRunResponse` |
| GET | `/api/maintenance/merge-candidates` | — | `MergeCandidatesResponse` |

`{job}` is one of `consolidate | entity-resolution | decay | audit |
migrate-index` (unknown → 404). `MaintenanceStatusResponse` carries `cron`,
`scheduler_running`, and `jobs: MaintenanceJobStatus[]` (each with optional
`last_report: MaintenanceReportOut`). `MergeCandidate` is the FR-MNT-4 HITL
entity-resolution review row.

### Eval (PRD §4.8 — incl. F4 browser bootstrap)

| Method | Path | Body | Response schema |
|--------|------|------|-----------------|
| GET | `/api/eval` | — | `EvalSummaryResponse` |
| GET | `/api/eval/bootstrap` | — | `BootstrapCandidatesResponse` |
| POST | `/api/eval/bootstrap/{candidate_id}/label` | `BootstrapLabelRequest` | `BootstrapCandidate` |
| POST | `/api/eval/bootstrap/finish` | — | `EvalSummaryResponse` |

`EvalSummaryResponse` = `{ gold_set, passed, metrics: EvalMetric[], ran_at? }`.
The bootstrap trio closes **F4**: label `keep | drop | unreviewed` (+ optional
`corrected_type`) in the browser, then `finish` folds the kept candidates into the
gold set.

---

## TypeScript-facing shape (for the frontend foundation)

- **Codegen, don't hand-type.** Generate the TS client from `GET /openapi.json`
  (e.g. `openapi-typescript` for types + `openapi-fetch`, or `orval` for a
  TanStack Query client). The schema names above are stable; treat them as the
  type names.
- **Naming maps 1:1.** `MemoryListItem`, `MemoryDetail`, `GraphResponse`,
  `RecallResponse`, etc. become the TS interface names. `*Response` types are the
  top-level envelopes your `useQuery` hooks return; `*Request` types are the
  bodies your `useMutation` hooks send.
- **Enums** (`MemoryType` = `preference|project_fact|idea_seed`, `Tier` =
  `hot|archive`, `ActivityKind`, `WriteDecision`, `ScopeKind`) serialize as their
  string values — model them as TS string-literal unions.
- **Dates** are ISO-8601 strings on the wire (`valid_from`, `valid_to`, `ts`,
  `last_accessed`, `next_run`, …); `valid_to: null` / `last_accessed: null` mean
  active / never. The frontend renders the validity timeline and the
  struck-through superseded state off these (`active` booleans are provided so the
  UI never has to derive them).
- **Pagination.** Every list response has `page: { total, limit, offset }`; lists
  take `limit` / `offset` query params. Default page sizes: memories 50, activity
  100, crossrefs 50.
- **Scope strings** are `"global"` or `"project:<id>"` everywhere (request and
  response). The `scope` filter and `RecallRequest.scope` also accept a bare
  project id as a convenience.
- **Auth.** When a token is configured, send `Authorization: Bearer <token>` on
  every `/api` request (or `?token=` for static fetches). When unset (the default
  localhost bind) no header is needed.
- **Errors.** Standard FastAPI `{ "detail": "<message>" }` with conventional
  status codes (404 unknown id, 422 invalid patch/label, 401 bad token).

---

## Auth, CORS & exposure (Q5)

- **Bind:** `web.host=127.0.0.1`, `web.port=8765` by default. Never bind publicly.
- **Token:** optional static bearer (`MNEMOZINE_WEB__TOKEN`). Unset → API open on
  the bound interface (fine for localhost). Set → every `/api` request must carry
  it (constant-time compared). No multi-user / RBAC (PRD §7).
- **CORS:** locked to `web.cors_origins` (`MNEMOZINE_WEB__CORS_ORIGINS`). Empty
  (default) → no CORS middleware → same-origin only, which is correct for the
  single-image SPA served by this app.
- **Activity log:** off by default (`web.enable_activity_log=false`,
  `NullActivityLog`), so the existing pipeline + its 442 tests are unaffected. The
  WebUI run path turns it on; it then persists into the **same** FalkorDB store
  (never a new source of truth).

## Config (added in `mnemozine/config.py` → `Settings.web: WebSettings`)

| Env var | Default | Meaning |
|---------|---------|---------|
| `MNEMOZINE_WEB__HOST` | `127.0.0.1` | Bind host |
| `MNEMOZINE_WEB__PORT` | `8765` | Bind port |
| `MNEMOZINE_WEB__TOKEN` | _(unset)_ | Optional static bearer token |
| `MNEMOZINE_WEB__CORS_ORIGINS` | `[]` | Allowed CORS origins (JSON list) |
| `MNEMOZINE_WEB__STATIC_DIR` | _(unset)_ | SPA static dir override (else bundled `web/static`) |
| `MNEMOZINE_WEB__ENABLE_ACTIVITY_LOG` | `false` | Persist the activity log (Q3) |

## Run

`mnemozine-web` (console script → `mnemozine.app:run_web`) wires the live
`Container` into `create_app(container)` and serves it with uvicorn on the
configured host/port.
