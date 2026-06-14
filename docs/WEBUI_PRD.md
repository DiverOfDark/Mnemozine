# PRD: Mnemozine WebUI (Operator Console)

**Status:** Approved — implementing (decisions locked 2026-06-14)
**Owner:** operator (DiverOfDark)
**Implementer:** Claude Code
**Depends on:** the existing Mnemozine memory layer (see `PRD.md`)

---

## 1. Purpose

A **local, single-operator web console** to *observe and steer* the memory layer. Today everything is CLI/MCP-only; there is no way to see what the system actually knows, why it surfaced something, or to correct it. The WebUI makes the memory graph legible and gives the operator human-in-the-loop control.

It must let the operator:
- Browse, search, and filter current memories; inspect any memory's full detail, provenance, validity window, and supersession history.
- Explore the knowledge graph (entities, relations, idea-seeds, cross-references) visually.
- Watch activity: ingestion, extraction decisions (add/reinforce/supersede/no-op), maintenance runs, and what got injected into which session.
- Run `recall()` interactively and preview the SessionStart injection — for debugging precision.
- Make corrections (the R1/R5 mitigations): reclassify, re-scope, archive/restore, suppress a cross-reference.
- Label the **eval bootstrap set in the browser** (closes F4) instead of editing CLI markdown.

## 2. Principles

- **Read-first, write-where-it-matters.** Mostly observation; the only writes are HITL corrections and eval labels. The UI is never a new source of truth — it goes through the existing `StorageBackend`/`Container`.
- **Local-only, never public.** Memories can contain credentials (per the project's threat model). The UI binds to localhost / homelab network and is never exposed publicly.
- **Surface the signature feature.** Temporal validity + supersession is what makes Mnemozine special — show validity windows and superseded chains everywhere, not buried.
- **Honest state.** Always distinguish hot vs archive, active vs superseded, real vs derived.

## 3. Architecture

- **Backend:** a FastAPI app (`mnemozine.web`) exposing a JSON API over the existing composition root (`Container` → `StorageBackend`, maintenance jobs, evals, `recall`). New console script `mnemozine-web`; new docker-compose + Helm service. Mostly read endpoints + a small set of mutations (reclassify, re-scope, archive/restore, suppress, trigger-maintenance, eval-label).
- **Frontend (DECIDED — Q1):** React + Vite + TypeScript + Tailwind SPA, served as static assets by the same FastAPI container (single image). Graph viz via Cytoscape.js; data via TanStack Query.
- **Activity log (DECIDED — Q3):** a lightweight append-only `ActivityEvent` log (ingest / extract-decision / maintenance / injection), written from the existing pipeline seams and queried by the Logs screen + dashboard feed.
- **Auth/exposure:** bind to `127.0.0.1` by default; optional static bearer token (`MNEMOZINE_WEB__TOKEN`); CORS locked. README documents "do not expose publicly."

## 4. Information Architecture (screens)

App shell: left sidebar nav · top bar (global search + scope filter + live store stats) · main content (table/detail/graph).

1. **Dashboard** — totals (memories by type, hot vs archive), store-growth sparkline, source breakdown (claude_code/openai/hermes), recent activity feed, maintenance job status, infra health (FalkorDB / Ollama / LLM endpoint).
2. **Memories** *(core)* — filterable table: type · content · scope · entities · confidence · tier · valid_from/valid_to · last_accessed · access_count. Filters: type, scope, tier, entity, active-vs-superseded, source, date. Row → detail.
3. **Memory detail** — full content; classification with **reclassify** control; scope with **re-scope**; entity chips (→ graph); confidence; **provenance** link to source session/message + raw transcript; **validity-window timeline**; **supersession chain** (replaced / replaced-by); access stats; tier with **archive/restore**.
4. **Graph explorer** — interactive entity/idea-seed graph; edges = relations (weight); click node → its memories + neighborhood; **cross-reference connections highlighted with their human-readable reason**; filter by scope/entity-type; vector-similar neighbors.
5. **Search / Recall playground** — enter a query + scope; see exactly what `recall()` returns (ranked, with scores + why) and a **preview of the ~500-token SessionStart index** that would be injected. The precision-debugging tool.
6. **Activity / Logs** — chronological, filterable feed: ingestion (which session/chunk, which source), extraction (units/entities produced, the 4-way write decision), maintenance (consolidate/decay/entity-resolution/migrate-index), injections (what surfaced where). Entries link to affected memories.
7. **Maintenance / Ops** — scheduler status + last/next runs; trigger jobs on demand (consolidate, decay, entity-resolution, migrate-index); **entity-resolution review** (merge candidates, HITL); **suppression list** management (dismissed cross-refs).
8. **Eval** — view results (precision, classifier accuracy, latency, no-leak, scaling); run the harness; **bootstrap labeling (F4):** present auto-proposed candidates, label `preference` / `project_fact` / `idea_seed` / `not-a-memory` with keyboard shortcuts, save the gold set.

## 5. Visual direction *(DECIDED — Q4)*

**Dark, dense "observability console"** (calm Grafana × Linear). Monospace for IDs/content snippets; color-coded by memory type and tier (hot vivid, archive muted); validity as timelines; superseded as struck-through/greyed. Keyboard-first (search palette, `j/k` nav, label shortcuts). Desktop-primary, responsive.

## 6. API surface (illustrative)

`GET /api/memories` (filters, pagination) · `GET /api/memories/{id}` (detail + provenance + supersession) · `PATCH /api/memories/{id}` (reclassify/re-scope/tier) · `POST /api/recall` · `GET /api/graph` (scoped subgraph) · `GET /api/crossrefs` · `POST /api/crossrefs/{id}/suppress` · `GET /api/activity` · `GET /api/maintenance` + `POST /api/maintenance/{job}/run` · `GET /api/eval` + eval bootstrap propose/label/finish · `GET /api/health`.

## 7. Out of scope (v1)

Multi-user / RBAC · free-form editing of memory content (only classification/scope/tier + suppression) · public/hosted mode · write access to raw transcripts.

## 8. Decisions (locked 2026-06-14)

- **Q1 Frontend stack** → **React + Vite + TypeScript + Tailwind SPA**, served as static by FastAPI.
- **Q2 v1 scope** → **all 8 screens** (full console), built core-first in priority order.
- **Q3 Activity log** → **add a persisted `ActivityEvent` log** written from the pipeline seams.
- **Q4 Visual style** → **dark observability console**.
- **Q5 Auth** → **localhost-bind + optional static bearer token** (`MNEMOZINE_WEB__TOKEN`); never exposed publicly.
