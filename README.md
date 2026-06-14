# Mnemozine

A self-hosted **unified conversational memory layer**. Mnemozine ingests
conversations from every AI tool the operator uses (Claude Code, OpenAI-format
agents, Hermes), distills them into a **temporal knowledge graph** (Graphiti on
FalkorDB), and serves that memory to every agent through a single **MCP server**
— proactively at session start and on demand mid-session.

The defining constraint: it **consolidates rather than accumulates** — retrieval
precision stays flat as the store grows, because retrieval is always scoped
(current project + global preferences + entity neighborhood) instead of
searching the whole graph.

See [`PRD.md`](./PRD.md) for the full specification and
[`INTERFACES.md`](./INTERFACES.md) for the shared Protocol contracts every module
builds against.

---

## What it is

| Layer | What it does | Where |
|-------|--------------|-------|
| **Ingestion** | Normalize Claude Code JSONL transcripts, OpenAI-format gateway turns, and Hermes turns into one common event schema; strip `tool_calls`; chunk per session into Graphiti episodes; de-dup on `(source, session_id, content-hash)`. | `mnemozine/ingestion/` |
| **Typed extraction** | Classify each memory unit as `preference` / `project_fact` / `idea_seed`; extract entities + relationships; record confidence + provenance. | `mnemozine/extract/` |
| **Storage** | Graphiti temporal knowledge graph on FalkorDB (graph **and** vector embeddings in one store); validity windows; scopes (`global`, `project:<id>`); hot/archive tiers. | `mnemozine/storage/` |
| **Retrieval & delivery** | One MCP server exposing `recall()` plus session-start / mid-session index tools; scoped retrieval; ~500-token injection budget. | `mnemozine/retrieval/` |
| **Cross-reference** | Surface related `idea_seed`/project nodes via shared-entity graph traversal (vector fallback), with explainable reasons. | `mnemozine/crossref/` |
| **Maintenance** | Scheduled consolidate / entity-resolve / decay / audit; 4-way dedup-reinforce-supersede-noop write decision. | `mnemozine/maintenance/` |
| **Evals** | §9 eval harness + gold-set bootstrap + synthetic distractor generator. | `mnemozine/evals/` |

---

## Architecture

```
[ Conversation sources ]
  Claude Code (JSONL transcripts)   OpenAI-format agents   Hermes
            |                            |                    |
            |                  (LiteLLM gateway + capture callback)
            v                            v                    v
[ 1. Ingestion ]  -- normalize to the common event schema; strip tool_calls --
            |
            v
[ 2. Typed Extraction ]  -- classify preference / project_fact / idea_seed --
            |
            v
[ 3. Storage ]  -- Graphiti temporal KG on FalkorDB (graph + bge-m3 vectors) --
            |
            v
[ 4. Retrieval & Delivery ]  -- single MCP server + Claude Code hooks --
            |
            v
[ 5. Maintenance ]  -- dedup, consolidation, decay, entity resolution (scheduled) --
```

**Stack** (PRD §5.5, pinned in `pyproject.toml`):

| Concern | Choice |
|---------|--------|
| Graph + vector backend | **FalkorDB** (single store; no Postgres) |
| Temporal KG engine | **Graphiti** — `graphiti-core[falkordb]==0.29.2` (exact pin) |
| Extraction LLM | Pluggable **OpenAI-format `base_url`**; default local **Qwen2.5** |
| Embedding model | **bge-m3 via Ollama**, self-hosted (1024-d) |
| OpenAI-format gateway | **LiteLLM** proxy + a custom logging callback |
| MCP server | official `mcp` SDK (`FastMCP`) |
| Maintenance scheduler | APScheduler (or a k8s `CronJob`) |
| Language / packaging | Python ≥3.11, hatchling, `pydantic-settings` config |

The whole system runs **end-to-end on local models with no cloud dependency**.
The extraction/embedding endpoints are pluggable, so the extraction LLM MAY point
at a cloud model later on cost grounds — a one-line `base_url`/`model` swap.

### Console scripts

Installed by the package (`pyproject.toml [project.scripts]`):

| Script | Purpose |
|--------|---------|
| `mnemozine-mcp` | the single MCP server (FR-RET-1) |
| `mnemozine-ingest` | source → chunk → extract → store loop (FR-ING-*) |
| `mnemozine-maintenance` | scheduled consolidate/resolve/decay/audit (FR-MNT-*) |
| `mnemozine-eval` | §9 eval harness + gold-set bootstrap |
| `mnemozine-hook-session-start` | Claude Code `SessionStart` hook (FR-RET-3) |
| `mnemozine-hook-user-prompt-submit` | Claude Code `UserPromptSubmit` hook (FR-RET-5) |
| `mnemozine-hook-stop` | Claude Code `Stop` hook — flush session (FR-ING-6) |
| `mnemozine-hook-pre-compact` | Claude Code `PreCompact` hook — flush before compaction (FR-ING-6) |

The three service workloads (`mnemozine-mcp` / `-ingest` / `-maintenance`) share
**one** container image and differ only in the command they run.

---

## Setup

There are two supported deployment paths, sharing one image definition
(`deploy/Dockerfile`):

- **docker-compose** — local dev / running the eval harness without a cluster.
- **Helm chart** — homelab Kubernetes.

Both are documented in detail in [`deploy/README.md`](./deploy/README.md); the
essentials are below.

### Path A — bare-metal dev (Python only)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env       # then edit endpoints/keys
python -c "import mnemozine; print(mnemozine.__version__)"
pytest
```

This installs the console scripts but assumes you supply FalkorDB, Ollama
(bge-m3), and a Qwen/OpenAI-format endpoint yourself (the `.env` defaults point
at `localhost`). For a turnkey stack, use docker-compose.

### Path B — docker-compose (local full stack + eval)

```bash
# from the repo root
cp .env.example .env                                   # edit endpoints/keys if needed
docker compose -f deploy/docker-compose.yml up -d --build
```

Brings up every service with no cluster:

| Service | Purpose |
|---------|---------|
| `falkordb` | single graph + vector store, persisted to named volume `falkordb-data` (`/data`) |
| `ollama` + `ollama-init` | bge-m3 embeddings; `ollama-init` pulls the model into `ollama-data` on first `up` |
| `qwen` | local OpenAI-format extraction LLM (llama.cpp server by default), weights in `qwen-models` |
| `litellm` | OpenAI-format gateway + logging callback, on `:4000` |
| `mnemozine-mcp` | the MCP server, published on `:8765` |
| `mnemozine-ingest` | Claude Code watcher + hooks; mounts `~/.claude` read-only at `/claude` |
| `mnemozine-maintenance` | scheduled consolidate/resolve/decay/audit |

Inter-service URLs are set under each service's `environment:` (which overrides
`env_file` in Compose), so containers reach each other by service name
(`redis://falkordb:6379`, `http://ollama:11434`, `http://litellm:4000/v1`) even
though `.env` ships `localhost` defaults for bare-metal dev. Override any of them
with the `MZ_COMPOSE_*` interpolation vars, e.g.:

```bash
MZ_COMPOSE_EXTRACTION_URL=https://api.openai.com/v1 \
MZ_COMPOSE_EXTRACTION_MODEL=openai/gpt-4o-mini \
MZ_COMPOSE_EXTRACTION_API_KEY=sk-... \
docker compose -f deploy/docker-compose.yml up -d
```

**Local Qwen model.** The `qwen` service runs a llama.cpp OpenAI-compatible
server; drop a GGUF into the `qwen-models` volume (or bind-mount one) and set
`QWEN_MODEL` to its filename (default `qwen2.5-7b-instruct-q4_k_m.gguf`). To use a
cloud extraction endpoint instead, point the extraction URL at it (above) and the
`qwen` service becomes optional.

**Claude Code transcripts.** `mnemozine-ingest` mounts the host Claude Code
config dir read-only. Override the host path with `HOST_CLAUDE_CONFIG_DIR`
(defaults to `$HOME/.claude`).

**Open the WebUI (operator console).** The dark observability console is a
**separate** local-only FastAPI surface — it is **not** part of
`docker-compose.yml`. Run it from your venv (Path A) once the store is up:

```bash
mnemozine-web        # binds http://127.0.0.1:8765 by default; ⌘-click to open
```

It binds `MNEMOZINE_WEB__HOST` / `MNEMOZINE_WEB__PORT` (`127.0.0.1:8765`) and
serves the bundled SPA from `mnemozine/web/static` if present. The `/api` is
**open on the bound host** unless you set a static bearer token
`MNEMOZINE_WEB__TOKEN`.

> **Port clash:** the WebUI and the MCP server **share default port 8765**. Do
> not run `mnemozine-web` and `mnemozine-mcp` on the same host without moving one
> — set `MNEMOZINE_WEB__PORT` or `MNEMOZINE_MCP_PORT`.

### Path B′ — frontend dev loop (Vite)

When you are iterating on the WebUI itself, run the FastAPI backend
(`mnemozine-web`) on `:8765` and the Vite dev server with hot-reload from `web/`:

```bash
cd web
npm install
npm run dev          # serves the SPA on :5173, proxies /api → http://127.0.0.1:8765
```

Point the dev server at a remote backend by overriding the proxy target:

```bash
MNEMOZINE_API_TARGET=http://my-backend:8765 npm run dev
```

Build the production bundle (emitted into `mnemozine/web/static`, where
`mnemozine-web` serves it from) with:

```bash
npm run build
```

### Path C — Helm (homelab k8s)

```bash
helm lint deploy/helm/mnemozine
helm install mz deploy/helm/mnemozine -n mnemozine --create-namespace
# render without installing:
helm template mz deploy/helm/mnemozine
```

Rendered objects:

- **FalkorDB** — `StatefulSet` + headless `Service` + `volumeClaimTemplate`
  (graph + vector persistence at `/data`).
- **Ollama / Qwen / LiteLLM** — `Deployment` + `Service` (+ PVCs for model
  storage). Ollama pulls bge-m3 via an init container on first start.
- **mcp / ingest / maintenance** — `Deployment`s from the shared image.
  Maintenance can render as a k8s `CronJob` instead (`maintenance.asCronJob=true`).
- **ConfigMap** — all non-secret `MNEMOZINE_*` env, including every §6.6 tuning
  param from `.Values.tuning`; mounted into every workload via `envFrom`.
- **Secret** — FalkorDB password + extraction API key (+ `extraSecrets`).

When a bundled dependency is `enabled`, its in-cluster Service DNS is wired
automatically. To use something you run elsewhere, set `<dep>.enabled=false` and
the matching `endpoints.external.*`:

```bash
helm install mz deploy/helm/mnemozine \
  --set falkordb.enabled=false --set endpoints.external.falkordbUrl=redis://my-falkor:6379 \
  --set ollama.enabled=false   --set endpoints.external.ollamaBaseUrl=http://my-ollama:11434 \
  --set litellm.enabled=false  --set qwen.enabled=false \
  --set endpoints.external.extractionBaseUrl=https://api.openai.com/v1 \
  --set extraSecrets.MNEMOZINE_EXTRACTION__API_KEY=sk-...
```

Reach the MCP server in-cluster at
`http://<release>-mcp.<namespace>.svc:8765`, or port-forward it:

```bash
kubectl -n mnemozine port-forward svc/mz-mnemozine-mcp 8765:8765
```

---

## Configuration (environment variables)

All runtime configuration lives in `mnemozine/config.py` (a
`pydantic-settings` `Settings`) and is overridable via environment variables —
**prefix `MNEMOZINE_`, nested delimiter `__`**. The full, authoritative list is
[`.env.example`](./.env.example). Nothing is a hard-coded constant; in particular
the §6.6 tuning parameters are config so they can be calibrated against the eval
set. Setting `get_settings()` is cached process-wide.

### FalkorDB connection (FR-STO-2)

| Variable | Default | Meaning |
|----------|---------|---------|
| `MNEMOZINE_FALKORDB__URL` | `redis://localhost:6379` | FalkorDB (Redis protocol) connection URL |
| `MNEMOZINE_FALKORDB__GRAPH_NAME` | `mnemozine` | Graphiti graph/keyspace name |
| `MNEMOZINE_FALKORDB__PASSWORD` | _(unset)_ | optional FalkorDB/Redis password |

### Extraction LLM — pluggable OpenAI-format `base_url`, default local Qwen (§5.5)

| Variable | Default | Meaning |
|----------|---------|---------|
| `MNEMOZINE_EXTRACTION__BASE_URL` | `http://localhost:8000/v1` | OpenAI-format base URL (local Qwen by default; swap to a cloud `/v1` to use cloud) |
| `MNEMOZINE_EXTRACTION__MODEL` | `openai/qwen2.5` | LiteLLM `provider/model` id |
| `MNEMOZINE_EXTRACTION__API_KEY` | `not-needed` | API key (local servers ignore it) |
| `MNEMOZINE_EXTRACTION__TEMPERATURE` | `0.0` | extraction wants determinism |
| `MNEMOZINE_EXTRACTION__TIMEOUT_S` | `120` | per-request timeout (s) |

### Embedding endpoint — bge-m3 via Ollama (OQ3)

| Variable | Default | Meaning |
|----------|---------|---------|
| `MNEMOZINE_EMBEDDING__BASE_URL` | `http://localhost:11434` | Ollama base URL |
| `MNEMOZINE_EMBEDDING__MODEL` | `bge-m3` | Ollama embedding model |
| `MNEMOZINE_EMBEDDING__DIMENSIONS` | `1024` | vector dimensionality (bge-m3 is 1024-d) |
| `MNEMOZINE_EMBEDDING__TIMEOUT_S` | `60` | per-request timeout (s) |

### Claude Code ingestion — `CLAUDE_CONFIG_DIR` / `cleanupPeriodDays` (FR-ING-2/R4)

| Variable | Default | Meaning |
|----------|---------|---------|
| `MNEMOZINE_INGEST__CLAUDE_CONFIG_DIR` | `~/.claude` | root of Claude Code config/transcripts (the **`CLAUDE_CONFIG_DIR`** override) |
| `MNEMOZINE_INGEST__CLEANUP_PERIOD_DAYS` | `30` | Claude Code's local-transcript retention (**`cleanupPeriodDays`**) before cleanup |
| `MNEMOZINE_INGEST__STRIP_TOOL_CALLS` | `true` | strip `tool_calls`/tool results on ingest (FR-ING-7) |
| `MNEMOZINE_INGEST__CHUNK_MAX_CHARS` | `8000` | §6.6 `chunk.max_size` (chars) per episode |
| `MNEMOZINE_INGEST__CHUNK_MAX_MESSAGES` | `40` | §6.6 `chunk.max_size` (messages) per episode |

> **Note on `cleanupPeriodDays`:** Claude Code deletes local transcripts after
> `cleanupPeriodDays` (default 30). The ingester runs as a near-real-time watcher
> plus `Stop`/`PreCompact` hooks so nothing is lost before deletion; you may also
> raise Claude Code's own `cleanupPeriodDays` as a safety net. The mnemozine
> setting here records that retention window for the ingest layer.

### MCP server (FR-RET-1)

| Variable | Default | Meaning |
|----------|---------|---------|
| `MNEMOZINE_MCP_HOST` | `127.0.0.1` | MCP bind host (compose/Helm set `0.0.0.0`) |
| `MNEMOZINE_MCP_PORT` | `8765` | MCP bind port |
| `MNEMOZINE_LOG_LEVEL` | `INFO` | logging level |

### §6.6 tuning parameters (config, not constants)

These are deliberately calibrated against the eval set, not guessed. Initial
values match the PRD's initial guesses.

**Injection budget (FR-RET-3 / FR-RET-5)**

| Variable | Default | §6.6 |
|----------|---------|------|
| `MNEMOZINE_INJECT__TOKEN_BUDGET` | `500` | `inject.token_budget` — hard cap; truncate, never overflow |
| `MNEMOZINE_INJECT__MAX_PREFERENCE_SNIPPETS` | `5` | max top-preference snippets in the index |

**Cross-reference engine (FR-RET-6)**

| Variable | Default | §6.6 |
|----------|---------|------|
| `MNEMOZINE_CROSSREF__RELEVANCE_THRESHOLD` | `0.8` | `crossref.relevance_threshold` — start high (precision over recall) |
| `MNEMOZINE_CROSSREF__MAX_SUGGESTIONS` | `2` | `crossref.max_suggestions` (1–2) |
| `MNEMOZINE_CROSSREF__VECTOR_FALLBACK_THRESHOLD` | `0.75` | min cosine sim for the FR-RET-6 vector fallback (distinct from the surfacing threshold) |

**Maintenance / dedup / decay (FR-MNT-*)**

| Variable | Default | §6.6 |
|----------|---------|------|
| `MNEMOZINE_MAINTENANCE__DEDUP_EQUIVALENCE_THRESHOLD` | `0.9` | `dedup.equivalence_threshold` — reinforce-vs-add |
| `MNEMOZINE_MAINTENANCE__EDGE_WEIGHT_FLOOR` | `0.1` | `maintenance.edge_weight_floor` — low-weight edge pruning |
| `MNEMOZINE_MAINTENANCE__MAX_NODE_DEGREE` | `64` | `maintenance.max_node_degree` — traversal-bound cap |
| `MNEMOZINE_MAINTENANCE__CONTRADICTION_CANDIDATE_CAP` | `5` | FR-MNT-1 supersede-LLM candidate cap |
| `MNEMOZINE_MAINTENANCE__DECAY_HALF_LIFE_DAYS` | `30` | `decay.half_life` (days) |
| `MNEMOZINE_MAINTENANCE__DECAY_ARCHIVE_AFTER_DAYS` | `90` | `decay.archive_after` — hot→archive demotion (days unused) |
| `MNEMOZINE_MAINTENANCE__CRON` | `0 3 * * *` | scheduled maintenance cadence (FR-MNT-5) |

**Retrieval (FR-RET-2)**

| Variable | Default | §6.6 |
|----------|---------|------|
| `MNEMOZINE_RETRIEVAL__P95_LATENCY_TARGET_MS` | `500` | `retrieval.p95_latency_target` — baseline set in Phase 1 |
| `MNEMOZINE_RETRIEVAL__TOP_K` | `10` | default results per scoped query |
| `MNEMOZINE_RETRIEVAL__NEIGHBORHOOD_HOPS` | `1` | FR-RET-2 entity-neighborhood traversal depth |

In Helm these same knobs live under `.Values.tuning` (camelCase) and render into
the ConfigMap, e.g.:

```bash
helm upgrade mz deploy/helm/mnemozine \
  --set tuning.crossref.relevanceThreshold=0.85 \
  --set tuning.inject.tokenBudget=400 \
  --set tuning.maintenance.cron='0 4 * * *'
```

---

## Registering the Claude Code hooks

Claude Code invokes a hook as a subprocess, passing a JSON payload on **stdin**
and reading the hook's response (JSON `hookSpecificOutput`) from **stdout**. The
four hook entrypoints are installed as console scripts by the package:

| Hook event | Script | Does |
|------------|--------|------|
| `SessionStart` | `mnemozine-hook-session-start` | inject the compact, ~500-token memory index (FR-RET-3) |
| `UserPromptSubmit` | `mnemozine-hook-user-prompt-submit` | inject finer-grained prompt-scoped memory mid-session (FR-RET-5) |
| `Stop` | `mnemozine-hook-stop` | flush the session's chunk into ingestion at session end (FR-ING-6) |
| `PreCompact` | `mnemozine-hook-pre-compact` | flush the chunk before compaction (FR-ING-6) |

Register all four in Claude Code's `settings.json` `hooks` block. Each entry is
a `command`-type hook; the four entrypoints read the hook JSON from **stdin** and
take **no command-line arguments**, so the `command` is just the path to the
installed console script (no flags). `SessionStart` / `UserPromptSubmit` /
`Stop` / `PreCompact` are not tool-matched events, so no `matcher` is needed.

Drop this into `~/.claude/settings.json` (user-global) or a project's
`.claude/settings.json`. Use the **absolute path** to the installed scripts —
i.e. the path that `which mnemozine-hook-session-start` prints inside the
environment where you ran `pip install -e .` (typically `…/.venv/bin/…`):

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          { "type": "command", "command": "/abs/path/to/.venv/bin/mnemozine-hook-session-start" }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          { "type": "command", "command": "/abs/path/to/.venv/bin/mnemozine-hook-user-prompt-submit" }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          { "type": "command", "command": "/abs/path/to/.venv/bin/mnemozine-hook-stop" }
        ]
      }
    ],
    "PreCompact": [
      {
        "hooks": [
          { "type": "command", "command": "/abs/path/to/.venv/bin/mnemozine-hook-pre-compact" }
        ]
      }
    ]
  }
}
```

If the scripts are on `PATH` for the shell Claude Code spawns hooks in, you may
use the bare names (`"command": "mnemozine-hook-session-start"`), but an absolute
path is the robust default since the hook subprocess does not inherit your
interactive shell's activated venv. Resolve the four absolute paths at once with:

```bash
for h in session-start user-prompt-submit stop pre-compact; do
  command -v "mnemozine-hook-$h"
done
```

Notes:

- The hooks are **fail-safe**: an empty/invalid payload, an unwired backend, or
  any internal error yields an empty injection (or no-op flush) rather than
  raising — a hook must never break the session.
- Injected memory is wrapped in `<mnemozine-memory>…</mnemozine-memory>`
  delimiters so the model treats it as advisory background, and is truncated to
  `inject.token_budget` (~500 tokens).
- The hooks call into the same wired retriever + ingest service the
  `mnemozine-ingest` process owns; running that daemon installs the loader the
  hooks use. The `Stop`/`PreCompact` flush is idempotent — flushing a session the
  watcher already tailed is a no-op (de-dup on the FR-ING-5 content hash).

### Registering the MCP server (the `recall` tool)

The hooks give Claude Code memory **proactively** (session start + prompt
submit). To let the model also pull memory **on demand** mid-session, register
the same `mnemozine-mcp` server with Claude Code. It exposes `recall(query,
scope=None, top_k=10)` plus the two index tools.

For a local Claude Code, run the MCP server over **stdio** (it speaks stdio by
default). Add it with the CLI:

```bash
claude mcp add --transport stdio mnemozine -- mnemozine-mcp
```

…or declare it by hand in `~/.claude.json` (user scope) / `.mcp.json` (project
scope). Use the **absolute path** to the installed script and point it at the
**same FalkorDB the hooks write to**:

```json
{
  "mcpServers": {
    "mnemozine": {
      "command": "/abs/path/to/.venv/bin/mnemozine-mcp",
      "args": [],
      "env": {
        "MNEMOZINE_FALKORDB__URL": "redis://localhost:6379"
      }
    }
  }
}
```

If you are instead running the server over the network with a networked
transport — `mnemozine-mcp --transport streamable-http` (or `sse`), bound to
`MNEMOZINE_MCP_HOST` / `MNEMOZINE_MCP_PORT` (e.g. the compose `mnemozine-mcp`
service publishes `:8765`; note its bundled command runs the default `stdio`
transport, so add the flag to expose HTTP) — register it as an HTTP server
instead of spawning a fresh stdio process:

```bash
claude mcp add --transport http mnemozine http://localhost:8765
```

> **Same store, both ways.** The hooks and the MCP server **must read the same
> `MNEMOZINE_FALKORDB__URL`** — if hooks write to one FalkorDB and the MCP server
> reads from another, memory will not flow. Keep both pulling the URL from the
> same `.env` or environment.

---

## Pointing OpenAI-format agents and Hermes at the gateway

Capture happens through the **LiteLLM** OpenAI-format gateway with a registered
logging callback. The reference proxy config is
[`mnemozine/ingestion/gateway/config.yaml`](./mnemozine/ingestion/gateway/config.yaml)
(docker-compose uses [`deploy/litellm.config.yaml`](./deploy/litellm.config.yaml)).

> **Phase-2, default-off.** Both the gateway (FR-ING-3) and Hermes (FR-ING-4)
> sources are **off by default** — a fresh install only runs the Claude Code
> watcher. Turn them on with the `MNEMOZINE_INGEST__ENABLE_*` flags below; the
> ingest loop (`build_ingest_sources()` in `mnemozine/ingestion/loop.py`) reads
> them and fans every enabled source into one serialized consumer. The gateway
> callback uses an **in-process** `asyncio.Queue`, so it must run in the same
> process as `mnemozine-ingest`.

### OpenAI-format agents (FR-ING-3)

1. Enable the gateway source on the `mnemozine-ingest` process:

   ```bash
   MNEMOZINE_INGEST__ENABLE_GATEWAY=true
   MNEMOZINE_INGEST__GATEWAY_DEFAULT_PROJECT=my-project   # fallback project
   MNEMOZINE_INGEST__GATEWAY_QUEUE_MAX=10000              # in-process buffer
   ```

2. Run the gateway:

   ```bash
   litellm --config mnemozine/ingestion/gateway/config.yaml --port 4000
   ```

   (docker-compose / Helm run the `litellm` service for you.) The callback is
   registered in `litellm_settings.callbacks` as the dotted path
   `mnemozine.ingestion.gateway.litellm_register.gateway_callback` (LiteLLM
   resolves it by **string lookup** at runtime — the path must match exactly).
   The proxy's own upstream models come from the yaml
   (`os.environ/MNEMOZINE_GATEWAY_QWEN_BASE_URL`, `…_QWEN_API_KEY`):

   ```yaml
   model_list:
     - model_name: qwen
       litellm_params:
         model: openai/qwen2.5
         api_base: os.environ/MNEMOZINE_GATEWAY_QWEN_BASE_URL
         api_key: os.environ/MNEMOZINE_GATEWAY_QWEN_API_KEY
   litellm_settings:
     callbacks: mnemozine.ingestion.gateway.litellm_register.gateway_callback
   ```

3. Point any **operator-controlled, repointable** OpenAI-format agent at the
   gateway by setting its OpenAI `base_url` to `http://<gateway-host>:4000/v1`
   (port 4000 is the LiteLLM default; `--port` overrides it) and any `api_key`
   the proxy expects. Every completion that agent makes is then captured and
   emitted as common-schema events (`source=openai`), with `tool_calls` stripped
   (FR-ING-7).

   To route a turn to a specific project/session, thread it through LiteLLM's
   **metadata** dict — there is no request-path routing otherwise:

   ```python
   metadata={"mnemozine_project": "my-project", "mnemozine_session_id": "sess-123"}
   ```

   The callback resolves `project` from `mnemozine_project` → `project` → the
   configured default, and `session_id` from `mnemozine_session_id` →
   `session_id` → `user` → the LiteLLM call id.

4. The gateway's own upstream (the model it proxies to) is the local Qwen by
   default; swap to a cloud backend by editing the `model_list` `api_base`/
   `api_key` (a single line) — capture still works.

> **Explicit non-capability (FR-ING-3):** third-party apps that cannot be
> repointed at the gateway `base_url` (ChatGPT desktop, Cursor, …) are **not**
> captured by this path.

### Hermes (FR-ING-4)

Hermes is the self-hosted Nous Research Hermes agent on a homelab VM. Two paths:

- **Preferred — direct instrumentation.** Enable the Hermes source:

  ```bash
  MNEMOZINE_INGEST__ENABLE_HERMES=true
  MNEMOZINE_INGEST__HERMES_DEFAULT_PROJECT=hermes        # fallback project
  MNEMOZINE_INGEST__HERMES_QUEUE_MAX=10000               # in-process buffer
  ```

  Then instrument the VM to push each completed turn into the
  `HermesAdapter` (`mnemozine.ingestion.hermes.HermesAdapter`, an
  `IngestSource`), which normalizes Hermes-native payloads into the common schema
  (`source=hermes`), stripping `tool_calls`:

  ```python
  hermes.feed(payload)          # sync, returns the emitted IngestEvent list
  await hermes.afeed(payload)   # async, awaits queue space
  ```

  The adapter is field-name tolerant — `conversation_id` / `session_id` / `id`
  for the session, `messages` / `turns` for the turn list, `content` / `text`
  for text, `timestamp` / `created_at` for time. Recorded turns replay via
  `backfill` for the Phase-1 historical import.

- **Fallback — front it with a gateway.** If direct instrumentation is
  impractical, enable the gateway source and run a **second** LiteLLM proxy whose
  upstream `api_base` is Hermes' OpenAI-compatible endpoint and whose callback
  references `mnemozine.ingestion.gateway.litellm_register.hermes_gateway_callback`
  (note: **not** `gateway_callback` — that stamps `source=openai`):

  ```bash
  MNEMOZINE_INGEST__ENABLE_GATEWAY=true
  MNEMOZINE_INGEST__HERMES_BASE_URL=https://hermes-agent.nousresearch.com/
  MNEMOZINE_INGEST__HERMES_API_KEY=<api-key-if-needed>
  ```

  ```yaml
  model_list:
    - model_name: hermes
      litellm_params:
        model: openai/hermes
        api_base: https://hermes-agent.nousresearch.com/v1
        api_key: os.environ/MNEMOZINE_HERMES_API_KEY
  litellm_settings:
    callbacks: mnemozine.ingestion.gateway.litellm_register.hermes_gateway_callback
  ```

  The Hermes variant is sketched (commented) at the bottom of
  `gateway/config.yaml`.

### Reading memory back

All agents — Claude Code and OpenAI/Hermes alike — read from the **single MCP
server** (`mnemozine-mcp`). It exposes:

- `recall(query, scope=None, top_k=10)` — on-demand consolidated recall
  (FR-RET-4). `scope` is optional: omit for current project + global, or pass
  `global` / `project:<id>` / a bare project id.
- `session_start_index(...)` — the FR-RET-3 compact index as a tool (so non-hook
  agents can request it too).
- `mid_session_index(prompt, project=None)` — the FR-RET-5 finer-grained index.

Transports: `stdio` (Claude Code local default) and `streamable-http` / `sse`
(networked OpenAI/Hermes agents), selected with `mnemozine-mcp --transport ...`.

---

## Eval harness and bootstrapping the eval set

The §9 eval harness is the `mnemozine-eval` console script. It runs **offline**
against a committed gold-set fixture and a packaged in-memory fake store, so it
needs no FalkorDB/Ollama/Qwen.

```bash
mnemozine-eval run                  # every §9 metric once; exits non-zero on failure
mnemozine-eval run -x 10            # same, with a 10x distractor inflation
mnemozine-eval scaling              # headline: injection precision at 1x/10x/100x
mnemozine-eval show-gold            # summarize the gold set
```

`scaling` is the headline §9 assertion — that precision **does not decline** as
the store is inflated with synthetic plausible-but-irrelevant distractors
(`--levels 1,10,100`, `--tolerance` for allowed drop). It exits non-zero if
precision declines.

### Bootstrapping the eval set (operator task)

The eval set encodes the operator's own preferences across their own projects, so
**only the operator can label it** (PRD §9 — this is an operator deliverable, ≈40
cases, ~2–3 hrs). Two-step flow:

```bash
# 1. Auto-propose extracted candidates and write a Markdown review sheet.
mnemozine-eval bootstrap-propose --out eval_review.md

# 2. Edit eval_review.md by hand: tick "- [x] keep" on candidates to keep,
#    optionally correcting the proposed type/scope (human-in-the-loop, R1).

# 3. Fold the labeled sheet into a committed gold set.
mnemozine-eval bootstrap-finish --in eval_review.md --out mnemozine/evals/fixtures/gold_set.json
```

`bootstrap-finish` reads the ticked candidates back, builds a `GoldSet` (seed
memories + classifier cases), and writes it to the gold-set JSON (default the
committed fixture at `mnemozine/evals/fixtures/gold_set.json`). Commit that file
and run `mnemozine-eval run` on every change and on a schedule.

The offline `bootstrap-propose` uses a tiny demo backlog so the command is
exercisable out of the box; the integration pass can point it at the real
`IngestSource.backfill` + `Extractor` to propose from your actual historical
import.

---

## Operations

### Maintenance schedule (FR-MNT-5)

Maintenance is a separate, idempotent, repeatable pass (consolidate → resolve
entities → decay/archive → audit, in that order):

```bash
mnemozine-maintenance run      # run the full pass once and exit
mnemozine-maintenance serve    # run on the configured cron until interrupted
```

- The cron cadence is `MNEMOZINE_MAINTENANCE__CRON` (default `0 3 * * *`); the
  `serve` mode uses APScheduler.
- In docker-compose the `mnemozine-maintenance` service runs `serve` continuously.
- In Helm it is a long-lived `Deployment` by default; set
  `maintenance.asCronJob=true` to render a Kubernetes `CronJob` (schedule from
  `maintenance.cronSchedule`, defaulting to `tuning.maintenance.cron`).
- Each job is isolated — a failure in one is recorded as a note but does not abort
  the rest of the pass.
- Demotion to the archive tier is governed by `decay.archive_after`
  (`DECAY_ARCHIVE_AFTER_DAYS`, default 90 days unused); the system **archives,
  never hard-deletes** by default.

### Backing up the FalkorDB volume

FalkorDB is the single source of truth (graph **and** vectors). Its data lives at
`/data`:

- **docker-compose** — the named volume `falkordb-data` (mounted at `/data`).
- **Helm** — the StatefulSet's `data` PVC (the `volumeClaimTemplate`, mounted at
  `/data`).

FalkorDB speaks the Redis protocol, so back up the on-disk RDB. Trigger a save
then copy the dump out:

```bash
# docker-compose — trigger a save, then copy /data out of the falkordb container.
# (The named volume is <project>_falkordb-data; the project name defaults to the
#  compose file's directory, so `docker compose ... config --volumes` /
#  `docker inspect` resolve the exact volume name if you back it up by volume.)
docker compose -f deploy/docker-compose.yml exec falkordb redis-cli SAVE
docker compose -f deploy/docker-compose.yml cp falkordb:/data ./falkordb-backup-$(date +%F)

# kubernetes (StatefulSet pod <release>-mnemozine-falkordb-0, e.g. mz-mnemozine-falkordb-0)
kubectl -n mnemozine exec mz-mnemozine-falkordb-0 -- redis-cli SAVE
kubectl -n mnemozine cp mz-mnemozine-falkordb-0:/data ./falkordb-backup-$(date +%F)
```

If the FalkorDB password is set, pass `-a "$MNEMOZINE_FALKORDB__PASSWORD"` to
`redis-cli`. Restore by stopping FalkorDB, replacing the contents of the volume /
PVC with a backed-up `/data`, and restarting. Snapshotting the underlying volume
(or PVC `VolumeSnapshot`) while FalkorDB is quiesced is an equivalent approach.

> Superseded/decayed memories are kept (archive tier) rather than deleted, so the
> store grows slowly over time; size the FalkorDB volume (compose volume / Helm
> `falkordb.persistence.size`, default 10Gi) and Ollama/Qwen model volumes
> accordingly.

### Health checks

- `mnemozine-mcp` exposes an HTTP surface on its bind port; compose/Helm probe it
  via TCP/HTTP.
- `mnemozine-ingest` and `mnemozine-maintenance` have no HTTP surface — liveness
  is "the watcher/scheduler process is still running" (`pgrep`).

---

## Configuration reference

The single source of truth for config is `mnemozine/config.py`; the full env-var
list (with the `MNEMOZINE_` prefix and `__` nesting) is
[`.env.example`](./.env.example). Deployment specifics — image overrides, Helm
`values.yaml` knobs, the `MZ_COMPOSE_*` compose overrides — are in
[`deploy/README.md`](./deploy/README.md).
