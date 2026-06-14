# Mnemozine — Deployment (`deploy/`)

PRD Deliverable #1: self-hostable deployment of the full memory layer as **both**
a `docker-compose` stack (local dev / eval harness) and a **Helm chart** (homelab
k8s), sharing one image definition (`deploy/Dockerfile`).

```
deploy/
  Dockerfile                  multi-stage (uv) build of the mnemozine image
  docker-compose.yml          default 3-service stack (FalkorDB, Ollama, all-in-one mnemozine);
                              optional `gateway` (LiteLLM) + `qwen-llamacpp` (llama.cpp) profiles
  docker-compose.ingest.yml   ingest-only half for a split deployment (run on the main PC)
  litellm.config.yaml         LiteLLM gateway config (used by the `gateway` profile)
  helm/mnemozine/             the Helm chart (Chart.yaml, values.yaml, templates/)
```

The image is built once and shared by every mnemozine workload. The default
**all-in-one** `mnemozine` console script (= `mnemozine.app:run_all`) builds the
container **once** and concurrently runs every *enabled* component (MCP + ingest +
maintenance + web) under one asyncio loop, with graceful shutdown on
SIGINT/SIGTERM (compose's default `SIGTERM`/`stop_signal` works). That is what
collapses the compose stack to **~3 containers**. The five single-component
scripts (`mnemozine-mcp` / `-ingest` / `-maintenance` / `-web` / `-eval`, declared
in `pyproject.toml [project.scripts]`) are unchanged and each still run exactly
**one** component — use a standalone script (or the all-in-one with a single
toggle) to split a component onto another machine.

The four component toggles (booleans, all default `true`, prefix `MNEMOZINE_`,
nested delimiter `__`) select what the all-in-one process runs:
`MNEMOZINE_RUN__MCP` / `__INGEST` / `__MAINTENANCE` / `__WEB`. When `RUN__WEB` and
`RUN__MCP` are both true, the WebUI and the MCP streamable-http transport share
**one** port — `MNEMOZINE_WEB__PORT` (default **8765**), with the MCP app mounted
at **`/mcp`** — so the all-in-one default exposes only 8765 (WebUI at `/`, `/api`;
MCP at `/mcp`). If `RUN__WEB=false` but `RUN__MCP=true`, MCP runs standalone on
`MNEMOZINE_MCP_HOST` / `MNEMOZINE_MCP_PORT` (default `127.0.0.1:8765`) — expose
`MNEMOZINE_MCP_PORT` instead. For a container, set `MNEMOZINE_WEB__HOST=0.0.0.0`
(and `MNEMOZINE_MCP_HOST=0.0.0.0` for the standalone-MCP case) so the port is
reachable; gate `/api` with `MNEMOZINE_WEB__TOKEN`.

All runtime configuration is the `MNEMOZINE_*` env contract from
`mnemozine/config.py` (full list in `.env.example`). The §6.6 tuning parameters
are config, not constants — overridable in both deployments.

---

## Docker Compose (local dev / eval)

```bash
# from the repo root
cp .env.example .env            # then edit endpoints/keys if needed
docker compose -f deploy/docker-compose.yml up -d --build
```

The **default** services (3) and what they do:

| Service | Purpose | PRD |
|---------|---------|-----|
| `falkordb` | single graph + vector store (named volume `falkordb-data`) | FR-STO-2 |
| `ollama` (+ `ollama-init`) | serves **both** the bge-m3 **embeddings** *and* the **qwen extraction** model; init pulls them | §5.5 / OQ3/OQ5 |
| `mnemozine` | the all-in-one app — MCP + ingest + maintenance + WebUI on `:8765` (WebUI `/`, `/api`; MCP `/mcp`); mounts `~/.claude` read-only | FR-RET-1 / FR-ING-* / FR-MNT-* |

Optional services, behind compose profiles (off by a plain `up`):

| Profile | Service | Purpose | PRD |
|---------|---------|---------|-----|
| `gateway` | `litellm` | OpenAI-format gateway + logging callback (capture OpenAI/Hermes agents) | FR-ING-3 |
| `qwen-llamacpp` | `qwen` | a dedicated llama.cpp OpenAI-format extraction server | §5.5 / OQ5 |

```bash
docker compose -f deploy/docker-compose.yml up -d --build                      # default 3-service stack
docker compose -f deploy/docker-compose.yml --profile gateway up -d            # + LiteLLM gateway
docker compose -f deploy/docker-compose.yml --profile qwen-llamacpp up -d      # extraction on llama.cpp, not Ollama
```

Extraction runs **on Ollama** alongside embeddings by default, so neither `qwen`
nor `litellm` is required. The recommended extraction env (use verbatim):

```bash
MNEMOZINE_EXTRACTION__BASE_URL=http://ollama:11434/v1   # /v1 suffix REQUIRED (Ollama's OpenAI-compatible endpoint)
MNEMOZINE_EXTRACTION__MODEL=openai/qwen2.5             # openai/ provider hits the /v1 OpenAI path; "qwen2.5" is the Ollama tag (e.g. openai/qwen2.5:7b). NOTE: ollama/ would 404 on /v1.
MNEMOZINE_EXTRACTION__API_KEY=not-needed
MNEMOZINE_EMBEDDING__BASE_URL=http://ollama:11434
MNEMOZINE_EMBEDDING__MODEL=bge-m3
MNEMOZINE_EMBEDDING__DIMENSIONS=1024
```

> The extraction model id is a **LiteLLM** id — for Ollama it **must** be prefixed
> `ollama/`. The config default `openai/qwen2.5` targets an *OpenAI-format* server
> (llama.cpp / LiteLLM), **not** Ollama.

Inter-service URLs are set under each service's `environment:` (which overrides
`env_file`), so the containers reach each other by compose service name
(`redis://falkordb:6379`, `http://ollama:11434`, `http://ollama:11434/v1` for
extraction) even though `.env` ships `localhost` defaults for bare-metal dev.
Override any of them with the `MZ_COMPOSE_*` interpolation vars (e.g.
`MZ_COMPOSE_EXTRACTION_URL`).

### Local Qwen on llama.cpp (`qwen-llamacpp` profile)

The `qwen` service runs a llama.cpp OpenAI-compatible server. Place a GGUF model
into the `qwen-models` volume (or bind-mount one) and set `QWEN_MODEL` to its
filename. To use a **cloud / external extraction endpoint instead**, point
`MZ_COMPOSE_EXTRACTION_URL` (and `litellm.config.yaml` / `QWEN_API_BASE`) at it —
the system still runs end-to-end on local models with no cloud dependency
(PRD §3 exception).

### Claude Code transcripts

The `mnemozine` app mounts the host Claude Code config dir read-only at `/claude`
for its ingest component (FR-ING-2). Override the host path with
`HOST_CLAUDE_CONFIG_DIR` (defaults to `$HOME/.claude`).

**Host-user mapping (so the mount is readable).** `~/.claude/projects` is usually
mode `700`, owned by your login user — a non-root container can't read it. So the
`mnemozine` and `mnemozine-ingest` compose services run as `${MNEMOZINE_UID:-1000}:${MNEMOZINE_GID:-1000}`,
which is the typical single-user Linux uid and works out of the box. If `id -u` /
`id -g` differ on your host, set `MNEMOZINE_UID` / `MNEMOZINE_GID` in `.env`.
(On k8s you instead control this via volume ownership / `fsGroup`, or run ingest
off-cluster — see the split deployment below.)

---

## Helm chart (homelab k8s)

```bash
helm lint deploy/helm/mnemozine
helm install mz deploy/helm/mnemozine -n mnemozine --create-namespace
# render without installing:
helm template mz deploy/helm/mnemozine
```

Rendered objects:

- **FalkorDB** — `StatefulSet` + headless `Service` + `volumeClaimTemplate`
  (graph + vector persistence).
- **Ollama / Qwen / LiteLLM** — `Deployment` + `Service` (+ PVCs for model
  storage). Ollama pulls bge-m3 via an init container on first start.
- **mcp / ingest / maintenance** — `Deployment`s from the shared mnemozine image.
  Maintenance can instead render as a k8s `CronJob` (`maintenance.asCronJob=true`).
- **ConfigMap** — all non-secret `MNEMOZINE_*` env, including every §6.6 tuning
  param from `.Values.tuning`; mounted into every workload via `envFrom`.
- **Secret** — FalkorDB password + extraction API key (+ `extraSecrets`).

### Parameterization (`values.yaml`)

- **Images** — `image.*` (mnemozine) and `<dep>.image.*` for each dependency.
- **Endpoints** — when a bundled dependency is `enabled`, its in-cluster Service
  DNS is wired automatically. Set `<dep>.enabled=false` and the matching
  `endpoints.external.*` to use something you run elsewhere (e.g. a cloud
  extraction endpoint via `litellm.enabled=false` +
  `endpoints.external.extractionBaseUrl`).
- **§6.6 tuning** — under `.Values.tuning`. Examples:
  ```bash
  helm upgrade mz deploy/helm/mnemozine \
    --set tuning.crossref.relevanceThreshold=0.85 \
    --set tuning.inject.tokenBudget=400 \
    --set tuning.maintenance.cron='0 4 * * *'
  ```
- **Resources** — every workload has `requests`/`limits` defaults; override per
  component.
- **Disable ingest** — for a [split deployment](#split-deployment--ingest-on-the-main-pc)
  set `ingest.enabled=false` so the homelab renders MCP / web / maintenance but
  **not** the ingest workload (run ingest on the main PC instead).

### External-endpoint example (no bundled backends)

```bash
helm install mz deploy/helm/mnemozine \
  --set falkordb.enabled=false  --set endpoints.external.falkordbUrl=redis://my-falkor:6379 \
  --set ollama.enabled=false    --set endpoints.external.ollamaBaseUrl=http://my-ollama:11434 \
  --set litellm.enabled=false   --set qwen.enabled=false \
  --set endpoints.external.extractionBaseUrl=https://api.openai.com/v1 \
  --set extraSecrets.MNEMOZINE_EXTRACTION__API_KEY=sk-... 
```

---

## Split deployment — ingest on the main PC

The operator scenario: the **homelab** runs the always-on memory layer (FalkorDB
+ Ollama + MCP/web/maintenance) and the **main PC** runs **ingest** (where the
Claude Code transcripts and the OpenAI/Hermes agents actually live). Running the
all-in-one `mnemozine` with only `MNEMOZINE_RUN__INGEST=true` is **exactly
equivalent** to the standalone `mnemozine-ingest` script (same `_run_ingest`), so
the split is two opposite toggle sets pointed at one FalkorDB.

**1. Homelab — disable ingest.**

- *docker-compose:* set `MNEMOZINE_RUN__INGEST=false` on the `mnemozine` service
  (it then serves MCP + web + maintenance only).
- *Helm:* `helm install … --set ingest.enabled=false`.

> The homelab **FalkorDB must be network-reachable from the main PC** — compose
> publishes `:6379`; in k8s expose it (Service/NodePort/port-forward) so the
> remote ingester can write to it.

**2. Main PC — ingest only, pointed at the homelab.** Run
`deploy/docker-compose.ingest.yml` (or the `mnemozine-ingest` script in a venv)
with only the ingest component on and the three remote endpoints set:

```bash
# ingest-only toggles (== `mnemozine-ingest`):
MNEMOZINE_RUN__INGEST=true
MNEMOZINE_RUN__MCP=false
MNEMOZINE_RUN__MAINTENANCE=false
MNEMOZINE_RUN__WEB=false

# remote endpoints (point at the homelab box):
MNEMOZINE_FALKORDB__URL=redis://<homelab-host>:6379
MNEMOZINE_EMBEDDING__BASE_URL=http://<ollama-host>:11434
MNEMOZINE_EXTRACTION__BASE_URL=http://<extraction-host>/v1
```

```bash
docker compose -f deploy/docker-compose.ingest.yml up -d --build
# …or, in a venv, the identical standalone script:
mnemozine-ingest
```

When extraction is served by the homelab's Ollama (the default), set
`MNEMOZINE_EXTRACTION__BASE_URL=http://<ollama-host>:11434/v1` and
`MNEMOZINE_EXTRACTION__MODEL=openai/qwen2.5` (the `/v1` suffix and `openai/`
prefix are required — the `openai/` LiteLLM provider hits Ollama's OpenAI `/v1`
path, whereas the `ollama/` provider speaks the native /api/* surface and would
404 on /v1). The ingester mounts the host's `~/.claude` read-only at
`/claude` so the watcher tails the main PC's real transcripts; the memory it
writes flows into the same FalkorDB the homelab's MCP server reads from.

---

## Validation

`helm lint deploy/helm/mnemozine` and `docker compose -f deploy/docker-compose.yml
config` both pass. The structural tests in `tests/deploy/` assert the artifacts
offline and additionally invoke `helm` / `docker` when those binaries are present
(skipped otherwise).
