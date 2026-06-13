# Mnemozine — Deployment (`deploy/`)

PRD Deliverable #1: self-hostable deployment of the full memory layer as **both**
a `docker-compose` stack (local dev / eval harness) and a **Helm chart** (homelab
k8s), sharing one image definition (`deploy/Dockerfile`).

```
deploy/
  Dockerfile              multi-stage (uv) build of the mnemozine image
  docker-compose.yml      full local stack (FalkorDB, Ollama, Qwen, LiteLLM, mcp/ingest/maintenance)
  litellm.config.yaml     LiteLLM gateway config used by docker-compose
  helm/mnemozine/         the Helm chart (Chart.yaml, values.yaml, templates/)
```

The image is built once and shared by all three mnemozine workloads; each only
differs in the `console_script` it runs (`mnemozine-mcp` / `mnemozine-ingest` /
`mnemozine-maintenance`, declared in `pyproject.toml [project.scripts]`).

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

Services and what they do:

| Service | Purpose | PRD |
|---------|---------|-----|
| `falkordb` | single graph + vector store (named volume `falkordb-data`) | FR-STO-2 |
| `ollama` + `ollama-init` | bge-m3 embeddings; init pulls the model | §5.5 / OQ3 |
| `qwen` | local OpenAI-format extraction LLM (llama.cpp server by default) | §5.5 / OQ5 |
| `litellm` | OpenAI-format gateway + logging callback | FR-ING-3 |
| `mnemozine-mcp` | the single MCP server (published on `:8765`) | FR-RET-1 |
| `mnemozine-ingest` | Claude Code watcher + hooks; mounts `~/.claude` read-only | FR-ING-* |
| `mnemozine-maintenance` | consolidate / resolve / decay / audit | FR-MNT-* |

Inter-service URLs are set under each service's `environment:` (which overrides
`env_file`), so the containers reach each other by compose service name
(`redis://falkordb:6379`, `http://ollama:11434`, `http://litellm:4000/v1`) even
though `.env` ships `localhost` defaults for bare-metal dev. Override any of them
with the `MZ_COMPOSE_*` interpolation vars (e.g. `MZ_COMPOSE_EXTRACTION_URL`).

### Local Qwen model

The `qwen` service runs a llama.cpp OpenAI-compatible server. Place a GGUF model
into the `qwen-models` volume (or bind-mount one) and set `QWEN_MODEL` to its
filename. To use a **cloud / external extraction endpoint instead**, point
`MZ_COMPOSE_EXTRACTION_URL` (and `litellm.config.yaml` / `QWEN_API_BASE`) at it —
the system still runs end-to-end on local models with no cloud dependency
(PRD §3 exception).

### Claude Code transcripts

`mnemozine-ingest` mounts the host Claude Code config dir read-only at `/claude`
(FR-ING-2). Override the host path with `HOST_CLAUDE_CONFIG_DIR` (defaults to
`$HOME/.claude`).

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

## Validation

`helm lint deploy/helm/mnemozine` and `docker compose -f deploy/docker-compose.yml
config` both pass. The structural tests in `tests/deploy/` assert the artifacts
offline and additionally invoke `helm` / `docker` when those binaries are present
(skipped otherwise).
