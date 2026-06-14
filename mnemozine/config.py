"""Runtime configuration for Mnemozine (pydantic-settings).

Every operationally-relevant value lives here as a typed field so that nothing
is a hard-coded constant — in particular the §6.6 tuning parameters, which the
PRD insists be *config, not constants* because they must be empirically
calibrated against the eval set (see PRD §6.6 and §9).

Configuration is nested. Each subsection is its own ``BaseModel`` and is
populated from environment variables using a double-underscore delimiter, e.g.::

    MNEMOZINE_FALKORDB__URL=redis://falkordb:6379
    MNEMOZINE_EXTRACTION__MODEL=openai/qwen2.5
    MNEMOZINE_CROSSREF__RELEVANCE_THRESHOLD=0.8

All variables are also enumerated in ``.env.example``.

Importing this module is side-effect free; constructing :class:`Settings`
reads the environment / ``.env`` file but never opens a network connection.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Storage backend (FalkorDB — single store for graph + vectors, PRD §5.5/FR-STO-2)
# ---------------------------------------------------------------------------


class FalkorDBSettings(BaseModel):
    """Connection settings for the FalkorDB graph+vector store."""

    url: str = Field(
        default="redis://localhost:6379",
        description="FalkorDB (Redis protocol) connection URL.",
    )
    graph_name: str = Field(
        default="mnemozine",
        description="Name of the Graphiti graph/keyspace within FalkorDB.",
    )
    password: str | None = Field(
        default=None,
        description="Optional FalkorDB/Redis password.",
    )


# ---------------------------------------------------------------------------
# Extraction LLM (pluggable OpenAI-format base_url; default local Qwen, PRD §5.5)
# ---------------------------------------------------------------------------


class ExtractionLLMSettings(BaseModel):
    """The extraction/classification LLM (FR-EXT-*).

    Pluggable via an OpenAI-format ``base_url``. The dev/default target is a
    locally self-hosted Qwen; pointing this at a cloud model is a drop-in env
    swap (PRD §3 exception, §5.5, OQ5). The model id is given in LiteLLM's
    ``provider/model`` form so the same field works for local and cloud.
    """

    base_url: str = Field(
        default="http://localhost:8000/v1",
        description="OpenAI-format base URL for the extraction LLM (local Qwen by default).",
    )
    model: str = Field(
        default="openai/qwen2.5",
        description="LiteLLM model id (provider/model) for extraction/classification.",
    )
    api_key: str = Field(
        default="not-needed",
        description="API key for the extraction endpoint; local servers usually ignore it.",
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature; extraction wants determinism.",
    )
    timeout_s: float = Field(
        default=120.0,
        description="Per-request timeout in seconds for extraction calls.",
    )


# ---------------------------------------------------------------------------
# Embedding provider (bge-m3 via Ollama — self-hosted, PRD §5.5/OQ3)
# ---------------------------------------------------------------------------


class EmbeddingSettings(BaseModel):
    """The embedding model (bge-m3 served by Ollama, FR-STO-2).

    Embeddings are the highest-volume LLM cost and stay local (PRD §5.5/OQ3).
    """

    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama base URL serving the embedding model.",
    )
    model: str = Field(
        default="bge-m3",
        description="Ollama embedding model name.",
    )
    dimensions: int = Field(
        default=1024,
        description="Embedding vector dimensionality (bge-m3 is 1024-d).",
    )
    timeout_s: float = Field(
        default=60.0,
        description="Per-request timeout in seconds for embedding calls.",
    )


# ---------------------------------------------------------------------------
# Injection budget (FR-RET-3 / FR-RET-5 — the 500-token hard cap)
# ---------------------------------------------------------------------------


class InjectionSettings(BaseModel):
    """SessionStart / mid-session injection budget (FR-RET-3, FR-RET-5)."""

    # §6.6 tuning param `inject.token_budget`, initial value 500.
    token_budget: int = Field(
        default=500,
        description="Hard cap (tokens) on injected memory context. Truncate, never overflow.",
    )
    max_preference_snippets: int = Field(
        default=5,
        description="Max number of top-preference content snippets in the index.",
    )


# ---------------------------------------------------------------------------
# Cross-reference engine (FR-RET-6)
# ---------------------------------------------------------------------------


class CrossRefSettings(BaseModel):
    """Cross-reference / serendipity engine tuning (FR-RET-6, §6.6)."""

    # §6.6 `crossref.relevance_threshold`: "start high (precision over recall)".
    relevance_threshold: float = Field(
        default=0.8,
        description="Minimum relevance for a cross-reference to surface (high = precision-first).",
    )
    # §6.6 `crossref.max_suggestions`: initial guess 1-2.
    max_suggestions: int = Field(
        default=2,
        description="Max cross-reference suggestions surfaced per context.",
    )
    # FR-RET-6 vector-similarity fallback gate — distinct from
    # `relevance_threshold` (which gates final surfacing); this gates whether the
    # vector fallback path considers a candidate at all when graph traversal over
    # shared entities finds nothing.
    vector_fallback_threshold: float = Field(
        default=0.75,
        description=(
            "Min cosine similarity for the FR-RET-6 vector-similarity fallback to "
            "consider a candidate (distinct from relevance_threshold)."
        ),
    )


# ---------------------------------------------------------------------------
# Maintenance / dedup / decay (FR-MNT-* and §6.6)
# ---------------------------------------------------------------------------


class MaintenanceSettings(BaseModel):
    """Scheduled maintenance + dedup/decay tuning (FR-MNT-*, §6.6).

    Values flagged "calibrate" in §6.6 carry placeholder initial guesses; they
    are expected to be tuned against the eval set during Phase 1.
    """

    # §6.6 `dedup.equivalence_threshold` — reinforce-vs-add cutoff (FR-MNT-1).
    dedup_equivalence_threshold: float = Field(
        default=0.9,
        description="Cosine-similarity threshold above which a write reinforces rather than adds.",
    )
    # §6.6 `maintenance.edge_weight_floor` — low-weight edge pruning (FR-MNT-4).
    edge_weight_floor: float = Field(
        default=0.1,
        description="Edges below this weight are pruned during entity resolution.",
    )
    # §6.6 `maintenance.max_node_degree` — traversal-bound cap (FR-MNT-4).
    max_node_degree: int = Field(
        default=64,
        description="Cap on node degree to keep graph traversal bounded.",
    )
    # FR-MNT-1 contradiction-candidate cap — how many `type=preference`
    # candidates in the same scope/entity neighborhood are fed to the single
    # cheap contradiction LLM call. Bounds cost/latency of the supersede check.
    contradiction_candidate_cap: int = Field(
        default=5,
        description=(
            "Max type=preference candidates fed to the FR-MNT-1 cheap "
            "contradiction LLM call per write."
        ),
    )
    # §6.6 `decay.half_life` — recency ranking (FR-MNT-3), in days.
    decay_half_life_days: float = Field(
        default=30.0,
        description="Half-life (days) for the recency component of memory ranking.",
    )
    # §6.6 `decay.archive_after` — hot->archive demotion (FR-MNT-3), in days.
    decay_archive_after_days: int = Field(
        default=90,
        description="Demote a hot memory to the archive tier after this many days unused.",
    )
    # Cron-like schedule for the scheduled maintenance pass (FR-MNT-5).
    cron: str = Field(
        default="0 3 * * *",
        description="Cron expression for the scheduled maintenance run (APScheduler).",
    )


# ---------------------------------------------------------------------------
# Ingestion / chunking (FR-ING-*, §6.6)
# ---------------------------------------------------------------------------


class IngestSettings(BaseModel):
    """Ingestion-layer settings (FR-ING-*, §6.6)."""

    # --- source enablement (which sources the ingest loop wires up) ---------
    # The ingest loop (app.py `_run_ingest`) consults these to decide which
    # IngestSource(s) to wire into the source -> chunk -> extract -> store
    # pipeline. Claude Code is the Phase-1 default-on path (FR-ING-2); the
    # gateway (FR-ING-3) and Hermes (FR-ING-4) are Phase-2 and default off so a
    # fresh install does not require a running LiteLLM proxy or Hermes VM.
    enable_claude_code: bool = Field(
        default=True,
        description="Enable the Claude Code JSONL watcher source (FR-ING-2); on by default.",
    )
    enable_gateway: bool = Field(
        default=False,
        description="Enable the LiteLLM gateway source (FR-ING-3); off by default (Phase 2).",
    )
    enable_hermes: bool = Field(
        default=False,
        description="Enable the Hermes ingestion source (FR-ING-4). Off by default (Phase 2).",
    )

    # --- gateway (FR-ING-3) connection / queue settings ---------------------
    # The in-process GatewayCallback buffers emitted events on an asyncio.Queue
    # and stamps a default `project` when an agent does not thread one through
    # LiteLLM metadata. These let the ingest loop construct it without magic
    # numbers; the model base_url(s) themselves live in the LiteLLM proxy
    # config.yaml (os.environ/MNEMOZINE_GATEWAY_*), not here.
    gateway_default_project: str = Field(
        default="default",
        description="Fallback `project` for gateway turns lacking LiteLLM metadata (FR-ING-3).",
    )
    gateway_queue_max: int = Field(
        default=10_000,
        description="Max buffered events in the in-process gateway callback queue (FR-ING-3).",
    )

    # --- Hermes (FR-ING-4) connection / queue settings ----------------------
    # Direct VM instrumentation is preferred (HermesAdapter, an in-process
    # queue); `hermes_base_url` is for the fallback path that FRONTS Hermes'
    # OpenAI-compatible endpoint through the gateway (hermes_gateway_source).
    hermes_base_url: str = Field(
        default="https://hermes-agent.nousresearch.com/",
        description="Hermes OpenAI-compatible base URL for the FR-ING-4 gateway-fronting fallback.",
    )
    hermes_api_key: str = Field(
        default="not-needed",
        description="API key for the Hermes endpoint when fronted via the gateway (FR-ING-4).",
    )
    hermes_default_project: str = Field(
        default="hermes",
        description="Fallback `project` for Hermes turns lacking an explicit project (FR-ING-4).",
    )
    hermes_queue_max: int = Field(
        default=10_000,
        description="Max buffered events in the in-process Hermes adapter queue (FR-ING-4).",
    )

    # §6.6 `chunk.max_size` — episode size, calibrate vs Qwen context (FR-ING-6).
    chunk_max_chars: int = Field(
        default=8000,
        description="Max characters accumulated per chunk/episode before flush (FR-ING-6).",
    )
    chunk_max_messages: int = Field(
        default=40,
        description="Max messages accumulated per chunk before flush (FR-ING-6).",
    )
    # FR-ING-2: location of Claude Code JSONL transcripts.
    claude_config_dir: Path = Field(
        default_factory=lambda: Path.home() / ".claude",
        description="CLAUDE_CONFIG_DIR — root of Claude Code config/transcripts (FR-ING-2).",
    )
    # FR-ING-2 / R4: local transcripts are cleaned up after this many days; the
    # watcher must run frequently enough that nothing is lost. Optionally bumped
    # as a safety net.
    cleanup_period_days: int = Field(
        default=30,
        description="Claude Code transcript retention before local cleanup (FR-ING-2/R4).",
    )
    # FR-ING-7: strip tool_calls / tool results on ingest.
    strip_tool_calls: bool = Field(
        default=True,
        description="Strip tool_calls and tool results from events on ingest (FR-ING-7).",
    )


# ---------------------------------------------------------------------------
# Retrieval (FR-RET-2, §6.6)
# ---------------------------------------------------------------------------


class RetrievalSettings(BaseModel):
    """Retrieval-layer settings (FR-RET-2, §6.6)."""

    # §6.6 `retrieval.p95_latency_target` — set baseline in Phase 1.
    p95_latency_target_ms: int = Field(
        default=500,
        description="Target p95 retrieval latency in ms (calibrate baseline in Phase 1).",
    )
    top_k: int = Field(
        default=10,
        description="Default number of memory units returned by a scoped query.",
    )
    # FR-RET-2 entity-neighborhood traversal depth — how many hops out from the
    # active entities the scoped retrieve expands before semantic search. Bounds
    # the search subset (distinct from `maintenance.max_node_degree`, which caps
    # per-node fan-out).
    neighborhood_hops: int = Field(
        default=1,
        description=(
            "Entity-neighborhood traversal depth (hops) for FR-RET-2 scoped "
            "retrieve; bounds the searched subset."
        ),
    )
    # FR-RET-2 index-backed KNN over-fetch tuning (§6.6 "config, not constants").
    # FalkorDB's `db.idx.vector.queryNodes` applies the scope/tier/entity WHERE
    # *after* the KNN cut, so the backend over-fetches K = top_k * factor so the
    # post-filter is not starved by nearer out-of-scope neighbours, bounded by an
    # absolute cap so a large top_k can't ask the index for an effectively
    # unbounded scan (which would defeat the flat-search-space Goal-5). Previously
    # hard-coded as `_KNN_OVERFETCH`/`_KNN_MAX_K` in storage/backend.py.
    knn_overfetch_factor: int = Field(
        default=10,
        description="KNN over-fetch multiple of top_k before the scope/tier filter (FR-RET-2).",
    )
    knn_overfetch_cap: int = Field(
        default=512,
        description="Absolute cap on the over-fetched KNN K, bounding the index scan (FR-RET-2).",
    )


# ---------------------------------------------------------------------------
# WebUI / Operator console (PRD WEBUI §3 — local-only FastAPI console)
# ---------------------------------------------------------------------------


class WebSettings(BaseModel):
    """Operator-console WebUI server settings (WEBUI PRD §3, Q5).

    The console is a **local, single-operator** surface that can contain
    credentials (project threat model), so it binds to localhost by default and
    is never exposed publicly. An optional static bearer ``token`` gates every
    ``/api`` request when set (``MNEMOZINE_WEB__TOKEN=...``); when unset the API
    is open on the bound interface (fine for a localhost bind). CORS is locked to
    the configured ``cors_origins`` (empty = same-origin only, the default for the
    single-image SPA served by this same app).
    """

    host: str = Field(
        default="127.0.0.1",
        description="WebUI bind host. Defaults to localhost; never bind publicly (Q5).",
    )
    port: int = Field(
        default=8765,
        description="WebUI bind port.",
    )
    token: str | None = Field(
        default=None,
        description=(
            "Optional static bearer token gating /api requests "
            "(MNEMOZINE_WEB__TOKEN). When unset, the API is open on the bound host."
        ),
    )
    cors_origins: list[str] = Field(
        default_factory=list,
        description=(
            "Allowed CORS origins for the API. Empty = same-origin only "
            "(the default: the SPA is served by this same FastAPI app)."
        ),
    )
    static_dir: Path | None = Field(
        default=None,
        description=(
            "Directory of built SPA static assets to serve. None = serve the "
            "package's bundled web/static dir if present, else API-only."
        ),
    )
    enable_activity_log: bool = Field(
        default=False,
        description=(
            "Persist the ActivityEvent log (Q3). Off by default so the existing "
            "pipeline + tests use the NullActivityLog no-op seam; the WebUI run "
            "path turns it on (FalkorDB-backed)."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Top-level Mnemozine configuration.

    Read from the process environment and an optional ``.env`` file. Nested
    sections use a ``__`` delimiter (e.g. ``MNEMOZINE_FALKORDB__URL``). See
    ``.env.example`` for the full, authoritative variable list.
    """

    model_config = SettingsConfigDict(
        env_prefix="MNEMOZINE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    falkordb: FalkorDBSettings = Field(default_factory=FalkorDBSettings)
    extraction: ExtractionLLMSettings = Field(default_factory=ExtractionLLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    inject: InjectionSettings = Field(default_factory=InjectionSettings)
    crossref: CrossRefSettings = Field(default_factory=CrossRefSettings)
    maintenance: MaintenanceSettings = Field(default_factory=MaintenanceSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    web: WebSettings = Field(default_factory=WebSettings)

    # MCP server bind settings (FR-RET-1).
    mcp_host: str = Field(default="127.0.0.1", description="MCP server bind host.")
    mcp_port: int = Field(default=8765, description="MCP server bind port.")

    log_level: str = Field(default="INFO", description="Logging level.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    Cached so configuration is parsed once. Tests that need a fresh instance
    should call ``get_settings.cache_clear()`` or construct ``Settings(...)``
    directly with overrides.
    """

    return Settings()
