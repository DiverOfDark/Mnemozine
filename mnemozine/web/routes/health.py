"""Health + store-stats routes (PRD §4.1 / §6 GET /api/health, /api/stats).

Health pings the three infra dependencies the console depends on — FalkorDB (the
graph+vector store), Ollama (embeddings), and the extraction LLM endpoint — and
reports each as ``ok`` / ``degraded`` / ``down`` / ``unknown`` without ever
failing the request (a console that 500s because Ollama is down is useless). The
overall status is the worst component status, ``ok`` when all are healthy.

Stats compute the top-bar / Dashboard totals from the live store via
:meth:`StorageBackend.store_stats` — per-category, per-scope-decision, per-tier,
and per-source counts plus active/superseded/entity/raw-chunk counts, all from
Cypher ``COUNT`` / grouping aggregates (NEVER a whole-store stream that
loads/parses the 1024-float embedding). Both run identically against the
in-memory fake in tests; the infra probes degrade to ``down``/``unknown`` offline
rather than raising.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import httpx
from fastapi import APIRouter

from mnemozine.web.deps import ContainerDep, SettingsDep, StorageDep
from mnemozine.web.schemas import (
    ComponentHealth,
    HealthResponse,
    StoreStatsResponse,
)

router = APIRouter(prefix="/api", tags=["health"])

# Worst-status ordering so the overall rollup can pick the least-healthy.
_STATUS_RANK = {"ok": 0, "unknown": 1, "degraded": 2, "down": 3}
_HTTP_PROBE_TIMEOUT_S = 2.0


def _pkg_version() -> str:
    try:
        return version("mnemozine")
    except PackageNotFoundError:  # pragma: no cover - editable/uninstalled
        return "0.0.0"


def _overall(components: list[ComponentHealth]) -> str:
    """Roll the per-component statuses up to one overall status (worst wins)."""

    if not components:
        return "unknown"
    worst = max(components, key=lambda c: _STATUS_RANK.get(c.status, 1))
    rank = _STATUS_RANK.get(worst.status, 1)
    if rank == 0:
        return "ok"
    if rank == _STATUS_RANK["down"]:
        return "down"
    return "degraded"


async def _check_falkordb(container: ContainerDep) -> ComponentHealth:
    """Probe FalkorDB by opening the backend and running a trivial Cypher ping.

    Reuses the Container's lazily-built storage backend (the same connection every
    route uses), so a successful read elsewhere implies this passes. A failure to
    connect is reported as ``down`` with the error detail, never raised.
    """

    try:
        storage = await container.build_storage()
    except Exception as exc:  # noqa: BLE001 - health must not raise
        return ComponentHealth(name="falkordb", status="down", detail=str(exc)[:200])
    client = getattr(storage, "_client", None)
    execute = getattr(client, "execute_query", None)
    if execute is None:
        # The in-memory fake / a backend without a Cypher seam: it built, so the
        # store is reachable, but we cannot ping it — report ok with a note.
        return ComponentHealth(
            name="falkordb", status="ok", detail="no Cypher ping seam (in-memory backend)"
        )
    try:
        await execute("RETURN 1")
    except Exception as exc:  # noqa: BLE001 - health must not raise
        return ComponentHealth(name="falkordb", status="down", detail=str(exc)[:200])
    return ComponentHealth(name="falkordb", status="ok")


async def _probe_http(name: str, url: str, detail_ok: str | None = None) -> ComponentHealth:
    """GET ``url`` with a short timeout; map the outcome to a ComponentHealth.

    A 2xx/3xx/4xx response means the endpoint is reachable (``ok``); any 5xx is
    ``degraded``; a connection/timeout error is ``down``. Used for the Ollama and
    LLM-endpoint liveness checks (we only need reachability, not a real call).
    """

    try:
        async with httpx.AsyncClient(timeout=_HTTP_PROBE_TIMEOUT_S) as client:
            resp = await client.get(url)
    except Exception as exc:  # noqa: BLE001 - health must not raise
        return ComponentHealth(name=name, status="down", detail=str(exc)[:200])
    if resp.status_code >= 500:
        return ComponentHealth(name=name, status="degraded", detail=f"HTTP {resp.status_code}")
    return ComponentHealth(name=name, status="ok", detail=detail_ok)


@router.get("/health", response_model=HealthResponse, summary="Liveness + infra health")
async def health(container: ContainerDep, settings: SettingsDep) -> HealthResponse:
    """Overall WebUI + infra health (PRD §4.1).

    Pings FalkorDB (via the backend connection), Ollama (embeddings base URL), and
    the extraction LLM endpoint. None of the probes can fail the request — a down
    dependency is reported, not raised — so the console stays usable to diagnose.
    """

    falkordb = await _check_falkordb(container)
    ollama = await _probe_http("ollama", settings.embedding.base_url.rstrip("/"))
    # The extraction LLM is an OpenAI-format base_url; '/models' is the cheap
    # reachability probe (no completion call, no tokens).
    llm_base = settings.extraction.base_url.rstrip("/")
    llm = await _probe_http("llm", f"{llm_base}/models")

    components = [falkordb, ollama, llm]
    return HealthResponse(
        status=_overall(components),
        version=_pkg_version(),
        components=components,
        activity_log_enabled=settings.web.enable_activity_log,
    )


@router.get("/stats", response_model=StoreStatsResponse, summary="Top-bar store stats")
async def stats(storage: StorageDep) -> StoreStatsResponse:
    """Live store totals for the top bar + Dashboard (PRD §4.1).

    Delegates to :meth:`StorageBackend.store_stats`, which computes every total
    with Cypher ``COUNT`` / grouping aggregates over scalar properties — the
    embedding is never touched and the store is never streamed into Python.
    """

    s = await storage.store_stats()
    return StoreStatsResponse(
        total_memories=s.total_memories,
        by_category=s.by_category,
        by_scope_decision=s.by_scope_decision,
        by_tier=s.by_tier,
        by_source=s.by_source,
        active_count=s.active_count,
        superseded_count=s.superseded_count,
        entity_count=s.entity_count,
    )


__all__ = ["router"]
