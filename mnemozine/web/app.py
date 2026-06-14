"""The WebUI FastAPI app factory ``create_app(container)`` (WEBUI PRD §3).

This is the backend foundation everything else hangs off. ``create_app``:

1. stashes the live :class:`~mnemozine.app.Container` on ``app.state`` so every
   route reuses the existing composition root (``StorageBackend`` / retriever /
   maintenance / evals / activity log) — the UI is never a new source of truth;
2. installs **auth**: an optional static bearer-token dependency on every ``/api``
   router (no-op when ``web.token`` is unset; localhost-bind is the real fence, Q5);
3. **locks CORS**: allows only ``web.cors_origins`` (empty -> no cross-origin, the
   default for the single-image SPA served by this same app);
4. registers **all** routers (memories, mutations, graph, recall, crossrefs,
   activity, maintenance, eval, health) so the OpenAPI is complete now;
5. serves the **static SPA** from ``web.static_dir`` (or the bundled
   ``web/static`` dir) with an SPA-fallback to ``index.html`` for client routes,
   when present — the built SPA lands later, so a missing dir is fine (API-only).

Construction is pure (no network): the backend connects lazily on first request,
so ``create_app`` is import- and test-safe with no live FalkorDB.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from mnemozine.web.auth import require_auth
from mnemozine.web.routes import ALL_ROUTERS

if TYPE_CHECKING:
    from mnemozine.app import Container

logger = logging.getLogger(__name__)

# The bundled static dir (the built SPA is copied here in the build step). Absent
# in a source checkout, which is fine — the app serves API-only then.
_BUNDLED_STATIC = Path(__file__).parent / "static"


def _resolve_static_dir(container: Container) -> Path | None:
    """Pick the SPA static dir: configured override, else the bundled dir if present."""

    configured = container.settings.web.static_dir
    if configured is not None:
        path = Path(configured)
        return path if path.is_dir() else None
    return _BUNDLED_STATIC if _BUNDLED_STATIC.is_dir() else None


def create_app(container: Container) -> FastAPI:
    """Build the operator-console FastAPI app over the given Container (WEBUI PRD).

    Pure construction — does not open a FalkorDB connection (routes connect lazily
    on first use), so this is safe to call in tests and at import time.
    """

    app = FastAPI(
        title="Mnemozine Operator Console",
        version="0.0.1",
        summary="Local, single-operator console over the Mnemozine memory layer.",
        description=(
            "Read-first JSON API + static SPA host over the existing Container. "
            "Local-only; never expose publicly (WEBUI PRD §2/Q5)."
        ),
    )
    app.state.container = container

    # --- locked CORS (4) --------------------------------------------------
    origins = container.settings.web.cors_origins
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )
    # When cors_origins is empty we add NO CORS middleware: same-origin only,
    # which is the default for the single-image SPA served by this app.

    # --- auth + all routers (2, 4) ---------------------------------------
    # The bearer-token dependency is attached to every /api router; it is a no-op
    # until web.token is set (Q5). Health is included so a probe still authenticates
    # uniformly when a token is configured.
    auth_dep = [Depends(require_auth)]
    for router in ALL_ROUTERS:
        app.include_router(router, dependencies=auth_dep)

    # --- static SPA (5) ---------------------------------------------------
    static_dir = _resolve_static_dir(container)
    if static_dir is not None:
        _mount_spa(app, static_dir)
        logger.info("mnemozine-web: serving SPA from %s", static_dir)
    else:
        logger.info("mnemozine-web: no static dir; serving API only")

    return app


def _mount_spa(app: FastAPI, static_dir: Path) -> None:
    """Serve the built SPA with a client-side-routing fallback to index.html.

    Static assets (``/assets/*`` etc.) are served directly; any other non-``/api``
    GET falls back to ``index.html`` so deep links into the SPA's client routes
    work on refresh. Mounted last so it never shadows the API routers.
    """

    index = static_dir / "index.html"

    # Direct asset serving for the built bundle's asset folder, if present.
    assets = static_dir / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/", include_in_schema=False, response_model=None)
    async def _spa_root() -> FileResponse | JSONResponse:
        return _spa_file_response(static_dir, index, "")

    @app.get("/{full_path:path}", include_in_schema=False, response_model=None)
    async def _spa_fallback(full_path: str, request: Request) -> FileResponse | JSONResponse:
        # Never hijack the API or docs surfaces.
        if full_path.startswith(("api/", "docs", "redoc", "openapi.json")):
            return JSONResponse({"detail": "not found"}, status_code=404)
        return _spa_file_response(static_dir, index, full_path)


def _spa_file_response(
    static_dir: Path, index: Path, full_path: str
) -> FileResponse | JSONResponse:
    """Resolve a SPA path to a file (sync — keeps pathlib off the async routes).

    Serves a real static file under ``static_dir`` when one exists (path-traversal
    guarded), else falls back to ``index.html`` so client-side routes resolve on
    refresh, else 404 when the SPA is not built.
    """

    candidate = (static_dir / full_path).resolve()
    try:
        candidate.relative_to(static_dir.resolve())
    except ValueError:
        return JSONResponse({"detail": "not found"}, status_code=404)
    if candidate.is_file():
        return FileResponse(str(candidate))
    if index.is_file():
        return FileResponse(str(index))
    return JSONResponse({"detail": "SPA not built"}, status_code=404)


__all__ = ["create_app"]
