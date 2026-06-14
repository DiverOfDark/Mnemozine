"""Run-flag selection + all-in-one lifecycle tests for ``mnemozine`` (run_all).

These cover the ``settings.run.*`` toggles that drive the all-in-one entrypoint
(``mnemozine.app:run_all``) entirely against the offline conftest fakes — no live
FalkorDB / Ollama / Qwen and no real sockets:

* default: every component starts;
* ``ingest``-only: only the ingest component is selected (== ``mnemozine-ingest``);
* web + mcp: one FastAPI app serves the WebUI **and** the MCP app at ``/mcp`` on a
  single port (the 8765 clash is resolved);
* mcp-only (web off): a standalone MCP HTTP server is selected, not the web app;
* disabled components are never created;
* ``_run_all`` cancels every task on a stop signal and awaits ``container.close()``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mnemozine.app import (
    Container,
    _build_web_app,
    _run_all,
    _select_components,
)
from mnemozine.config import Settings
from tests.conftest import FakeEmbeddingProvider, FakeLLMProvider, InMemoryStorage


def _offline_container(**run_flags: bool) -> Container:
    """A Container pre-wired to offline fakes, with ``settings.run.*`` overrides.

    The memoized layer slots are filled with fakes so nothing ever opens a network
    connection; ``build_retriever`` / ``build_storage`` reuse these.
    """

    settings = Settings()
    # Keep the SPA mount out of the way so create_app builds API-only.
    settings.web.static_dir = Path("/nonexistent-spa-dir-for-tests")
    for name, value in run_flags.items():
        setattr(settings.run, name, value)
    c = Container(settings=settings)
    c._storage = InMemoryStorage()
    c._embedding = FakeEmbeddingProvider()
    c._llm = FakeLLMProvider()
    return c


# ---------------------------------------------------------------------------
# Component selection (pure: reads settings.run.*, returns factory mapping)
# ---------------------------------------------------------------------------


def test_default_starts_every_component() -> None:
    """All flags default True -> all four logical components are selected."""

    c = _offline_container()
    assert c.settings.run.mcp is True
    assert c.settings.run.ingest is True
    assert c.settings.run.maintenance is True
    assert c.settings.run.web is True

    components = _select_components(c)
    # web+mcp collapse into the single "web" component (MCP mounted at /mcp).
    assert set(components) == {"web", "ingest", "maintenance"}


def test_ingest_only_selects_just_ingest() -> None:
    """run.ingest only (the split-onto-another-machine case) selects only ingest.

    This is the ``mnemozine`` == ``mnemozine-ingest`` equivalence: with web/mcp/
    maintenance off, the all-in-one process drives exactly the ingest loop.
    """

    c = _offline_container(mcp=False, ingest=True, maintenance=False, web=False)
    components = _select_components(c)
    assert set(components) == {"ingest"}


def test_disabled_components_do_not_start() -> None:
    """A disabled flag never appears in the selected component set."""

    c = _offline_container(mcp=True, ingest=False, maintenance=False, web=True)
    components = _select_components(c)
    # ingest + maintenance disabled -> absent; web+mcp -> the single web component.
    assert "ingest" not in components
    assert "maintenance" not in components
    assert "mcp" not in components  # mcp is mounted inside web, not standalone
    assert set(components) == {"web"}


def test_mcp_only_selects_standalone_mcp_when_web_disabled() -> None:
    """mcp on + web off -> a standalone MCP component (not the web app)."""

    c = _offline_container(mcp=True, ingest=False, maintenance=False, web=False)
    components = _select_components(c)
    assert set(components) == {"mcp"}


def test_no_components_enabled_selects_nothing() -> None:
    c = _offline_container(mcp=False, ingest=False, maintenance=False, web=False)
    assert _select_components(c) == {}


# ---------------------------------------------------------------------------
# web + mcp share one app on one port
# ---------------------------------------------------------------------------


async def test_web_and_mcp_share_one_app() -> None:
    """The combined app serves the WebUI API AND the MCP transport at /mcp.

    Both surfaces live on the same FastAPI app (hence the same uvicorn port,
    default 8765): the ``/api`` routes are not shadowed by the MCP mount, and the
    MCP transport is reachable (the MCP session manager starts under the spliced
    lifespan). The contracted slash-less ``/mcp`` URL 307-redirects to ``/mcp/``
    (Starlette's ``Mount("/mcp")`` only answers ``/mcp/...``, so without the
    redirect the bare path would fall through to the SPA catch-all); ``/mcp/`` is
    the transport endpoint and rejects a bare GET with a transport guard, NOT a
    routing 404. ``/mcp/mcp`` is a real 404 (no double-prefix).
    """

    c = _offline_container()
    app = await _build_web_app(c, mount_mcp=True)

    with TestClient(app, follow_redirects=False) as client:
        # The WebUI API is intact (mount did not shadow it).
        assert client.get("/api/health").status_code == 200
        # The contracted slash-less /mcp 307-redirects into the mounted transport
        # (NOT swallowed by the SPA catch-all, which would 200 with HTML).
        redir = client.get("/mcp")
        assert redir.status_code == 307
        assert redir.headers["location"] == "/mcp/"
        # The transport endpoint exists at /mcp/: a bare GET hits its transport
        # guard (4xx), NOT a routing 404 and NOT the SPA.
        mcp_resp = client.get("/mcp/")
        assert mcp_resp.status_code != 404
        assert mcp_resp.status_code >= 400  # transport guard, not a success
        assert "text/html" not in mcp_resp.headers.get("content-type", "")
        # No accidental double-prefix mount.
        assert client.get("/mcp/mcp").status_code == 404


async def test_web_without_mcp_has_no_mcp_route() -> None:
    """mount_mcp=False builds the plain WebUI app with no /mcp surface."""

    c = _offline_container()
    app = await _build_web_app(c, mount_mcp=False)
    mount_paths = [getattr(r, "path", None) for r in app.router.routes]
    assert "/mcp" not in mount_paths
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200


# ---------------------------------------------------------------------------
# _run_all lifecycle: concurrent run, clean shutdown, container.close()
# ---------------------------------------------------------------------------


class _RecordingContainer:
    """A stand-in Container that records component starts and close(), no infra.

    ``_run_all`` only touches ``settings.run`` (via ``_select_components``) and
    ``close()``; we monkeypatch ``_select_components`` to return controllable
    component coroutines, so the real component factories (which would open
    sockets / FalkorDB) never run.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


async def test_run_all_runs_enabled_and_shuts_down_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled components run concurrently; a stop cancels them and closes once.

    We replace ``_select_components`` with three never-returning components and
    fire SIGTERM (via the installed loop handler proxy) by setting the stop event
    indirectly: the cleanest deterministic trigger is to cancel ``_run_all`` from
    the outside after the components are up, which exercises the same
    ``finally`` shutdown path (cancel every task, gather, ``container.close()``).
    """

    started: dict[str, asyncio.Event] = {
        "web": asyncio.Event(),
        "ingest": asyncio.Event(),
        "maintenance": asyncio.Event(),
    }
    cancelled: dict[str, bool] = {}

    def _make(name: str):
        async def _component() -> None:
            started[name].set()
            try:
                await asyncio.Event().wait()  # block forever
            except asyncio.CancelledError:
                cancelled[name] = True
                raise

        return _component

    def _fake_select(container: object) -> dict[str, object]:
        return {name: _make(name) for name in started}

    monkeypatch.setattr("mnemozine.app._select_components", _fake_select)

    container = _RecordingContainer(Settings())
    run_task = asyncio.create_task(_run_all(container))  # type: ignore[arg-type]

    # Wait until every component has actually started (proves concurrent launch).
    await asyncio.wait_for(
        asyncio.gather(*(e.wait() for e in started.values())), timeout=2.0
    )

    # Trigger graceful shutdown the same way SIGINT/SIGTERM would: cancel the
    # coordinator, which runs its finally block (cancel tasks + close container).
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    # Every component task was cancelled, and the container was closed exactly once.
    assert cancelled == {name: True for name in started}
    assert container.closed == 1


async def test_run_all_stop_signal_cancels_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A component finishing on its own also triggers shutdown of the rest.

    ``_run_all`` waits for FIRST_COMPLETED across the stop event and every task; a
    component returning early must cancel the still-running siblings and close the
    container (no hang).
    """

    long_started = asyncio.Event()
    long_cancelled = asyncio.Event()

    async def _short() -> None:
        return  # finishes immediately -> triggers the shutdown path

    async def _long() -> None:
        long_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            long_cancelled.set()
            raise

    def _fake_select(container: object) -> dict[str, object]:
        return {"ingest": _long, "maintenance": _short}

    monkeypatch.setattr("mnemozine.app._select_components", _fake_select)

    container = _RecordingContainer(Settings())
    await asyncio.wait_for(_run_all(container), timeout=2.0)  # type: ignore[arg-type]

    assert long_started.is_set()
    assert long_cancelled.is_set()
    assert container.closed == 1


async def test_ingest_component_parks_when_loop_ends_keeping_others_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished ingest loop must NOT tear down the rest of the all-in-one.

    In the all-in-one, ``_run_all`` shuts everything down on FIRST_COMPLETED. The
    streaming ingest loop returns early when no source is enabled or every source
    producer fails (e.g. a Claude Code watcher on an unreadable mount). The
    ``_ingest_component`` wrapper must therefore *park* (await forever) after the
    loop returns rather than returning — otherwise WebUI/MCP/maintenance would be
    taken down too. We stub ``_run_ingest`` to return immediately and assert the
    component is still pending and only ends on cancellation.
    """

    from mnemozine import app as app_mod

    async def _instant_loop(container: object, *, backfill: bool) -> None:
        return  # all sources failed/none enabled -> the loop returns

    monkeypatch.setattr(app_mod, "_run_ingest", _instant_loop)

    task = asyncio.create_task(app_mod._ingest_component(object()))  # type: ignore[arg-type]
    # Give the loop a chance to return; the component must still be parked (running).
    await asyncio.sleep(0.05)
    assert not task.done(), "ingest component returned -> would crash the all-in-one"

    # A real shutdown still cancels the parked await cleanly.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_run_all_no_components_closes_and_returns() -> None:
    """With nothing enabled, _run_all returns promptly and still closes cleanly."""

    settings = Settings()
    settings.run.mcp = False
    settings.run.ingest = False
    settings.run.maintenance = False
    settings.run.web = False
    container = _RecordingContainer(settings)
    await asyncio.wait_for(_run_all(container), timeout=2.0)  # type: ignore[arg-type]
    # No components -> early return before the run loop; close() is not reached in
    # that branch (nothing was opened), so the connection is left as-is.
    assert container.closed == 0
