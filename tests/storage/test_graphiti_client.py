"""Tests for :mod:`mnemozine.storage.graphiti_client` (OQ4 / F5).

The headline assertion (resolving OQ4 / the F5 follow-up): constructing the
Graphiti engine must NOT require a cloud ``OPENAI_API_KEY``. Graphiti 0.29.2's
``__init__`` eagerly builds a cloud ``OpenAIClient()`` (which raises
``OpenAIError`` when the key is unset) for any of ``llm_client`` / ``embedder`` /
``cross_encoder`` left ``None``; the client fixes that by injecting explicit
local no-op clients so a fully-local FalkorDB store needs no cloud key (PRD §3).

Two layers of coverage:

* **offline** — :func:`_build_noop_clients` + a stub ``GraphDriver`` prove the
  engine constructs with ``OPENAI_API_KEY`` unset, with no FalkorDB and no
  network. This is the deterministic, always-run proof of the fix.
* **live** (``live_falkordb`` marker, skipped when no server) — a full
  :meth:`GraphitiClient.connect` with the key unset asserts the *real* engine is
  built (``client.graphiti`` is non-None, no init error) end-to-end.
"""

from __future__ import annotations

import uuid

import pytest

from mnemozine.config import FalkorDBSettings
from mnemozine.storage.graphiti_client import (
    MEMORY_LABEL,
    GraphitiClient,
    _build_noop_clients,
    _find_dimension,
)


@pytest.fixture
def _no_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee ``OPENAI_API_KEY`` is unset for the duration of a test.

    This is the precise precondition the OQ4/F5 fix is about: the old code raised
    ``OpenAIError`` here; the new code must construct cleanly.
    """

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Offline: explicit no-op clients let Graphiti construct with no cloud key
# ---------------------------------------------------------------------------


def _stub_driver():  # type: ignore[no-untyped-def]
    """A minimal in-process ``GraphDriver`` so ``Graphiti(...)`` validates offline.

    Graphiti's ``GraphitiClients`` pydantic model validates the driver is a
    ``GraphDriver`` instance, so a bare object will not do — but a subclass with
    its abstract methods stubbed needs neither FalkorDB nor a network. We only
    need *construction* to succeed (the no-cloud-key assertion); the stubbed
    methods are never called.
    """

    from graphiti_core.driver.driver import GraphDriver

    class _StubDriver(GraphDriver):  # type: ignore[misc]
        provider = "falkordb"

        def __init__(self) -> None:  # bypass the real connection
            pass

        async def execute_query(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def session(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        async def close(self) -> None:
            return None

        async def delete_all_indexes(self) -> None:
            return None

        async def build_indices_and_constraints(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            return None

    return _StubDriver()


def test_build_noop_clients_construct_without_openai_key(_no_openai_key: None) -> None:
    """The three no-op clients build with ``OPENAI_API_KEY`` unset (OQ4/F5)."""

    llm_client, embedder, cross_encoder = _build_noop_clients()
    assert llm_client is not None
    assert embedder is not None
    assert cross_encoder is not None


def test_graphiti_constructs_with_noop_clients_no_cloud_key(_no_openai_key: None) -> None:
    """``Graphiti(...)`` with explicit no-op clients needs no cloud key (OQ4/F5).

    This is the core fix: with all three client slots filled, Graphiti's
    constructor never builds its cloud ``OpenAIClient()`` default and so never
    touches ``OPENAI_API_KEY``. Previously this path raised ``OpenAIError``.
    """

    from graphiti_core import Graphiti

    llm_client, embedder, cross_encoder = _build_noop_clients()
    engine = Graphiti(
        graph_driver=_stub_driver(),
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )
    assert engine is not None
    # The engine is wired to our explicit clients, not cloud defaults.
    assert engine.llm_client is llm_client
    assert engine.embedder is embedder
    assert engine.cross_encoder is cross_encoder


async def test_noop_clients_raise_if_actually_invoked(_no_openai_key: None) -> None:
    """The no-op clients are construct-only: invoking them is a surfaced bug.

    Mnemozine drives its own injected LLM/embedding providers and only uses
    Graphiti's FalkorDriver for raw Cypher, so any call into the engine's
    LLM/embedder/rerank pipeline indicates a wiring mistake and must fail loudly
    rather than silently no-op or attempt a cloud round-trip.
    """

    llm_client, embedder, cross_encoder = _build_noop_clients()
    with pytest.raises(RuntimeError):
        await embedder.create("hello")
    with pytest.raises(RuntimeError):
        await embedder.create_batch(["a", "b"])
    with pytest.raises(RuntimeError):
        await cross_encoder.rank("q", ["p1", "p2"])
    with pytest.raises(RuntimeError):
        await llm_client._generate_response([])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OQ3 — vector-index introspection / drop / recreate (offline, fake driver)
# ---------------------------------------------------------------------------


class _IndexFakeDriver:
    """Tiny driver that answers ``db.indexes()`` + ``DROP/CREATE VECTOR INDEX``.

    Just enough of FalkorDB's surface to exercise the client's OQ3 helpers
    (``current_vector_index_dimension`` / ``drop_vector_index`` /
    ``recreate_vector_index``) offline. ``indexes`` holds the rows
    ``db.indexes()`` would return; ``drop_raises`` lets a test simulate the
    "no such index" benign-absent case.
    """

    def __init__(self, indexes: list, *, drop_raises: str | None = None) -> None:
        self.indexes = indexes
        self.drop_raises = drop_raises
        self.dropped = False
        self.created = False
        self.queries: list[str] = []

    async def execute_query(self, cypher: str, **params: object):  # type: ignore[no-untyped-def]
        self.queries.append(cypher)
        if "db.indexes()" in cypher:
            return (list(self.indexes), ["type", "label", "properties"], None)
        if cypher.startswith("DROP VECTOR INDEX"):
            if self.drop_raises is not None:
                raise RuntimeError(self.drop_raises)
            self.dropped = True
            self.indexes = []
            return ([], [], None)
        if cypher.startswith("CREATE VECTOR INDEX"):
            self.created = True
            return ([], [], None)
        raise AssertionError(f"unexpected cypher: {cypher}")

    async def close(self) -> None:
        return None


def _client_with_driver(driver: _IndexFakeDriver, *, dim: int = 1024) -> GraphitiClient:
    client = GraphitiClient(FalkorDBSettings(), embedding_dimensions=dim)
    client._driver = driver  # type: ignore[attr-defined]
    return client


def test_find_dimension_walks_nested_options() -> None:
    # FalkorDB nests the vector dimension inside an options/info map.
    row = {
        "label": MEMORY_LABEL,
        "type": "VECTOR",
        "properties": ["embedding"],
        "options": {"embedding": {"dimension": "768", "similarityFunction": "cosine"}},
    }
    assert _find_dimension(row) == 768


async def test_current_vector_index_dimension_reads_live_width() -> None:
    rows = [
        {
            "label": MEMORY_LABEL,
            "type": "VECTOR",
            "properties": ["embedding"],
            "info": {"dimension": 768},
        },
    ]
    client = _client_with_driver(_IndexFakeDriver(rows))
    assert await client.current_vector_index_dimension() == 768


async def test_current_vector_index_dimension_none_when_absent() -> None:
    # Only an unrelated range index present -> no memory vector index.
    rows = [{"label": "Other", "type": "RANGE", "properties": ["x"]}]
    client = _client_with_driver(_IndexFakeDriver(rows))
    assert await client.current_vector_index_dimension() is None


async def test_drop_vector_index_returns_true_then_false_when_absent() -> None:
    driver = _IndexFakeDriver([])
    client = _client_with_driver(driver)
    assert await client.drop_vector_index() is True
    assert driver.dropped is True

    # Simulate FalkorDB reporting the index is already gone -> benign False.
    driver2 = _IndexFakeDriver([], drop_raises="No such index ...")
    client2 = _client_with_driver(driver2)
    assert await client2.drop_vector_index() is False


async def test_drop_vector_index_reraises_unexpected_error() -> None:
    driver = _IndexFakeDriver([], drop_raises="some other failure")
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="some other failure"):
        await client.drop_vector_index()


async def test_recreate_vector_index_drops_then_creates() -> None:
    driver = _IndexFakeDriver(
        [{"label": MEMORY_LABEL, "type": "VECTOR", "properties": ["embedding"]}]
    )
    client = _client_with_driver(driver, dim=1024)
    await client.recreate_vector_index()
    assert driver.dropped is True
    assert driver.created is True
    # The recreate CREATE carries the configured width.
    create = [q for q in driver.queries if q.startswith("CREATE VECTOR INDEX")][0]
    assert "dimension: 1024" in create


# ---------------------------------------------------------------------------
# Live: a full connect() builds the real engine with no cloud key
# ---------------------------------------------------------------------------


async def _falkordb_reachable(url: str) -> bool:
    try:
        from falkordb.asyncio import FalkorDB
    except ImportError:  # pragma: no cover - dep present in this repo
        return False
    try:
        db = FalkorDB.from_url(url)
        g = db.select_graph("__mnemozine_ping__")
        await g.query("RETURN 1")
        await db.connection.aclose()
        return True
    except Exception:  # noqa: BLE001 - any failure => skip
        return False


@pytest.mark.live_falkordb
async def test_connect_succeeds_without_openai_key(_no_openai_key: None) -> None:
    """End-to-end: ``connect()`` builds the engine with ``OPENAI_API_KEY`` unset.

    The deliberate, documented OQ4/F5 contract, asserted against a real FalkorDB:
    no cloud key is needed to stand up the engine. ``client.graphiti`` is the real
    engine (not the driver-only fallback) and no init error was captured.
    """

    url = "redis://localhost:6379"
    if not await _falkordb_reachable(url):
        pytest.skip(f"no FalkorDB reachable at {url}")

    graph_name = f"mnemozine_oq4_{uuid.uuid4().hex[:12]}"
    client = GraphitiClient(
        FalkorDBSettings(url=url, graph_name=graph_name),
        embedding_dimensions=4,
    )
    try:
        await client.connect()  # must NOT raise OpenAIError
        assert client.graphiti is not None
        assert client._graphiti_init_error is None
    finally:
        try:
            from falkordb.asyncio import FalkorDB

            db = FalkorDB.from_url(url)
            await db.select_graph(graph_name).delete()
            await db.connection.aclose()
        except Exception:  # noqa: BLE001 - cleanup is best-effort
            pass
        await client.close()
