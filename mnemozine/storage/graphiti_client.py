"""Thin Graphiti + FalkorDB client wrapper (FR-STO-1/2, OQ4).

Graphiti is the temporal-knowledge-graph engine and FalkorDB is the single store
that holds *both* the graph and the vector embeddings (PRD §5.5, FR-STO-2). This
module is the only place that imports ``graphiti_core`` / ``falkordb``; the rest
of the storage layer talks Cypher through :meth:`GraphitiClient.execute_query`
and model objects through the backend, so the heavy graph dependency is
quarantined behind one seam.

OQ4 (resolved): ``graphiti-core[falkordb]==0.29.2`` is pinned in
``pyproject.toml`` and ships ``graphiti_core.driver.falkordb_driver.FalkorDriver``
— Graphiti supports FalkorDB as a first-class backend at this version (the
``[falkordb]`` extra pulls the ``falkordb`` redis client; the upstream README and
MCP server both document a FalkorDB ``docker compose up`` default). The driver is
constructed from :class:`FalkorDBSettings` and passed to ``Graphiti(graph_driver=...)``.

Import policy
-------------
``graphiti_core`` and ``falkordb`` are imported **lazily** inside
:meth:`GraphitiClient.connect` (not at module top) so that:

* unit tests run fully offline against the in-memory fake without those packages
  installed (the storage backend imports this module but only *touches* the
  driver when a real backend is built), and
* an import error surfaces with an actionable message naming the extra to install
  rather than an opaque ``ModuleNotFoundError`` at process start.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from mnemozine.config import FalkorDBSettings, get_settings

# Label used for the MemoryUnit nodes this project owns in FalkorDB. Kept distinct
# from Graphiti's own ``Entity``/``Episodic`` labels so the two schemas coexist in
# one graph without colliding.
MEMORY_LABEL = "MnemozineMemory"
ENTITY_LABEL = "MnemozineEntity"
SESSION_LABEL = "MnemozineSession"
SUPPRESSION_LABEL = "MnemozineSuppression"
# Relationship type for the weighted, temporal entity-entity edges (§7 Edge).
RELATES_TYPE = "MNEMOZINE_RELATES"

# Name of the FalkorDB vector index over MemoryUnit embeddings (FR-STO-2).
MEMORY_VECTOR_INDEX = "mnemozine_memory_vec"

_GRAPH_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _parse_redis_url(url: str) -> tuple[str, int]:
    """Split a ``redis://host:port`` URL into ``(host, port)`` for FalkorDriver.

    FalkorDriver takes host/port rather than a URL; this keeps the single
    ``MNEMOZINE_FALKORDB__URL`` config knob authoritative.
    """

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    return host, port


class GraphitiClient:
    """Owns the Graphiti engine + FalkorDB connection for the storage backend.

    Responsibilities:

    * construct the :class:`FalkorDriver` from :class:`FalkorDBSettings` and the
      :class:`Graphiti` engine around it (lazy import; see module docstring),
    * expose :meth:`execute_query` so the backend can run the project's own
      MemoryUnit/Entity/Edge Cypher (the §7 model is richer than Graphiti's native
      nodes, so the backend owns its own labels — see :data:`MEMORY_LABEL`),
    * create the FalkorDB vector index over memory embeddings (FR-STO-2), and
    * own connection lifecycle (:meth:`connect` / :meth:`close`).

    It does **not** implement the :class:`StorageBackend` Protocol itself — that
    is :class:`mnemozine.storage.backend.GraphitiStorageBackend`, which composes
    this client. Keeping them separate means the backend's query logic is
    testable against any object exposing ``execute_query`` (see the fake driver in
    the storage tests).
    """

    def __init__(
        self,
        settings: FalkorDBSettings | None = None,
        *,
        embedding_dimensions: int = 1024,
    ) -> None:
        self._settings = settings or get_settings().falkordb
        if not _GRAPH_NAME_RE.match(self._settings.graph_name):
            # graph_name is interpolated into Cypher/label context; constrain it
            # to an identifier-safe charset to avoid injection via config.
            raise ValueError(
                f"falkordb.graph_name {self._settings.graph_name!r} must match "
                f"{_GRAPH_NAME_RE.pattern}"
            )
        self._embedding_dimensions = embedding_dimensions
        self._graphiti: Any | None = None
        self._driver: Any | None = None

    @property
    def graph_name(self) -> str:
        return self._settings.graph_name

    @property
    def graphiti(self) -> Any:
        """The underlying :class:`graphiti_core.Graphiti` engine (after connect)."""

        if self._graphiti is None:
            raise RuntimeError("GraphitiClient.connect() must be called first")
        return self._graphiti

    @property
    def driver(self) -> Any:
        """The underlying FalkorDB graph driver (after connect)."""

        if self._driver is None:
            raise RuntimeError("GraphitiClient.connect() must be called first")
        return self._driver

    async def connect(self) -> None:
        """Construct the FalkorDB driver + Graphiti engine and prepare indices.

        Idempotent: a second call is a no-op. Imports ``graphiti_core`` lazily so
        offline unit tests never need it (see module docstring).
        """

        if self._graphiti is not None:
            return
        try:
            from graphiti_core import Graphiti
            from graphiti_core.driver.falkordb_driver import FalkorDriver
        except ImportError as exc:  # pragma: no cover - exercised only with the dep absent
            raise RuntimeError(
                "graphiti-core[falkordb] is required for the live storage backend; "
                "install it (it is pinned in pyproject.toml as "
                "graphiti-core[falkordb]==0.29.2)"
            ) from exc

        host, port = _parse_redis_url(self._settings.url)
        self._driver = FalkorDriver(
            host=host,
            port=port,
            password=self._settings.password,
            database=self._settings.graph_name,
        )
        self._graphiti = Graphiti(graph_driver=self._driver)
        await self._build_indices()

    async def _build_indices(self) -> None:
        """Create Graphiti's own indices + the Mnemozine memory vector index.

        Both are idempotent in FalkorDB (``IF NOT EXISTS`` / build_indices guards),
        so this is safe to call on every connect (FR-MNT-5 spirit: re-runnable).
        """

        # Let Graphiti set up its native fulltext/range indices + constraints.
        graphiti = self.graphiti  # raises if connect() not yet called
        if hasattr(graphiti, "build_indices_and_constraints"):
            await graphiti.build_indices_and_constraints()
        await self.ensure_vector_index()

    async def ensure_vector_index(self) -> None:
        """Create the FalkorDB vector index over MemoryUnit embeddings (FR-STO-2).

        FalkorDB exposes vector search via a node vector index on a property; this
        indexes ``MnemozineMemory.embedding`` with cosine similarity at the
        configured dimensionality so :meth:`scoped_query` can do semantic search
        inside the composed-scope subset (FR-RET-2). ``IF NOT EXISTS`` makes it
        idempotent.
        """

        cypher = (
            f"CREATE VECTOR INDEX IF NOT EXISTS FOR (m:{MEMORY_LABEL}) "
            f"ON (m.embedding) OPTIONS {{dimension: $dim, similarityFunction: 'cosine'}}"
        )
        await self.execute_query(cypher, dim=self._embedding_dimensions)

    async def execute_query(self, cypher: str, **params: Any) -> Any:
        """Run a raw Cypher statement against the FalkorDB graph.

        Delegates to the Graphiti FalkorDB driver's ``execute_query`` so all of
        the project's own MemoryUnit/Entity/Edge Cypher (the §7 model the backend
        owns on top of Graphiti) goes through one connection/pool. Returns the
        driver's native result (records, summary, keys) untouched — the backend
        normalizes it.
        """

        return await self.driver.execute_query(cypher, **params)

    async def close(self) -> None:
        """Close the FalkorDB driver/connection (idempotent)."""

        if self._driver is not None:
            await self._driver.close()
        self._graphiti = None
        self._driver = None
