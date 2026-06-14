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

OQ4 / F5 — no cloud key required to construct the engine
--------------------------------------------------------
``Graphiti.__init__`` (verified live against ``graphiti-core==0.29.2``, see its
``__init__`` body) eagerly constructs a cloud ``OpenAIClient()`` /
``OpenAIEmbedder()`` / ``OpenAIRerankerClient()`` for every one of
``llm_client`` / ``embedder`` / ``cross_encoder`` left ``None``, and the
``OpenAIClient()`` default raises ``OpenAIError`` when ``OPENAI_API_KEY`` is
unset. Mnemozine never drives Graphiti's extraction/embedding/rerank pipeline —
it talks raw Cypher through the ``FalkorDriver`` (see :meth:`execute_query`) and
runs its *own* injected LLM/embedding providers — so rather than *tolerating* the
eager-client failure we inject explicit local **no-op** clients
(:class:`_NoopLLMClient` / :class:`_NoopEmbedder` / :class:`_NoopCrossEncoder`)
into ``Graphiti(...)``. Because all three slots are filled, none of the cloud
defaults are constructed and engine construction never touches ``OPENAI_API_KEY``
— a fully-local FalkorDB store needs no cloud key (PRD §3: "MUST run end-to-end
against local models with no cloud dependency"). The no-op clients raise if ever
*called*, which is correct: any code path that actually invokes Graphiti's
LLM/embedder is a bug here, since Mnemozine owns those layers itself.

Import policy
-------------
``graphiti_core`` and ``falkordb`` are imported **lazily** inside
:meth:`GraphitiClient.connect` (not at module top) so that:

* unit tests run fully offline against the in-memory fake without those packages
  installed (the storage backend imports this module but only *touches* the
  driver when a real backend is built), and
* an import error surfaces with an actionable message naming the extra to install
  rather than an opaque ``ModuleNotFoundError`` at process start.

The no-op client classes are likewise built **lazily** (they subclass Graphiti's
abstract base clients, so their *definition* needs ``graphiti_core`` importable)
via :func:`_build_noop_clients`, keeping the offline import policy intact.
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


def _result_rows(result: Any) -> list[Any]:
    """Normalize a FalkorDB driver result into a list of rows (OQ3 introspection).

    The real ``FalkorDriver.execute_query`` returns ``(records, header, summary)``;
    the fake / a QueryResult-shaped object exposes rows on ``.result_set``. This
    keeps :meth:`GraphitiClient.current_vector_index_dimension` agnostic to which
    one it got (it mirrors the backend's own ``_rows`` normalization, but kept
    local so the client has no dependency on the backend module).
    """

    if result is None:
        return []
    if isinstance(result, tuple):
        return list(result[0] or [])
    records = getattr(result, "result_set", result)
    return list(records or [])


def _row_mapping(row: Any) -> dict[Any, Any]:
    """Best-effort coerce a ``db.indexes()`` row into a flat ``{field: value}`` map.

    ``db.indexes()`` rows come back either as a dict keyed by the YIELD field
    names or as a positional list ``[type, label, properties, ...options...]``.
    For positional rows we name the first three stable columns and keep the rest
    under integer keys so :func:`_find_dimension` can still scan them for a vector
    ``dimension`` (hence the ``dict[Any, Any]`` key type).
    """

    if isinstance(row, dict):
        return dict(row)
    if isinstance(row, (list, tuple)):
        names = ["type", "label", "properties"]
        mapping: dict[Any, Any] = {}
        for i, value in enumerate(row):
            mapping[names[i] if i < len(names) else i] = value
        return mapping
    return {}


def _is_memory_vector_index(mapping: dict[Any, Any]) -> bool:
    """True if a ``db.indexes()`` row describes the MnemozineMemory vector index."""

    label = mapping.get("label")
    if label != MEMORY_LABEL:
        return False
    # The row must concern a vector index over the ``embedding`` property; the
    # type/entity-type/property fields vary by version, so accept any row whose
    # serialized form names both "vector" and "embedding".
    blob = " ".join(str(v) for v in mapping.values()).lower()
    return "vector" in blob and "embedding" in blob


def _find_dimension(value: Any) -> int | None:
    """Recursively search a ``db.indexes()`` row for a vector ``dimension`` int.

    FalkorDB nests the vector ``dimension`` inside an options/info mapping whose
    shape has shifted across versions; rather than hard-code a path we walk the
    structure and return the first ``dimension``-keyed integer found.
    """

    if isinstance(value, dict):
        for key, sub in value.items():
            if isinstance(key, str) and key.lower() == "dimension":
                coerced = _as_int(sub)
                if coerced is not None:
                    return coerced
            found = _find_dimension(sub)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_dimension(item)
            if found is not None:
                return found
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_noop_clients() -> tuple[Any, Any, Any]:
    """Build local no-op ``(llm_client, embedder, cross_encoder)`` for Graphiti.

    OQ4 / F5: passing explicit clients into ``Graphiti(...)`` stops its
    constructor from eagerly building the cloud ``OpenAIClient``/``OpenAIEmbedder``
    /``OpenAIRerankerClient`` defaults — which is what would otherwise raise
    ``OpenAIError`` when ``OPENAI_API_KEY`` is unset (see module docstring). The
    classes are defined here (lazily, inside :meth:`GraphitiClient.connect`) rather
    than at module top so importing this module never requires ``graphiti_core``
    (the offline-import policy): their *definition* needs the abstract base
    classes from ``graphiti_core``, so they are built only when a live engine is
    constructed.

    Each method raises if actually invoked. That is deliberate: Mnemozine drives
    its own injected LLM/embedding providers and only uses Graphiti's
    ``FalkorDriver`` for raw Cypher (see :meth:`execute_query`), so any call into
    Graphiti's LLM/embedder/rerank pipeline is a programming error to surface, not
    a silent cloud round-trip.
    """

    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.llm_client.client import LLMClient
    from graphiti_core.llm_client.config import LLMConfig, ModelSize

    _UNUSED = (
        "Mnemozine never drives Graphiti's built-in {kind}; it uses its own "
        "injected provider and only Graphiti's FalkorDriver for raw Cypher. This "
        "no-op client exists solely to keep Graphiti construction free of a cloud "
        "OPENAI_API_KEY (OQ4/F5). Reaching it means a code path unexpectedly "
        "invoked the engine's {kind}."
    )

    class _NoopLLMClient(LLMClient):
        """Local stand-in for Graphiti's LLM client; never called (OQ4/F5)."""

        def __init__(self) -> None:
            # A non-None config with a dummy key avoids any env/key lookup.
            super().__init__(config=LLMConfig(api_key="not-needed"), cache=False)

        async def _generate_response(
            self,
            messages: Any,
            response_model: Any = None,
            max_tokens: Any = None,
            model_size: Any = ModelSize.medium,
        ) -> dict[str, Any]:
            raise RuntimeError(_UNUSED.format(kind="LLM client"))

    class _NoopEmbedder(EmbedderClient):
        """Local stand-in for Graphiti's embedder; never called (OQ4/F5)."""

        async def create(self, input_data: Any) -> list[float]:
            raise RuntimeError(_UNUSED.format(kind="embedder"))

        async def create_batch(self, input_data_list: Any) -> list[list[float]]:
            raise RuntimeError(_UNUSED.format(kind="embedder"))

    class _NoopCrossEncoder(CrossEncoderClient):
        """Local stand-in for Graphiti's reranker; never called (OQ4/F5)."""

        async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
            raise RuntimeError(_UNUSED.format(kind="cross-encoder"))

    return _NoopLLMClient(), _NoopEmbedder(), _NoopCrossEncoder()


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
        # Retained for diagnostics only. With the explicit no-op clients (OQ4/F5)
        # the engine constructs without a cloud key, so this should stay None; it
        # captures an *unexpected* engine-construction failure (e.g. a future
        # graphiti-core API change) so callers can introspect rather than guess.
        self._graphiti_init_error: Exception | None = None

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
        # OQ4 / F5 (verified live against graphiti-core[falkordb]==0.29.2): pass
        # explicit local no-op LLM/embedder/cross-encoder clients so Graphiti's
        # __init__ does NOT eagerly construct its cloud ``OpenAIClient()`` default
        # (which raises ``OpenAIError`` when ``OPENAI_API_KEY`` is unset). Filling
        # all three client slots is what removes the cloud-key dependency: the
        # engine then constructs against a fully-local FalkorDB store with no
        # cloud round-trip. Mnemozine never drives Graphiti's
        # extraction/embedding/rerank pipeline (it uses its own injected providers
        # and only the ``FalkorDriver`` for raw Cypher; see :meth:`execute_query`),
        # so the no-op clients are never invoked — see :func:`_build_noop_clients`.
        llm_client, embedder, cross_encoder = _build_noop_clients()
        try:
            self._graphiti = Graphiti(
                graph_driver=self._driver,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=cross_encoder,
            )
        except Exception as exc:  # noqa: BLE001 - keep raw Cypher usable on any engine-init surprise
            # Should not happen now that no cloud key is required; retained as a
            # defensive guard so an unexpected future graphiti-core change degrades
            # to "driver-only" (raw Cypher still works) rather than aborting
            # connect(). The error is captured for diagnostics.
            self._graphiti = None
            self._graphiti_init_error = exc
        await self._build_indices()

    async def _build_indices(self) -> None:
        """Create Graphiti's own indices + the Mnemozine memory vector index.

        Both are idempotent in FalkorDB (``IF NOT EXISTS`` / build_indices guards),
        so this is safe to call on every connect (FR-MNT-5 spirit: re-runnable).

        Graphiti's native fulltext/range indices are built through whichever object
        exposes ``build_indices_and_constraints`` — the Graphiti engine (now always
        constructed, thanks to the OQ4/F5 explicit no-op clients in
        :meth:`connect`), else the FalkorDriver directly as a defensive fallback if
        the engine somehow failed to build. The Mnemozine vector index (FR-STO-2)
        is always created via :meth:`ensure_vector_index`.
        """

        builder = self._graphiti if self._graphiti is not None else self._driver
        if builder is not None and hasattr(builder, "build_indices_and_constraints"):
            await builder.build_indices_and_constraints()
        await self.ensure_vector_index()

    async def ensure_vector_index(self) -> None:
        """Create the FalkorDB vector index over MemoryUnit embeddings (FR-STO-2).

        FalkorDB exposes vector search via a node vector index on a property; this
        indexes ``MnemozineMemory.embedding`` with cosine similarity at the
        configured dimensionality so :meth:`scoped_query` can do an index-backed
        KNN inside the composed-scope subset (FR-RET-2).

        Idempotency: FalkorDB's ``CREATE VECTOR INDEX`` does **not** accept the
        ``IF NOT EXISTS`` qualifier (verified live against falkordb v4.x — it is a
        syntax error). Re-creating an existing index instead raises
        ``Attribute 'embedding' is already indexed``; we swallow exactly that so
        the call stays safe to run on every connect (FR-MNT-5: re-runnable). The
        ``dimension``/``similarityFunction`` OPTIONS must be inline literals, so
        the dimension is interpolated from the integer ``embedding_dimensions``
        (not a query param) — it is an ``int`` field, never user text.
        """

        cypher = (
            f"CREATE VECTOR INDEX FOR (m:{MEMORY_LABEL}) "
            f"ON (m.embedding) "
            f"OPTIONS {{dimension: {int(self._embedding_dimensions)}, "
            f"similarityFunction: 'cosine'}}"
        )
        try:
            await self.execute_query(cypher)
        except Exception as exc:  # noqa: BLE001 - only the already-indexed case is benign
            if "already indexed" not in str(exc).lower():
                raise

    @property
    def embedding_dimensions(self) -> int:
        """The dimensionality this client builds the memory vector index at.

        OQ3 ``migrate-index`` compares this *configured* width against the live
        index width (:meth:`current_vector_index_dimension`) to decide whether the
        index must be dropped + recreated and the hot tier re-embedded.
        """

        return self._embedding_dimensions

    async def current_vector_index_dimension(self) -> int | None:
        """Return the dimension of the *live* ``MnemozineMemory.embedding`` vector
        index, or ``None`` if no such vector index exists yet (OQ3 migrate-index).

        Reads ``CALL db.indexes()`` and picks out the vector index on
        ``MnemozineMemory.embedding``. FalkorDB exposes per-index metadata, with
        the vector ``dimension`` living in an ``options``/``info`` mapping whose
        exact key has shifted across versions; this scans the returned row for a
        ``dimension`` value defensively (and recurses into nested maps) rather than
        hard-coding a column position, so a FalkorDB minor-version change does not
        silently return the wrong width. Returns ``None`` when the index is absent
        or its dimension cannot be determined — the caller treats an undeterminable
        width as "do not migrate" (safe default).
        """

        try:
            result = await self.execute_query("CALL db.indexes()")
        except Exception:  # noqa: BLE001 - introspection is best-effort; absent => None
            return None

        rows = _result_rows(result)
        for row in rows:
            mapping = _row_mapping(row)
            if not _is_memory_vector_index(mapping):
                continue
            dim = _find_dimension(mapping)
            if dim is not None:
                return dim
        return None

    async def drop_vector_index(self) -> bool:
        """Drop the ``MnemozineMemory.embedding`` vector index if present (OQ3).

        Mirrors :meth:`ensure_vector_index`'s ``CREATE VECTOR INDEX FOR (m:Label)
        ON (m.prop)`` with the matching ``DROP VECTOR INDEX FOR (m:Label) ON
        (m.prop)``. Returns ``True`` if an index was dropped, ``False`` if there
        was nothing to drop (so re-running migrate-index is idempotent). A
        "no such index"/"not indexed" error is treated as the benign already-absent
        case; anything else surfaces.
        """

        cypher = f"DROP VECTOR INDEX FOR (m:{MEMORY_LABEL}) ON (m.embedding)"
        try:
            await self.execute_query(cypher)
            return True
        except Exception as exc:  # noqa: BLE001 - only the already-absent case is benign
            msg = str(exc).lower()
            if any(marker in msg for marker in ("no such index", "not indexed", "no index")):
                return False
            raise

    async def recreate_vector_index(self) -> None:
        """Drop then recreate the memory vector index at the configured width (OQ3).

        The fixed-dimension FalkorDB vector index cannot change its dimensionality
        in place, so an embedding-model/dimension change requires a drop +
        recreate. Idempotent and safe to re-run: :meth:`drop_vector_index` no-ops
        when absent and :meth:`ensure_vector_index` swallows the already-indexed
        case. After this, every memory must be re-embedded (the old vectors are at
        the wrong width and are no longer indexed) — the migrate-index job drives
        that re-embed via the StorageBackend.
        """

        await self.drop_vector_index()
        await self.ensure_vector_index()

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
