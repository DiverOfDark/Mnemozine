"""Embeddings adapter for the bge-m3 model served by Ollama (FR-STO-2, OQ3).

This is the concrete :class:`mnemozine.interfaces.EmbeddingProvider` the storage
layer (and, via the container, the rest of the system) uses to turn memory-unit
content and queries into vectors. Embeddings are the highest-volume LLM cost in
the system and stay local on Ollama for stability and cost control (PRD §5.5).

The adapter is intentionally thin: it owns an :class:`ollama.AsyncClient`,
batches where it can (R3), and validates that the model actually returns vectors
of the configured ``dimensions`` (a mismatch would silently break the FalkorDB
vector index, so it is caught at the boundary). The ``ollama`` import is kept at
module top because the package is import-light and pure-python; constructing the
provider does not open a connection (the client connects lazily on first call),
so importing this module remains side-effect free.
"""

from __future__ import annotations

from collections.abc import Sequence

from ollama import AsyncClient

from mnemozine.config import EmbeddingSettings, get_settings


class OllamaEmbeddingProvider:
    """bge-m3 embeddings via a self-hosted Ollama endpoint (FR-STO-2).

    Implements :class:`mnemozine.interfaces.EmbeddingProvider` structurally. The
    endpoint, model, and dimensionality come from :class:`EmbeddingSettings`
    (``MNEMOZINE_EMBEDDING__*`` env vars), so pointing at a different host/model
    is a config swap with no code change.
    """

    def __init__(
        self,
        settings: EmbeddingSettings | None = None,
        *,
        client: AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings().embedding
        # A caller may inject a pre-built/instrumented client; otherwise build one
        # pointed at the configured Ollama base URL with the per-request timeout.
        self._client = client or AsyncClient(
            host=self._settings.base_url,
            timeout=self._settings.timeout_s,
        )

    @property
    def dimensions(self) -> int:
        """Embedding vector dimensionality (bge-m3 = 1024 by default)."""

        return self._settings.dimensions

    @property
    def model(self) -> str:
        """The configured Ollama embedding model name (e.g. ``bge-m3``)."""

        return self._settings.model

    def _validate(self, vector: Sequence[float]) -> list[float]:
        """Coerce to ``list[float]`` and assert the configured dimensionality.

        A dimension mismatch (wrong model pulled, or a stale ``dimensions``
        config) would silently corrupt the FalkorDB vector index, so it is a hard
        error at the boundary rather than a confusing downstream failure.
        """

        out = [float(x) for x in vector]
        if len(out) != self._settings.dimensions:
            raise ValueError(
                f"embedding model {self._settings.model!r} returned "
                f"{len(out)} dims, expected {self._settings.dimensions}"
            )
        return out

    async def embed(self, text: str) -> list[float]:
        """Embed a single text into a vector (FR-STO-2)."""

        resp = await self._client.embed(model=self._settings.model, input=text)
        embeddings = list(resp.embeddings)
        if not embeddings:
            raise ValueError(
                f"embedding model {self._settings.model!r} returned no vector"
            )
        return self._validate(embeddings[0])

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many texts in one round-trip where the endpoint supports it (R3).

        Ollama's ``/api/embed`` accepts a list ``input`` and returns one vector
        per item, so a batch is a single request. An empty input short-circuits
        to avoid a needless call.
        """

        items = list(texts)
        if not items:
            return []
        resp = await self._client.embed(model=self._settings.model, input=items)
        vectors = list(resp.embeddings)
        if len(vectors) != len(items):
            raise ValueError(
                f"embedding model {self._settings.model!r} returned "
                f"{len(vectors)} vectors for {len(items)} inputs"
            )
        return [self._validate(v) for v in vectors]
