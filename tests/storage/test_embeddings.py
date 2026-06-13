"""Unit tests for the bge-m3/Ollama embeddings adapter (FR-STO-2, OQ3).

No live Ollama: a fake ``AsyncClient`` is injected so the adapter's batching,
dimension validation, and EmbeddingProvider conformance are tested offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from mnemozine.config import EmbeddingSettings
from mnemozine.interfaces import EmbeddingProvider
from mnemozine.storage.embeddings import OllamaEmbeddingProvider


class _FakeEmbedResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class _FakeOllama:
    """Records calls and returns deterministic vectors of a fixed dimension."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[Any] = []

    async def embed(self, *, model: str, input: str | Sequence[str]) -> _FakeEmbedResponse:
        self.calls.append((model, input))
        items = [input] if isinstance(input, str) else list(input)
        return _FakeEmbedResponse([[float(i)] * self.dim for i in range(len(items))])


@pytest.fixture
def emb_settings() -> EmbeddingSettings:
    return EmbeddingSettings(model="bge-m3", dimensions=4)


async def test_satisfies_protocol(emb_settings: EmbeddingSettings) -> None:
    provider = OllamaEmbeddingProvider(emb_settings, client=_FakeOllama())
    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimensions == 4
    assert provider.model == "bge-m3"


async def test_embed_single(emb_settings: EmbeddingSettings) -> None:
    fake = _FakeOllama(dim=4)
    provider = OllamaEmbeddingProvider(emb_settings, client=fake)
    vec = await provider.embed("hello")
    assert vec == [0.0, 0.0, 0.0, 0.0]
    assert len(vec) == 4
    assert fake.calls == [("bge-m3", "hello")]


async def test_embed_batch_single_round_trip(emb_settings: EmbeddingSettings) -> None:
    fake = _FakeOllama(dim=4)
    provider = OllamaEmbeddingProvider(emb_settings, client=fake)
    vecs = await provider.embed_batch(["a", "b", "c"])
    assert len(vecs) == 3
    # R3: a batch is ONE request, not three.
    assert len(fake.calls) == 1
    assert fake.calls[0][1] == ["a", "b", "c"]


async def test_embed_batch_empty_short_circuits(emb_settings: EmbeddingSettings) -> None:
    fake = _FakeOllama()
    provider = OllamaEmbeddingProvider(emb_settings, client=fake)
    assert await provider.embed_batch([]) == []
    assert fake.calls == []


async def test_dimension_mismatch_raises() -> None:
    # Model returns 4-d vectors but config claims 1024 -> hard error at the boundary.
    settings = EmbeddingSettings(model="bge-m3", dimensions=1024)
    provider = OllamaEmbeddingProvider(settings, client=_FakeOllama(dim=4))
    with pytest.raises(ValueError, match="expected 1024"):
        await provider.embed("x")
