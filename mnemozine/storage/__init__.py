"""Storage layer — Graphiti temporal knowledge graph on FalkorDB (FR-STO-*, FR-MNT-1).

Public surface for the integration pass and sibling layers (which only import
from here, never from another layer's internals):

* :class:`GraphitiStorageBackend` — the concrete
  :class:`mnemozine.interfaces.StorageBackend`: persists MemoryUnit / Entity /
  Edge / SourceSession into FalkorDB, with temporal validity windows (FR-STO-1),
  scope tagging + scope-composing queries (FR-STO-3), vector embeddings in
  FalkorDB for semantic search (FR-STO-2), hot/archive tiering with a default-hot
  retrieval path (FR-STO-4), entity upsert/merge primitives (FR-MNT-4), and the
  4-way write decision (FR-MNT-1).
* :class:`GraphitiClient` — thin Graphiti+FalkorDB connection/Cypher wrapper; the
  only module that imports ``graphiti_core`` (OQ4: ``graphiti-core[falkordb]``
  supports FalkorDB at the pinned 0.29.2).
* :class:`OllamaEmbeddingProvider` — the bge-m3/Ollama
  :class:`mnemozine.interfaces.EmbeddingProvider` (FR-STO-2).
* ``ContradictsFn`` — the injected async contradiction predicate the integration
  pass wires to the FR-MNT-1 cheap LLM call.
* :func:`cosine_similarity` — the shared ranking helper.
"""

from __future__ import annotations

from mnemozine.storage.backend import ContradictsFn, GraphitiStorageBackend
from mnemozine.storage.cosine import cosine_similarity
from mnemozine.storage.embeddings import OllamaEmbeddingProvider
from mnemozine.storage.graphiti_client import (
    ENTITY_NAME_KEY_INDEX,
    MEMORY_VECTOR_INDEX,
    GraphitiClient,
)

__all__ = [
    "GraphitiStorageBackend",
    "GraphitiClient",
    "OllamaEmbeddingProvider",
    "ContradictsFn",
    "MEMORY_VECTOR_INDEX",
    "ENTITY_NAME_KEY_INDEX",
    "cosine_similarity",
]
