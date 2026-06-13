"""Typed extraction layer (FR-EXT-1..4) — the make-or-break component.

Given a chunk/episode of :class:`~mnemozine.schema.events.IngestEvent`s, the
extraction layer calls a pluggable :class:`~mnemozine.interfaces.LLMProvider`
(local Qwen by default, OpenAI-format) to:

* classify each memory unit into exactly one
  :class:`~mnemozine.schema.models.MemoryType`
  (``preference`` | ``project_fact`` | ``idea_seed``) — FR-EXT-1;
* set the scope **at extraction time** — ``preference``/``idea_seed`` -> global,
  ``project_fact`` -> ``project:<id>`` — FR-EXT-3;
* extract entities + relationship triples — FR-EXT-2;
* stamp confidence + provenance back to the source session/chunk — FR-EXT-4;
  ``idea_seed`` becomes its own first-class node (with its own embedding + the
  same entity links) when persisted by the storage layer.

The concrete implementation is :class:`TypedExtractor`, which satisfies the
:class:`mnemozine.interfaces.Extractor` Protocol. Prompts live in
:mod:`mnemozine.extract.prompts` so they are independently evaluable (the §9
classifier-accuracy eval, R1).
"""

from __future__ import annotations

from mnemozine.extract.extractor import (
    ExtractedRelationship,
    ExtractionResult,
    TypedExtractor,
)

__all__ = [
    "ExtractedRelationship",
    "ExtractionResult",
    "TypedExtractor",
]
