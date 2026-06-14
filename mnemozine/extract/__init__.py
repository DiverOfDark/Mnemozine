"""Typed extraction layer (FR-EXT-1..4) — the make-or-break component.

Given a chunk/episode of :class:`~mnemozine.schema.events.IngestEvent`s, the
extraction layer calls a pluggable :class:`~mnemozine.interfaces.LLMProvider`
(local Qwen by default, OpenAI-format) to emit, per memory unit, the
category-split signals (core data-model redesign):

* the CONTROLLED :class:`~mnemozine.schema.models.ScopeDecision`
  (``global`` vs ``project``) — FR-EXT-1/3 — from which the final hierarchical
  :class:`~mnemozine.schema.models.Scope` is derived in Python (``global`` -> the
  root scope, ``project`` -> the transcript-derived, roll-up project scope), never
  trusted from the model (no-leak, FR-EXT-3);
* a FREE-FORM emergent ``category`` slug (the semantic role; no enum) — FR-EXT-1;
* a ``cross_ref_candidate`` flag preserving the old idea_seed cross-reference
  behavior (FR-RET-6);
* extracted entities + relationship triples — FR-EXT-2;
* confidence + provenance back to the source session/chunk — FR-EXT-4.

The normalized extraction-input chunk is also retained as a first-class
:class:`~mnemozine.schema.models.RawChunk` (the raw tier) via
:func:`build_raw_chunk` so the store can re-extract / reindex offline and survive
Claude's 30-day local cleanup (R4).

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
    build_raw_chunk,
)
from mnemozine.extract.raw_tier import extract_with_raw_retention

__all__ = [
    "ExtractedRelationship",
    "ExtractionResult",
    "TypedExtractor",
    "build_raw_chunk",
    "extract_with_raw_retention",
]
