"""Cross-reference engine — FR-RET-6 / UC-3.

Surfaces serendipitous, *explainable* connections between the current working
context and earlier ``idea_seed``/project memories. The primary path is graph
traversal over shared entities (explainable — the shared entities and the
connecting edges *are* the reason), with a vector-similarity fallback for when
no shared-entity path exists. Only connections above a (deliberately high)
relevance threshold surface, each carrying a human-readable reason, and
dismissed suggestions are suppressed so they stop resurfacing (R2).

The public class is :class:`CrossReferenceEngine`, a concrete implementation of
:class:`mnemozine.interfaces.CrossReferencer`.
"""

from __future__ import annotations

from mnemozine.crossref.engine import CrossReferenceEngine, context_key_for

__all__ = ["CrossReferenceEngine", "context_key_for"]
