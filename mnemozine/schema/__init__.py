"""Shared pydantic schema for Mnemozine.

* :mod:`mnemozine.schema.events` — FR-ING-1 common ingest event schema and the
  content-hash idempotency helpers (FR-ING-5).
* :mod:`mnemozine.schema.models` — the §7 data model: MemoryUnit, Entity, Edge,
  SourceSession, and the supporting enums (MemoryType, Tier) and Scope helper.
"""

from __future__ import annotations

from mnemozine.schema.events import (
    IngestEvent,
    Role,
    Source,
    content_hash,
    idempotency_key,
)
from mnemozine.schema.models import (
    CLASSIFY_SOURCE,
    DEFAULT_CATEGORY,
    GLOBAL_SCOPE,
    SCOPE_DELIMITER,
    Edge,
    Entity,
    MemoryType,
    MemoryUnit,
    Provenance,
    RawChunk,
    Scope,
    ScopeDecision,
    SourceSession,
    Suppression,
    Tier,
)

__all__ = [
    # events
    "IngestEvent",
    "Role",
    "Source",
    "content_hash",
    "idempotency_key",
    # models
    "CLASSIFY_SOURCE",
    "DEFAULT_CATEGORY",
    "GLOBAL_SCOPE",
    "SCOPE_DELIMITER",
    "Edge",
    "Entity",
    "MemoryType",
    "MemoryUnit",
    "Provenance",
    "RawChunk",
    "Scope",
    "ScopeDecision",
    "SourceSession",
    "Suppression",
    "Tier",
]
