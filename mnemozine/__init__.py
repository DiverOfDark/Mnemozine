"""Mnemozine — a self-hosted unified conversational memory layer.

Mnemozine ingests conversations from multiple AI agent surfaces (Claude Code,
OpenAI-format agents, Hermes), distills them into a temporal knowledge graph
(Graphiti on FalkorDB), and serves the result to every agent through a single
MCP server. See ``PRD.md`` for the full specification.

This top-level package re-exports the shared contracts that the rest of the
system is built against:

* :mod:`mnemozine.config`        — runtime configuration (pydantic-settings).
* :mod:`mnemozine.schema.events` — FR-ING-1 common ingest event schema.
* :mod:`mnemozine.schema.models` — §7 data model (MemoryUnit, Entity, ...).
* :mod:`mnemozine.interfaces`    — Protocol contracts for every layer.

Importing this package must never require a live FalkorDB, Ollama, or Qwen
endpoint; it only defines contracts, schema, and configuration.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.0.1"
