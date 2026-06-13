"""Hermes ingestion (FR-ING-4) — self-hosted Nous Research Hermes agent.

Public surface:

* :class:`HermesAdapter` — the preferred direct-instrumentation
  :class:`~mnemozine.interfaces.IngestSource`: the instrumented Hermes VM pushes
  native turn payloads in, the adapter emits common-schema
  :class:`~mnemozine.schema.events.IngestEvent`s (``source=hermes``).
* :func:`events_from_hermes_turn` — the pure, testable normalization of one
  Hermes-native turn payload into events (strips ``tool_calls``, FR-ING-7).
* :func:`hermes_gateway_source` — the fallback path: a LiteLLM
  :class:`~mnemozine.ingestion.gateway.callback.GatewayCallback` configured with
  ``source=hermes`` to front Hermes' OpenAI-compatible endpoint.
"""

from __future__ import annotations

from mnemozine.ingestion.hermes.adapter import (
    HermesAdapter,
    events_from_hermes_turn,
    hermes_gateway_source,
)

__all__ = [
    "HermesAdapter",
    "events_from_hermes_turn",
    "hermes_gateway_source",
]
