"""OpenAI-format LiteLLM gateway ingestion (FR-ING-3).

Public surface:

* :class:`GatewayCallback` — the LiteLLM custom-logger callback that captures
  each completion turn and emits common-schema
  :class:`~mnemozine.schema.events.IngestEvent`s. It is also an
  :class:`~mnemozine.interfaces.IngestSource`.
* :func:`make_gateway_callback` — factory that wires the callback as a LiteLLM
  ``CustomLogger`` subclass when LiteLLM is importable (offline-safe otherwise).
* :func:`events_from_completion` — the pure, testable mapping from a LiteLLM
  ``(kwargs, response_obj)`` payload to ordered events.

A reference LiteLLM proxy ``config.yaml`` ships alongside this module
(``mnemozine/ingestion/gateway/config.yaml``): it points at a local Qwen
``base_url`` by default and shows how to add a cloud backend behind the same
proxy, registering :class:`GatewayCallback` via ``litellm_settings.callbacks``.
"""

from __future__ import annotations

from mnemozine.ingestion.gateway.callback import (
    GatewayCallback,
    events_from_completion,
    make_gateway_callback,
)

__all__ = [
    "GatewayCallback",
    "events_from_completion",
    "make_gateway_callback",
]
