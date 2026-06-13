"""Module-level callback instance for LiteLLM proxy registration (FR-ING-3).

LiteLLM's proxy ``config.yaml`` registers a custom logger by a dotted path to a
**module-level instance**::

    litellm_settings:
      callbacks: mnemozine.ingestion.gateway.litellm_register.gateway_callback

This module exposes exactly that instance so the bundled ``config.yaml`` can
reference it without any glue code. The instance is the OpenAI-source gateway
(FR-ING-3); a Hermes-fronting deployment registers
:data:`hermes_gateway_callback` instead (FR-ING-4) — both share the same
:func:`~mnemozine.ingestion.gateway.callback.events_from_completion` mapping and
differ only in the ``source`` they stamp.

The ingestion service reads events off these instances via
``async for e in gateway_callback.stream(): ...``. In a multi-process proxy
deployment the consumer must run **in the same process** as the LiteLLM workers
(the queue is in-process); see this module's integration notes for the
cross-process variant (a small HTTP/redis sink) if the proxy is scaled out.
"""

from __future__ import annotations

from mnemozine.ingestion.gateway.callback import GatewayCallback, make_gateway_callback
from mnemozine.schema.events import Source

# FR-ING-3: the OpenAI-format gateway callback (source=openai). Registered by the
# bundled config.yaml. Built via the factory so it subclasses LiteLLM's
# CustomLogger when LiteLLM is importable.
gateway_callback: GatewayCallback = make_gateway_callback(source=Source.OPENAI)

# FR-ING-4: the same callback configured to stamp source=hermes, for fronting
# Hermes' OpenAI-compatible endpoint through a (separate) LiteLLM proxy instance.
hermes_gateway_callback: GatewayCallback = make_gateway_callback(source=Source.HERMES)

__all__ = ["gateway_callback", "hermes_gateway_callback"]
