"""Ingestion subpackage for the OpenAI-format gateway and Hermes (FR-ING-3/4).

This package owns the two non-Claude-Code ingestion paths:

* :mod:`mnemozine.ingestion.gateway` — the LiteLLM custom logging callback that
  fronts the operator's repointable OpenAI-format agents and emits common-schema
  :class:`~mnemozine.schema.events.IngestEvent`s (FR-ING-3). Ships with an
  example LiteLLM proxy ``config.yaml`` pointing at local Qwen and allowing a
  cloud backend behind a ``base_url``.
* :mod:`mnemozine.ingestion.hermes` — the adapter for the self-hosted Nous
  Research Hermes agent (FR-ING-4): a direct-instrumentation
  :class:`~mnemozine.interfaces.IngestSource` (preferred) plus a thin helper to
  front Hermes' OpenAI-compatible endpoint via the same gateway callback
  (``source=hermes``).

Both paths normalize into the FR-ING-1 common schema and strip ``tool_calls``
per FR-ING-7 before anything downstream sees an event.

.. note::
   ``INTERFACES.md`` names the ingestion layer root ``mnemozine/ingest/**``; this
   module's owning task assigns the explicit paths ``mnemozine/ingestion/gateway``
   and ``mnemozine/ingestion/hermes``. The two are flagged for the integration
   pass to reconcile (see this module's integration notes) — the public symbols
   are stable regardless of the final root name.
"""

from __future__ import annotations
