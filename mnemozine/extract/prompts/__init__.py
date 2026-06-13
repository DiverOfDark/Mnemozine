"""Independently-evaluable prompts for the typed extraction layer (FR-EXT-*).

The PRD calls the extraction classifier the **make-or-break** component (R1,
FR-EXT-3): its accuracy on ``preference`` vs ``project_fact`` is the single
biggest driver of system quality. To make the prompts independently auditable
and tunable, they live here — separate from the orchestration in
:mod:`mnemozine.extract.extractor` — so they can be reviewed, diffed, and
swapped without touching the classifier control flow, and so the §9
classifier-accuracy eval can target the exact text shipped in production.

Two prompt families are exposed:

* **chunk extraction** (:func:`build_extract_prompt`) — turns a whole
  chunk/episode of :class:`~mnemozine.schema.events.IngestEvent`s into a list of
  classified, scoped, entity-linked memory units (FR-EXT-1/2/3/4). Backs
  :meth:`Extractor.extract`.
* **single-statement classification** (:func:`build_classify_prompt`) — scores a
  single bare statement into one :class:`~mnemozine.schema.models.MemoryType`
  with a scope, entities, and a confidence (FR-EXT-3, the R1 eval path). Backs
  :meth:`Extractor.classify`.

Both ship a JSON schema (:data:`EXTRACT_JSON_SCHEMA`,
:data:`CLASSIFY_JSON_SCHEMA`) handed to
:meth:`mnemozine.interfaces.LLMProvider.complete_json` so the model returns a
parseable structured object rather than free text.
"""

from __future__ import annotations

from mnemozine.extract.prompts.classify import (
    CLASSIFY_JSON_SCHEMA,
    CLASSIFY_SYSTEM_PROMPT,
    build_classify_prompt,
)
from mnemozine.extract.prompts.extract import (
    EXTRACT_JSON_SCHEMA,
    EXTRACT_SYSTEM_PROMPT,
    build_extract_prompt,
)

__all__ = [
    "CLASSIFY_JSON_SCHEMA",
    "CLASSIFY_SYSTEM_PROMPT",
    "EXTRACT_JSON_SCHEMA",
    "EXTRACT_SYSTEM_PROMPT",
    "build_classify_prompt",
    "build_extract_prompt",
]
