"""Concrete OpenAI-format :class:`~mnemozine.interfaces.LLMProvider` (LiteLLM).

The extraction/classification/contradiction LLM is pluggable via an OpenAI-format
``base_url`` (default local Qwen; a cloud model is a drop-in env swap — PRD §3
exception, §5.5, OQ5). This module is the single concrete adapter behind the
:class:`~mnemozine.interfaces.LLMProvider` Protocol; everything else codes against
the Protocol and is tested against the ``FakeLLMProvider`` fake.

Import policy
-------------
``litellm`` is imported **lazily** inside the call methods (not at module top) so
that importing this module — and therefore the composition root in
:mod:`mnemozine.app` — never requires ``litellm`` to be importable. The package
imports and unit-tests fully offline; the dependency is only touched when a live
completion is actually requested.

JSON mode
---------
:meth:`LiteLLMProvider.complete_json` requests ``response_format={"type":
"json_object"}`` and parses the returned content. A pydantic model class passed as
``schema`` is converted to its JSON schema and embedded in the system prompt as a
shape hint (OpenAI-compatible servers vary in native schema support, so the
prompt hint is the portable path). The result is defensively parsed: a fenced or
prose-wrapped JSON object is recovered, and an unparseable response yields ``{}``
rather than raising (the callers — extraction/contradiction — all tolerate an
empty dict and degrade to a benign default).
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from mnemozine.config import ExtractionLLMSettings, get_settings

# Matches the first ``{...}`` object in a possibly fenced/prose-wrapped reply.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class LiteLLMProvider:
    """OpenAI-format LLM via LiteLLM (FR-EXT-*, FR-MNT-1/2, §5.5).

    Implements :class:`mnemozine.interfaces.LLMProvider` structurally. The model
    id, base URL, API key, temperature and timeout come from
    :class:`ExtractionLLMSettings` (``MNEMOZINE_EXTRACTION__*``), so pointing at a
    different endpoint/model is a config swap with no code change.
    """

    def __init__(self, settings: ExtractionLLMSettings | None = None) -> None:
        self._settings = settings or get_settings().extraction

    @property
    def model(self) -> str:
        """The configured LiteLLM model id (``provider/model``)."""

        return self._settings.model

    def _common_kwargs(self) -> dict[str, Any]:
        """Per-call kwargs shared by both completion paths."""

        return {
            "model": self._settings.model,
            "api_base": self._settings.base_url,
            "api_key": self._settings.api_key,
            "timeout": self._settings.timeout_s,
        }

    @staticmethod
    def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _content(response: Any) -> str:
        """Extract the assistant message text from a LiteLLM response object."""

        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError, KeyError, TypeError):
            # Some providers/dicts shape the response differently; be tolerant.
            if isinstance(response, dict):
                try:
                    return response["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError):
                    return ""
            return ""

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Return a plain-text completion (FR-EXT-*)."""

        import litellm

        kwargs = self._common_kwargs()
        kwargs.update(
            messages=self._messages(prompt, system),
            temperature=temperature,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = await litellm.acompletion(**kwargs)
        return self._content(response)

    @staticmethod
    def _schema_hint(schema: dict[str, Any] | type[Any]) -> dict[str, Any] | None:
        """Coerce ``schema`` into a JSON-schema dict for the prompt hint."""

        if isinstance(schema, dict):
            return schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_json_schema()
        return None

    def _json_system(self, system: str | None, schema: dict[str, Any] | type[Any]) -> str:
        """Build the system prompt for JSON mode, embedding the schema shape."""

        parts: list[str] = []
        if system:
            parts.append(system)
        hint = self._schema_hint(schema)
        if hint is not None:
            parts.append(
                "Respond with a single JSON object conforming to this JSON schema. "
                "Output ONLY the JSON object, no prose, no code fences:\n"
                + json.dumps(hint)
            )
        else:
            parts.append("Respond with a single JSON object only — no prose, no code fences.")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Best-effort parse of a JSON object from a model reply.

        Returns ``{}`` on failure: the callers (typed extraction, the contradiction
        check) all tolerate an empty dict and fall back to a benign default rather
        than crashing the pipeline on a malformed model reply.
        """

        if not text or not text.strip():
            return {}
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            match = _JSON_OBJECT_RE.search(text)
            if match is None:
                return {}
            try:
                value = json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                return {}
        return value if isinstance(value, dict) else {}

    async def complete_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | type[Any],
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Return a structured (JSON) completion conforming to ``schema``."""

        import litellm

        kwargs = self._common_kwargs()
        kwargs.update(
            messages=self._messages(prompt, self._json_system(system, schema)),
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            response = await litellm.acompletion(**kwargs)
        except Exception:  # noqa: BLE001 - some servers reject response_format
            kwargs.pop("response_format", None)
            response = await litellm.acompletion(**kwargs)
        return self._parse_json(self._content(response))


__all__ = ["LiteLLMProvider"]
