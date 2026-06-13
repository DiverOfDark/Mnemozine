"""Claude Code hook entrypoints (deliverable #5; FR-ING-2/6, FR-RET-3/5).

Claude Code invokes a hook as a subprocess, passing a JSON payload on **stdin**
and reading the hook's response from **stdout** (and exit code). These scripts
are thin: they decode the payload, then delegate to the retrieval + ingestion
service Protocols (:class:`~mnemozine.ingestion.claude_code.hooks.runtime.HookServices`)
so the *integration pass* can wire the concrete services without these modules
importing any sibling module's internals.

Hooks
-----
* :mod:`~mnemozine.ingestion.claude_code.hooks.session_start` — **SessionStart**:
  derive the working context (cwd / git / recent turns) and inject the compact,
  token-budgeted index (FR-RET-3) by calling ``Retriever.build_index``.
* :mod:`~mnemozine.ingestion.claude_code.hooks.user_prompt_submit` —
  **UserPromptSubmit**: inject finer-grained, prompt-scoped memory mid-session
  (FR-RET-5).
* :mod:`~mnemozine.ingestion.claude_code.hooks.stop` — **Stop**: flush the current
  chunk to the ingestion service at session end (FR-ING-6).
* :mod:`~mnemozine.ingestion.claude_code.hooks.pre_compact` — **PreCompact**:
  flush the current chunk before compaction (FR-ING-6).

Each module exposes a ``main()`` (sync console entrypoint) and an async
``run(payload, services)`` so the behavior is unit-testable without a subprocess.
"""

from __future__ import annotations

from mnemozine.ingestion.claude_code.hooks.runtime import (
    HookPayload,
    HookServices,
    IngestService,
    context_from_payload,
    emit_additional_context,
    read_payload,
)

__all__ = [
    "HookPayload",
    "HookServices",
    "IngestService",
    "context_from_payload",
    "emit_additional_context",
    "read_payload",
]
