"""UserPromptSubmit hook — mid-session fine-grained injection (FR-RET-5, deliverable #5).

As the conversation moves into new topics, inject finer-grained memory relevant
to the *specific* prompt (FR-RET-5). Draws from the same ~500-token budget
envelope as SessionStart (FR-RET-3 format contract) and should be even smaller
per-injection — the retriever owns the budget; this hook passes the current
prompt through ``RetrievalContext.recent_text`` so the index is prompt-scoped.

Like SessionStart, this calls :meth:`Retriever.build_index`, which does not record
access (passive read).
"""

from __future__ import annotations

import asyncio

from mnemozine.ingestion.claude_code.hooks.runtime import (
    HookPayload,
    HookServices,
    context_from_payload,
    emit_additional_context,
    load_services,
    read_payload,
    render_injection,
)

HOOK_EVENT_NAME = "UserPromptSubmit"


async def run(payload: HookPayload, services: HookServices) -> str:
    """Build the prompt-scoped mid-session injection (FR-RET-5).

    Returns the delimited injection string (empty when there is nothing relevant
    or no retriever is wired). The current prompt is folded into the retrieval
    context so the index targets the new topic.
    """

    if services.retriever is None:
        return ""
    context = context_from_payload(payload, services.settings)
    index = await services.retriever.build_index(
        context, token_budget=services.settings.inject.token_budget
    )
    return render_injection(index)


def main() -> None:
    """Console entrypoint: read stdin payload, inject prompt-scoped memory."""

    payload = read_payload()
    services = load_services()
    try:
        text = asyncio.run(run(payload, services))
    except Exception:  # noqa: BLE001 - a hook must never break the session
        text = ""
    emit_additional_context(text, hook_event_name=HOOK_EVENT_NAME)


if __name__ == "__main__":
    main()
