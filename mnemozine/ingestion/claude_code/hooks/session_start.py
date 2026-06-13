"""SessionStart hook — inject the compact memory index (FR-RET-3, deliverable #5).

On every Claude Code ``SessionStart``, derive the working context and inject a
compact, token-budgeted index (counts + entity tags + idea-seed hints + top
preference snippets) by calling :meth:`Retriever.build_index`. The retriever
truncates to ``inject.token_budget`` (~500); this hook only frames the result as
clearly-delimited advisory background and writes it to stdout in Claude Code's
hook-output shape.

``build_index`` deliberately does **not** record access (its reads are passive,
firing on every SessionStart) — the contract is enforced by the retriever, not
this hook.
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

HOOK_EVENT_NAME = "SessionStart"


async def run(payload: HookPayload, services: HookServices) -> str:
    """Build and return the injection text for this session start (FR-RET-3).

    Returns the delimited injection string (empty if there is nothing to surface
    or no retriever is wired yet). Separated from :func:`main` so it is testable
    without a subprocess/stdin.
    """

    if services.retriever is None:
        return ""
    context = context_from_payload(payload, services.settings)
    index = await services.retriever.build_index(
        context, token_budget=services.settings.inject.token_budget
    )
    return render_injection(index)


def main() -> None:
    """Console entrypoint: read stdin payload, inject, write stdout (FR-RET-3)."""

    payload = read_payload()
    services = load_services()
    try:
        text = asyncio.run(run(payload, services))
    except Exception:  # noqa: BLE001 - a hook must never break the session
        text = ""
    emit_additional_context(text, hook_event_name=HOOK_EVENT_NAME)


if __name__ == "__main__":
    main()
