"""Stop hook — flush the current chunk at session end (FR-ING-6, deliverable #5).

Claude Code's ``Stop`` hook fires when a session ends. Per FR-ING-6 (and R4: the
30-day cleanup loss risk) the hook flushes the current session's accumulated
chunk to the ingestion service so end-of-session memory is captured even if the
watcher lagged. The flush is idempotent — it de-dups on the FR-ING-5 content
hash — so flushing a session the watcher already tailed is a safe no-op.
"""

from __future__ import annotations

import asyncio

from mnemozine.ingestion.claude_code.hooks.runtime import (
    HookPayload,
    HookServices,
    load_services,
    read_payload,
    session_and_project,
)

HOOK_EVENT_NAME = "Stop"


async def run(payload: HookPayload, services: HookServices) -> int:
    """Flush the session's chunk(s) into the pipeline; return chunks submitted.

    Returns 0 when no ingest service is wired or nothing new was submitted (e.g.
    the watcher already ingested the session). Separated from :func:`main` for
    testability.
    """

    if services.ingest is None:
        return 0
    session_id, project = session_and_project(payload, services.settings)
    if not session_id:
        return 0
    return await services.ingest.flush_session(
        session_id=session_id,
        transcript_path=payload.transcript_path,
        project=project,
    )


def main() -> None:
    """Console entrypoint: read stdin payload, flush the session (FR-ING-6)."""

    payload = read_payload()
    services = load_services()
    try:
        asyncio.run(run(payload, services))
    except Exception:  # noqa: BLE001 - a hook must never break the session
        pass


if __name__ == "__main__":
    main()
