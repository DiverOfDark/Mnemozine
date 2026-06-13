"""PreCompact hook — flush the current chunk before compaction (FR-ING-6, deliverable #5).

Claude Code's ``PreCompact`` hook fires just before the transcript is compacted
(summarized + truncated), which would otherwise lose the pre-compaction turns to
the ingester. Per FR-ING-6 the hook flushes the current session's accumulated
chunk to the ingestion service first. Like the ``Stop`` flush it is idempotent
(de-dups on the FR-ING-5 content hash), so it is safe even when the watcher
already tailed the session.
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

HOOK_EVENT_NAME = "PreCompact"


async def run(payload: HookPayload, services: HookServices) -> int:
    """Flush the session's chunk(s) before compaction; return chunks submitted.

    Returns 0 when no ingest service is wired or nothing new was submitted.
    Separated from :func:`main` for testability.
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
    """Console entrypoint: read stdin payload, flush before compaction (FR-ING-6)."""

    payload = read_payload()
    services = load_services()
    try:
        asyncio.run(run(payload, services))
    except Exception:  # noqa: BLE001 - a hook must never break the session
        pass


if __name__ == "__main__":
    main()
