"""Schema-valid sample data for the Phase-1 route stubs (WEBUI).

The stubs return these so the app boots, the OpenAPI is complete, and the
frontend foundation can render against realistic shapes before Phase-2 wires the
real backend. Everything here is deterministic and offline (no FalkorDB).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mnemozine.schema.models import MemoryType, Tier
from mnemozine.web.schemas import (
    MemoryDetail,
    MemoryListItem,
    Provenance,
    SupersessionLink,
    ValidityWindow,
)

_NOW = datetime(2026, 6, 14, 12, 0, 0, tzinfo=UTC)


def sample_list_item() -> MemoryListItem:
    """A representative active global preference row."""

    return MemoryListItem(
        id="sample-0001",
        type=MemoryType.PREFERENCE,
        content="Prefers thiserror over anyhow for Rust error handling.",
        scope="global",
        entities=["rust", "error-handling", "thiserror"],
        confidence=0.92,
        tier=Tier.HOT,
        active=True,
        valid_from=_NOW - timedelta(days=10),
        valid_to=None,
        last_accessed=_NOW - timedelta(hours=3),
        access_count=7,
        source="claude_code",
    )


def sample_detail() -> MemoryDetail:
    """A representative full memory detail with a supersession chain."""

    return MemoryDetail(
        id="sample-0001",
        type=MemoryType.PREFERENCE,
        content="Prefers thiserror over anyhow for Rust error handling.",
        scope="global",
        entities=["rust", "error-handling", "thiserror"],
        confidence=0.92,
        tier=Tier.HOT,
        validity=ValidityWindow(
            valid_from=_NOW - timedelta(days=10),
            valid_to=None,
            active=True,
        ),
        provenance=Provenance(
            source="claude_code",
            session_id="sess-1",
            chunk_hash="deadbeef",
            raw_path="~/.claude/projects/demo/sess-1.jsonl",
        ),
        supersedes=[
            SupersessionLink(
                memory_id="sample-0000",
                content="Prefers anyhow for Rust error handling.",
                valid_from=_NOW - timedelta(days=40),
                valid_to=_NOW - timedelta(days=10),
            )
        ],
        superseded_by=[],
        last_accessed=_NOW - timedelta(hours=3),
        access_count=7,
    )


__all__ = ["sample_list_item", "sample_detail"]
