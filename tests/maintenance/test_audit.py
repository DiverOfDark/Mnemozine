"""R5 audit tests — read-only integrity walk reports counts and anomalies."""

from __future__ import annotations

import pytest

from mnemozine.config import Settings
from mnemozine.maintenance.audit import AuditJob
from mnemozine.schema.models import MemoryType, MemoryUnit, Provenance, Scope, Tier
from tests.conftest import InMemoryStorage


@pytest.mark.asyncio
async def test_audit_counts_and_flags_anomalies() -> None:
    storage = InMemoryStorage()

    good = MemoryUnit(
        type=MemoryType.PREFERENCE,
        content="solid pref",
        scope=Scope.global_(),
        entities=["rust"],
        confidence=0.9,
        provenance=Provenance(source="claude_code", session_id="s1"),
    )
    # Anomaly: classify-sentinel provenance + no entities + low confidence.
    sketchy = MemoryUnit(
        type=MemoryType.PREFERENCE,
        content="sketchy pref",
        scope=Scope.global_(),
        entities=[],
        confidence=0.05,
    )
    for m in (good, sketchy):
        await storage.upsert_memory(m)
    # One superseded + archived unit.
    await storage.upsert_memory(
        MemoryUnit(
            type=MemoryType.PREFERENCE,
            content="old pref",
            scope=Scope.global_(),
            entities=["rust"],
            confidence=0.9,
            provenance=Provenance(source="claude_code", session_id="s1"),
        )
    )
    old_id = next(m.id for m in storage.memories.values() if m.content == "old pref")
    await storage.close_validity_window(old_id)
    await storage.archive(old_id)

    job = AuditJob(storage, settings=Settings())
    report = await job.run()

    notes = " ".join(report.notes)
    assert "total=3" in notes
    assert "active=2" in notes
    assert "superseded=1" in notes
    assert "archived=1" in notes
    assert "classify-sentinel" in notes  # sketchy's provenance gap
    assert "no linked entities" in notes
    assert "confidence floor" in notes


@pytest.mark.asyncio
async def test_audit_is_read_only_and_idempotent() -> None:
    storage = InMemoryStorage()
    await storage.upsert_memory(
        MemoryUnit(
            type=MemoryType.PREFERENCE,
            content="a pref",
            scope=Scope.global_(),
            entities=["rust"],
            confidence=0.9,
            provenance=Provenance(source="claude_code", session_id="s1"),
        )
    )
    snapshot = {m.id: (m.tier, m.valid_to, m.confidence) for m in storage.memories.values()}
    job = AuditJob(storage, settings=Settings())
    await job.run()
    await job.run()
    after = {m.id: (m.tier, m.valid_to, m.confidence) for m in storage.memories.values()}
    # The audit mutates nothing.
    assert after == snapshot
    assert all(m.tier is Tier.HOT for m in storage.memories.values())
