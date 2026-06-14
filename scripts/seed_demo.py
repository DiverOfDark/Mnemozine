"""Seed the live FalkorDB with demo data so the WebUI screens render real content.

Builds the real GraphitiStorageBackend over the live FalkorDB + Ollama embeddings,
upserts a handful of memories across all types/scopes (incl. a superseded pair and
an idea_seed), entities + weighted edges, a suppression, and emits a few
ActivityEvents into the FalkorDB-backed activity log.

Run with the venv active and FalkorDB + Ollama (bge-m3) reachable:
    python scripts/seed_demo.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from mnemozine.activity.log import build_activity_log
from mnemozine.activity.models import (
    ingest_event,
    injection_event,
    maintenance_event,
    write_decision_event,
)
from mnemozine.app import Container
from mnemozine.config import get_settings
from mnemozine.schema.models import (
    Edge,
    Entity,
    MemoryType,
    MemoryUnit,
    Provenance,
    Scope,
    Tier,
)

NOW = datetime.now(UTC)


def days_ago(n: float) -> datetime:
    return NOW - timedelta(days=n)


async def main() -> None:
    settings = get_settings()
    container = Container(settings)
    storage = await container.build_storage()
    print("connected to storage backend")

    # ---- entities --------------------------------------------------------
    entities = [
        Entity(id="ent-rust", canonical_name="Rust", type="language"),
        Entity(id="ent-thiserror", canonical_name="thiserror", type="tool"),
        Entity(id="ent-anyhow", canonical_name="anyhow", type="tool"),
        Entity(id="ent-tokio", canonical_name="tokio", type="tool"),
        Entity(id="ent-falkordb", canonical_name="FalkorDB", type="tool"),
        Entity(id="ent-graphiti", canonical_name="Graphiti", type="tool"),
        Entity(id="ent-memory", canonical_name="memory-layer", type="concept"),
        Entity(id="ent-cli", canonical_name="cli", type="concept"),
    ]
    for e in entities:
        await storage.upsert_entity(e)
    print(f"upserted {len(entities)} entities")

    # ---- weighted, temporal structural edges -----------------------------
    edges = [
        Edge(from_entity="ent-rust", to_entity="ent-thiserror", relation="uses", weight=0.9),
        Edge(from_entity="ent-rust", to_entity="ent-anyhow", relation="uses", weight=0.4),
        Edge(from_entity="ent-rust", to_entity="ent-tokio", relation="uses", weight=0.85),
        Edge(from_entity="ent-memory", to_entity="ent-falkordb", relation="stored_in", weight=0.95),
        Edge(from_entity="ent-memory", to_entity="ent-graphiti", relation="built_on", weight=0.8),
        Edge(from_entity="ent-falkordb", to_entity="ent-graphiti", relation="backs", weight=0.7),
        Edge(from_entity="ent-cli", to_entity="ent-rust", relation="written_in", weight=0.6),
    ]
    for edge in edges:
        await storage.upsert_edge(edge)
    print(f"upserted {len(edges)} edges")

    # ---- memories: preferences (incl. a superseded pair) -----------------
    prov_cc = Provenance(
        source="claude_code",
        session_id="sess-demo-1",
        chunk_hash="deadbeef01",
        raw_path="~/.claude/projects/demo/sess-demo-1.jsonl",
    )
    prov_cc2 = Provenance(
        source="claude_code",
        session_id="sess-demo-2",
        chunk_hash="deadbeef02",
        raw_path="~/.claude/projects/demo/sess-demo-2.jsonl",
    )
    prov_gw = Provenance(
        source="gateway",
        session_id="sess-gw-1",
        chunk_hash="cafe1234",
    )

    memories: list[MemoryUnit] = []

    # Superseded pair: old preference (anyhow) closed, new (thiserror) active.
    old_pref = MemoryUnit(
        id="mem-pref-old",
        type=MemoryType.PREFERENCE,
        content="Prefers the anyhow crate for error handling in Rust.",
        scope=Scope.global_(),
        entities=["Rust", "anyhow"],
        confidence=0.9,
        provenance=prov_cc,
        valid_from=days_ago(45),
        valid_to=days_ago(10),  # superseded
        tier=Tier.HOT,
        last_accessed=days_ago(12),
        access_count=4,
    )
    new_pref = MemoryUnit(
        id="mem-pref-new",
        type=MemoryType.PREFERENCE,
        content="Prefers the thiserror crate over anyhow for error handling in Rust.",
        scope=Scope.global_(),
        entities=["Rust", "thiserror"],
        confidence=0.97,
        provenance=prov_cc2,
        valid_from=days_ago(10),
        valid_to=None,  # active, supersedes old_pref
        tier=Tier.HOT,
        last_accessed=days_ago(1),
        access_count=11,
    )
    memories += [old_pref, new_pref]

    # More global preferences
    memories.append(
        MemoryUnit(
            id="mem-pref-tests",
            type=MemoryType.PREFERENCE,
            content="Always wants tests written before declaring a task done; "
            "prefers small, focused commits.",
            scope=Scope.global_(),
            entities=["cli"],
            confidence=0.88,
            provenance=prov_cc,
            valid_from=days_ago(30),
            last_accessed=days_ago(2),
            access_count=7,
        )
    )

    # ---- project_facts (project scope) -----------------------------------
    memories.append(
        MemoryUnit(
            id="mem-fact-tokio",
            type=MemoryType.PROJECT_FACT,
            content="The mnemozine project pins tokio to 1.38 and uses async throughout.",
            scope=Scope.project("mnemozine"),
            entities=["tokio", "Rust"],
            confidence=0.95,
            provenance=prov_cc2,
            valid_from=days_ago(20),
            last_accessed=days_ago(3),
            access_count=5,
        )
    )
    memories.append(
        MemoryUnit(
            id="mem-fact-falkor",
            type=MemoryType.PROJECT_FACT,
            content="Mnemozine stores the temporal knowledge graph and vectors "
            "in a single FalkorDB instance via Graphiti.",
            scope=Scope.project("mnemozine"),
            entities=["FalkorDB", "Graphiti", "memory-layer"],
            confidence=0.96,
            provenance=prov_cc,
            valid_from=days_ago(25),
            last_accessed=days_ago(1),
            access_count=9,
        )
    )
    # An archived project fact in another project scope.
    memories.append(
        MemoryUnit(
            id="mem-fact-archived",
            type=MemoryType.PROJECT_FACT,
            content="Project atlas used a Postgres + pgvector store before migrating.",
            scope=Scope.project("atlas"),
            entities=["memory-layer"],
            confidence=0.7,
            provenance=prov_gw,
            valid_from=days_ago(120),
            tier=Tier.ARCHIVE,
            last_accessed=days_ago(100),
            access_count=2,
        )
    )

    # ---- idea_seeds (first-class graph nodes) ----------------------------
    memories.append(
        MemoryUnit(
            id="mem-idea-crossref",
            type=MemoryType.IDEA_SEED,
            content="Idea: a serendipity engine that surfaces cross-project "
            "connections between idea seeds and active work via shared entities.",
            scope=Scope.global_(),
            entities=["memory-layer", "Graphiti"],
            confidence=0.6,
            provenance=prov_cc2,
            valid_from=days_ago(15),
            last_accessed=days_ago(4),
            access_count=3,
        )
    )
    memories.append(
        MemoryUnit(
            id="mem-idea-cli",
            type=MemoryType.IDEA_SEED,
            content="Idea: a single CLI in Rust that wraps the whole memory layer "
            "for local operators.",
            scope=Scope.project("mnemozine"),
            entities=["cli", "Rust", "memory-layer"],
            confidence=0.55,
            provenance=prov_cc,
            valid_from=days_ago(8),
            last_accessed=days_ago(5),
            access_count=1,
        )
    )

    for m in memories:
        await storage.upsert_memory(m)
    print(f"upserted {len(memories)} memories")

    # ---- a suppressed cross-reference (R2) -------------------------------
    await storage.record_suppression("mem-idea-cli", "project:mnemozine")
    print("recorded 1 suppression")

    # ---- activity log: emit a few events ---------------------------------
    client = getattr(storage, "_client", None)
    activity = build_activity_log(enable=True, client=client)
    events = [
        ingest_event(
            source="claude_code",
            session_id="sess-demo-2",
            project="mnemozine",
            summary="Ingested 3 chunks from Claude Code transcript sess-demo-2",
            ref_memory_ids=["mem-pref-new", "mem-fact-tokio"],
            detail={"chunks": 3, "messages": 24},
        ),
        write_decision_event(
            decision="supersede",
            memory_id="mem-pref-new",
            source="claude_code",
            summary="Preference reversed: thiserror now preferred over anyhow",
            superseded_ids=["mem-pref-old"],
            detail={"similarity": 0.91},
        ),
        write_decision_event(
            decision="add",
            memory_id="mem-fact-falkor",
            source="claude_code",
            summary="New project fact recorded for mnemozine",
        ),
        injection_event(
            session_id="sess-demo-3",
            project="mnemozine",
            summary="Injected SessionStart index (3 preferences, 2 project facts)",
            ref_memory_ids=["mem-pref-new", "mem-pref-tests", "mem-fact-tokio"],
            detail={"token_estimate": 142, "token_budget": 500},
        ),
        maintenance_event(
            job_name="consolidate",
            summary="Consolidation pass: 0 merged, 1 archived",
            detail={"archived": 1, "merged": 0},
        ),
        ingest_event(
            source="gateway",
            session_id="sess-gw-1",
            project="atlas",
            summary="Ingested 1 chunk from LiteLLM gateway",
            ref_memory_ids=["mem-fact-archived"],
            detail={"chunks": 1},
        ),
    ]
    for ev in events:
        await activity.append(ev)
    print(f"emitted {len(events)} activity events (enabled={activity.enabled})")

    await container.close()
    print("seed complete")


if __name__ == "__main__":
    asyncio.run(main())
