"""Synthetic distractor generator (PRD §9, deliverable #2 tail).

Proves *precision-stays-flat at 10x/100x* before a large real store exists. It
inflates the store with **plausible-but-irrelevant** memories (fake projects,
fake cross-domain preferences, fake idea-seeds) at 1x / 10x / 100x volume around
the fixed gold set, so the precision eval can run at each inflation level and
assert no decline (PRD §9 "synthetic scaling"). It doubles as FalkorDB traversal
load-testing.

Two generation backends, both implemented here:

* **Template/deterministic** (default, offline) — combinatorially expands a bank
  of fake projects × topics × stances into distractors. Seeded, so a given
  ``(multiplier, seed)`` always yields the same set: reproducible eval runs and
  unit tests with **no LLM call**.
* **LLM-backed** — when an :class:`~mnemozine.interfaces.LLMProvider` is supplied,
  asks the model for fresh plausible-but-irrelevant statements (PRD §9
  "LLM-produced"), falling back to the template bank to top up to the requested
  count and on any malformed/empty response. So it never blocks offline.

Crucially the generator is told the gold set's entities/projects/contents so it
can *avoid colliding* with them: a distractor must be irrelevant to every gold
case, otherwise it would corrupt the precision measurement.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from mnemozine.evals.goldset import GoldSet
from mnemozine.interfaces import LLMProvider
from mnemozine.schema.models import (
    MemoryType,
    MemoryUnit,
    Provenance,
    Scope,
)

# Inflation levels the precision eval must hold flat across (PRD §9).
DEFAULT_INFLATION_LEVELS: tuple[int, ...] = (1, 10, 100)

# A bank of plausible-but-irrelevant building blocks. These are deliberately in
# domains/projects unrelated to the gold fixture (rust/python/tokio/postgres/
# async-cli/recipes) so generated distractors never collide with a gold case.
_FAKE_PROJECTS: tuple[str, ...] = (
    "ledger-sync",
    "photo-organizer",
    "weather-dash",
    "invoice-bot",
    "garden-planner",
    "fleet-tracker",
    "podcast-clipper",
    "habit-tracker",
    "menu-scanner",
    "trip-budget",
)

_FAKE_PREFERENCE_TOPICS: tuple[tuple[str, list[str]], ...] = (
    ("Go modules layout", ["go", "modules", "layout"]),
    ("Terraform workspace naming", ["terraform", "infra", "naming"]),
    ("Kotlin coroutine scoping", ["kotlin", "coroutines", "scoping"]),
    ("Svelte store conventions", ["svelte", "frontend", "stores"]),
    ("gRPC proto versioning", ["grpc", "proto", "versioning"]),
    ("Bash strict-mode flags", ["bash", "shell", "strict-mode"]),
    ("Makefile phony targets", ["make", "build", "phony"]),
    ("YAML anchor usage", ["yaml", "config", "anchors"]),
)

_FAKE_FACT_TOPICS: tuple[tuple[str, list[str]], ...] = (
    ("uses redis 7 for caching", ["redis", "cache", "infra"]),
    ("targets node 20 LTS", ["node", "runtime", "lts"]),
    ("deploys via fly.io", ["flyio", "deploy", "infra"]),
    ("stores blobs in S3", ["s3", "storage", "blobs"]),
    ("uses pnpm workspaces", ["pnpm", "monorepo", "workspaces"]),
    ("pins kotlin 1.9", ["kotlin", "version", "build"]),
)

_FAKE_IDEAS: tuple[tuple[str, list[str]], ...] = (
    ("a CLI that lints commit messages", ["commits", "lint", "git"]),
    ("a TUI dashboard for home sensors", ["tui", "iot", "sensors"]),
    ("a bookmarklet that summarizes pages", ["browser", "summary", "bookmarklet"]),
    ("a board-game score tracker", ["games", "scoring", "mobile"]),
    ("a CSV-to-chart web tool", ["csv", "charts", "web"]),
)


def gold_blocklist(gold_set: GoldSet) -> set[str]:
    """Tokens (entities + projects) a distractor must avoid, to stay irrelevant.

    Returns the lower-cased entity names and project ids used anywhere in the
    gold set. The generator filters any candidate distractor that shares a token
    with this set so the synthetic inflation cannot accidentally make a gold case
    easier/harder and corrupt the precision measurement.
    """

    block: set[str] = set()
    for m in gold_set.memories:
        for e in m.entities:
            block.add(e.lower())
        scope = Scope.parse(m.scope)
        if scope.project_id:
            block.add(scope.project_id.lower())
    for case in gold_set.injection_cases:
        if case.project:
            block.add(case.project.lower())
        for e in case.entities:
            block.add(e.lower())
    return block


def _collides(entities: Sequence[str], project: str | None, block: set[str]) -> bool:
    toks = {e.lower() for e in entities}
    if project:
        toks.add(project.lower())
    return bool(toks & block)


class DistractorGenerator:
    """Generates plausible-but-irrelevant memories around a gold set (PRD §9).

    Deterministic given a ``seed``; an optional :class:`LLMProvider` switches on
    the LLM-backed path (with template top-up/fallback so it still works offline).
    """

    def __init__(
        self,
        gold_set: GoldSet,
        *,
        llm: LLMProvider | None = None,
        seed: int = 1234,
    ) -> None:
        self.gold_set = gold_set
        self.llm = llm
        self.seed = seed
        self._block = gold_blocklist(gold_set)

    # --- deterministic template path -------------------------------------

    def _template_candidates(self, count: int, rng: random.Random) -> list[MemoryUnit]:
        """Combinatorially expand the fake bank into ``count`` distractors."""

        out: list[MemoryUnit] = []
        now = datetime.now(UTC)
        # Build a large candidate pool then sample, so each multiplier draws a
        # stable, non-trivial subset.
        pool: list[tuple[MemoryType, str, list[str], str | None]] = []

        for proj in _FAKE_PROJECTS:
            for desc, ents in _FAKE_FACT_TOPICS:
                pool.append(
                    (
                        MemoryType.PROJECT_FACT,
                        f"The {proj} project {desc}.",
                        [*ents, proj],
                        proj,
                    )
                )
        for topic, ents in _FAKE_PREFERENCE_TOPICS:
            pool.append(
                (
                    MemoryType.PREFERENCE,
                    f"Prefers a specific {topic} convention.",
                    list(ents),
                    None,
                )
            )
        for idea, ents in _FAKE_IDEAS:
            pool.append(
                (
                    MemoryType.IDEA_SEED,
                    f"Idea: {idea}.",
                    list(ents),
                    None,
                )
            )

        # Drop any pool entry that collides with the gold set.
        pool = [
            (t, c, ents, proj)
            for (t, c, ents, proj) in pool
            if not _collides(ents, proj, self._block)
        ]
        if not pool:  # pragma: no cover - bank is large enough in practice
            return out

        for i in range(count):
            mtype, content, c_ents, c_proj = pool[rng.randrange(len(pool))]
            # Disambiguate content so high multipliers don't all reinforce into
            # one another (distinct content => distinct nodes).
            unique_content = f"{content} [distractor #{i}]"
            scope = Scope.project(c_proj) if c_proj else Scope.global_()
            out.append(
                MemoryUnit(
                    id=f"distractor-{self.seed}-{i}",
                    type=mtype,
                    content=unique_content,
                    scope=scope,
                    entities=list(c_ents),
                    confidence=round(rng.uniform(0.4, 0.85), 3),
                    provenance=Provenance(source="distractor", session_id=f"synthetic:{i}"),
                    valid_from=now - timedelta(days=rng.randint(1, 120)),
                )
            )
        return out

    # --- LLM-backed path -------------------------------------------------

    async def _llm_candidates(self, count: int) -> list[MemoryUnit]:
        """Ask the LLM for fresh plausible-but-irrelevant statements (PRD §9)."""

        assert self.llm is not None
        avoid = sorted(self._block)
        prompt = (
            "Generate plausible-but-IRRELEVANT software memory statements for a "
            "distractor set used to load-test a memory store. They must read like "
            "real developer preferences, project facts, or project ideas, but be "
            "unrelated to these topics/projects (do NOT mention any of them): "
            f"{', '.join(avoid)}.\n"
            f'Return JSON: {{"items": [{{"type": "preference|project_fact|'
            'idea_seed", "content": str, "entities": [str], "project": '
            "str|null}}, ...]}}. "
            f"Produce {count} items."
        )
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "content": {"type": "string"},
                            "entities": {"type": "array", "items": {"type": "string"}},
                            "project": {"type": ["string", "null"]},
                        },
                        "required": ["type", "content"],
                    },
                }
            },
            "required": ["items"],
        }
        try:
            raw = await self.llm.complete_json(prompt, schema=schema)
        except Exception:
            return []
        items = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return []

        out: list[MemoryUnit] = []
        now = datetime.now(UTC)
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            entities = [str(e) for e in item.get("entities", []) if isinstance(e, (str, int))]
            project = item.get("project")
            project = project if isinstance(project, str) and project else None
            if _collides(entities, project, self._block):
                continue
            try:
                mtype = MemoryType(item.get("type", "idea_seed"))
            except ValueError:
                mtype = MemoryType.IDEA_SEED
            scope = (
                Scope.project(project)
                if project and mtype is MemoryType.PROJECT_FACT
                else Scope.global_()
            )
            out.append(
                MemoryUnit(
                    id=f"distractor-llm-{self.seed}-{i}",
                    type=mtype,
                    content=content.strip(),
                    scope=scope,
                    entities=entities or ["misc"],
                    confidence=0.6,
                    provenance=Provenance(source="distractor", session_id=f"llm:{i}"),
                    valid_from=now,
                )
            )
        return out

    # --- public API ------------------------------------------------------

    async def generate(self, count: int) -> list[MemoryUnit]:
        """Generate exactly ``count`` distractor memories (LLM + template top-up)."""

        if count <= 0:
            return []
        rng = random.Random(self.seed)
        result: list[MemoryUnit] = []
        if self.llm is not None:
            result.extend(await self._llm_candidates(count))
            result = result[:count]
        if len(result) < count:
            # Top up / full template path. Re-id template items past any LLM ones
            # so ids stay unique.
            needed = count - len(result)
            templates = self._template_candidates(needed, rng)
            for offset, unit in enumerate(templates):
                unit.id = f"distractor-{self.seed}-{len(result) + offset}"
            result.extend(templates)
        return result[:count]

    async def inflate_store(
        self,
        storage: object,
        *,
        multiplier: int,
    ) -> int:
        """Insert ``multiplier`` distractors per gold memory into ``storage``.

        At ``multiplier=N`` the store grows by ``N * len(gold memories)``
        distractors — the 1x / 10x / 100x scaling of PRD §9. Returns the number
        of distractors actually inserted. Inserts directly via ``upsert_memory``
        (the only write path on the backend Protocol).
        """

        gold_count = len(self.gold_set.memories)
        target = multiplier * gold_count
        distractors = await self.generate(target)
        inserted = 0
        for unit in distractors:
            await storage.upsert_memory(unit)  # type: ignore[attr-defined]
            inserted += 1
        return inserted


def serialize_distractors(units: Sequence[MemoryUnit]) -> str:
    """JSON-serialize generated distractors (for caching a synthetic corpus)."""

    return json.dumps([u.model_dump(mode="json") for u in units], indent=2)
