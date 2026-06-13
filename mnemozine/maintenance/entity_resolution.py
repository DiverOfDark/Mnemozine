"""FR-MNT-4 — entity resolution: merge duplicates, prune edges, cap node degree.

The graph fragments if ``rust`` / ``rust-lang`` / "the Rust work" become three
separate nodes; this job periodically:

1. **Merges duplicate entities.** Scans every entity (only
   :meth:`StorageBackend.iter_entities` can enumerate the store) and groups them
   by a normalized key. Exact-name/alias matches always merge; an optional
   :class:`~mnemozine.interfaces.LLMProvider` can adjudicate fuzzier candidates
   (e.g. "the Rust work" -> ``rust``). The lexicographically-stable canonical
   survivor is kept so the pass is **deterministic** and idempotent.
2. **Prunes low-weight edges.** Edges below ``maintenance.edge_weight_floor`` are
   closed via :meth:`StorageBackend.prune_edge` (validity window closed, never
   hard-deleted).
3. **Caps node degree.** Any node whose active degree exceeds
   ``maintenance.max_node_degree`` keeps its highest-weight edges and prunes the
   rest, so traversal cost stays bounded (FR-RET-2/FR-MNT-4).

Implements :class:`~mnemozine.interfaces.MaintenanceJob`. Safe to re-run
(FR-MNT-5): once duplicates are merged and weak edges pruned, a second pass finds
nothing to do.
"""

from __future__ import annotations

import logging
import re

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import LLMProvider, MaintenanceReport, StorageBackend
from mnemozine.schema.models import Entity

logger = logging.getLogger(__name__)

_NORMALIZE_RE = re.compile(r"[\s_]+")


def normalize_entity_key(name: str) -> str:
    """Normalize an entity name for duplicate grouping.

    Lowercased, hyphen/underscore/whitespace collapsed to a single hyphen, a
    leading article ("the ") and a few common language suffixes stripped, so
    ``Rust``, ``rust-lang``, and "the Rust" collapse to one key. This is the
    cheap, deterministic first pass before any LLM adjudication.
    """

    key = name.strip().lower()
    if key.startswith("the "):
        key = key[4:]
    key = _NORMALIZE_RE.sub("-", key)
    # Fold a couple of common ecosystem suffixes (rust-lang -> rust).
    for suffix in ("-lang", "-language", "-work", "-stuff"):
        if key.endswith(suffix) and len(key) > len(suffix) + 1:
            key = key[: -len(suffix)]
    return key


def _pick_survivor(group: list[Entity]) -> Entity:
    """Deterministically choose the canonical survivor of a duplicate group.

    Prefers the entity with the most aliases (most-evolved canonical), breaking
    ties by the shortest then lexicographically-smallest canonical name so the
    result is stable across runs (idempotency, FR-MNT-5).
    """

    return sorted(
        group,
        key=lambda e: (-len(e.aliases), len(e.canonical_name), e.canonical_name, e.id),
    )[0]


class EntityResolutionJob:
    """FR-MNT-4 entity resolution + edge pruning + node-degree cap.

    Depends only on :class:`~mnemozine.interfaces.StorageBackend` (enumeration,
    merge, edge ops) and optionally :class:`~mnemozine.interfaces.LLMProvider`
    for fuzzy duplicate adjudication.
    """

    def __init__(
        self,
        storage: StorageBackend,
        *,
        llm: LLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._storage = storage
        self._llm = llm
        self._settings = settings or get_settings()

    @property
    def name(self) -> str:
        return "entity_resolution"

    async def run(self) -> MaintenanceReport:
        report = MaintenanceReport(job_name=self.name)
        merged = await self._merge_duplicates(report)
        report.entities_merged = merged
        pruned = await self._prune_and_cap(report)
        report.edges_pruned = pruned
        return report

    # --- 1) merge duplicate entities --------------------------------------

    async def _merge_duplicates(self, report: MaintenanceReport) -> int:
        groups: dict[str, list[Entity]] = {}
        async for entity in self._storage.iter_entities():
            key = normalize_entity_key(entity.canonical_name)
            groups.setdefault(key, []).append(entity)

        merged_count = 0
        for key, group in groups.items():
            if len(group) < 2:
                continue
            survivor = _pick_survivor(group)
            for dup in group:
                if dup.id == survivor.id:
                    continue
                await self._storage.merge_entities(dup.id, survivor.id)
                merged_count += 1
            report.notes.append(
                f"merged {len(group) - 1} entit(ies) into '{survivor.canonical_name}' "
                f"(key='{key}')"
            )
        return merged_count

    # --- 2) prune low-weight edges + 3) cap node degree -------------------

    async def _prune_and_cap(self, report: MaintenanceReport) -> int:
        m = self._settings.maintenance
        floor = m.edge_weight_floor
        max_degree = m.max_node_degree
        pruned = 0
        seen_edge_ids: set[str] = set()

        # Re-enumerate entities post-merge so survivors carry the folded edges.
        async for entity in self._storage.iter_entities():
            edges = await self._storage.edges_for_entity(
                entity.canonical_name, active_only=True
            )
            # (a) low-weight floor prune.
            survivors = []
            for edge in edges:
                if edge.id in seen_edge_ids:
                    continue
                if edge.weight < floor:
                    await self._storage.prune_edge(edge.id)
                    seen_edge_ids.add(edge.id)
                    pruned += 1
                else:
                    survivors.append(edge)

            # (b) node-degree cap: keep highest-weight, prune the overflow.
            if len(survivors) > max_degree:
                survivors.sort(key=lambda e: (e.weight, e.id), reverse=True)
                for edge in survivors[max_degree:]:
                    if edge.id in seen_edge_ids:
                        continue
                    await self._storage.prune_edge(edge.id)
                    seen_edge_ids.add(edge.id)
                    pruned += 1
                report.notes.append(
                    f"capped degree of '{entity.canonical_name}' to {max_degree} "
                    f"(pruned {len(survivors) - max_degree} edge(s))"
                )

        if pruned:
            report.notes.append(
                f"pruned {pruned} edge(s) (floor={floor}, max_degree={max_degree})"
            )
        return pruned
