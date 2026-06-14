"""Shared projection helpers for the READ routes (WEBUI BE-READ stream).

The wire schemas in :mod:`mnemozine.web.schemas` are a flat JSON projection of the
§7 domain models (:mod:`mnemozine.schema.models`). The read endpoints all need the
same few projections — a :class:`~mnemozine.schema.models.MemoryUnit` to a
:class:`~mnemozine.web.schemas.MemoryListItem`, the validity window, the
provenance, and the (derived) supersession chain — so they live here once rather
than being re-implemented per route.

Everything here is pure and offline (no FalkorDB / LLM): it operates on
already-fetched :class:`MemoryUnit`s and on the ``StorageBackend`` Protocol's
enumeration/traversal methods only (never a concrete backend), so the routes and
their tests run identically against the live Graphiti backend and the in-memory
fake.

Why iterate rather than key-read for detail/counts? The
:class:`~mnemozine.interfaces.StorageBackend` Protocol exposes
:meth:`iter_memories` as its only whole-store entry point — there is no
``get_memory`` / ``count`` on the Protocol (the concrete backends have a private
``get_memory`` but the fake does not), so the portable read path streams
``iter_memories`` and filters in Python. The store is single-operator and small
(PRD §1), so this is acceptable for the console.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from mnemozine.interfaces import StorageBackend
from mnemozine.schema.models import GLOBAL_SCOPE, MemoryUnit, Scope
from mnemozine.web.schemas import (
    MemoryDetail,
    MemoryListItem,
    Provenance,
    ScopeTreeNode,
    SupersessionLink,
    ValidityWindow,
)


def normalize_scope(scope: str | None) -> str | None:
    """Normalize a scope filter string to its persisted form.

    Accepts the persisted forms ``'global'`` / ``'project:<id>'`` and the
    convenience bare project id (``'<id>'`` -> ``'project:<id>'``, per the API
    contract). Returns the canonical persisted string, or ``None`` when no scope
    filter was supplied.
    """

    if scope is None:
        return None
    value = scope.strip()
    if not value:
        return None
    if value == Scope.global_().as_str():
        return value
    if value.startswith("project:"):
        return value
    # Bare project id convenience form.
    return Scope.project(value).as_str()


def scope_to_obj(scope: str | None) -> Scope | None:
    """Parse a (possibly bare) scope filter string into a :class:`Scope`."""

    normalized = normalize_scope(scope)
    if normalized is None:
        return None
    return Scope.parse(normalized)


def memory_to_list_item(memory: MemoryUnit) -> MemoryListItem:
    """Project a :class:`MemoryUnit` onto the flat table-row wire model."""

    return MemoryListItem(
        id=memory.id,
        category=memory.category,
        cross_ref_candidate=memory.cross_ref_candidate,
        scope_decision=memory.scope_decision,
        content=memory.content,
        scope=memory.scope.as_str(),
        entities=list(memory.entities),
        confidence=memory.confidence,
        tier=memory.tier,
        active=memory.is_active,
        valid_from=memory.valid_from,
        valid_to=memory.valid_to,
        last_accessed=memory.last_accessed,
        access_count=memory.access_count,
        source=memory.provenance.source,
    )


def _supersession_link(memory: MemoryUnit) -> SupersessionLink:
    """Project a related unit onto one end of a supersession chain link."""

    return SupersessionLink(
        memory_id=memory.id,
        content=memory.content,
        valid_from=memory.valid_from,
        valid_to=memory.valid_to,
    )


def _overlaps(a: Sequence[str], b: Sequence[str]) -> bool:
    return bool({e.lower() for e in a} & {e.lower() for e in b})


def derive_supersession_chain(
    memory: MemoryUnit, universe: Sequence[MemoryUnit]
) -> tuple[list[SupersessionLink], list[SupersessionLink]]:
    """Derive the (supersedes, superseded_by) chain for ``memory`` (PRD §2/§4.3).

    Supersession is not a stored edge in the §7 model — a supersede simply closes
    the older unit's validity window (``valid_to = now``) and inserts the new one
    active (FR-MNT-1). So the chain is reconstructed temporally from the units in
    the **same scope sharing at least one entity** (the FR-MNT-1 comparison set):

    * ``supersedes`` — older closed units whose window closed at/around this unit's
      ``valid_from`` (this unit replaced them);
    * ``superseded_by`` — units that became active at/around this unit's
      ``valid_to`` (they replaced this one), only when this unit is itself closed.

    The match is on validity-window adjacency (a closed older window meeting a
    newer ``valid_from``), which is exactly how a real supersede leaves the graph.
    Links are ordered newest-first so the UI renders the most recent step first.
    """

    scope = memory.scope.as_str()
    related = [
        m
        for m in universe
        if m.id != memory.id
        and m.scope.as_str() == scope
        and m.category == memory.category
        and _overlaps(m.entities, memory.entities)
    ]

    supersedes: list[MemoryUnit] = []
    superseded_by: list[MemoryUnit] = []

    for other in related:
        # `other` is an older fact this unit replaced: it is closed, started before
        # this unit, and its window closed no later than this unit became valid.
        if (
            other.valid_to is not None
            and other.valid_from <= memory.valid_from
            and other.valid_to <= memory.valid_from
        ):
            supersedes.append(other)
            continue
        # `other` replaced this unit: only meaningful when this unit is closed, and
        # `other` started at/after this unit's window closed.
        if memory.valid_to is not None and other.valid_from >= memory.valid_to:
            superseded_by.append(other)

    supersedes.sort(key=lambda m: m.valid_from, reverse=True)
    superseded_by.sort(key=lambda m: m.valid_from)
    return (
        [_supersession_link(m) for m in supersedes],
        [_supersession_link(m) for m in superseded_by],
    )


def memory_to_detail(
    memory: MemoryUnit, universe: Sequence[MemoryUnit]
) -> MemoryDetail:
    """Project a :class:`MemoryUnit` (+ its peers) onto the full detail wire model.

    ``universe`` is the set of units the supersession chain is derived against —
    the same-scope/entity neighborhood (or the whole store) the caller already
    fetched. Carries the validity window, provenance link, and the derived
    supersession chain — the signature temporal feature, first-class (PRD §2).
    """

    supersedes, superseded_by = derive_supersession_chain(memory, universe)
    return MemoryDetail(
        id=memory.id,
        category=memory.category,
        cross_ref_candidate=memory.cross_ref_candidate,
        scope_decision=memory.scope_decision,
        content=memory.content,
        scope=memory.scope.as_str(),
        entities=list(memory.entities),
        confidence=memory.confidence,
        tier=memory.tier,
        validity=ValidityWindow(
            valid_from=memory.valid_from,
            valid_to=memory.valid_to,
            active=memory.is_active,
        ),
        provenance=Provenance(
            source=memory.provenance.source,
            session_id=memory.provenance.session_id,
            chunk_hash=memory.provenance.chunk_hash,
            raw_path=memory.provenance.raw_path,
        ),
        supersedes=supersedes,
        superseded_by=superseded_by,
        last_accessed=memory.last_accessed,
        access_count=memory.access_count,
    )


def build_scope_tree(memories: Iterable[MemoryUnit]) -> ScopeTreeNode:
    """Build the hierarchical project/sub-scope tree with per-node counts.

    Walks every memory's stored :class:`~mnemozine.schema.models.Scope` path and
    materializes the ordered-segment hierarchy as a tree rooted at ``global``.
    Each :class:`~mnemozine.web.schemas.ScopeTreeNode` records:

    * ``count``       — memories stored *exactly* at that scope, and
    * ``total_count`` — that node plus every descendant (the ancestor-composed
      roll-up a query at that scope sees from this subtree).

    Intermediate nodes are synthesized even when no memory is stored exactly
    there (a memory at ``project:P/auth/api`` creates the ``project:P`` and
    ``project:P/auth`` nodes with ``count=0``), so the navigator can always drill
    the full path. Children are sorted by descending roll-up then segment name.
    """

    # A mutable scratch tree keyed by full scope-path string.
    @dataclass
    class _N:
        segment: str
        path: str
        depth: int
        count: int = 0
        children: dict[str, _N] = field(default_factory=dict)

    root = _N(segment=GLOBAL_SCOPE, path=GLOBAL_SCOPE, depth=0)

    for mem in memories:
        segments = mem.scope.segments
        node = root
        if not segments:
            node.count += 1
            continue
        for i, seg in enumerate(segments):
            child = node.children.get(seg)
            if child is None:
                path = Scope(segments=segments[: i + 1]).as_str()
                child = _N(segment=seg, path=path, depth=i + 1)
                node.children[seg] = child
            node = child
        node.count += 1

    def _to_node(scratch: _N) -> ScopeTreeNode:
        children = [_to_node(c) for c in scratch.children.values()]
        total = scratch.count + sum(c.total_count for c in children)
        children.sort(key=lambda c: (-c.total_count, c.segment))
        return ScopeTreeNode(
            segment=scratch.segment,
            path=scratch.path,
            depth=scratch.depth,
            count=scratch.count,
            total_count=total,
            children=children,
        )

    return _to_node(root)


async def collect_memories(
    storage: StorageBackend, *, scope: Scope | None = None
) -> list[MemoryUnit]:
    """Stream the whole (optionally scope-bounded) store into a list.

    Uses the only whole-store enumeration entry point on the Protocol
    (:meth:`StorageBackend.iter_memories`), leaving its ``active_only`` default
    (False) so the table can show superseded rows too; the caller applies the
    remaining filters.
    """

    out: list[MemoryUnit] = []
    async for memory in storage.iter_memories(scope=scope):
        out.append(memory)
    return out


__all__ = [
    "normalize_scope",
    "scope_to_obj",
    "memory_to_list_item",
    "memory_to_detail",
    "derive_supersession_chain",
    "build_scope_tree",
    "collect_memories",
]
