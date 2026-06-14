"""The WebUI wire contract — pydantic request/response models (WEBUI PRD §6).

These models ARE the API contract. Everything downstream (the Phase-2 route
bodies, the React/TypeScript SPA, the generated OpenAPI) is written against the
shapes here, so they are deliberately explicit and self-documenting. They are a
**projection** of the §7 domain models (:mod:`mnemozine.schema.models`) onto a
flat, JSON-friendly wire surface — the UI never sees a raw ``MemoryUnit``; it
sees a :class:`MemoryListItem` / :class:`MemoryDetail`.

Naming convention (honored by the frontend codegen):

* ``*ListItem``  — a compact row for a table.
* ``*Detail``    — the full single-entity view.
* ``*Request``   — a request body (mutations, recall).
* ``*Response``  — a top-level response envelope (lists carry paging).

Signature feature (PRD §2): temporal **validity windows** + **supersession** are
first-class everywhere — see :class:`ValidityWindow`, :class:`SupersessionLink`,
and the ``active`` / ``superseded`` flags on memory items.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from mnemozine.activity.models import ActivityKind
from mnemozine.interfaces import WriteDecision
from mnemozine.schema.models import MemoryType, ScopeDecision, Tier

# ---------------------------------------------------------------------------
# Shared / common
# ---------------------------------------------------------------------------


class Page(BaseModel):
    """Pagination envelope echoed back on every list response."""

    total: int = Field(description="Total matching rows (before limit/offset).")
    limit: int = Field(description="Page size used.")
    offset: int = Field(description="Offset used.")


class ValidityWindow(BaseModel):
    """A temporal validity window ``(valid_from, valid_to)`` (FR-STO-1, PRD §2).

    ``valid_to=None`` means the fact is currently active; a timestamp means it was
    superseded/closed at that time and has left the hot retrieval path.
    """

    valid_from: datetime = Field(description="When the fact became valid.")
    valid_to: datetime | None = Field(
        default=None,
        description="None = active/current; a timestamp = superseded/closed.",
    )
    active: bool = Field(description="True when valid_to is None (still current).")


class Provenance(BaseModel):
    """Where a memory came from (FR-EXT-4) — the detail-screen provenance link."""

    source: str = Field(description="Originating source, e.g. 'claude_code'.")
    session_id: str = Field(description="Originating session id.")
    chunk_hash: str | None = Field(default=None, description="Content hash of the source chunk.")
    raw_path: str | None = Field(default=None, description="Path to the raw transcript (archive).")


class SupersessionLink(BaseModel):
    """One end of a supersession relationship (UC-2, PRD §2/§4.3).

    A memory that ``replaced`` an older fact, or was ``replaced_by`` a newer one.
    Carries enough to render the chain and link to the other unit.
    """

    memory_id: str = Field(description="The id of the linked (replaced/replacing) memory.")
    content: str = Field(description="The linked memory's content snippet.")
    valid_from: datetime = Field(description="The linked memory's validity start.")
    valid_to: datetime | None = Field(default=None, description="The linked memory's validity end.")


# ---------------------------------------------------------------------------
# Memories (PRD §4.2 / §4.3)
# ---------------------------------------------------------------------------


class MemoryListItem(BaseModel):
    """A compact memory row for the Memories table (PRD §4.2)."""

    id: str
    category: str = Field(
        description="Free-form emergent category (e.g. 'preference', 'decision', 'gotcha')."
    )
    cross_ref_candidate: bool = Field(
        default=False, description="True if flagged as a cross-reference seed (FR-RET-6)."
    )
    scope_decision: ScopeDecision = Field(
        description="Controlled scope decision implied by the scope: global | project."
    )
    content: str = Field(description="The distilled memory statement (snippet in the table).")
    scope: str = Field(description="Persisted scope string: 'global' or 'project:<id>'.")
    entities: list[str] = Field(default_factory=list, description="Linked entity names.")
    confidence: float = Field(description="Extraction confidence (0..1).")
    tier: Tier = Field(description="hot | archive.")
    active: bool = Field(description="True if the validity window is open (current).")
    valid_from: datetime
    valid_to: datetime | None = None
    last_accessed: datetime | None = Field(default=None, description="Last retrieval time.")
    access_count: int = Field(description="Times retrieved (decay ranking).")
    source: str = Field(description="Originating source (from provenance).")


class MemoryDetail(BaseModel):
    """The full single-memory view (PRD §4.3): content + provenance + validity + chain."""

    id: str
    category: str = Field(description="Free-form emergent category.")
    cross_ref_candidate: bool = Field(
        default=False, description="True if flagged as a cross-reference seed (FR-RET-6)."
    )
    scope_decision: ScopeDecision = Field(
        description="Controlled scope decision implied by the scope: global | project."
    )
    content: str
    scope: str
    entities: list[str] = Field(default_factory=list)
    confidence: float
    tier: Tier
    validity: ValidityWindow = Field(
        description="The temporal validity window (signature feature)."
    )
    provenance: Provenance = Field(description="Link back to the source session/message.")
    supersedes: list[SupersessionLink] = Field(
        default_factory=list,
        description="Older facts this memory replaced (replaced chain).",
    )
    superseded_by: list[SupersessionLink] = Field(
        default_factory=list,
        description="Newer facts that replaced this one (replaced-by chain).",
    )
    last_accessed: datetime | None = None
    access_count: int = 0


class MemoryListResponse(BaseModel):
    """Paged Memories table response (PRD §4.2)."""

    items: list[MemoryListItem]
    page: Page


# ---------------------------------------------------------------------------
# Category facets + scope tree (the discovery surface for the open-ended model)
# ---------------------------------------------------------------------------


class CategoryFacet(BaseModel):
    """One discovered free-form category with its memory count.

    Categories are now open-ended (the old 3-value ``MemoryType`` enum is gone),
    so the UI cannot ship a fixed list — it must discover the in-use categories
    from the store. Each facet is a ``(category, count)`` pair the frontend
    renders as a filter chip in the dynamic CATEGORY facet.
    """

    category: str = Field(description="The free-form, normalized category slug.")
    count: int = Field(description="How many memories carry this category.")


class CategoryFacetsResponse(BaseModel):
    """The discovered category facets (distinct categories + counts).

    Backs the dynamic CATEGORY filter that replaced the fixed type enum: the
    frontend lists exactly the categories that exist in the store with their
    counts, ordered most-frequent first.
    """

    facets: list[CategoryFacet] = Field(default_factory=list)
    total: int = Field(default=0, description="Number of distinct categories.")


class ScopeTreeNode(BaseModel):
    """One node in the hierarchical scope tree (a project / sub-scope segment).

    Mirrors :class:`~mnemozine.schema.models.Scope`'s ordered path: a node's
    :attr:`path` is the full canonical scope string (``'global'`` /
    ``'project:<P>'`` / ``'project:<P>/<sub>...'``) so the frontend can drill in
    and select it directly. :attr:`count` is the number of memories stored
    *exactly* at this scope; :attr:`total_count` rolls up this node plus all of
    its descendants (the ancestor-composed view a query at this scope would see
    from this subtree). Children are the immediate sub-scopes.
    """

    segment: str = Field(
        description="This node's own segment label ('global' for the root)."
    )
    path: str = Field(
        description="Canonical scope string for this node ('global'/'project:<P>'/...)."
    )
    depth: int = Field(description="Path depth (0 = global root).")
    count: int = Field(default=0, description="Memories stored exactly at this scope.")
    total_count: int = Field(
        default=0,
        description="Memories at this scope plus all descendant sub-scopes (rolled up).",
    )
    children: list[ScopeTreeNode] = Field(
        default_factory=list, description="Immediate sub-scope nodes."
    )


class ScopeTreeResponse(BaseModel):
    """The project/sub-scope hierarchy with per-node counts (the scope navigator).

    A single :attr:`root` (``global``) holds the whole tree: each project is a
    child of the root and each sub-scope is a child of its project. Powers the
    SCOPE TREE navigator that replaced the flat scope filter — selecting a node
    shows its ancestor-composed memories.
    """

    root: ScopeTreeNode = Field(description="The global root of the scope tree.")


# ---------------------------------------------------------------------------
# Graph explorer (PRD §4.4)
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """A node in the graph explorer (entity or idea-seed) — Cytoscape-friendly."""

    id: str = Field(description="Stable node id (entity id or memory id).")
    label: str = Field(description="Display label (canonical name / content snippet).")
    kind: str = Field(description="Node kind: 'entity' | 'idea_seed' | 'memory'.")
    entity_type: str | None = Field(default=None, description="Optional entity category.")
    scope: str | None = Field(default=None, description="Scope for memory/idea-seed nodes.")
    memory_count: int = Field(default=0, description="How many memories link this node.")


class GraphEdge(BaseModel):
    """A weighted relationship edge between two graph nodes (FR-EXT-2)."""

    id: str
    source: str = Field(description="Source node id (Cytoscape convention).")
    target: str = Field(description="Target node id.")
    relation: str = Field(description="Relation label.")
    weight: float = Field(description="Edge weight (FR-MNT-4 pruning).")
    active: bool = Field(description="True if the edge's validity window is open.")
    is_crossref: bool = Field(
        default=False,
        description="True if this edge is a surfaced cross-reference connection (PRD §4.4).",
    )
    reason: str | None = Field(
        default=None,
        description="Human-readable reason for a cross-reference edge (FR-RET-6).",
    )


class GraphResponse(BaseModel):
    """A scoped subgraph for the explorer (PRD §4.4 / §6 GET /api/graph)."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool = Field(
        default=False,
        description="True if the subgraph was capped (more nodes/edges exist).",
    )


# ---------------------------------------------------------------------------
# Recall playground (PRD §4.5 / §6 POST /api/recall)
# ---------------------------------------------------------------------------


class RecallRequest(BaseModel):
    """A recall() query from the playground (PRD §4.5)."""

    query: str = Field(description="The free-text recall query.")
    scope: str | None = Field(
        default=None,
        description=(
            "Optional scope: 'global', 'project:<id>', or a bare project id. "
            "None = default composed scope."
        ),
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return.")
    include_index_preview: bool = Field(
        default=True,
        description="Also return the ~500-token SessionStart index preview (FR-RET-3).",
    )


class ScoredMemory(BaseModel):
    """One ranked recall result with its relevance score + a why-it-surfaced note."""

    memory: MemoryListItem = Field(description="The retrieved memory (compact projection).")
    score: float = Field(description="Relevance score (higher = more relevant).")
    why: str | None = Field(
        default=None,
        description="Optional explanation of why it surfaced (matched entities / lexical).",
    )


class InjectionIndexPreview(BaseModel):
    """The compact, token-budgeted SessionStart index that would be injected (FR-RET-3)."""

    text: str = Field(description="The final injected text (already truncated to budget).")
    token_estimate: int = Field(description="Estimated token size of the index.")
    token_budget: int = Field(description="The configured hard cap (inject.token_budget).")
    global_count: int = Field(default=0, description="Global-scope memories in the index.")
    project_count: int = Field(default=0, description="Project-scope memories in the index.")
    cross_ref_hints: list[str] = Field(
        default_factory=list, description="One-line cross-reference seed hints (FR-RET-6)."
    )
    entity_tags: list[str] = Field(default_factory=list)


class RecallResponse(BaseModel):
    """The recall playground result: ranked memories + the injection preview (PRD §4.5)."""

    query: str
    scope: str | None = None
    results: list[ScoredMemory]
    index_preview: InjectionIndexPreview | None = Field(
        default=None,
        description="The SessionStart index preview, when requested (FR-RET-3 debugging).",
    )


# ---------------------------------------------------------------------------
# Cross-references (PRD §4.4 / §6 GET /api/crossrefs)
# ---------------------------------------------------------------------------


class CrossRefItem(BaseModel):
    """A surfaced serendipitous connection with its mandatory reason (FR-RET-6)."""

    memory: MemoryListItem = Field(description="The related memory/idea-seed surfaced.")
    score: float = Field(description="Connection relevance (above relevance_threshold).")
    reason: str = Field(description="Human-readable reason — never empty (FR-RET-6).")
    shared_entities: list[str] = Field(default_factory=list)
    suppressed: bool = Field(
        default=False,
        description="True if this connection was dismissed/suppressed (R2).",
    )
    context_key: str | None = Field(
        default=None,
        description="The working-context key a suppression applies to (FR-RET-6).",
    )


class CrossRefResponse(BaseModel):
    """Cross-references for a context + the suppression list (PRD §4.7)."""

    items: list[CrossRefItem]
    page: Page


# ---------------------------------------------------------------------------
# Activity / Logs (PRD §4.6 / §6 GET /api/activity) — mirrors the activity log
# ---------------------------------------------------------------------------


class ActivityEventOut(BaseModel):
    """One activity-feed entry (PRD §4.6) — wire projection of an ActivityEvent."""

    id: str
    kind: ActivityKind = Field(description="ingest | extract_decision | maintenance | injection.")
    source: str | None = None
    summary: str = Field(description="One-line human-readable summary.")
    ref_memory_ids: list[str] = Field(
        default_factory=list, description="Affected memory ids (UI links to them)."
    )
    session_id: str | None = None
    project: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict, description="Kind-specific extras.")
    ts: datetime = Field(description="Event timestamp (UTC).")


class ActivityResponse(BaseModel):
    """Paged activity feed (PRD §4.6)."""

    items: list[ActivityEventOut]
    page: Page


# ---------------------------------------------------------------------------
# Maintenance / Ops (PRD §4.7 / §6 GET /api/maintenance)
# ---------------------------------------------------------------------------


class MaintenanceJobStatus(BaseModel):
    """The schedule/last-run status of one maintenance job (PRD §4.7)."""

    name: str = Field(description="Stable job name (consolidate/decay/entity-resolution/...).")
    enabled: bool = Field(default=True, description="Whether the job is in the scheduled set.")
    last_run: datetime | None = Field(default=None, description="Last completed run time.")
    next_run: datetime | None = Field(default=None, description="Next scheduled run time.")
    last_report: MaintenanceReportOut | None = Field(
        default=None, description="Summary of the last run."
    )


class MaintenanceReportOut(BaseModel):
    """Summary counts of one maintenance pass (mirrors MaintenanceReport, FR-MNT-5)."""

    job_name: str
    consolidated: int = 0
    entities_merged: int = 0
    archived: int = 0
    edges_pruned: int = 0
    notes: list[str] = Field(default_factory=list)


class MaintenanceStatusResponse(BaseModel):
    """Scheduler + per-job status (PRD §4.7 / §6 GET /api/maintenance)."""

    cron: str = Field(description="The configured maintenance cron expression.")
    scheduler_running: bool = Field(default=False)
    jobs: list[MaintenanceJobStatus] = Field(default_factory=list)


class MaintenanceRunResponse(BaseModel):
    """The result of triggering a maintenance job on demand (POST /api/maintenance/{job}/run)."""

    job: str
    started: bool
    report: MaintenanceReportOut | None = None


class MergeCandidate(BaseModel):
    """An entity-resolution merge candidate awaiting HITL review (PRD §4.7, FR-MNT-4)."""

    source_id: str
    source_name: str
    target_id: str
    target_name: str
    similarity: float = Field(description="How similar the two entities are (0..1).")
    shared_neighbors: int = Field(default=0, description="Count of shared neighbor entities.")


class MergeCandidatesResponse(BaseModel):
    """Entity-resolution review queue (PRD §4.7)."""

    candidates: list[MergeCandidate]


# ---------------------------------------------------------------------------
# Eval (PRD §4.8 / §6 GET /api/eval + bootstrap)
# ---------------------------------------------------------------------------


class EvalMetric(BaseModel):
    """One eval metric result (precision / classifier-accuracy / latency / ...)."""

    name: str = Field(description="Metric name, e.g. 'injection_precision'.")
    value: float = Field(description="Measured value.")
    threshold: float | None = Field(default=None, description="Pass threshold, if any.")
    passed: bool = Field(description="Whether the metric passed.")
    detail: str | None = Field(default=None, description="Optional human note.")


class EvalSummaryResponse(BaseModel):
    """The eval results summary (PRD §4.8 / §6 GET /api/eval)."""

    gold_set: str = Field(description="Name of the gold set evaluated against.")
    passed: bool = Field(description="Overall pass/fail.")
    metrics: list[EvalMetric] = Field(default_factory=list)
    ran_at: datetime | None = Field(default=None, description="When this report was produced.")


class BootstrapCandidate(BaseModel):
    """One auto-proposed eval bootstrap candidate to label in the browser (F4, PRD §4.8)."""

    candidate_id: str
    content: str
    proposed_type: MemoryType
    scope: str
    entities: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    source_session: str = ""
    label: str = Field(default="unreviewed", description="unreviewed | keep | drop.")
    corrected_type: MemoryType | None = Field(
        default=None, description="Operator-corrected type (R1 HITL)."
    )


class BootstrapCandidatesResponse(BaseModel):
    """The bootstrap labeling queue (F4 — label in the browser, PRD §4.8)."""

    candidates: list[BootstrapCandidate]


class BootstrapLabelRequest(BaseModel):
    """Label one bootstrap candidate from the browser (F4)."""

    label: str = Field(description="keep | drop | unreviewed.")
    corrected_type: MemoryType | None = Field(
        default=None, description="Optional reclassification of the candidate."
    )


# ---------------------------------------------------------------------------
# Mutations (HITL corrections — PRD §4.3 / §6 PATCH /api/memories/{id})
# ---------------------------------------------------------------------------


class MemoryPatchRequest(BaseModel):
    """A HITL correction to one memory (reclassify / re-scope / tier) — PRD §4.3, R1/R5.

    All fields optional; only the supplied ones are applied. Content is NOT
    editable (PRD §7 out-of-scope) — only classification, scope, and tier. The old
    ``type`` reclassify is split into the free-form ``category`` re-label and the
    ``cross_ref_candidate`` flag (the controlled scope decision follows the scope).
    """

    category: str | None = Field(default=None, description="Re-label free-form category (R1 HITL).")
    cross_ref_candidate: bool | None = Field(
        default=None, description="Toggle the cross-reference seed flag (FR-RET-6)."
    )
    scope: str | None = Field(default=None, description="Re-scope ('global'/'project:<id>').")
    tier: Tier | None = Field(default=None, description="Archive (archive) / restore (hot).")


class SuppressRequest(BaseModel):
    """Suppress a cross-reference suggestion in a working context (R2, PRD §4.7)."""

    context_key: str = Field(description="The working-context key the dismissal applies to.")


class MutationResponse(BaseModel):
    """The result of a mutation: the affected entity + what changed."""

    ok: bool = Field(default=True)
    memory: MemoryDetail | None = Field(default=None, description="The updated memory, if any.")
    changed: list[str] = Field(default_factory=list, description="Field names that changed.")


class WriteDecisionOut(BaseModel):
    """The outcome of a write (echoed where a mutation triggers the 4-way write)."""

    decision: WriteDecision
    memory_id: str
    superseded_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Health / Dashboard (PRD §4.1 / §6 GET /api/health)
# ---------------------------------------------------------------------------


class ComponentHealth(BaseModel):
    """Health of one infra dependency (FalkorDB / Ollama / LLM endpoint) — PRD §4.1."""

    name: str = Field(description="Component name: 'falkordb' | 'ollama' | 'llm'.")
    status: str = Field(description="'ok' | 'degraded' | 'down' | 'unknown'.")
    detail: str | None = Field(default=None, description="Optional human detail (latency/error).")


class HealthResponse(BaseModel):
    """Overall WebUI + infra health (PRD §4.1 / §6 GET /api/health)."""

    status: str = Field(description="Overall: 'ok' | 'degraded' | 'down'.")
    version: str = Field(description="Mnemozine package version.")
    components: list[ComponentHealth] = Field(default_factory=list)
    activity_log_enabled: bool = Field(
        default=False, description="Whether the persisted activity log is on (Q3)."
    )


class StoreStatsResponse(BaseModel):
    """Top-bar live store stats + Dashboard totals (PRD §4.1)."""

    total_memories: int = 0
    by_category: dict[str, int] = Field(
        default_factory=dict, description="Count per free-form category."
    )
    by_scope_decision: dict[str, int] = Field(
        default_factory=dict, description="Count per controlled scope decision (global/project)."
    )
    by_tier: dict[str, int] = Field(default_factory=dict, description="hot vs archive counts.")
    by_source: dict[str, int] = Field(default_factory=dict, description="Count per ingest source.")
    active_count: int = Field(default=0, description="Active (open validity window) memories.")
    superseded_count: int = Field(default=0, description="Superseded (closed window) memories.")
    entity_count: int = 0


# Resolve forward references (MaintenanceJobStatus -> MaintenanceReportOut;
# ScopeTreeNode -> ScopeTreeNode for the recursive children list).
MaintenanceJobStatus.model_rebuild()
ScopeTreeNode.model_rebuild()


class ScopeKind(str, Enum):
    """Helper enum exposed in the contract for the frontend scope filter."""

    GLOBAL = "global"
    PROJECT = "project"


__all__ = [
    "Page",
    "ValidityWindow",
    "Provenance",
    "SupersessionLink",
    "MemoryListItem",
    "MemoryDetail",
    "MemoryListResponse",
    "CategoryFacet",
    "CategoryFacetsResponse",
    "ScopeTreeNode",
    "ScopeTreeResponse",
    "GraphNode",
    "GraphEdge",
    "GraphResponse",
    "RecallRequest",
    "ScoredMemory",
    "InjectionIndexPreview",
    "RecallResponse",
    "CrossRefItem",
    "CrossRefResponse",
    "ActivityEventOut",
    "ActivityResponse",
    "MaintenanceJobStatus",
    "MaintenanceReportOut",
    "MaintenanceStatusResponse",
    "MaintenanceRunResponse",
    "MergeCandidate",
    "MergeCandidatesResponse",
    "EvalMetric",
    "EvalSummaryResponse",
    "BootstrapCandidate",
    "BootstrapCandidatesResponse",
    "BootstrapLabelRequest",
    "MemoryPatchRequest",
    "SuppressRequest",
    "MutationResponse",
    "WriteDecisionOut",
    "ComponentHealth",
    "HealthResponse",
    "StoreStatsResponse",
    "ScopeKind",
]
