/**
 * TypeScript wire types — a 1:1 mirror of mnemozine/web/schemas.py (the pydantic
 * contract) and the enums in mnemozine/schema/models.py, mnemozine/activity/models.py,
 * mnemozine/interfaces.py.
 *
 * These are HAND-MIRRORED for the foundation phase so screen agents have stable
 * names immediately. They are intentionally identical to the schema names in
 * API_CONTRACT.md §"TypeScript-facing shape" so a future `openapi-typescript`
 * codegen drops in without churn. Dates are ISO-8601 strings on the wire; `null`
 * on valid_to / last_accessed means active / never (the contract guarantees an
 * `active` boolean so the UI never derives it).
 *
 * DO NOT add UI-only fields here. This file is the contract surface only.
 */

// ---------------------------------------------------------------------------
// Enums (string-literal unions — they serialize as their string values)
// ---------------------------------------------------------------------------

/**
 * DEPRECATED legacy 3-value classification. The MemoryUnit `type` enum was split
 * into a FREE-FORM `category` string + a `cross_ref_candidate` boolean (see the
 * core data-model redesign). `MemoryType` survives ONLY for the eval-bootstrap
 * `proposed_type`/`corrected_type` path, whose Python `Candidate` still carries
 * the legacy enum. New code must NOT branch on it for memories.
 */
export type MemoryType = "preference" | "project_fact" | "idea_seed";
export type Tier = "hot" | "archive";
export type ActivityKind = "ingest" | "extract_decision" | "maintenance" | "injection";
export type WriteDecision = "add" | "reinforce" | "supersede" | "no-op";
/** The controlled scope decision derived from a memory's hierarchical scope. */
export type ScopeDecision = "global" | "project";
export type ScopeKind = "global" | "project";

/** ISO-8601 datetime string (e.g. "2026-06-14T09:30:00Z"). */
export type ISODateTime = string;

/** Legacy eval-bootstrap enum only — NOT a memory facet (categories are free-form). */
export const MEMORY_TYPES: readonly MemoryType[] = ["preference", "project_fact", "idea_seed"];
export const TIERS: readonly Tier[] = ["hot", "archive"];
export const ACTIVITY_KINDS: readonly ActivityKind[] = [
  "ingest",
  "extract_decision",
  "maintenance",
  "injection",
];
export const WRITE_DECISIONS: readonly WriteDecision[] = ["add", "reinforce", "supersede", "no-op"];

// ---------------------------------------------------------------------------
// Shared / common
// ---------------------------------------------------------------------------

export interface Page {
  total: number;
  limit: number;
  offset: number;
}

export interface ValidityWindow {
  valid_from: ISODateTime;
  valid_to: ISODateTime | null;
  active: boolean;
}

export interface Provenance {
  source: string;
  session_id: string;
  chunk_hash: string | null;
  raw_path: string | null;
}

export interface SupersessionLink {
  memory_id: string;
  content: string;
  valid_from: ISODateTime;
  valid_to: ISODateTime | null;
}

// ---------------------------------------------------------------------------
// Memories
// ---------------------------------------------------------------------------

export interface MemoryListItem {
  id: string;
  /** Free-form, emergent category (e.g. "preference", "decision", "gotcha"). */
  category: string;
  /** True if flagged as a cross-reference seed (replaces the old idea_seed type). */
  cross_ref_candidate: boolean;
  /** Controlled scope decision implied by the scope: "global" | "project". */
  scope_decision: ScopeDecision;
  content: string;
  /** Persisted hierarchical scope path: "global" | "project:<P>[/<sub>...]". */
  scope: string;
  entities: string[];
  confidence: number;
  tier: Tier;
  active: boolean;
  valid_from: ISODateTime;
  valid_to: ISODateTime | null;
  last_accessed: ISODateTime | null;
  access_count: number;
  source: string;
}

export interface MemoryDetail {
  id: string;
  category: string;
  cross_ref_candidate: boolean;
  scope_decision: ScopeDecision;
  content: string;
  scope: string;
  entities: string[];
  confidence: number;
  tier: Tier;
  validity: ValidityWindow;
  provenance: Provenance;
  supersedes: SupersessionLink[];
  superseded_by: SupersessionLink[];
  last_accessed: ISODateTime | null;
  access_count: number;
}

export interface MemoryListResponse {
  items: MemoryListItem[];
  page: Page;
}

// ---------------------------------------------------------------------------
// Category facets + scope tree (the discovery surface for the open-ended model)
// ---------------------------------------------------------------------------

export interface CategoryFacet {
  category: string;
  count: number;
}

export interface CategoryFacetsResponse {
  facets: CategoryFacet[];
  total: number;
}

export interface ScopeTreeNode {
  /** This node's own segment label ("global" for the root). */
  segment: string;
  /** Canonical scope string for this node ("global"/"project:<P>"/...). */
  path: string;
  /** Path depth (0 = global root). */
  depth: number;
  /** Memories stored exactly at this scope. */
  count: number;
  /** Memories at this scope plus all descendant sub-scopes (rolled up). */
  total_count: number;
  children: ScopeTreeNode[];
}

export interface ScopeTreeResponse {
  root: ScopeTreeNode;
}

// ---------------------------------------------------------------------------
// Graph explorer
// ---------------------------------------------------------------------------

export interface GraphNode {
  id: string;
  label: string;
  kind: string; // "entity" | "idea_seed" | "memory"
  entity_type: string | null;
  scope: string | null;
  memory_count: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  weight: number;
  active: boolean;
  is_crossref: boolean;
  reason: string | null;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated: boolean;
}

// ---------------------------------------------------------------------------
// Recall playground
// ---------------------------------------------------------------------------

export interface RecallRequest {
  query: string;
  scope?: string | null;
  top_k?: number;
  include_index_preview?: boolean;
}

export interface ScoredMemory {
  memory: MemoryListItem;
  score: number;
  why: string | null;
}

export interface InjectionIndexPreview {
  text: string;
  token_estimate: number;
  token_budget: number;
  /** Global-scope memories in the index (renamed from preference_count). */
  global_count: number;
  /** Project-scope memories in the index (renamed from project_fact_count). */
  project_count: number;
  /** Cross-reference seed hints (renamed from idea_seed_hints). */
  cross_ref_hints: string[];
  entity_tags: string[];
}

export interface RecallResponse {
  query: string;
  scope: string | null;
  results: ScoredMemory[];
  index_preview: InjectionIndexPreview | null;
}

// ---------------------------------------------------------------------------
// Cross-references
// ---------------------------------------------------------------------------

export interface CrossRefItem {
  memory: MemoryListItem;
  score: number;
  reason: string;
  shared_entities: string[];
  suppressed: boolean;
  context_key: string | null;
}

export interface CrossRefResponse {
  items: CrossRefItem[];
  page: Page;
}

// ---------------------------------------------------------------------------
// Activity / Logs
// ---------------------------------------------------------------------------

export interface ActivityEventOut {
  id: string;
  kind: ActivityKind;
  source: string | null;
  summary: string;
  ref_memory_ids: string[];
  session_id: string | null;
  project: string | null;
  detail: Record<string, unknown>;
  ts: ISODateTime;
}

export interface ActivityResponse {
  items: ActivityEventOut[];
  page: Page;
}

// ---------------------------------------------------------------------------
// Maintenance / Ops
// ---------------------------------------------------------------------------

export interface MaintenanceReportOut {
  job_name: string;
  consolidated: number;
  entities_merged: number;
  archived: number;
  edges_pruned: number;
  notes: string[];
}

export interface MaintenanceJobStatus {
  name: string;
  enabled: boolean;
  last_run: ISODateTime | null;
  next_run: ISODateTime | null;
  last_report: MaintenanceReportOut | null;
}

export interface MaintenanceStatusResponse {
  cron: string;
  scheduler_running: boolean;
  jobs: MaintenanceJobStatus[];
}

export interface MaintenanceRunResponse {
  job: string;
  started: boolean;
  report: MaintenanceReportOut | null;
}

export interface MergeCandidate {
  source_id: string;
  source_name: string;
  target_id: string;
  target_name: string;
  similarity: number;
  shared_neighbors: number;
}

export interface MergeCandidatesResponse {
  candidates: MergeCandidate[];
}

/** Valid {job} path values for POST /api/maintenance/{job}/run (API_CONTRACT.md). */
export type MaintenanceJobName =
  | "consolidate"
  | "entity-resolution"
  | "decay"
  | "audit"
  | "migrate-index";

export const MAINTENANCE_JOBS: readonly MaintenanceJobName[] = [
  "consolidate",
  "entity-resolution",
  "decay",
  "audit",
  "migrate-index",
];

// ---------------------------------------------------------------------------
// Eval
// ---------------------------------------------------------------------------

export interface EvalMetric {
  name: string;
  value: number;
  threshold: number | null;
  passed: boolean;
  detail: string | null;
}

export interface EvalSummaryResponse {
  gold_set: string;
  passed: boolean;
  metrics: EvalMetric[];
  ran_at: ISODateTime | null;
}

export interface BootstrapCandidate {
  candidate_id: string;
  content: string;
  proposed_type: MemoryType;
  scope: string;
  entities: string[];
  confidence: number;
  source_session: string;
  label: string; // "unreviewed" | "keep" | "drop"
  corrected_type: MemoryType | null;
}

export interface BootstrapCandidatesResponse {
  candidates: BootstrapCandidate[];
}

export type BootstrapLabel = "keep" | "drop" | "unreviewed";

export interface BootstrapLabelRequest {
  label: BootstrapLabel;
  corrected_type?: MemoryType | null;
}

// ---------------------------------------------------------------------------
// Mutations (HITL corrections)
// ---------------------------------------------------------------------------

export interface MemoryPatchRequest {
  /** Re-label the free-form category (R1 HITL). */
  category?: string | null;
  /** Toggle the cross-reference seed flag (FR-RET-6). */
  cross_ref_candidate?: boolean | null;
  scope?: string | null;
  tier?: Tier | null;
}

export interface SuppressRequest {
  context_key: string;
}

export interface MutationResponse {
  ok: boolean;
  memory: MemoryDetail | null;
  changed: string[];
}

export interface WriteDecisionOut {
  decision: WriteDecision;
  memory_id: string;
  superseded_ids: string[];
}

// ---------------------------------------------------------------------------
// Health / Dashboard
// ---------------------------------------------------------------------------

export interface ComponentHealth {
  name: string; // "falkordb" | "ollama" | "llm"
  status: string; // "ok" | "degraded" | "down" | "unknown"
  detail: string | null;
}

export interface HealthResponse {
  status: string; // "ok" | "degraded" | "down"
  version: string;
  components: ComponentHealth[];
  activity_log_enabled: boolean;
}

export interface StoreStatsResponse {
  total_memories: number;
  /** Count per free-form category (replaces the old fixed by_type). */
  by_category: Record<string, number>;
  /** Count per controlled scope decision ("global" | "project"). */
  by_scope_decision: Record<string, number>;
  by_tier: Record<string, number>;
  by_source: Record<string, number>;
  active_count: number;
  superseded_count: number;
  entity_count: number;
}

/**
 * GET /api/stats/growth → memory-creation trend over a trailing `days` window.
 * Arrays are DENSE / parallel / oldest-first and the same length as `days`
 * (zero-filled by the server); `total === sum(daily)`. Backed by a cheap grouped
 * count of memories by DAY of valid_from — real retroactive data, independent of
 * the activity log.
 */
export interface GrowthResponse {
  /** Day labels (YYYY-MM-DD), oldest → newest, length == window. */
  days: string[];
  /** Memories created per day, oldest → newest (parallel to `days`). */
  daily: number[];
  /** Running cumulative total of `daily`, oldest → newest. */
  cumulative: number[];
  /** Sum of `daily` over the window. */
  total: number;
}

// ---------------------------------------------------------------------------
// Query-param shapes (the filter objects the hooks accept)
// ---------------------------------------------------------------------------

export interface MemoriesQuery {
  /** Free-form category filter (replaces the old fixed type enum filter). */
  category?: string;
  scope?: string;
  tier?: Tier;
  entity?: string;
  source?: string;
  /** tri-state: true=active only, false=superseded only, undefined=both. */
  active?: boolean;
  q?: string;
  limit?: number;
  offset?: number;
}

export interface GraphQuery {
  scope?: string;
  entity?: string;
  entity_type?: string;
  depth?: number;
  include_crossrefs?: boolean;
  limit?: number;
}

export interface CrossRefsQuery {
  project?: string;
  entity?: string;
  include_suppressed?: boolean;
  limit?: number;
  offset?: number;
}

export interface ActivityQuery {
  /** repeatable in the URL: ?kind=ingest&kind=maintenance */
  kind?: ActivityKind[];
  source?: string;
  session_id?: string;
  project?: string;
  ref_memory_id?: string;
  since?: ISODateTime;
  until?: ISODateTime;
  limit?: number;
  offset?: number;
}
