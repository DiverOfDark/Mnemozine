/**
 * TanStack Query hooks — one per endpoint in API_CONTRACT.md, plus mutation hooks.
 *
 * THIS IS THE FRONTEND DATA CONTRACT. Screen agents import these hooks by name and
 * never touch src/api/client.ts or the router/theme. The hook names map 1:1 to the
 * PRD screens:
 *
 *   Dashboard / top bar : useHealth, useStats
 *   Memories table      : useMemories
 *   Memory detail       : useMemory, usePatchMemory (reclassify / re-scope / archive-restore)
 *   Graph explorer      : useGraph
 *   Recall playground   : useRecall (query-driven) + useRecallMutation (button-driven)
 *   Cross-references    : useCrossRefs, useSuppressCrossRef
 *   Activity / Logs     : useActivity
 *   Maintenance / Ops   : useMaintenance, useMergeCandidates, useRunMaintenance
 *   Eval                : useEval, useBootstrapCandidates, useLabelBootstrap, useFinishBootstrap
 *
 * Every list hook accepts its typed *Query filter object and returns the *Response
 * envelope (items + page). Endpoints, params and bodies are exactly the API_CONTRACT.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationResult,
  type UseQueryOptions,
  type UseQueryResult,
} from "@tanstack/react-query";

import { api } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import type {
  ActivityQuery,
  ActivityResponse,
  BootstrapCandidate,
  BootstrapCandidatesResponse,
  BootstrapLabelRequest,
  CategoryFacetsResponse,
  CrossRefResponse,
  CrossRefsQuery,
  EvalSummaryResponse,
  GraphQuery,
  GraphResponse,
  GrowthResponse,
  HealthResponse,
  MaintenanceJobName,
  MaintenanceRunResponse,
  MaintenanceStatusResponse,
  MemoriesQuery,
  MemoryDetail,
  MemoryListResponse,
  MemoryPatchRequest,
  MergeCandidatesResponse,
  MutationResponse,
  RecallRequest,
  RecallResponse,
  ScopeTreeResponse,
  StoreStatsResponse,
  SuppressRequest,
} from "@/api/types";

/** Options passthrough so screens can tune enabled/refetchInterval per-mount. */
type QueryOpts<T> = Omit<UseQueryOptions<T, Error, T>, "queryKey" | "queryFn">;

// ---------------------------------------------------------------------------
// Health & stats (Dashboard / top bar)
// ---------------------------------------------------------------------------

/** GET /api/health → HealthResponse (infra components + activity_log_enabled). */
export function useHealth(opts?: QueryOpts<HealthResponse>): UseQueryResult<HealthResponse> {
  return useQuery({
    queryKey: queryKeys.health(),
    queryFn: ({ signal }) => api.get<HealthResponse>("/health", undefined, signal),
    ...opts,
  });
}

/** GET /api/stats → StoreStatsResponse (top-bar live counts + Dashboard totals). */
export function useStats(opts?: QueryOpts<StoreStatsResponse>): UseQueryResult<StoreStatsResponse> {
  return useQuery({
    queryKey: queryKeys.stats(),
    queryFn: ({ signal }) => api.get<StoreStatsResponse>("/stats", undefined, signal),
    ...opts,
  });
}

/**
 * GET /api/stats/growth → GrowthResponse (Dashboard store-growth trend). Returns a
 * dense, zero-filled, oldest-first series of memories created per day over the
 * trailing `days` window. `scope` accepts the canonical scope string
 * ("global" | "project:<id>") and rolls up sub-scopes; omit / null for all scopes.
 */
export function useGrowth(
  scope: string | null,
  days = 14,
  opts?: QueryOpts<GrowthResponse>,
): UseQueryResult<GrowthResponse> {
  return useQuery({
    queryKey: queryKeys.growth(scope, days),
    queryFn: ({ signal }) =>
      api.get<GrowthResponse>(
        "/stats/growth",
        { ...(scope ? { scope } : {}), days },
        signal,
      ),
    ...opts,
  });
}

// ---------------------------------------------------------------------------
// Memories (table + detail + HITL patch)
// ---------------------------------------------------------------------------

/** GET /api/memories → MemoryListResponse (filterable, paged Memories table). */
export function useMemories(
  params: MemoriesQuery = {},
  opts?: QueryOpts<MemoryListResponse>,
): UseQueryResult<MemoryListResponse> {
  return useQuery({
    queryKey: queryKeys.memories.list(params),
    queryFn: ({ signal }) =>
      api.get<MemoryListResponse>("/memories", params, signal),
    ...opts,
  });
}

/**
 * GET /api/memories/facets/categories → CategoryFacetsResponse. The discovered,
 * open-ended categories (+ counts) that back the dynamic CATEGORY filter chips —
 * categories are no longer a fixed enum, so the UI lists what the store contains.
 */
export function useCategoryFacets(
  opts?: QueryOpts<CategoryFacetsResponse>,
): UseQueryResult<CategoryFacetsResponse> {
  return useQuery({
    queryKey: queryKeys.memories.categoryFacets(),
    queryFn: ({ signal }) =>
      api.get<CategoryFacetsResponse>("/memories/facets/categories", undefined, signal),
    ...opts,
  });
}

/**
 * GET /api/memories/facets/scope-tree → ScopeTreeResponse. The hierarchical
 * project/sub-scope tree (+ per-node counts) that backs the SCOPE TREE navigator;
 * selecting a node filters the table to that scope's ancestor-composed memories.
 */
export function useScopeTree(
  opts?: QueryOpts<ScopeTreeResponse>,
): UseQueryResult<ScopeTreeResponse> {
  return useQuery({
    queryKey: queryKeys.memories.scopeTree(),
    queryFn: ({ signal }) =>
      api.get<ScopeTreeResponse>("/memories/facets/scope-tree", undefined, signal),
    ...opts,
  });
}

/** GET /api/memories/{id} → MemoryDetail (content + provenance + validity + chain). */
export function useMemory(
  id: string | undefined,
  opts?: QueryOpts<MemoryDetail>,
): UseQueryResult<MemoryDetail> {
  return useQuery({
    queryKey: queryKeys.memories.detail(id ?? ""),
    queryFn: ({ signal }) => api.get<MemoryDetail>(`/memories/${id}`, undefined, signal),
    enabled: Boolean(id),
    ...opts,
  });
}

/**
 * PATCH /api/memories/{id} → MutationResponse. The R1/R5 HITL correction:
 * re-label (`category`), toggle the cross-ref seed flag (`cross_ref_candidate`),
 * re-scope (`scope`), archive/restore (`tier`). On success it invalidates that
 * memory's detail, the memories lists/facets/scope-tree and the stats counts (a
 * re-label or re-scope shifts the category facets and the scope tree).
 */
export function usePatchMemory(): UseMutationResult<
  MutationResponse,
  Error,
  { id: string; patch: MemoryPatchRequest }
> {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, patch }) => api.patch<MutationResponse>(`/memories/${id}`, patch),
    onSuccess: (_data, { id }) => {
      void qc.invalidateQueries({ queryKey: queryKeys.memories.detail(id) });
      void qc.invalidateQueries({ queryKey: queryKeys.memories.all() });
      void qc.invalidateQueries({ queryKey: queryKeys.stats() });
    },
  });
}

// ---------------------------------------------------------------------------
// Graph explorer
// ---------------------------------------------------------------------------

/** GET /api/graph → GraphResponse (scoped subgraph; include_crossrefs overlays). */
export function useGraph(
  params: GraphQuery = {},
  opts?: QueryOpts<GraphResponse>,
): UseQueryResult<GraphResponse> {
  return useQuery({
    queryKey: queryKeys.graph(params),
    queryFn: ({ signal }) =>
      api.get<GraphResponse>("/graph", params, signal),
    ...opts,
  });
}

// ---------------------------------------------------------------------------
// Recall playground (POST — exposed both as a query and a mutation)
// ---------------------------------------------------------------------------

/**
 * POST /api/recall → RecallResponse, modeled as a query keyed by the request so a
 * given (query, scope, top_k) result is cached. Disabled until `enabled` is true
 * (the playground only fires when the operator submits). Prefer this for "live as
 * you type"; use {@link useRecallMutation} for a pure button-press model.
 */
export function useRecall(
  req: RecallRequest,
  opts?: QueryOpts<RecallResponse> & { enabled?: boolean },
): UseQueryResult<RecallResponse> {
  return useQuery({
    queryKey: queryKeys.recall(req),
    queryFn: () => api.post<RecallResponse>("/recall", req),
    enabled: opts?.enabled ?? Boolean(req.query),
    ...opts,
  });
}

/** POST /api/recall as a mutation (imperative submit; returns RecallResponse). */
export function useRecallMutation(): UseMutationResult<RecallResponse, Error, RecallRequest> {
  return useMutation({
    mutationFn: (req) => api.post<RecallResponse>("/recall", req),
  });
}

// ---------------------------------------------------------------------------
// Cross-references
// ---------------------------------------------------------------------------

/** GET /api/crossrefs → CrossRefResponse (surfaced connections + suppression flags). */
export function useCrossRefs(
  params: CrossRefsQuery = {},
  opts?: QueryOpts<CrossRefResponse>,
): UseQueryResult<CrossRefResponse> {
  return useQuery({
    queryKey: queryKeys.crossrefs.list(params),
    queryFn: ({ signal }) =>
      api.get<CrossRefResponse>("/crossrefs", params, signal),
    ...opts,
  });
}

/** POST /api/crossrefs/{memory_id}/suppress → MutationResponse (R2 dismissal). */
export function useSuppressCrossRef(): UseMutationResult<
  MutationResponse,
  Error,
  { memoryId: string; body: SuppressRequest }
> {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ memoryId, body }) =>
      api.post<MutationResponse>(`/crossrefs/${memoryId}/suppress`, body),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.crossrefs.all() });
    },
  });
}

// ---------------------------------------------------------------------------
// Activity / Logs
// ---------------------------------------------------------------------------

/**
 * GET /api/activity → ActivityResponse. `kind` is repeatable (passed as an array).
 * NOTE (API_CONTRACT): live data only when the activity log is enabled
 * (MNEMOZINE_WEB__ENABLE_ACTIVITY_LOG=1); otherwise the feed is empty by design.
 */
export function useActivity(
  params: ActivityQuery = {},
  opts?: QueryOpts<ActivityResponse>,
): UseQueryResult<ActivityResponse> {
  return useQuery({
    queryKey: queryKeys.activity.list(params),
    queryFn: ({ signal }) =>
      api.get<ActivityResponse>("/activity", params, signal),
    ...opts,
  });
}

// ---------------------------------------------------------------------------
// Maintenance / Ops
// ---------------------------------------------------------------------------

/** GET /api/maintenance → MaintenanceStatusResponse (scheduler + per-job status). */
export function useMaintenance(
  opts?: QueryOpts<MaintenanceStatusResponse>,
): UseQueryResult<MaintenanceStatusResponse> {
  return useQuery({
    queryKey: queryKeys.maintenance.status(),
    queryFn: ({ signal }) =>
      api.get<MaintenanceStatusResponse>("/maintenance", undefined, signal),
    ...opts,
  });
}

/** GET /api/maintenance/merge-candidates → MergeCandidatesResponse (FR-MNT-4 HITL). */
export function useMergeCandidates(
  opts?: QueryOpts<MergeCandidatesResponse>,
): UseQueryResult<MergeCandidatesResponse> {
  return useQuery({
    queryKey: queryKeys.maintenance.mergeCandidates(),
    queryFn: ({ signal }) =>
      api.get<MergeCandidatesResponse>("/maintenance/merge-candidates", undefined, signal),
    ...opts,
  });
}

/** POST /api/maintenance/{job}/run → MaintenanceRunResponse (trigger on demand). */
export function useRunMaintenance(): UseMutationResult<
  MaintenanceRunResponse,
  Error,
  MaintenanceJobName
> {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (job) => api.post<MaintenanceRunResponse>(`/maintenance/${job}/run`),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.maintenance.status() });
      void qc.invalidateQueries({ queryKey: queryKeys.maintenance.mergeCandidates() });
      void qc.invalidateQueries({ queryKey: queryKeys.stats() });
    },
  });
}

// ---------------------------------------------------------------------------
// Eval (incl. F4 browser bootstrap labeling)
// ---------------------------------------------------------------------------

/** GET /api/eval → EvalSummaryResponse (precision / classifier / latency / ...). */
export function useEval(opts?: QueryOpts<EvalSummaryResponse>): UseQueryResult<EvalSummaryResponse> {
  return useQuery({
    queryKey: queryKeys.eval.summary(),
    queryFn: ({ signal }) => api.get<EvalSummaryResponse>("/eval", undefined, signal),
    ...opts,
  });
}

/** GET /api/eval/bootstrap → BootstrapCandidatesResponse (F4 labeling queue). */
export function useBootstrapCandidates(
  opts?: QueryOpts<BootstrapCandidatesResponse>,
): UseQueryResult<BootstrapCandidatesResponse> {
  return useQuery({
    queryKey: queryKeys.eval.bootstrap(),
    queryFn: ({ signal }) =>
      api.get<BootstrapCandidatesResponse>("/eval/bootstrap", undefined, signal),
    ...opts,
  });
}

/** POST /api/eval/bootstrap/{candidate_id}/label → BootstrapCandidate (keep/drop + corrected_type). */
export function useLabelBootstrap(): UseMutationResult<
  BootstrapCandidate,
  Error,
  { candidateId: string; body: BootstrapLabelRequest }
> {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ candidateId, body }) =>
      api.post<BootstrapCandidate>(`/eval/bootstrap/${candidateId}/label`, body),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.eval.bootstrap() });
    },
  });
}

/** POST /api/eval/bootstrap/finish → EvalSummaryResponse (fold kept into gold set). */
export function useFinishBootstrap(): UseMutationResult<EvalSummaryResponse, Error, void> {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post<EvalSummaryResponse>("/eval/bootstrap/finish"),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queryKeys.eval.summary() });
      void qc.invalidateQueries({ queryKey: queryKeys.eval.bootstrap() });
    },
  });
}
