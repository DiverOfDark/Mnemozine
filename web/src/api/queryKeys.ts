/**
 * Centralized TanStack Query keys. Hooks and mutations reference these so cache
 * invalidation is consistent (e.g. a memory PATCH invalidates the matching list +
 * detail + stats keys). Screen agents that need to invalidate cache should import
 * from here rather than typing key arrays by hand.
 */

import type {
  ActivityQuery,
  CrossRefsQuery,
  GraphQuery,
  MemoriesQuery,
  RecallRequest,
} from "@/api/types";

export const queryKeys = {
  health: () => ["health"] as const,
  stats: () => ["stats"] as const,

  memories: {
    all: () => ["memories"] as const,
    list: (params: MemoriesQuery) => ["memories", "list", params] as const,
    detail: (id: string) => ["memories", "detail", id] as const,
    categoryFacets: () => ["memories", "facets", "categories"] as const,
    scopeTree: () => ["memories", "facets", "scope-tree"] as const,
  },

  graph: (params: GraphQuery) => ["graph", params] as const,

  recall: (req: RecallRequest) => ["recall", req] as const,

  crossrefs: {
    all: () => ["crossrefs"] as const,
    list: (params: CrossRefsQuery) => ["crossrefs", "list", params] as const,
  },

  activity: {
    all: () => ["activity"] as const,
    list: (params: ActivityQuery) => ["activity", "list", params] as const,
  },

  maintenance: {
    status: () => ["maintenance", "status"] as const,
    mergeCandidates: () => ["maintenance", "merge-candidates"] as const,
  },

  eval: {
    summary: () => ["eval", "summary"] as const,
    bootstrap: () => ["eval", "bootstrap"] as const,
  },
} as const;
