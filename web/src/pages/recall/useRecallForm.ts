/**
 * useRecallForm (FE-B / PRD §4.5) — local form state for the Recall playground +
 * the imperative submit over the shared `useRecallMutation`. Holds the query text,
 * the scope (defaulted from the top-bar `useScope`), top_k, and the
 * include-index-preview toggle; exposes `run()` which fires the recall with the
 * current request and surfaces `{ data, error, pending, lastReq }`.
 *
 * We use the mutation (button-driven submit) rather than the live query so a recall
 * only fires on explicit submit — recall is a real backend call (embeddings + graph
 * traversal) and the operator drives it deliberately (the "precision-debugging tool").
 */

import { useCallback, useEffect, useState } from "react";
import { useRecallMutation } from "@/api";
import type { RecallRequest, RecallResponse } from "@/api";
import { useScope } from "@/state/scope";

export const RECALL_DEFAULT_TOP_K = 10;

export interface RecallFormState {
  query: string;
  setQuery: (q: string) => void;
  scope: string;
  setScope: (s: string) => void;
  topK: number;
  setTopK: (n: number) => void;
  includeIndexPreview: boolean;
  setIncludeIndexPreview: (v: boolean) => void;
  /** Fire the recall with the current request (no-op for an empty query). */
  run: () => void;
  /** Reset query + result. */
  reset: () => void;
  pending: boolean;
  error: Error | null;
  data: RecallResponse | undefined;
  /** The request that produced the current `data` (for echoing the searched scope). */
  lastReq: RecallRequest | null;
}

export function useRecallForm(): RecallFormState {
  const { scope: topBarScope } = useScope();
  const mutation = useRecallMutation();

  const [query, setQuery] = useState("");
  // Default the scope field from the top-bar scope ("" = all scopes / unscoped recall).
  const [scope, setScope] = useState<string>(topBarScope ?? "");
  const [topK, setTopK] = useState(RECALL_DEFAULT_TOP_K);
  const [includeIndexPreview, setIncludeIndexPreview] = useState(true);
  const [lastReq, setLastReq] = useState<RecallRequest | null>(null);

  // Adopt the top-bar scope when it changes (operator switched working scope).
  useEffect(() => {
    setScope(topBarScope ?? "");
  }, [topBarScope]);

  const run = useCallback(() => {
    const q = query.trim();
    if (!q) return;
    const req: RecallRequest = {
      query: q,
      scope: scope.trim() ? scope.trim() : null,
      top_k: topK,
      include_index_preview: includeIndexPreview,
    };
    setLastReq(req);
    mutation.mutate(req);
  }, [query, scope, topK, includeIndexPreview, mutation]);

  const reset = useCallback(() => {
    setQuery("");
    setLastReq(null);
    mutation.reset();
  }, [mutation]);

  return {
    query,
    setQuery,
    scope,
    setScope,
    topK,
    setTopK,
    includeIndexPreview,
    setIncludeIndexPreview,
    run,
    reset,
    pending: mutation.isPending,
    error: mutation.error,
    data: mutation.data,
    lastReq,
  };
}
