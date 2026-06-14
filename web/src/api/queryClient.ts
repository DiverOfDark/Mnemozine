/**
 * The shared TanStack Query client. Tuned for an observation console: data is
 * mostly read, refetch on focus is on for "live" feeling, retries are limited so
 * a 4xx (e.g. 401 bad token, 404) fails fast instead of hammering the backend.
 */

import { QueryClient } from "@tanstack/react-query";
import { ApiError } from "@/api/client";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 15_000,
      gcTime: 5 * 60_000,
      refetchOnWindowFocus: true,
      retry: (failureCount, error) => {
        // Don't retry client errors (401/403/404/422) — only transient ones.
        if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
          return false;
        }
        return failureCount < 2;
      },
    },
    mutations: {
      retry: false,
    },
  },
});
