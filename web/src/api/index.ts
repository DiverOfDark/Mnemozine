/** Public barrel for the API layer — screen agents import from "@/api". */
export * from "@/api/types";
export * from "@/api/hooks";
export { api, ApiError, API_BASE, buildQuery, hasToken } from "@/api/client";
export { queryClient } from "@/api/queryClient";
export { queryKeys } from "@/api/queryKeys";
