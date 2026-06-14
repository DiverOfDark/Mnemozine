/**
 * The low-level typed HTTP client for the Mnemozine API.
 *
 * Same-origin by default (the SPA is served by the FastAPI app; dev uses Vite's
 * /api proxy → 127.0.0.1:8765, see vite.config.ts). Auth (API_CONTRACT.md §Auth):
 * when a static bearer token is configured we send `Authorization: Bearer <t>`
 * on every /api request. The token is read from (in order): a Vite env var
 * `VITE_MNEMOZINE_TOKEN`, a `?token=` URL param (persisted to localStorage), or
 * localStorage. When unset (the default localhost bind) no header is sent.
 *
 * Errors surface as `ApiError` with the FastAPI `{ detail }` message + status.
 * Screen agents should NOT call these directly — use the TanStack Query hooks in
 * src/api/hooks.ts. This module is the single fetch seam (auth, base URL, error
 * shape) so the hooks stay declarative.
 */

const TOKEN_STORAGE_KEY = "mnemozine.token";

/** API base. Same-origin '/api' in both dev (proxied) and prod (served by FastAPI). */
export const API_BASE = "/api";

function resolveToken(): string | null {
  // 1) build-time env (set in CI/local .env as VITE_MNEMOZINE_TOKEN)
  const envToken = import.meta.env.VITE_MNEMOZINE_TOKEN as string | undefined;
  if (envToken) return envToken;

  // 2) ?token=... in the URL → persist then strip from history
  if (typeof window !== "undefined") {
    const url = new URL(window.location.href);
    const urlToken = url.searchParams.get("token");
    if (urlToken) {
      try {
        window.localStorage.setItem(TOKEN_STORAGE_KEY, urlToken);
        url.searchParams.delete("token");
        window.history.replaceState({}, "", url.toString());
      } catch {
        /* ignore storage failures */
      }
      return urlToken;
    }
    // 3) previously-persisted token
    try {
      return window.localStorage.getItem(TOKEN_STORAGE_KEY);
    } catch {
      return null;
    }
  }
  return null;
}

/** A structured API error carrying the HTTP status + FastAPI `detail`. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;
  constructor(status: number, detail: unknown, fallback: string) {
    const message =
      typeof detail === "string"
        ? detail
        : detail && typeof detail === "object" && "detail" in detail
          ? String((detail as { detail: unknown }).detail)
          : fallback;
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export type QueryValue = string | number | boolean | undefined | null | (string | number)[];

/** A loose query-param bag. The typed *Query objects in api/types.ts are assignable. */
export type QueryParams = Record<string, QueryValue>;

/**
 * Build a query string, dropping null/undefined and expanding arrays into repeated
 * keys (`?kind=ingest&kind=maintenance`). Accepts any plain object whose values are
 * QueryValue-shaped — the typed *Query interfaces in api/types.ts pass directly with
 * no cast (values are read as unknown and serialized defensively).
 */
export function buildQuery(params?: object): string {
  if (!params) return "";
  const sp = new URLSearchParams();
  for (const [key, value] of Object.entries(params as Record<string, unknown>)) {
    if (value === undefined || value === null) continue;
    if (Array.isArray(value)) {
      for (const v of value) sp.append(key, String(v));
    } else {
      sp.append(key, String(value));
    }
  }
  const qs = sp.toString();
  return qs ? `?${qs}` : "";
}

interface RequestOptions {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  body?: unknown;
  signal?: AbortSignal;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, signal } = options;
  const headers: Record<string, string> = { Accept: "application/json" };
  const token = resolveToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";

  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal,
  });

  if (!res.ok) {
    let detail: unknown = null;
    try {
      detail = await res.json();
    } catch {
      detail = await res.text().catch(() => null);
    }
    throw new ApiError(res.status, detail, `Request failed: ${res.status} ${res.statusText}`);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/**
 * Typed verb helpers. `path` is relative to API_BASE ('/api'). `params` accepts any
 * plain object (the typed *Query interfaces); see {@link buildQuery}.
 */
export const api = {
  get: <T>(path: string, params?: object, signal?: AbortSignal) =>
    request<T>(`${path}${buildQuery(params)}`, { method: "GET", signal }),
  post: <T>(path: string, body?: unknown, params?: object) =>
    request<T>(`${path}${buildQuery(params)}`, { method: "POST", body }),
  patch: <T>(path: string, body?: unknown) => request<T>(path, { method: "PATCH", body }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};

/** Whether a token is currently configured (for the top-bar lock indicator). */
export function hasToken(): boolean {
  return resolveToken() !== null;
}

export { TOKEN_STORAGE_KEY };
