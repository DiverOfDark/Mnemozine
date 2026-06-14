/**
 * Global scope filter state (PRD §4 top-bar scope filter). The TopBar sets it; any
 * screen reads it via useScope() to scope its queries (e.g. memories list, graph,
 * recall default). `scope` is a scope string ("global" | "project:<id>" | a bare
 * project id) or null for "all scopes". Persisted to localStorage so the operator's
 * working scope survives reloads.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

const STORAGE_KEY = "mnemozine.scope";

interface ScopeContextValue {
  /** null = all scopes; otherwise a scope string. */
  scope: string | null;
  setScope: (scope: string | null) => void;
}

const ScopeContext = createContext<ScopeContextValue | null>(null);

export function ScopeProvider({ children }: { children: ReactNode }) {
  const [scope, setScopeState] = useState<string | null>(() => {
    try {
      return window.localStorage.getItem(STORAGE_KEY) || null;
    } catch {
      return null;
    }
  });

  const setScope = useCallback((next: string | null) => {
    setScopeState(next);
  }, []);

  useEffect(() => {
    try {
      if (scope) window.localStorage.setItem(STORAGE_KEY, scope);
      else window.localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
  }, [scope]);

  const value = useMemo(() => ({ scope, setScope }), [scope, setScope]);
  return <ScopeContext.Provider value={value}>{children}</ScopeContext.Provider>;
}

export function useScope(): ScopeContextValue {
  const ctx = useContext(ScopeContext);
  if (!ctx) throw new Error("useScope must be used within <ScopeProvider>");
  return ctx;
}
