/**
 * Display formatters shared across screens. Keep rendering logic here (not in the
 * components) so the table, drawer and timeline format dates/scopes identically.
 */

import type { ScopeKind } from "@/api/types";

/** Compact absolute timestamp, e.g. "2026-06-14 09:30". null → "—". */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** Date only, e.g. "2026-06-14". */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/** Relative "time ago", e.g. "3m ago", "2h ago", "5d ago". null → "never". */
export function formatRelative(iso: string | null | undefined): string {
  if (!iso) return "never";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const secs = Math.round((Date.now() - d.getTime()) / 1000);
  if (secs < 0) return "in the future";
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.round(days / 30);
  if (months < 12) return `${months}mo ago`;
  return `${Math.round(months / 12)}y ago`;
}

/** A 0..1 confidence/score as a percentage string, e.g. "0.87" → "87%". */
export function formatPercent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** Truncate the leading chars of an id for compact mono display ("a1b2c3…"). */
export function shortId(id: string, len = 8): string {
  return id.length > len ? `${id.slice(0, len)}…` : id;
}

export interface ParsedScope {
  kind: ScopeKind;
  /** Project id for project scopes, undefined for global. */
  project?: string;
  /** A short display label, e.g. "global" or "myproj". */
  label: string;
}

/** Parse a scope string ("global" | "project:<id>" | bare "<id>") into parts. */
export function parseScope(scope: string | null | undefined): ParsedScope {
  if (!scope || scope === "global") return { kind: "global", label: "global" };
  const project = scope.startsWith("project:") ? scope.slice("project:".length) : scope;
  return { kind: "project", project, label: project };
}

/** Pluralize an integer with its noun: pluralize(1,'memory','memories'). */
export function pluralize(n: number, singular: string, plural?: string): string {
  return `${n.toLocaleString()} ${n === 1 ? singular : (plural ?? `${singular}s`)}`;
}
