/**
 * Runtime mirror of the design tokens defined in tailwind.config.js.
 *
 * Tailwind classes cover styling; this module exposes the *semantic* maps that
 * logic needs at runtime — e.g. picking the Badge color class for a given
 * MemoryType / Tier, or interpolating the recall ScoreBar gradient. Keep this in
 * lockstep with tailwind.config.js. Screen agents should consume the helpers
 * (TYPE_BADGE, TIER_BADGE, scoreColor) rather than literal hex/class strings.
 */

import type { MemoryType, Tier } from "@/api/types";

/** Tailwind class bundles for the three memory TYPE badges (PRD §5). */
export const TYPE_BADGE: Record<MemoryType, { text: string; bg: string; dot: string; label: string }> = {
  preference: {
    text: "text-type-preference",
    bg: "bg-type-bg-preference",
    dot: "bg-type-preference",
    label: "preference",
  },
  project_fact: {
    text: "text-type-project_fact",
    bg: "bg-type-bg-project_fact",
    dot: "bg-type-project_fact",
    label: "project_fact",
  },
  idea_seed: {
    text: "text-type-idea_seed",
    bg: "bg-type-bg-idea_seed",
    dot: "bg-type-idea_seed",
    label: "idea_seed",
  },
};

/** Tailwind class bundles for the two TIER badges (hot vivid / archive muted). */
export const TIER_BADGE: Record<Tier, { text: string; bg: string; dot: string; label: string }> = {
  hot: { text: "text-tier-hot", bg: "bg-tier-bg-hot", dot: "bg-tier-hot", label: "hot" },
  archive: {
    text: "text-tier-archive",
    bg: "bg-tier-bg-archive",
    dot: "bg-tier-archive",
    label: "archive",
  },
};

/** Raw hex values (for Cytoscape stylesheet + canvas drawing, where Tailwind classes don't apply). */
export const HEX = {
  bg: "#0a0c10",
  bgRaised: "#10141b",
  bgInset: "#161b24",
  border: "#222a36",
  borderStrong: "#2e3848",
  text: "#d7dde6",
  textMuted: "#8b95a5",
  textFaint: "#5b6675",
  accent: "#4f8cff",
  ok: "#3fb950",
  warn: "#d29922",
  danger: "#f85149",
  type: {
    preference: "#a78bfa",
    project_fact: "#38bdf8",
    idea_seed: "#fbbf24",
  } as Record<MemoryType, string>,
  tier: {
    hot: "#34d399",
    archive: "#6b7280",
  } as Record<Tier, string>,
  active: "#34d399",
  superseded: "#6b7280",
  crossref: "#fb7185", // rose — cross-reference overlay edges (PRD §4.4)
  edge: "#3a4556",
  score: { low: "#3b4a5e", mid: "#4f8cff", high: "#34d399" },
} as const;

/** Health status → token class (used by ComponentHealth chips on the Dashboard / top bar). */
export const HEALTH_STATUS: Record<string, { text: string; dot: string }> = {
  ok: { text: "text-ok", dot: "bg-ok" },
  degraded: { text: "text-warn", dot: "bg-warn" },
  down: { text: "text-danger", dot: "bg-danger" },
  unknown: { text: "text-text-faint", dot: "bg-text-faint" },
};

/** Activity-kind → token color (Logs screen feed dots). */
export const ACTIVITY_KIND_COLOR: Record<string, string> = {
  ingest: "text-info",
  extract_decision: "text-type-preference",
  maintenance: "text-warn",
  injection: "text-tier-hot",
};

/** 4-way write decision → token color (Logs / mutation echoes). */
export const WRITE_DECISION_COLOR: Record<string, string> = {
  add: "text-ok",
  reinforce: "text-info",
  supersede: "text-warn",
  "no-op": "text-text-faint",
};

/**
 * Interpolate a 0..1 (or arbitrary, clamped) relevance score to a hex color
 * along the low→mid→high score gradient. Used by <ScoreBar>.
 */
export function scoreColor(score: number): string {
  const t = Math.max(0, Math.min(1, score));
  if (t < 0.5) return HEX.score.mid;
  return t < 0.8 ? HEX.score.mid : HEX.score.high;
}

/** Layout constants mirrored from tailwind spacing tokens (px). */
export const LAYOUT = {
  sidebarWidth: 208,
  topbarHeight: 48,
  drawerWidth: 560,
} as const;
