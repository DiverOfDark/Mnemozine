/**
 * Logs screen — presentation metadata for ActivityEvent kinds + write decisions.
 *
 * Maps the four ActivityKind values to a human label + a dot color class drawn from
 * the ACTIVITY_KIND_COLOR token map (never raw hex). Also extracts the optional
 * 4-way WriteDecision from an extract_decision event's `detail` payload so the feed
 * can color it with WRITE_DECISION_COLOR. All look-ups are token-driven.
 */

import type { ActivityEventOut, ActivityKind, WriteDecision } from "@/api";
import { WRITE_DECISIONS } from "@/api";
import { ACTIVITY_KIND_COLOR, WRITE_DECISION_COLOR } from "@/theme/tokens";

/** Short, human label for each activity kind (matches PRD §4.6 vocabulary). */
export const ACTIVITY_KIND_LABEL: Record<ActivityKind, string> = {
  ingest: "ingest",
  extract_decision: "extract",
  maintenance: "maintenance",
  injection: "injection",
};

/** The text-color token class for a kind's dot/label (falls back to faint). */
export function kindColorClass(kind: ActivityKind): string {
  return ACTIVITY_KIND_COLOR[kind] ?? "text-text-faint";
}

/**
 * The matching dot (bg) token class for a kind. Written as literal strings (NOT
 * derived at runtime) so Tailwind's content scanner emits each class — a string
 * built via .replace() would otherwise be purged. Mirrors ACTIVITY_KIND_COLOR.
 */
const ACTIVITY_KIND_DOT: Record<ActivityKind, string> = {
  ingest: "bg-info",
  extract_decision: "bg-type-preference",
  maintenance: "bg-warn",
  injection: "bg-tier-hot",
};

export function kindDotClass(kind: ActivityKind): string {
  return ACTIVITY_KIND_DOT[kind] ?? "bg-text-faint";
}

/** True if a value is a recognised 4-way write decision. */
function isWriteDecision(value: unknown): value is WriteDecision {
  return typeof value === "string" && (WRITE_DECISIONS as readonly string[]).includes(value);
}

/**
 * Pull the write decision (add / reinforce / supersede / no-op) out of an
 * extract_decision event's structured detail, if present. Checks the common keys a
 * write_decision_event builder may use. Returns null when absent or off-kind.
 */
export function extractWriteDecision(event: ActivityEventOut): WriteDecision | null {
  if (event.kind !== "extract_decision") return null;
  const detail = event.detail;
  for (const key of ["decision", "write_decision", "action"]) {
    const candidate = detail[key];
    if (isWriteDecision(candidate)) return candidate;
  }
  return null;
}

/** The text-color token class for a write decision (faint fallback). */
export function writeDecisionColorClass(decision: WriteDecision): string {
  return WRITE_DECISION_COLOR[decision] ?? "text-text-faint";
}
