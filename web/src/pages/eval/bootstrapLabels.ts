/**
 * Bootstrap labeling mapping (PRD §4.8, F4) — page-local.
 *
 * The PRD's four operator choices are `preference / project_fact / idea_seed /
 * not-a-memory`, but the wire contract (BootstrapLabelRequest) is a {label, corrected_type}
 * pair where `label` ∈ keep|drop|unreviewed and `corrected_type` ∈ MemoryType|null.
 * This module is the single place that maps between the two so the card UI, the keyboard
 * handler, and the mutation body all stay consistent:
 *
 *   preference   → { label: "keep", corrected_type: "preference"   }   key 1
 *   project_fact → { label: "keep", corrected_type: "project_fact" }   key 2
 *   idea_seed    → { label: "keep", corrected_type: "idea_seed"    }   key 3
 *   not-a-memory → { label: "drop", corrected_type: null           }   key 0 / n
 *   (clear)      → { label: "unreviewed", corrected_type: null     }   key u
 *
 * A candidate's effective "choice" for highlighting is derived from its persisted
 * {label, corrected_type} via choiceFromCandidate().
 */

import type { BootstrapCandidate, BootstrapLabelRequest, MemoryType } from "@/api";

/** The four operator-facing choices (PRD §4.8). */
export type LabelChoice = MemoryType | "not_a_memory";

export interface ChoiceMeta {
  choice: LabelChoice;
  /** Display label. */
  label: string;
  /** Keyboard shortcut (single char). */
  key: string;
  /** Tailwind text color class for the choice. */
  textClass: string;
  /** Tailwind bg color class for the choice (badge fill). */
  bgClass: string;
}

export const CHOICES: readonly ChoiceMeta[] = [
  {
    choice: "preference",
    label: "preference",
    key: "1",
    textClass: "text-type-preference",
    bgClass: "bg-type-bg-preference",
  },
  {
    choice: "project_fact",
    label: "project_fact",
    key: "2",
    textClass: "text-type-project_fact",
    bgClass: "bg-type-bg-project_fact",
  },
  {
    choice: "idea_seed",
    label: "idea_seed",
    key: "3",
    textClass: "text-type-idea_seed",
    bgClass: "bg-type-bg-idea_seed",
  },
  {
    choice: "not_a_memory",
    label: "not-a-memory",
    key: "0",
    textClass: "text-danger",
    bgClass: "bg-danger/10",
  },
] as const;

/** Map a single keypress to a choice (supports the "n" alias for not-a-memory). */
export function choiceForKey(key: string): LabelChoice | null {
  if (key === "n") return "not_a_memory";
  const match = CHOICES.find((c) => c.key === key);
  return match ? match.choice : null;
}

/** Build the wire label body for a choice. */
export function bodyForChoice(choice: LabelChoice): BootstrapLabelRequest {
  if (choice === "not_a_memory") return { label: "drop", corrected_type: null };
  return { label: "keep", corrected_type: choice };
}

/** The "clear → unreviewed" body (key `u`). */
export const UNREVIEWED_BODY: BootstrapLabelRequest = { label: "unreviewed", corrected_type: null };

/** Derive the currently-applied choice from a candidate's persisted state (or null). */
export function choiceFromCandidate(c: BootstrapCandidate): LabelChoice | null {
  if (c.label === "keep") return c.corrected_type ?? c.proposed_type;
  if (c.label === "drop") return "not_a_memory";
  return null;
}

export function choiceMeta(choice: LabelChoice): ChoiceMeta {
  return CHOICES.find((c) => c.choice === choice)!;
}
