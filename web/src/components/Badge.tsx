/**
 * Badge — the canonical color-coded chip for memory TYPE and TIER (PRD §5).
 *
 * Screen agents MUST use <TypeBadge> / <TierBadge> (or the generic <Badge>) instead
 * of hand-rolling colored spans, so type/tier color coding stays consistent across
 * the table, detail drawer, recall results and graph side panel.
 */

import type { ReactNode } from "react";
import type { MemoryType, Tier } from "@/api/types";
import { TYPE_BADGE, TIER_BADGE } from "@/theme/tokens";
import { cn } from "@/lib/cn";

type BadgeSize = "sm" | "md";

interface BadgeProps {
  children: ReactNode;
  /** Tailwind text color class, e.g. "text-type-preference". */
  textClass?: string;
  /** Tailwind bg color class, e.g. "bg-type-bg-preference". */
  bgClass?: string;
  /** Optional leading dot color class, e.g. "bg-type-preference". */
  dotClass?: string;
  size?: BadgeSize;
  /** Render with a hairline border instead of a filled background. */
  outline?: boolean;
  title?: string;
  className?: string;
}

/** Generic semantic chip. Prefer the typed wrappers below for type/tier. */
export function Badge({
  children,
  textClass = "text-text-muted",
  bgClass = "bg-bg-inset",
  dotClass,
  size = "sm",
  outline = false,
  title,
  className,
}: BadgeProps) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1.5 rounded font-mono uppercase tracking-wide whitespace-nowrap",
        size === "sm" ? "px-1.5 py-0.5 text-2xs" : "px-2 py-1 text-xs",
        textClass,
        outline ? "border border-border-strong bg-transparent" : bgClass,
        className,
      )}
    >
      {dotClass && <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", dotClass)} />}
      {children}
    </span>
  );
}

/** TYPE badge: preference (violet) / project_fact (sky) / idea_seed (amber). */
export function TypeBadge({ type, size = "sm" }: { type: MemoryType; size?: BadgeSize }) {
  const t = TYPE_BADGE[type];
  return (
    <Badge textClass={t.text} bgClass={t.bg} dotClass={t.dot} size={size} title={`type: ${type}`}>
      {t.label}
    </Badge>
  );
}

/** TIER badge: hot (vivid green) / archive (muted grey). */
export function TierBadge({ tier, size = "sm" }: { tier: Tier; size?: BadgeSize }) {
  const t = TIER_BADGE[tier];
  return (
    <Badge textClass={t.text} bgClass={t.bg} dotClass={t.dot} size={size} title={`tier: ${tier}`}>
      {t.label}
    </Badge>
  );
}

/** Active/Superseded state pill (drives the struck-through superseded treatment). */
export function StatusBadge({ active, size = "sm" }: { active: boolean; size?: BadgeSize }) {
  return active ? (
    <Badge textClass="text-active" bgClass="bg-tier-bg-hot" dotClass="bg-active" size={size}>
      active
    </Badge>
  ) : (
    <Badge textClass="text-superseded" bgClass="bg-tier-bg-archive" dotClass="bg-superseded" size={size}>
      superseded
    </Badge>
  );
}
