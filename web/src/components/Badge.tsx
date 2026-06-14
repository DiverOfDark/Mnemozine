/**
 * Badge — the canonical color-coded chip for memory CATEGORY, TIER and SCOPE (PRD §5).
 *
 * Screen agents MUST use <CategoryBadge> / <TierBadge> / <ScopePath> (or the generic
 * <Badge>) instead of hand-rolling colored spans, so the color coding stays
 * consistent across the table, detail drawer, recall results and graph side panel.
 *
 * The free-form CATEGORY replaced the old fixed 3-value type enum: <CategoryBadge>
 * colors any emergent category deterministically (see theme/tokens categoryColor).
 * <TypeBadge> is retained ONLY for the eval-bootstrap legacy MemoryType path.
 */

import type { CSSProperties, ReactNode } from "react";
import type { MemoryType, Tier } from "@/api/types";
import { TYPE_BADGE, TIER_BADGE, categoryColor } from "@/theme/tokens";
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
  /** Inline style escape hatch for runtime colors (free-form category chips). */
  style?: CSSProperties;
  /** Inline dot color (used together with `style` for category chips). */
  dotColor?: string;
}

/** Generic semantic chip. Prefer the typed wrappers below for category/tier/scope. */
export function Badge({
  children,
  textClass = "text-text-muted",
  bgClass = "bg-bg-inset",
  dotClass,
  size = "sm",
  outline = false,
  title,
  className,
  style,
  dotColor,
}: BadgeProps) {
  return (
    <span
      title={title}
      style={style}
      className={cn(
        "inline-flex items-center gap-1.5 rounded font-mono uppercase tracking-wide whitespace-nowrap",
        size === "sm" ? "px-1.5 py-0.5 text-2xs" : "px-2 py-1 text-xs",
        textClass,
        outline ? "border border-border-strong bg-transparent" : bgClass,
        className,
      )}
    >
      {dotClass && <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", dotClass)} />}
      {dotColor && (
        <span
          className="h-1.5 w-1.5 shrink-0 rounded-full"
          style={{ backgroundColor: dotColor }}
        />
      )}
      {children}
    </span>
  );
}

/**
 * CATEGORY badge — the free-form, emergent category chip (replaces TypeBadge for
 * memories). Color is picked deterministically from the dark-theme category
 * palette by hashing the slug, so any category renders consistently.
 */
export function CategoryBadge({
  category,
  size = "sm",
}: {
  category: string;
  size?: BadgeSize;
}) {
  const c = categoryColor(category);
  return (
    <Badge
      size={size}
      title={`category: ${category}`}
      dotColor={c.fg}
      style={{ color: c.fg, backgroundColor: c.bg }}
    >
      {category}
    </Badge>
  );
}

/**
 * CROSS-REF seed flag chip — surfaces a memory flagged as a cross-reference seed
 * (the boolean that replaced the legacy idea_seed type). Renders nothing when the
 * flag is false so it stays unobtrusive in dense tables.
 */
export function CrossRefBadge({ size = "sm" }: { size?: BadgeSize }) {
  return (
    <Badge
      size={size}
      title="flagged as a cross-reference seed (FR-RET-6)"
      outline
      style={{ color: "#fb7185" }}
    >
      x-ref
    </Badge>
  );
}

/** TYPE badge — LEGACY, eval-bootstrap path only (preference/project_fact/idea_seed). */
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

/**
 * Parse a persisted scope string into its ordered segments for display.
 *
 * "global" -> []; "project:P" -> ["P"]; "project:P/auth/api" -> ["P","auth","api"].
 * Mirrors Scope.parse on the backend (the "project:" prefix is on the whole path,
 * sub-segments joined by "/").
 */
export function scopeSegments(scope: string): string[] {
  if (!scope || scope === "global") return [];
  const body = scope.startsWith("project:") ? scope.slice("project:".length) : scope;
  return body.split("/").filter(Boolean);
}

/**
 * SCOPE PATH — renders a memory's HIERARCHICAL scope as a breadcrumb chip:
 * "global", or "P / auth / api" with subtle separators. The global root and the
 * project segment are emphasized; deeper sub-scopes are muted, so the path reads
 * left-to-right from the broadest scope to the leaf.
 */
export function ScopePath({
  scope,
  size = "sm",
}: {
  scope: string;
  size?: BadgeSize;
}) {
  const segments = scopeSegments(scope);
  const textSize = size === "sm" ? "text-2xs" : "text-xs";
  if (segments.length === 0) {
    return (
      <span
        title="scope: global"
        className={cn(
          "inline-flex items-center rounded font-mono uppercase tracking-wide whitespace-nowrap",
          size === "sm" ? "px-1.5 py-0.5" : "px-2 py-1",
          textSize,
          "bg-bg-inset text-info",
        )}
      >
        global
      </span>
    );
  }
  return (
    <span
      title={`scope: ${scope}`}
      className={cn(
        "inline-flex items-center gap-1 rounded font-mono whitespace-nowrap",
        size === "sm" ? "px-1.5 py-0.5" : "px-2 py-1",
        textSize,
        "bg-bg-inset",
      )}
    >
      {segments.map((seg, i) => (
        <span key={`${seg}-${i}`} className="inline-flex items-center gap-1">
          {i > 0 && <span className="text-text-faint">/</span>}
          <span className={i === 0 ? "text-accent" : "text-text-muted"}>{seg}</span>
        </span>
      ))}
    </span>
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
