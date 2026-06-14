/**
 * SupersessionChain — renders a memory's supersession history (PRD §2/§4.3, UC-2):
 * the older facts this memory `supersedes` (replaced) and the newer facts that
 * `superseded_by` it (replaced-by). Superseded links are struck-through + greyed
 * per the design language. Each link is clickable (onSelect) to jump to that memory.
 */

import type { SupersessionLink } from "@/api/types";
import { formatDateTime, shortId } from "@/lib/format";
import { cn } from "@/lib/cn";

interface SupersessionChainProps {
  /** Older facts this memory replaced. */
  supersedes: SupersessionLink[];
  /** Newer facts that replaced this memory. */
  supersededBy: SupersessionLink[];
  /** The id of the memory being viewed (rendered as the "current" middle node). */
  currentId: string;
  currentContent: string;
  /** Jump to another memory in the chain. */
  onSelect?: (memoryId: string) => void;
  className?: string;
}

function ChainNode({
  link,
  variant,
  onSelect,
}: {
  link: SupersessionLink;
  variant: "older" | "newer";
  onSelect?: (id: string) => void;
}) {
  const closed = link.valid_to !== null;
  return (
    <button
      type="button"
      onClick={() => onSelect?.(link.memory_id)}
      className={cn(
        "group flex w-full items-start gap-2 rounded border border-border bg-bg px-2.5 py-2 text-left transition-colors hover:border-border-strong hover:bg-bg-hover",
      )}
    >
      <span
        className={cn(
          "mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full",
          variant === "older" ? "bg-superseded" : "bg-active",
        )}
      />
      <span className="min-w-0 flex-1">
        <span className={cn("block truncate text-xs", closed ? "superseded" : "text-text")}>
          {link.content}
        </span>
        <span className="mt-0.5 block font-mono text-2xs text-text-faint">
          {shortId(link.memory_id)} · {formatDateTime(link.valid_from)}
          {closed && ` → ${formatDateTime(link.valid_to)}`}
        </span>
      </span>
    </button>
  );
}

export function SupersessionChain({
  supersedes,
  supersededBy,
  currentId,
  currentContent,
  onSelect,
  className,
}: SupersessionChainProps) {
  const hasChain = supersedes.length > 0 || supersededBy.length > 0;
  if (!hasChain) {
    return <div className="text-xs text-text-faint">No supersession history — this is the sole version.</div>;
  }

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {/* newer (replaced-by) at the top — most recent first */}
      {supersededBy.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <span className="text-2xs uppercase tracking-wide text-active">replaced by (newer)</span>
          {supersededBy.map((link) => (
            <ChainNode key={link.memory_id} link={link} variant="newer" onSelect={onSelect} />
          ))}
        </div>
      )}

      {/* the current memory */}
      <div className="flex items-start gap-2 rounded border border-accent/40 bg-accent-muted px-2.5 py-2">
        <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-xs text-text">{currentContent}</span>
          <span className="mt-0.5 block font-mono text-2xs text-accent">
            {shortId(currentId)} · current view
          </span>
        </span>
      </div>

      {/* older (supersedes) below */}
      {supersedes.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <span className="text-2xs uppercase tracking-wide text-superseded">supersedes (older)</span>
          {supersedes.map((link) => (
            <ChainNode key={link.memory_id} link={link} variant="older" onSelect={onSelect} />
          ))}
        </div>
      )}
    </div>
  );
}
