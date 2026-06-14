/**
 * KeyboardHints — a compact legend of keyboard shortcuts for the keyboard-first
 * console (PRD §5). Renders a row of `<kbd>` chips with descriptions. Use it in a
 * screen's footer/toolbar (e.g. the Eval bootstrap labeler's k/d/u shortcuts, the
 * table's j/k nav) so shortcuts are discoverable.
 */

import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

export interface KeyHint {
  keys: string[]; // e.g. ["j"], ["⌘", "k"]
  label: string;
}

export function KeyboardHints({ hints, className }: { hints: KeyHint[]; className?: string }) {
  return (
    <div className={cn("flex flex-wrap items-center gap-x-4 gap-y-1.5", className)}>
      {hints.map((hint, i) => (
        <span key={i} className="inline-flex items-center gap-1.5 text-2xs text-text-faint">
          <span className="inline-flex items-center gap-0.5">
            {hint.keys.map((k, j) => (
              <Kbd key={j}>{k}</Kbd>
            ))}
          </span>
          <span>{hint.label}</span>
        </span>
      ))}
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex min-w-[18px] items-center justify-center rounded border border-border-strong bg-bg-inset px-1 py-0.5 font-mono text-2xs text-text-muted">
      {children}
    </kbd>
  );
}
