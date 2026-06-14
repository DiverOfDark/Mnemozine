/**
 * DetailDrawer — the right-side slide-in panel used for Memory detail (PRD §4.3),
 * graph node inspection, activity entry detail, etc. Renders a fixed-width overlay
 * drawer with a header (title + close), scrollable body, and optional footer for
 * actions. Closes on Escape and backdrop click.
 *
 * Screen agents render their detail content as `children`; the drawer owns the
 * chrome, animation and dismissal.
 */

import { useEffect } from "react";
import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { IconButton } from "@/components/primitives";
import { cn } from "@/lib/cn";

interface DetailDrawerProps {
  open: boolean;
  onClose: () => void;
  title?: ReactNode;
  subtitle?: ReactNode;
  /** Toolbar content rendered in the header right side (badges, actions). */
  headerActions?: ReactNode;
  footer?: ReactNode;
  children: ReactNode;
  /** Drawer width in px (default 560, from the `drawer` spacing token). */
  width?: number;
}

export function DetailDrawer({
  open,
  onClose,
  title,
  subtitle,
  headerActions,
  footer,
  children,
  width = 560,
}: DetailDrawerProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return createPortal(
    <div className="fixed inset-0 z-50 flex justify-end">
      {/* backdrop */}
      <div
        className="absolute inset-0 animate-fade-in bg-black/50"
        onClick={onClose}
        aria-hidden="true"
      />
      {/* panel */}
      <aside
        role="dialog"
        aria-modal="true"
        className="relative flex h-full animate-slide-in-right flex-col border-l border-border bg-bg-raised shadow-drawer"
        style={{ width }}
      >
        <header className="flex shrink-0 items-start justify-between gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0 flex-1">
            {title && <div className="truncate text-md font-medium text-text">{title}</div>}
            {subtitle && <div className="mt-0.5 truncate font-mono text-2xs text-text-muted">{subtitle}</div>}
          </div>
          <div className="flex items-center gap-2">
            {headerActions}
            <IconButton onClick={onClose} aria-label="Close" title="Close (Esc)">
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
              </svg>
            </IconButton>
          </div>
        </header>

        <div className={cn("min-h-0 flex-1 overflow-auto px-4 py-4")}>{children}</div>

        {footer && (
          <footer className="shrink-0 border-t border-border bg-bg-inset px-4 py-3">{footer}</footer>
        )}
      </aside>
    </div>,
    document.body,
  );
}

/** A labeled section block inside a drawer body. */
export function DrawerSection({
  title,
  children,
  actions,
}: {
  title: string;
  children: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <section className="mb-5">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-2xs font-semibold uppercase tracking-wider text-text-faint">{title}</h3>
        {actions}
      </div>
      {children}
    </section>
  );
}
