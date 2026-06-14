/**
 * AppShell — the top-level layout (PRD §4 wireframe): left Sidebar nav · top bar ·
 * main content. Renders the persistent chrome and an <Outlet/> for the active page.
 * Pages render INSIDE the scrollable main region; they should use <PageHeader> for
 * their title/toolbar so headers are consistent.
 *
 * Screen agents render their page as the routed element — they never re-create the
 * shell. App chrome only.
 */

import { Suspense, type ReactNode } from "react";
import { Outlet } from "react-router-dom";
import { Sidebar } from "@/components/Sidebar";
import { TopBar } from "@/components/TopBar";
import { Loading } from "@/components/primitives";
import { cn } from "@/lib/cn";

export function AppShell() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg text-text">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        <main className="min-h-0 flex-1 overflow-hidden">
          <Suspense fallback={<Loading label="Loading screen…" />}>
            <Outlet />
          </Suspense>
        </main>
      </div>
    </div>
  );
}

/**
 * PageHeader — a consistent page title bar with an optional subtitle + right-side
 * toolbar. Screen agents put their filters/actions in `actions`.
 */
export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="flex h-12 shrink-0 items-center justify-between gap-4 border-b border-border px-5">
      <div className="min-w-0">
        <h1 className="truncate text-md font-semibold text-text">{title}</h1>
        {subtitle && <p className="truncate text-2xs text-text-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

/**
 * Page — a standard page scaffold: a fixed PageHeader + a scrollable body. Pages
 * that want full control (e.g. the Graph canvas filling the viewport) can skip this
 * and compose PageHeader directly.
 */
export function Page({
  title,
  subtitle,
  actions,
  children,
  bodyClassName,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  bodyClassName?: string;
}) {
  return (
    <div className="flex h-full flex-col">
      <PageHeader title={title} subtitle={subtitle} actions={actions} />
      <div className={cn("min-h-0 flex-1 overflow-auto p-5", bodyClassName)}>{children}</div>
    </div>
  );
}
