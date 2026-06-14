/**
 * The canonical route table. The Sidebar nav and the React-Router config in
 * App.tsx both read from here so paths never drift. Screen agents do NOT edit this
 * file — they only fill their page body in src/pages/.
 *
 * Nav order matches the approved wireframe: Dash · Mem · Graph · Recall · Logs ·
 * Ops · Eval (Memory detail is a child route of Memories, not a top-level nav item).
 */

import type { ReactNode } from "react";

export interface NavRoute {
  /** Router path. */
  path: string;
  /** Short sidebar label. */
  label: string;
  /** Longer title (top bar / document title). */
  title: string;
  /** Inline SVG icon. */
  icon: ReactNode;
}

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
      {children}
    </svg>
  );
}

export const NAV_ROUTES: NavRoute[] = [
  {
    path: "/",
    label: "Dashboard",
    title: "Dashboard",
    icon: <Icon><rect x="2" y="2" width="5" height="5" /><rect x="9" y="2" width="5" height="5" /><rect x="2" y="9" width="5" height="5" /><rect x="9" y="9" width="5" height="5" /></Icon>,
  },
  {
    path: "/memories",
    label: "Memories",
    title: "Memories",
    icon: <Icon><path d="M2 4h12M2 8h12M2 12h8" /></Icon>,
  },
  {
    path: "/graph",
    label: "Graph",
    title: "Graph Explorer",
    icon: <Icon><circle cx="4" cy="4" r="2" /><circle cx="12" cy="6" r="2" /><circle cx="6" cy="12" r="2" /><path d="M5.5 5.2l5 .6M5.4 6.3l.8 4.1" /></Icon>,
  },
  {
    path: "/recall",
    label: "Recall",
    title: "Recall Playground",
    icon: <Icon><circle cx="7" cy="7" r="4.5" /><path d="M11 11l3 3" /></Icon>,
  },
  {
    path: "/logs",
    label: "Logs",
    title: "Activity / Logs",
    icon: <Icon><path d="M3 2v12M3 4h8M3 8h8M3 12h5" /></Icon>,
  },
  {
    path: "/maintenance",
    label: "Ops",
    title: "Maintenance / Ops",
    icon: <Icon><circle cx="8" cy="8" r="2" /><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.5 3.5l1.4 1.4M11.1 11.1l1.4 1.4M12.5 3.5l-1.4 1.4M4.9 11.1l-1.4 1.4" /></Icon>,
  },
  {
    path: "/eval",
    label: "Eval",
    title: "Eval",
    icon: <Icon><path d="M2 13l3-3 3 2 6-6" /><path d="M11 6h3v3" /></Icon>,
  },
];

/** Route paths as constants for programmatic navigation (e.g. row → detail). */
export const PATHS = {
  dashboard: "/",
  memories: "/memories",
  memoryDetail: (id: string) => `/memories/${id}`,
  graph: "/graph",
  recall: "/recall",
  logs: "/logs",
  maintenance: "/maintenance",
  eval: "/eval",
} as const;
