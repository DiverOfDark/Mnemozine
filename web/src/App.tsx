/**
 * App.tsx — the router. Wires all 8 screens (+ memory detail child + 404) to LAZY
 * page components in src/pages/, inside the persistent <AppShell>.
 *
 * THIS FILE IS FROZEN for screen agents: they REPLACE the default export of their
 * own page in src/pages/ and never touch the router, the api client/hooks, the
 * theme, or the shell. The lazy() boundaries mean each screen is code-split.
 *
 * Route map (matches NAV_ROUTES / the approved wireframe):
 *   /                      → Dashboard            (PRD §4.1)
 *   /memories              → Memories  (table)    (PRD §4.2)
 *   /memories/:memoryId    → MemoryDetail         (PRD §4.3)
 *   /graph                 → Graph                (PRD §4.4)
 *   /recall                → Recall               (PRD §4.5)
 *   /logs                  → Logs                 (PRD §4.6)
 *   /maintenance           → Maintenance          (PRD §4.7)
 *   /eval                  → Eval                 (PRD §4.8)
 *   *                      → NotFound
 */

import { lazy } from "react";
import { Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/AppShell";

// Lazy page bodies — screen agents fill these (default export per file).
const Dashboard = lazy(() => import("@/pages/Dashboard"));
const Memories = lazy(() => import("@/pages/Memories"));
const MemoryDetail = lazy(() => import("@/pages/MemoryDetail"));
const Graph = lazy(() => import("@/pages/Graph"));
const Recall = lazy(() => import("@/pages/Recall"));
const Logs = lazy(() => import("@/pages/Logs"));
const Maintenance = lazy(() => import("@/pages/Maintenance"));
const Eval = lazy(() => import("@/pages/Eval"));
const NotFound = lazy(() => import("@/pages/NotFound"));

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Dashboard />} />
        <Route path="memories" element={<Memories />} />
        <Route path="memories/:memoryId" element={<MemoryDetail />} />
        <Route path="graph" element={<Graph />} />
        <Route path="recall" element={<Recall />} />
        <Route path="logs" element={<Logs />} />
        <Route path="maintenance" element={<Maintenance />} />
        <Route path="eval" element={<Eval />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
