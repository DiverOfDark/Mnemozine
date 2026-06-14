/**
 * Component library barrel — screen agents import from "@/components".
 *
 * Design-system components (use these; do not restyle their internals):
 *   AppShell / Page / PageHeader  — layout chrome
 *   Sidebar / TopBar              — nav chrome (wired by App.tsx; rarely imported directly)
 *   DataTable                     — dense, keyboard-nav table
 *   DetailDrawer / DrawerSection  — right slide-in detail panel
 *   Badge / CategoryBadge / CrossRefBadge / TierBadge / StatusBadge / ScopePath
 *                                  — category/cross-ref/tier/state/scope chips
 *   ValidityTimeline              — signature validity window
 *   SupersessionChain             — supersedes / superseded-by chain
 *   ScoreBar                      — relevance/confidence bar
 *   GraphCanvas                   — Cytoscape.js wrapper
 *   JsonViewer                    — raw-payload pretty printer
 *   KeyboardHints / Kbd           — shortcut legend
 *   primitives: Button, IconButton, Input, Select, Field, KeyValue, Panel,
 *               Spinner, Loading, EmptyState, ErrorState
 */

export { AppShell, Page, PageHeader } from "@/components/AppShell";
export { Sidebar } from "@/components/Sidebar";
export { TopBar } from "@/components/TopBar";
export { DataTable, type Column } from "@/components/DataTable";
export { DetailDrawer, DrawerSection } from "@/components/DetailDrawer";
export {
  Badge,
  CategoryBadge,
  CrossRefBadge,
  TypeBadge,
  TierBadge,
  StatusBadge,
  ScopePath,
  scopeSegments,
} from "@/components/Badge";
export { ValidityTimeline } from "@/components/ValidityTimeline";
export { SupersessionChain } from "@/components/SupersessionChain";
export { ScoreBar } from "@/components/ScoreBar";
export { GraphCanvas, type GraphLayoutName } from "@/components/GraphCanvas";
export { JsonViewer } from "@/components/JsonViewer";
export { KeyboardHints, Kbd, type KeyHint } from "@/components/KeyboardHints";
export {
  Button,
  IconButton,
  Input,
  Select,
  Field,
  KeyValue,
  Panel,
  Spinner,
  Loading,
  EmptyState,
  ErrorState,
} from "@/components/primitives";
