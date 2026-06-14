# Mnemozine WebUI

Dark "observability console" SPA over the Mnemozine memory layer. React 18 + Vite 5
+ TypeScript (strict) + Tailwind 3 + TanStack Query 5 + Cytoscape, served as static
assets by the FastAPI app (`mnemozine.web`).

## Scripts

```bash
npm install        # install deps (live phase only)
npm run dev        # Vite dev server (proxies /api → 127.0.0.1:8765)
npm run build      # tsc --noEmit + vite build → ../mnemozine/web/static
npm run typecheck  # tsc --noEmit
npm run lint       # eslint
```

The dev proxy target is `MNEMOZINE_API_TARGET` (default `http://127.0.0.1:8765`,
the FastAPI bind from `API_CONTRACT.md`). `vite build` emits straight into
`../mnemozine/web/static`, which `mnemozine/web/app.py` serves (single image).

## Architecture (FROZEN — screen agents do not edit)

| Path | Owns |
|------|------|
| `src/main.tsx` | provider stack (Query / Router / Scope / ErrorBoundary) |
| `src/App.tsx` | the router — all 8 routes → lazy pages |
| `src/routes.tsx` | nav table (Sidebar + router read it) |
| `src/api/` | typed client, wire `types.ts`, `hooks.ts` (one hook per endpoint) |
| `src/theme/tokens.ts` + `tailwind.config.js` | design tokens (type/tier/tier colors) |
| `src/components/` | the design system |
| `src/state/scope.tsx` | the global scope filter context |

## Screen agents

Each screen agent **replaces the default export of exactly one file** in
`src/pages/` (Dashboard, Memories, MemoryDetail, Graph, Recall, Logs, Maintenance,
Eval). They:

- import data hooks from `@/api` (e.g. `useMemories`, `usePatchMemory`),
- compose components from `@/components` (e.g. `DataTable`, `DetailDrawer`,
  `TypeBadge`, `ValidityTimeline`, `SupersessionChain`, `ScoreBar`, `GraphCanvas`),
- read the active scope via `useScope()` from `@/state/scope`,
- use design tokens via the Tailwind classes / `@/theme/tokens` (never raw hex).

They must **not** edit: `App.tsx`, `routes.tsx`, anything under `src/api/`,
`src/theme/`, `src/components/`, `src/state/`, or `_Placeholder.tsx`.

## Design language (PRD §5)

Dark, dense observability console. Monospace for IDs/content/JSON. Color by memory
**type** (preference = violet, project_fact = sky, idea_seed = amber) and **tier**
(hot = vivid green, archive = muted grey). Superseded = struck-through + greyed
(`.superseded` class). Keyboard-first (`j/k` table nav, label shortcuts).
