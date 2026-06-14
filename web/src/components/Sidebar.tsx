/**
 * Sidebar — the left nav (PRD §4 / approved wireframe: Dash · Mem · Graph · Recall
 * · Logs · Ops · Eval). Reads NAV_ROUTES so it never drifts from the router. The
 * active route is highlighted. This is app chrome — screen agents don't touch it.
 */

import { NavLink } from "react-router-dom";
import { NAV_ROUTES } from "@/routes";
import { cn } from "@/lib/cn";

export function Sidebar() {
  return (
    <nav
      className="flex h-full w-sidebar shrink-0 flex-col border-r border-border bg-bg-raised"
      aria-label="Primary"
    >
      {/* brand */}
      <div className="flex h-topbar shrink-0 items-center gap-2 border-b border-border px-4">
        <span className="flex h-5 w-5 items-center justify-center rounded bg-accent text-2xs font-bold text-text-inverse">
          M
        </span>
        <span className="font-mono text-sm font-semibold tracking-tight text-text">mnemozine</span>
      </div>

      {/* nav items */}
      <div className="flex flex-1 flex-col gap-0.5 overflow-y-auto p-2">
        {NAV_ROUTES.map((route) => (
          <NavLink
            key={route.path}
            to={route.path}
            // The dashboard route ("/") must match exactly so it isn't always active.
            end={route.path === "/"}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-2.5 rounded px-2.5 py-1.5 text-sm transition-colors duration-fast",
                isActive
                  ? "bg-bg-active text-text"
                  : "text-text-muted hover:bg-bg-hover hover:text-text",
              )
            }
          >
            <span className="shrink-0">{route.icon}</span>
            <span>{route.label}</span>
          </NavLink>
        ))}
      </div>

      {/* footer marker */}
      <div className="shrink-0 border-t border-border px-3 py-2 font-mono text-2xs text-text-faint">
        operator console · local
      </div>
    </nav>
  );
}
