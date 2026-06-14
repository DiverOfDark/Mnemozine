/**
 * Infra health tiles (PRD §4.1): FalkorDB / Ollama / LLM endpoint component
 * health, plus overall status, version and whether the persisted activity log is
 * enabled. Page-local to the Dashboard; reads useHealth and colors each tile via
 * the shared HEALTH_STATUS token map.
 */

import { Panel, Loading, ErrorState, EmptyState } from "@/components/primitives";
import type { ComponentHealth } from "@/api/types";
import { useHealth } from "@/api/hooks";
import { HEALTH_STATUS } from "@/theme/tokens";
import { cn } from "@/lib/cn";

function HealthTile({ component }: { component: ComponentHealth }) {
  const s = HEALTH_STATUS[component.status] ?? HEALTH_STATUS.unknown!;
  return (
    <div className="flex flex-col gap-1 rounded border border-border bg-bg-inset px-2.5 py-2">
      <div className="flex items-center gap-1.5">
        <span className={cn("h-2 w-2 rounded-full", s.dot)} />
        <span className="font-mono text-xs text-text">{component.name}</span>
        <span className={cn("ml-auto font-mono text-2xs uppercase", s.text)}>{component.status}</span>
      </div>
      {component.detail && (
        <p className="truncate text-2xs text-text-faint" title={component.detail}>
          {component.detail}
        </p>
      )}
    </div>
  );
}

export function HealthTiles() {
  const { data, isLoading, error, refetch } = useHealth({ refetchInterval: 30_000 });
  const components = data?.components ?? [];
  const overall = data ? HEALTH_STATUS[data.status] ?? HEALTH_STATUS.unknown! : null;

  return (
    <Panel
      title="Infra health"
      actions={
        data && (
          <div className="flex items-center gap-2.5 text-2xs">
            <span className="font-mono text-text-faint" title="API version">
              v{data.version}
            </span>
            <span
              className={cn("font-mono", data.activity_log_enabled ? "text-ok" : "text-text-faint")}
              title={data.activity_log_enabled ? "activity log persisted" : "activity log off (NullActivityLog)"}
            >
              log {data.activity_log_enabled ? "on" : "off"}
            </span>
            {overall && (
              <span className={cn("flex items-center gap-1 font-mono uppercase", overall.text)}>
                <span className={cn("h-1.5 w-1.5 rounded-full", overall.dot)} />
                {data.status}
              </span>
            )}
          </div>
        )
      }
    >
      {isLoading ? (
        <Loading label="Checking health…" />
      ) : error ? (
        <ErrorState error={error} onRetry={() => void refetch()} />
      ) : components.length === 0 ? (
        <EmptyState title="No health components reported" />
      ) : (
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
          {components.map((component) => (
            <HealthTile key={component.name} component={component} />
          ))}
        </div>
      )}
    </Panel>
  );
}
