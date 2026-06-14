/**
 * EvalMetrics — eval results panels (PRD §4.8).
 *
 * Renders the EvalSummaryResponse: the gold-set name, overall pass/fail, when it last
 * ran, and a grid of metric cards (precision / classifier accuracy / latency / no-leak
 * / scaling). Each card shows the value vs its threshold with a pass/fail chip and a
 * ScoreBar where the metric is a 0..1 ratio; latency-style metrics render their raw
 * value with the threshold for context.
 *
 * Page-local: design-system components + the typed useEval hook only.
 */

import { useEval, type EvalMetric } from "@/api";
import { Badge, Button, ErrorState, Loading, Panel, ScoreBar } from "@/components";
import { formatRelative, formatDateTime } from "@/lib/format";

/** Metrics whose value is NOT a 0..1 ratio (don't render a ScoreBar for these). */
const NON_RATIO = /latency|ms|seconds|count|p50|p95|p99|scaling/i;

function isRatioMetric(m: EvalMetric): boolean {
  if (NON_RATIO.test(m.name)) return false;
  return m.value >= 0 && m.value <= 1;
}

export function EvalMetrics() {
  const { data, isLoading, error, refetch, isRefetching } = useEval();

  if (isLoading && !data) return <Loading label="Loading eval results…" />;
  if (error && !data) return <ErrorState error={error} onRetry={() => void refetch()} />;
  if (!data) return null;

  const metrics = data.metrics ?? [];

  return (
    <Panel
      title="Eval results"
      actions={
        <div className="flex items-center gap-2">
          <Badge
            textClass={data.passed ? "text-ok" : "text-danger"}
            bgClass={data.passed ? "bg-tier-bg-hot" : "bg-danger/10"}
            dotClass={data.passed ? "bg-ok" : "bg-danger"}
          >
            {data.passed ? "passing" : "failing"}
          </Badge>
          <span className="font-mono text-2xs text-text-faint" title={formatDateTime(data.ran_at)}>
            {data.ran_at ? `ran ${formatRelative(data.ran_at)}` : "not yet run"}
          </span>
          <Button variant="ghost" loading={isRefetching} onClick={() => void refetch()}>
            refresh
          </Button>
        </div>
      }
    >
      <div className="mb-3 flex items-center gap-2 text-2xs text-text-muted">
        <span className="uppercase tracking-wide text-text-faint">gold set</span>
        <code className="font-mono text-text">{data.gold_set || "—"}</code>
      </div>

      {metrics.length === 0 ? (
        <p className="py-6 text-center text-sm text-text-muted">
          No metrics yet. Run the eval harness to populate precision / classifier / latency / no-leak / scaling.
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {metrics.map((m) => (
            <MetricCard key={m.name} metric={m} />
          ))}
        </div>
      )}
    </Panel>
  );
}

function MetricCard({ metric }: { metric: EvalMetric }) {
  const ratio = isRatioMetric(metric);
  return (
    <div className="flex flex-col gap-2 rounded border border-border bg-bg-inset p-3">
      <div className="flex items-start justify-between gap-2">
        <span className="text-xs font-medium text-text">{prettyName(metric.name)}</span>
        <Badge
          textClass={metric.passed ? "text-ok" : "text-danger"}
          bgClass={metric.passed ? "bg-tier-bg-hot" : "bg-danger/10"}
          dotClass={metric.passed ? "bg-ok" : "bg-danger"}
        >
          {metric.passed ? "pass" : "fail"}
        </Badge>
      </div>

      <div className="flex items-baseline gap-2">
        <span className="font-mono text-xl tabular-nums text-text">
          {ratio ? `${(metric.value * 100).toFixed(1)}%` : formatValue(metric.value)}
        </span>
        {metric.threshold != null && (
          <span className="font-mono text-2xs text-text-faint">
            {comparator(metric)} {ratio ? `${(metric.threshold * 100).toFixed(0)}%` : formatValue(metric.threshold)}
          </span>
        )}
      </div>

      {ratio && <ScoreBar value={metric.value} showValue={false} width="100%" />}

      {metric.detail && <p className="text-2xs leading-relaxed text-text-muted">{metric.detail}</p>}
    </div>
  );
}

function prettyName(name: string): string {
  return name.replace(/[_-]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatValue(v: number): string {
  if (Number.isInteger(v)) return v.toLocaleString();
  return v.toFixed(v < 10 ? 2 : 0);
}

/** Cheap heuristic for the threshold comparator label (lower-is-better for latency). */
function comparator(metric: EvalMetric): string {
  return /latency|ms|p50|p95|p99/i.test(metric.name) ? "≤" : "≥";
}
