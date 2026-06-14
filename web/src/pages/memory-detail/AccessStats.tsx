/**
 * AccessStats (FE-B / PRD §4.3) — the memory's retrieval stats: access count, last
 * accessed (absolute + relative), and confidence rendered with the shared <ScoreBar>.
 * Pure presentation off MemoryDetail fields; no derivation of `active` (the contract
 * provides it).
 */

import { KeyValue, ScoreBar } from "@/components/index";
import type { MemoryDetail } from "@/api";
import { formatDateTime, formatRelative, pluralize } from "@/lib/format";

export function AccessStats({ memory }: { memory: MemoryDetail }) {
  return (
    <div className="flex flex-col">
      <KeyValue k="confidence">
        <ScoreBar value={memory.confidence} format="percent" width={120} />
      </KeyValue>
      <KeyValue k="access count">
        <span className="font-mono tabular-nums">{pluralize(memory.access_count, "hit", "hits")}</span>
      </KeyValue>
      <KeyValue k="last accessed">
        {memory.last_accessed ? (
          <span className="font-mono">
            {formatDateTime(memory.last_accessed)}
            <span className="ml-1.5 text-text-faint">({formatRelative(memory.last_accessed)})</span>
          </span>
        ) : (
          <span className="text-text-faint">never</span>
        )}
      </KeyValue>
    </div>
  );
}
