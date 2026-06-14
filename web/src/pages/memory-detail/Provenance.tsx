/**
 * ProvenanceBlock (FE-B / PRD §4.3) — where this memory came from: source channel,
 * the originating session, the chunk hash, and a link to the raw transcript path
 * (the "provenance link"). Renders the structured fields as <KeyValue> rows and the
 * full provenance object in a <JsonViewer> for raw inspection.
 *
 * Raw transcript content is out of scope for v1 (PRD §7 — no write access to raw
 * transcripts); we surface the `raw_path` pointer + a session activity deep-link so
 * the operator can trace it in the Logs screen.
 */

import { Link } from "react-router-dom";
import { JsonViewer, KeyValue } from "@/components/index";
import type { Provenance } from "@/api";
import { PATHS } from "@/routes";
import { shortId } from "@/lib/format";

export function ProvenanceBlock({ provenance }: { provenance: Provenance }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col">
        <KeyValue k="source">
          <span className="font-mono">{provenance.source}</span>
        </KeyValue>
        <KeyValue k="session">
          {provenance.session_id ? (
            <Link
              to={`${PATHS.logs}?session_id=${encodeURIComponent(provenance.session_id)}`}
              title="trace this session in the activity log"
              className="font-mono text-accent hover:text-accent-hover hover:underline"
            >
              {provenance.session_id}
            </Link>
          ) : (
            <span className="text-text-faint">—</span>
          )}
        </KeyValue>
        <KeyValue k="chunk hash">
          {provenance.chunk_hash ? (
            <span className="font-mono text-text-muted" title={provenance.chunk_hash}>
              {shortId(provenance.chunk_hash, 16)}
            </span>
          ) : (
            <span className="text-text-faint">—</span>
          )}
        </KeyValue>
        <KeyValue k="raw transcript">
          {provenance.raw_path ? (
            <span className="break-all font-mono text-text-muted" title={provenance.raw_path}>
              {provenance.raw_path}
            </span>
          ) : (
            <span className="text-text-faint">not retained</span>
          )}
        </KeyValue>
      </div>
      <JsonViewer value={provenance} maxHeight={180} />
    </div>
  );
}
