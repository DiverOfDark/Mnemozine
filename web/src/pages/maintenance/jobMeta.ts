/**
 * Page-local presentation metadata for the Maintenance / Ops screen (PRD §4.7).
 *
 * Maps the wire `MaintenanceJobName` values (the {job} path segment for
 * POST /api/maintenance/{job}/run) to operator-facing labels + descriptions, and
 * normalizes the loosely-typed `MaintenanceJobStatus.name` from the status payload
 * onto the canonical run-name so the "trigger" button lines up with each status row.
 *
 * This is page-local UI sugar only — it never edits the contract types.
 */

import type { MaintenanceJobName } from "@/api";

export interface JobMeta {
  name: MaintenanceJobName;
  label: string;
  description: string;
}

/** Ordered, canonical job metadata (drives the trigger button grid + status sort). */
export const JOB_META: readonly JobMeta[] = [
  {
    name: "consolidate",
    label: "Consolidate",
    description: "Merge duplicate / reinforcing memory units; collapse redundant facts.",
  },
  {
    name: "entity-resolution",
    label: "Entity resolution",
    description: "Detect & merge co-referent entities (HITL review below).",
  },
  {
    name: "decay",
    label: "Decay",
    description: "Demote stale hot memories to archive on the decay schedule.",
  },
  {
    name: "audit",
    label: "Audit",
    description: "Integrity sweep over validity windows + supersession chains.",
  },
  {
    name: "migrate-index",
    label: "Migrate index",
    description: "Rebuild / re-embed the vector + injection index.",
  },
] as const;

const META_BY_NAME = new Map<MaintenanceJobName, JobMeta>(JOB_META.map((m) => [m.name, m]));

/**
 * Resolve a status row's `name` (which may arrive as a job-name variant such as
 * "entity_resolution" or "migrate_index") to the canonical `MaintenanceJobName`,
 * so the Run button targets the right path. Falls back to null when unknown.
 */
export function canonicalJobName(name: string): MaintenanceJobName | null {
  const normalized = name.trim().toLowerCase().replace(/_/g, "-") as MaintenanceJobName;
  return META_BY_NAME.has(normalized) ? normalized : null;
}

/** Human label for a (possibly non-canonical) status row name. */
export function jobLabel(name: string): string {
  const canonical = canonicalJobName(name);
  const meta = canonical ? META_BY_NAME.get(canonical) : undefined;
  if (meta) return meta.label;
  // Title-case fallback for any job the backend reports that we don't model.
  return name.replace(/[_-]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
