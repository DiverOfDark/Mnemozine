/**
 * EntityChips (FE-B / PRD §4.3) — the memory's entities as clickable mono chips that
 * jump to the Graph explorer filtered by that entity (`/graph?entity=<e>`). Uses the
 * shared <Badge> for consistent chip chrome (accent dot to read as a "graph link"),
 * never a hand-rolled span.
 */

import { Link } from "react-router-dom";
import { Badge } from "@/components/index";
import { PATHS } from "@/routes";

export function EntityChips({ entities }: { entities: string[] }) {
  if (entities.length === 0) {
    return <span className="text-xs text-text-faint">No entities extracted.</span>;
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {entities.map((entity) => (
        <Link
          key={entity}
          to={`${PATHS.graph}?entity=${encodeURIComponent(entity)}`}
          title={`open ${entity} in graph`}
          className="rounded outline-none focus-visible:ring-1 focus-visible:ring-border-focus"
        >
          <Badge
            textClass="text-text"
            outline
            dotClass="bg-accent"
            className="cursor-pointer normal-case hover:border-accent hover:text-accent"
          >
            {entity}
          </Badge>
        </Link>
      ))}
    </div>
  );
}
