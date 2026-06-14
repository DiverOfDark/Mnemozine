/**
 * Shared placeholder body for the un-built screens. Each page in src/pages/ renders
 * this until its screen agent replaces the page body. It shows the screen name, the
 * PRD section, and the exact api hooks + components the agent should wire — so the
 * skeleton is self-documenting and the app boots end-to-end before any screen lands.
 *
 * Screen agents REPLACE their page's default export entirely; they must NOT edit the
 * router (App.tsx), the api client/hooks, the theme, or this file.
 */

import { Page } from "@/components/AppShell";
import { Panel } from "@/components/primitives";

export interface PlaceholderProps {
  screen: string;
  prdSection: string;
  description: string;
  /** api hook names this screen should use (verbatim from src/api/hooks.ts). */
  hooks: string[];
  /** design-system components this screen should compose. */
  components: string[];
}

export function ScreenPlaceholder({ screen, prdSection, description, hooks, components }: PlaceholderProps) {
  return (
    <Page title={screen} subtitle={`PRD ${prdSection} — placeholder (awaiting screen agent)`}>
      <div className="mx-auto flex max-w-2xl flex-col gap-4">
        <Panel title="Screen scaffold">
          <p className="text-sm text-text-muted">{description}</p>
        </Panel>
        <div className="grid grid-cols-2 gap-4">
          <Panel title="API hooks to use">
            <ul className="flex flex-col gap-1 font-mono text-xs text-accent">
              {hooks.map((h) => (
                <li key={h}>{h}()</li>
              ))}
            </ul>
          </Panel>
          <Panel title="Components to compose">
            <ul className="flex flex-col gap-1 font-mono text-xs text-text-muted">
              {components.map((c) => (
                <li key={c}>&lt;{c} /&gt;</li>
              ))}
            </ul>
          </Panel>
        </div>
        <p className="text-2xs text-text-faint">
          Replace this page's default export. Do not edit the router, api client/hooks, or theme.
        </p>
      </div>
    </Page>
  );
}
