/** 404 fallback for unknown client routes. App chrome — not a screen-agent file. */

import { Link } from "react-router-dom";
import { Page } from "@/components/AppShell";

export default function NotFound() {
  return (
    <Page title="Not found" subtitle="Unknown route">
      <div className="flex flex-col items-center gap-3 py-16 text-center">
        <div className="font-mono text-2xl text-text-faint">404</div>
        <Link to="/" className="text-xs text-accent hover:text-accent-hover">
          ← back to dashboard
        </Link>
      </div>
    </Page>
  );
}
