/**
 * App-level error boundary so a thrown render error in one screen doesn't blank the
 * whole console. Catches, shows the error, and offers a reload. App chrome.
 */

import { Component, type ErrorInfo, type ReactNode } from "react";

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // eslint-disable-next-line no-console
    console.error("Unhandled UI error:", error, info);
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="flex h-screen w-screen flex-col items-center justify-center gap-4 bg-bg text-text">
          <div className="font-mono text-sm text-danger">the console hit a render error</div>
          <pre className="max-w-xl overflow-auto rounded border border-border bg-bg-inset p-3 text-xs text-text-muted">
            {this.state.error.message}
          </pre>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="rounded border border-border-strong bg-bg-inset px-3 py-1 text-xs text-text hover:bg-bg-hover"
          >
            reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
