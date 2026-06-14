/**
 * SPA entry. Mounts the provider stack (frozen for screen agents):
 *   ErrorBoundary → QueryClientProvider → BrowserRouter → ScopeProvider → App
 *
 * BrowserRouter basename is "/" (the SPA is served from root by FastAPI; deep links
 * fall back to index.html, see mnemozine/web/app.py::_mount_spa). The QueryClient is
 * the shared one from src/api/queryClient.ts.
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";

import App from "@/App";
import { queryClient } from "@/api/queryClient";
import { ScopeProvider } from "@/state/scope";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import "@/index.css";

const rootEl = document.getElementById("root");
if (!rootEl) throw new Error("#root not found");

createRoot(rootEl).render(
  <StrictMode>
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <ScopeProvider>
            <App />
          </ScopeProvider>
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  </StrictMode>,
);
