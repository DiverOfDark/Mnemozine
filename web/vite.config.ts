import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

// The FastAPI app (mnemozine.web) binds to 127.0.0.1:8765 by default
// (API_CONTRACT.md → Config). In dev we proxy /api there so the SPA talks to the
// real backend with no CORS. The production build is emitted straight into the
// Python package's static dir so the same FastAPI image serves it.
const API_TARGET = process.env.MNEMOZINE_API_TARGET ?? "http://127.0.0.1:8765";

export default defineConfig({
  // Served from '/' by FastAPI (single image, same origin).
  base: "/",
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      // Dev proxy → FastAPI. Both the JSON API and the OpenAPI/docs surfaces.
      "/api": { target: API_TARGET, changeOrigin: true },
      "/openapi.json": { target: API_TARGET, changeOrigin: true },
      "/docs": { target: API_TARGET, changeOrigin: true },
    },
  },
  build: {
    // Emit the built SPA into the Python package so FastAPI's _BUNDLED_STATIC
    // (mnemozine/web/static) picks it up. assetsDir 'assets' matches the app's
    // StaticFiles mount at /assets (mnemozine/web/app.py::_mount_spa).
    outDir: fileURLToPath(new URL("../mnemozine/web/static", import.meta.url)),
    emptyOutDir: true,
    assetsDir: "assets",
    sourcemap: false,
    target: "es2022",
  },
});
