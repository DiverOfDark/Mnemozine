/**
 * Mnemozine Operator Console — DARK OBSERVABILITY CONSOLE theme.
 *
 * Design direction (WEBUI PRD §5, locked 2026-06-14): "calm Grafana × Linear".
 * Dark, dense; monospace for IDs/content snippets; color-coded by memory TYPE
 * (preference / project_fact / idea_seed) and TIER (hot vivid / archive muted);
 * superseded = struck-through + greyed.
 *
 * These tokens are the SINGLE SOURCE OF TRUTH for color/spacing/typography. The
 * TS mirror in src/theme/tokens.ts re-exports the semantic maps for logic that
 * needs them at runtime (Badge color lookups etc). Screen agents must use these
 * tokens / the <Badge> component and NEVER hand-pick hex values.
 *
 * @type {import('tailwindcss').Config}
 */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // --- surfaces (dark console chrome) ---
        bg: {
          DEFAULT: "#0a0c10", // app background (deepest)
          raised: "#10141b", // panels, sidebar, topbar
          inset: "#161b24", // cards, table header, drawer
          hover: "#1c2230", // row hover / focus surface
          active: "#232b3b", // selected row / active nav
        },
        border: {
          DEFAULT: "#222a36", // hairline dividers
          strong: "#2e3848", // emphasized borders
          focus: "#3b82f6", // focus ring
        },
        text: {
          DEFAULT: "#d7dde6", // primary body text
          muted: "#8b95a5", // secondary / labels
          faint: "#5b6675", // tertiary / disabled / superseded
          inverse: "#0a0c10",
        },
        // --- accent (interactive / brand) ---
        accent: {
          DEFAULT: "#4f8cff",
          hover: "#6ea0ff",
          muted: "#1e2c47",
        },
        // --- semantic status ---
        ok: "#3fb950",
        warn: "#d29922",
        danger: "#f85149",
        info: "#58a6ff",

        // --- memory TYPE tokens (PRD §5 color-by-type) ---
        type: {
          preference: "#a78bfa", // violet — durable cross-project preference
          project_fact: "#38bdf8", // sky — project-scoped fact
          idea_seed: "#fbbf24", // amber — idea seed / candidate concept
        },
        // dim backgrounds for type badges
        "type-bg": {
          preference: "#241d3b",
          project_fact: "#0e2738",
          idea_seed: "#2e2410",
        },

        // --- TIER tokens (PRD §5 hot vivid / archive muted) ---
        tier: {
          hot: "#34d399", // vivid green — on the hot retrieval path
          archive: "#6b7280", // muted grey — cold/archived
        },
        "tier-bg": {
          hot: "#0d2b22",
          archive: "#1b1f27",
        },

        // --- validity / supersession states ---
        active: "#34d399", // open validity window
        superseded: "#6b7280", // closed window (rendered struck + grey)

        // --- score bar gradient anchors (recall relevance) ---
        score: {
          low: "#3b4a5e",
          mid: "#4f8cff",
          high: "#34d399",
        },
      },
      fontFamily: {
        // UI sans for chrome; mono for IDs, content snippets, JSON, scores.
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      fontSize: {
        // dense type scale
        "2xs": ["10px", { lineHeight: "14px" }],
        xs: ["11px", { lineHeight: "16px" }],
        sm: ["12px", { lineHeight: "18px" }],
        base: ["13px", { lineHeight: "20px" }],
        md: ["14px", { lineHeight: "21px" }],
        lg: ["16px", { lineHeight: "24px" }],
        xl: ["20px", { lineHeight: "28px" }],
        "2xl": ["26px", { lineHeight: "32px" }],
      },
      spacing: {
        // dense layout rhythm
        sidebar: "208px",
        topbar: "48px",
        drawer: "560px",
      },
      borderRadius: {
        sm: "3px",
        DEFAULT: "5px",
        md: "6px",
        lg: "8px",
      },
      boxShadow: {
        drawer: "-8px 0 32px -8px rgba(0,0,0,0.6)",
        panel: "0 1px 0 0 rgba(255,255,255,0.02) inset",
        focus: "0 0 0 2px rgba(59,130,246,0.5)",
      },
      transitionDuration: {
        fast: "120ms",
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        "slide-in-right": {
          from: { transform: "translateX(100%)" },
          to: { transform: "translateX(0)" },
        },
        pulse: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.4" },
        },
      },
      animation: {
        "fade-in": "fade-in 120ms ease-out",
        "slide-in-right": "slide-in-right 160ms ease-out",
      },
    },
  },
  plugins: [],
};
