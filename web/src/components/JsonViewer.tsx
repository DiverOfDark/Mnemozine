/**
 * JsonViewer — a read-only, monospace, syntax-tinted JSON pretty-printer for raw
 * payloads (activity `detail`, provenance, raw memory inspection). Lightweight
 * (no dependency): renders pre-formatted JSON with token-colored keys/values and
 * a copy button. Use this for any "raw data" panel so they look consistent.
 */

import { useMemo, useState } from "react";
import { cn } from "@/lib/cn";

interface JsonViewerProps {
  value: unknown;
  /** Collapse to a single scrollable block with a max height. */
  maxHeight?: number | string;
  className?: string;
}

export function JsonViewer({ value, maxHeight = 360, className }: JsonViewerProps) {
  const text = useMemo(() => safeStringify(value), [value]);
  const [copied, setCopied] = useState(false);

  const copy = () => {
    void navigator.clipboard?.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    });
  };

  return (
    <div className={cn("relative rounded-md border border-border bg-bg", className)}>
      <button
        type="button"
        onClick={copy}
        className="absolute right-2 top-2 z-10 rounded border border-border-strong bg-bg-inset px-1.5 py-0.5 font-mono text-2xs text-text-muted hover:text-text"
      >
        {copied ? "copied" : "copy"}
      </button>
      <pre
        className="overflow-auto p-3 font-mono text-xs leading-relaxed text-text-muted"
        style={{ maxHeight: typeof maxHeight === "number" ? `${maxHeight}px` : maxHeight }}
      >
        <code dangerouslySetInnerHTML={{ __html: highlight(text) }} />
      </pre>
    </div>
  );
}

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

/** Tiny, self-contained JSON tokenizer → HTML with token colors. Escapes input first. */
function highlight(json: string): string {
  const escaped = json
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return escaped.replace(
    /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g,
    (match) => {
      let color = "#a5d6ff"; // string value
      if (/^"/.test(match)) {
        color = /:$/.test(match) ? "#79c0ff" : "#a5d6ff"; // key vs string
      } else if (/true|false/.test(match)) {
        color = "#d29922"; // boolean
      } else if (/null/.test(match)) {
        color = "#5b6675"; // null
      } else {
        color = "#f0883e"; // number
      }
      return `<span style="color:${color}">${match}</span>`;
    },
  );
}
