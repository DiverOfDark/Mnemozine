"""Minimal OpenAI-format /v1/models reachability stub for the health probe.

The WebUI /api/health LLM check only GETs {base_url}/models and treats any
non-5xx response as reachable (it never makes a completion call). This serves
exactly that so the Health screen shows the LLM endpoint as reachable during the
local demo, without standing up a real model server.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        body = json.dumps(
            {"object": "list", "data": [{"id": "demo-llm", "object": "model"}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # silence per-request logging
        return None


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
