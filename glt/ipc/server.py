"""Long-running HTTP worker. The dashboard spawns one of these on first
demand (`python -m glt --serve`), and talks to it via JSON HTTP from then
on. Keeps the InsightFace model warm between requests instead of paying the
~3 s load cost every time.

Protocol — all bodies are JSON, all responses too:

  GET  /health              → {"ok": true}
  POST /detect-faces        → body {"paths": [...], "recursive": bool?,
                                     "config_path": str?}
                              returns {"datasets": [...]} (same shape as
                              the old --detect-faces script).
  POST /shutdown            → schedules a graceful exit, returns {"ok": true}.

Output discipline:
  - stdout has EXACTLY one line at boot: `READY <port>\n`. After that,
    stdout is silent so the parent doesn't have to parse anything.
  - All operational logging goes to stderr.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import face_service


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class _Handler(BaseHTTPRequestHandler):
    server_version = "GltWorker/0.1"

    # silence the default access log (it writes to stderr line by line)
    def log_message(self, format, *args):  # noqa: A002
        # opt-in via env if we ever want it
        return

    def _send_json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("content-length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"error": f"unknown route {self.path}"})

    def do_POST(self):  # noqa: N802
        # /detect-faces-blob takes raw image bytes; every other POST is JSON.
        if self.path == "/detect-faces-blob":
            try:
                length = int(self.headers.get("content-length") or 0)
                if length <= 0:
                    raise ValueError("empty body")
                blob = self.rfile.read(length)
                config_path = self.headers.get("x-glt-config-path") or None
                # Caller-supplied CUDA mem cap (GiB). When present, it
                # overrides whatever's in config.json's face_recognition
                # section. Float OK for fractional GiB caps.
                gpu_mem_gb: float | None = None
                raw_cap = self.headers.get("x-glt-face-gpu-mem-gb")
                if raw_cap:
                    try:
                        v = float(raw_cap)
                        if v > 0:
                            gpu_mem_gb = v
                    except ValueError:
                        pass
                result = face_service.detect_faces_blob(
                    blob,
                    config_path=config_path,
                    gpu_mem_limit_gb=gpu_mem_gb,
                )
                self._send_json(200, result)
            except Exception as e:
                _stderr(f"[face-blob][error] {type(e).__name__}: {e}")
                self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
            return

        try:
            body = self._read_json()
        except Exception as e:
            self._send_json(400, {"error": f"bad request body: {e}"})
            return

        if self.path == "/detect-faces":
            try:
                paths = body.get("paths") or []
                if not isinstance(paths, list) or not paths:
                    raise ValueError("missing 'paths' (list of dataset folders)")
                recursive = bool(body.get("recursive", False))
                config_path = body.get("config_path")
                result = face_service.detect_faces(
                    paths=paths,
                    recursive=recursive,
                    config_path=config_path,
                )
                self._send_json(200, result)
            except Exception as e:
                _stderr(f"[face][error] {type(e).__name__}: {e}")
                self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
            return

        if self.path == "/shutdown":
            self._send_json(200, {"ok": True})
            # Defer shutdown so the response actually reaches the client.
            threading.Thread(
                target=lambda: (self.server.shutdown(), self.server.server_close()),
                daemon=True,
            ).start()
            return

        self._send_json(404, {"error": f"unknown route {self.path}"})


def _pick_port() -> int:
    """Pick an OS-assigned free port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="glt --serve",
        description="Run the long-lived HTTP worker the dashboard talks to.",
    )
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--port", type=int, default=0,
                        help="TCP port to bind on 127.0.0.1. 0 (default) lets the OS pick.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    port = args.port if args.port > 0 else _pick_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)

    # Announce readiness to the parent and then never touch stdout again.
    # The single READY line is the only thing the Node manager parses.
    sys.stdout.write(f"READY {port}\n")
    sys.stdout.flush()
    _stderr(f"[serve] listening on 127.0.0.1:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _stderr("[serve] stopped")
