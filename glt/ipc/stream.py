"""Per-job face-detection entrypoints — one short-lived subprocess per job,
spawned by the dashboard, instead of a long-lived shared HTTP worker.

Two modes, dispatched from `glt.cli`:

  `--detect-stream`   Framed request/response loop over stdin/stdout. Each
                      request frame is the raw bytes of ONE image; the reply
                      frame is the JSON result of `detect_faces_blob`. The
                      dashboard (which owns connector credentials and does the
                      downloads) streams bytes in; Python only runs the model.
                      Loops until the parent closes stdin (EOF), then exits.

  `--detect-folders`  Scan one or more dataset folders for faces and print the
                      whole `{"datasets": [...]}` result as a single JSON blob
                      on stdout. Used by the centroid job, which then does the
                      averaging + persistence in TS.

Output discipline (both modes):
  - stdout carries ONLY the protocol (framed replies / the final JSON blob).
    NOTHING else may print to stdout, or it corrupts the stream.
  - All human/operational logging goes to stderr — the dashboard pipes it
    line-by-line into the job's own log.

Frame format (--detect-stream, both directions):
    [4-byte big-endian unsigned length][payload of that many bytes]
A request payload is raw image bytes; a reply payload is UTF-8 JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

from . import face_service


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---- framed stdin/stdout protocol --------------------------------------
def _read_exact(buf, n: int) -> bytes | None:
    """Read exactly `n` bytes from a binary stream, or None on EOF/truncation."""
    out = bytearray()
    while len(out) < n:
        chunk = buf.read(n - len(out))
        if not chunk:
            return None
        out += chunk
    return bytes(out)


def _read_frame(buf) -> bytes | None:
    """Next length-prefixed frame, or None when the parent closed the pipe."""
    header = _read_exact(buf, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    if length == 0:
        return b""
    return _read_exact(buf, length)


def _write_frame(buf, payload: bytes) -> None:
    buf.write(struct.pack(">I", len(payload)))
    buf.write(payload)
    buf.flush()


def _hijack_stdout():
    """Hand back a private, clean handle on the REAL stdout for the protocol,
    and reroute fd 1 to stderr so every library that prints to stdout —
    insightface's "Applied providers:", torchao, onnxruntime's C++ chatter —
    lands on stderr (→ the job log) instead of corrupting the frame stream or
    JSON blob. We move the file descriptor, not just `sys.stdout`, so it covers
    C-level writes too. From the parent's side this makes child.stdout pure
    protocol and child.stderr all the logs."""
    real_stdout = os.fdopen(os.dup(1), "wb")  # buffered; we flush per write
    os.dup2(2, 1)
    return real_stdout


def detect_stream_main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="glt --detect-stream")
    parser.add_argument("--detect-stream", action="store_true", required=True)
    parser.add_argument("--config", default=None,
                        help="Optional config.json overriding face_recognition settings.")
    parser.add_argument("--gpu-mem-limit", dest="gpu_mem_limit", type=float, default=None,
                        help="ONNX CUDA arena cap in GiB (omit / 0 = no cap).")
    args = parser.parse_args(argv)
    gpu_mem = args.gpu_mem_limit if (args.gpu_mem_limit and args.gpu_mem_limit > 0) else None

    # Private clean stdout for the frame protocol; fd 1 → stderr so insightface
    # & friends can't corrupt the stream when the first image loads the model.
    stdout = _hijack_stdout()
    stdin = sys.stdin.buffer
    _stderr("[stream] ready — waiting for framed image requests on stdin")

    processed = 0
    while True:
        blob = _read_frame(stdin)
        if blob is None:
            break  # parent closed stdin — clean shutdown
        try:
            result = face_service.detect_faces_blob(
                blob, config_path=args.config, gpu_mem_limit_gb=gpu_mem,
            )
        except Exception as e:  # one bad image must not kill the whole job
            _stderr(f"[stream][error] {type(e).__name__}: {e}")
            result = {
                "image_width": None,
                "image_height": None,
                "faces": [],
                "error": f"{type(e).__name__}: {e}",
            }
        _write_frame(stdout, json.dumps(result).encode("utf-8"))
        processed += 1

    _stderr(f"[stream] stdin closed after {processed} image(s) — exiting")


def detect_folders_main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="glt --detect-folders")
    parser.add_argument("--detect-folders", action="store_true", required=True)
    parser.add_argument("--paths-json", dest="paths_json", required=True,
                        help="JSON array of dataset folder paths to scan.")
    parser.add_argument("--recursive", action="store_true",
                        help="Descend into subdirectories.")
    parser.add_argument("--config", default=None,
                        help="Optional config.json overriding face_recognition settings.")
    args = parser.parse_args(argv)

    try:
        paths = json.loads(args.paths_json)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--paths-json is not valid JSON: {e}")
    if not isinstance(paths, list) or not paths:
        raise SystemExit("--paths-json must be a non-empty JSON array of folder paths")

    # Private clean stdout BEFORE loading insightface (which prints "Applied
    # providers:" etc. to stdout) — otherwise that chatter corrupts the JSON.
    out = _hijack_stdout()
    result = face_service.detect_faces(
        paths=[str(p) for p in paths],
        recursive=args.recursive,
        config_path=args.config,
    )
    # The ONLY thing on the real stdout — the dashboard reads it whole + parses.
    out.write(json.dumps(result).encode("utf-8"))
    out.flush()
