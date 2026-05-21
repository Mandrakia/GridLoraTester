"""InsightFace facade used by the HTTP worker. Model is loaded lazily on the
first call and kept warm in a module-level singleton so subsequent
`/detect-faces` requests skip the ~3 s startup."""
from __future__ import annotations

import base64
import sys
import threading
from pathlib import Path

import numpy as np

from ..utils.config import load_config
from ..utils.cuda import preload_cuda12_libs


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_app = None
_app_lock = threading.Lock()
_app_cfg_fingerprint: str | None = None
_inference_count = 0

# Periodic session recycle bounds the worst-case VRAM accumulation. ORT's
# BFC arena never returns memory to CUDA during a session's lifetime, so
# the high-water mark across all images processed becomes permanent. For
# datasets with outliers (group photos / collages with 20-30+ faces — each
# face triggers a recognition + landmark + pose sub-model run, growing the
# arena), the peak quickly hits the GPU's hard limit and subsequent images
# OOM forever.
#
# We periodically drop the cached InsightFace app and let the next call
# reload it (~3s cost). This actually releases the arena back to CUDA.
#   - RECYCLE_EVERY_N_IMAGES: belt-and-braces bound regardless of payload.
#   - RECYCLE_AFTER_NFACES: immediate trigger after a "heavy" image, since
#     that's the one that just spiked the arena. Catches the bomb that the
#     periodic check would miss between cadence boundaries.
RECYCLE_EVERY_N_IMAGES = 200
RECYCLE_AFTER_NFACES = 10


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _recycle_app() -> None:
    """Drop the cached InsightFace app + force GC so the next call reloads
    fresh. Called either on a fixed cadence or after a heavy image. The
    actual reload happens lazily inside `_get_app` on the next request."""
    global _app, _app_cfg_fingerprint, _inference_count
    with _app_lock:
        prev = _app
        _app = None
        _app_cfg_fingerprint = None
        _inference_count = 0
    if prev is not None:
        del prev
    import gc
    gc.collect()
    _stderr('[face] session recycled — releasing arena back to CUDA')


def _cfg_fingerprint(face_cfg: dict) -> str:
    """Detect when the user changed model/providers/det_size/gpu_mem since
    last load — we drop the cached app so the next call picks up the new
    config. Includes `gpu_mem_limit_gb` because the provider-options
    monkey-patch is one-shot and only takes effect on session (re)creation."""
    return repr((
        face_cfg.get("model_name", "buffalo_l"),
        tuple(face_cfg.get("providers") or ()),
        tuple(face_cfg.get("det_size") or ()),
        face_cfg.get("model_root"),
        face_cfg.get("gpu_mem_limit_gb"),
    ))


def _get_app(face_cfg: dict):
    global _app, _app_cfg_fingerprint
    fp = _cfg_fingerprint(face_cfg)
    with _app_lock:
        if _app is not None and _app_cfg_fingerprint == fp:
            return _app

        providers = face_cfg.get("providers") or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in providers:
            preload_cuda12_libs(face_cfg.get("cuda12_search_dirs"))

        from insightface.app import FaceAnalysis
        model_name = face_cfg.get("model_name", "buffalo_l")
        det_size = tuple(face_cfg.get("det_size", [640, 640]))
        # Pass ONLY `providers` — NO provider_options. Forwarding CUDA-EP
        # provider_options through insightface here makes the CUDA EP silently
        # fall back to CPU, while the grid's FaceScorer (analysis/face.py),
        # which passes just `providers`, binds CUDA fine on the SAME image +
        # ORT. The gpu-mem cap those options carried is off-by-default anyway
        # (a hard cap OOMs ORT's BFC arena), so dropping them costs nothing.
        kwargs = {"name": model_name, "providers": providers}
        if face_cfg.get("model_root"):
            kwargs["root"] = face_cfg["model_root"]
        _stderr(f"[face] loading insightface model={model_name} providers={providers}")
        app = FaceAnalysis(**kwargs)
        app.prepare(ctx_id=0, det_size=det_size)

        # Surface the providers each ONNX session ACTUALLY bound. onnxruntime
        # silently drops CUDAExecutionProvider (→ CPU-only) when it can't load
        # its CUDA libs; that fallback is otherwise invisible — you only notice
        # via wall-clock. If we asked for CUDA but every session came back
        # CPU-only, shout about it so it shows up in the job log.
        try:
            active = sorted({
                p
                for m in app.models.values()
                if getattr(m, "session", None) is not None
                for p in m.session.get_providers()
            })
        except Exception as e:  # never let introspection break loading
            active = []
            _stderr(f"[face] could not read active providers: {e!r}")
        _stderr(f"[face] active ONNX providers={active}")
        if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in active:
            _stderr(
                "[face] WARNING: requested CUDA but ONNX bound CPU-only — "
                "onnxruntime could not load its CUDA libs (silent fallback). "
                "Running on CPU; check the [cuda] preload line above."
            )

        _app = app
        _app_cfg_fingerprint = fp
        return _app


def _b64_float32(arr) -> str:
    return base64.b64encode(np.asarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def _iter_images(folder: Path, recursive: bool):
    it = folder.rglob("*") if recursive else sorted(folder.iterdir())
    for p in it:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _detect_one(app, image_path: Path):
    """Returns (image_width, image_height, faces[]). The dims travel with
    the faces because the framing-distance proxy (face size relative to
    frame) needs to know what 'full frame' is for this image."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None, []
    h, w = img.shape[:2]
    faces = app.get(img)
    out = []
    for i, f in enumerate(faces):
        emb = np.asarray(f.embedding, dtype=np.float32)
        n = float(np.linalg.norm(emb))
        if n == 0:
            continue
        emb = emb / n
        # face.pose is (pitch, yaw, roll) in degrees, populated by the
        # landmark_3d_68 head in buffalo_l. It can be None on detectors that
        # don't ship a pose subnet — fall back to None for those slots.
        pose = getattr(f, "pose", None)
        if pose is not None:
            try:
                pitch, yaw, roll = [float(x) for x in pose]
            except (TypeError, ValueError):
                pitch = yaw = roll = None
        else:
            pitch = yaw = roll = None
        out.append({
            "face_index": i,
            "bbox": [float(x) for x in f.bbox.tolist()],
            "det_score": float(getattr(f, "det_score", 0.0) or 0.0),
            "embedding_b64": _b64_float32(emb),
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
        })
    return int(w), int(h), out


def detect_faces_blob(
    image_bytes: bytes,
    config_path: str | None = None,
    gpu_mem_limit_gb: float | None = None,
) -> dict:
    """Single-image variant of `detect_faces`, working on raw bytes.
    Used by jobs that stream pictures from a connector — we don't write the
    blob to disk just to read it back.

    `gpu_mem_limit_gb` is an optional per-request override (from the
    dashboard's settings) for the ONNX CUDA sessions' VRAM cap. When set
    it's merged into `face_cfg` and participates in the cache fingerprint,
    so a change reloads the sessions on the next call.

    Returns `{"image_width", "image_height", "faces"}` with the same face
    dict shape as `detect_faces`. Empty `faces` (or null dims) when the
    decoder can't parse the bytes."""
    import cv2
    import numpy as _np
    import time

    cfg_root = load_config(Path(config_path) if config_path else None)
    face_cfg = dict(cfg_root.get("face_recognition") or {})
    if gpu_mem_limit_gb is not None:
        face_cfg["gpu_mem_limit_gb"] = gpu_mem_limit_gb
    app = _get_app(face_cfg)

    # Phase timing — surfaced in the response so the dashboard can compute
    # p50/p95 across the run without an extra round-trip per image. perf_counter
    # gives wall-clock-monotonic ns precision.
    t0 = time.perf_counter()
    arr = _np.frombuffer(image_bytes, dtype=_np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    t_decoded = time.perf_counter()
    if img is None:
        return {
            "image_width": None,
            "image_height": None,
            "faces": [],
            "timing_ms": {
                "decode": (t_decoded - t0) * 1000.0,
                "detect": 0.0,
                "total": (t_decoded - t0) * 1000.0,
            },
        }
    h, w = img.shape[:2]

    faces = app.get(img)
    t_detected = time.perf_counter()
    out = []
    for i, f in enumerate(faces):
        emb = np.asarray(f.embedding, dtype=np.float32)
        n = float(np.linalg.norm(emb))
        if n == 0:
            continue
        emb = emb / n
        pose = getattr(f, "pose", None)
        if pose is not None:
            try:
                pitch, yaw, roll = [float(x) for x in pose]
            except (TypeError, ValueError):
                pitch = yaw = roll = None
        else:
            pitch = yaw = roll = None
        out.append({
            "face_index": i,
            "bbox": [float(x) for x in f.bbox.tolist()],
            "det_score": float(getattr(f, "det_score", 0.0) or 0.0),
            "embedding_b64": _b64_float32(emb),
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
        })
    t_done = time.perf_counter()
    result = {
        "image_width": int(w),
        "image_height": int(h),
        "faces": out,
        "timing_ms": {
            "decode": (t_decoded - t0) * 1000.0,
            "detect": (t_detected - t_decoded) * 1000.0,
            "total": (t_done - t0) * 1000.0,
        },
    }

    # Bound the VRAM high-water mark: recycle after `RECYCLE_EVERY_N_IMAGES`
    # OR immediately after a heavy image. Both checks live OUTSIDE the
    # response so the caller already has its result by the time we reload
    # — only the NEXT request pays the ~3 s reload cost. Increment with
    # the lock since multiple HTTP threads may share the worker.
    global _inference_count
    with _app_lock:
        _inference_count += 1
        triggered = (
            _inference_count >= RECYCLE_EVERY_N_IMAGES
            or len(out) >= RECYCLE_AFTER_NFACES
        )
    if triggered:
        _recycle_app()

    return result


def detect_faces(paths: list[str], recursive: bool = False,
                 config_path: str | None = None) -> dict:
    """Run detection + embedding for every image in every dataset folder.
    Returns the same `{"datasets": [{path, images: [...]}]}` shape the
    one-shot CLI used to print, but no JSON-on-stdout to parse anymore."""
    cfg_root = load_config(Path(config_path) if config_path else None)
    face_cfg = (cfg_root.get("face_recognition") or {})
    app = _get_app(face_cfg)

    folders: list[Path] = []
    for p in paths:
        d = Path(p).expanduser().resolve()
        if not d.is_dir():
            _stderr(f"[face][skip] not a directory: {d}")
            continue
        folders.append(d)
    if not folders:
        raise ValueError("no valid directories given")

    datasets_out = []
    for folder in folders:
        _stderr(f"[face] scanning {folder}")
        images_out = []
        for img in _iter_images(folder, recursive):
            w, h, faces = _detect_one(app, img)
            images_out.append({
                "image_path": str(img),
                "image_width": w,
                "image_height": h,
                "faces": faces,
            })
            label = f"{len(faces)} face(s)" if faces else "no face"
            _stderr(f"  {img.name:<60s} {label}")
        datasets_out.append({"path": str(folder), "images": images_out})

    return {"datasets": datasets_out}
