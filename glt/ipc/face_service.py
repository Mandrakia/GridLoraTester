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


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _cfg_fingerprint(face_cfg: dict) -> str:
    """Detect when the user changed model/providers/det_size since last load —
    we drop the cached app so the next call picks up the new config."""
    return repr((
        face_cfg.get("model_name", "buffalo_l"),
        tuple(face_cfg.get("providers") or ()),
        tuple(face_cfg.get("det_size") or ()),
        face_cfg.get("model_root"),
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
        kwargs = {"name": model_name, "providers": providers}
        if face_cfg.get("model_root"):
            kwargs["root"] = face_cfg["model_root"]
        _stderr(f"[face] loading insightface model={model_name} providers={providers}")
        app = FaceAnalysis(**kwargs)
        app.prepare(ctx_id=0, det_size=det_size)
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


def detect_faces_blob(image_bytes: bytes, config_path: str | None = None) -> dict:
    """Single-image variant of `detect_faces`, working on raw bytes.
    Used by jobs that stream pictures from a connector — we don't write the
    blob to disk just to read it back.

    Returns `{"image_width", "image_height", "faces"}` with the same face
    dict shape as `detect_faces`. Empty `faces` (or null dims) when the
    decoder can't parse the bytes."""
    import cv2
    import numpy as _np

    cfg_root = load_config(Path(config_path) if config_path else None)
    face_cfg = (cfg_root.get("face_recognition") or {})
    app = _get_app(face_cfg)

    arr = _np.frombuffer(image_bytes, dtype=_np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"image_width": None, "image_height": None, "faces": []}
    h, w = img.shape[:2]

    faces = app.get(img)
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
    return {"image_width": int(w), "image_height": int(h), "faces": out}


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
