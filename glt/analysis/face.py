"""Optional face-similarity scorer (InsightFace + ArcFace).

For each generated image, runs face detection + recognition with the
configured InsightFace model and returns the cosine similarity between the
best-matching face embedding and a configured centroid.
"""
from __future__ import annotations

import base64
from typing import Any

from ..utils.cuda import preload_cuda12_libs


class FaceScorer:
    """Score generated images by cosine similarity to a reference centroid.

    "Best-matching" = max similarity across all detected faces in the image.
    Returns `None` when no face is detected.

    Config schema (under `face_recognition` in `config.json`):

        centroid: [float, ...]            # 1D embedding (512-dim for buffalo_l)
        centroid_b64: "..."               # OR base64-encoded float32 blob
        model_name: "buffalo_l"           # InsightFace model pack
        model_root: "/path/to/models"     # optional cache dir
        providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        det_size: [640, 640]
        thresholds: {"good": 0.5, "ok": 0.35}   # for HTML color coding
        cuda12_search_dirs: ["/path/to/nvidia"] # optional override
    """

    def __init__(self, cfg: dict):
        # Preload cu12 libs BEFORE insightface imports onnxruntime, otherwise
        # CUDAExecutionProvider silently falls back to CPU when the venv only
        # has cu13 (e.g. via PyTorch's nvidia-cu13 wheels).
        providers = cfg.get("providers") or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in providers:
            preload_cuda12_libs(cfg.get("cuda12_search_dirs"))

        from insightface.app import FaceAnalysis
        import numpy as np

        self.model_name = cfg.get("model_name", "buffalo_l")
        self.model_root = cfg.get("model_root")
        self.providers = providers
        det_size = tuple(cfg.get("det_size", [640, 640]))
        self.thresholds = {
            "good": float(cfg.get("thresholds", {}).get("good", 0.5)),
            "ok": float(cfg.get("thresholds", {}).get("ok", 0.35)),
        }

        kwargs: dict[str, Any] = {"name": self.model_name, "providers": self.providers}
        if self.model_root:
            kwargs["root"] = self.model_root
        print(f"[face] loading insightface model={self.model_name} providers={self.providers}")
        self.app = FaceAnalysis(**kwargs)
        self.app.prepare(ctx_id=0, det_size=det_size)

        self.centroid = self._load_centroid(cfg)
        self._np = np
        print(f"[face] centroid loaded (dim={self.centroid.shape[0]}, norm=1.0) "
              f"thresholds=good>={self.thresholds['good']:.2f} "
              f"ok>={self.thresholds['ok']:.2f}")

    @staticmethod
    def _load_centroid(cfg: dict):
        import numpy as np
        if "centroid" in cfg and cfg["centroid"] is not None:
            arr = np.asarray(cfg["centroid"], dtype=np.float32)
        elif "centroid_b64" in cfg and cfg["centroid_b64"]:
            raw = base64.b64decode(cfg["centroid_b64"])
            arr = np.frombuffer(raw, dtype=np.float32).copy()
        else:
            raise ValueError(
                "face_recognition: provide either 'centroid' (list of floats) or "
                "'centroid_b64' (base64-encoded float32 blob)."
            )
        if arr.ndim != 1:
            raise ValueError(f"face_recognition: centroid must be 1D, got shape {arr.shape}")
        n = float(np.linalg.norm(arr))
        if n == 0:
            raise ValueError("face_recognition: centroid has zero norm")
        return arr / n

    def score_pil(self, pil) -> float | None:
        import cv2
        import numpy as np
        if pil is None:
            return None
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return self._score_bgr(bgr)

    def score_path(self, path) -> float | None:
        import cv2
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        return self._score_bgr(bgr)

    def _score_bgr(self, bgr) -> float | None:
        np = self._np
        faces = self.app.get(bgr)
        if not faces:
            return None
        best = -1.0
        for f in faces:
            emb = f.embedding
            n = float(np.linalg.norm(emb))
            if n == 0:
                continue
            emb = emb / n
            sim = float(np.dot(emb, self.centroid))
            if sim > best:
                best = sim
        return best if best > -1.0 else None
