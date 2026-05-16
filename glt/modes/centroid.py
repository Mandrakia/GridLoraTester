"""`glt --compute-centroid` — average face embeddings into a reference centroid."""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from ..utils.config import load_config
from ..utils.cuda import preload_cuda12_libs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="glt --compute-centroid",
        description="Compute a face-centroid embedding by averaging face embeddings "
                    "from one or more directories of reference images. Output is "
                    "ready to paste into `config.json` under "
                    "`face_recognition.centroid` (or `centroid_b64`). Reads insightface "
                    "settings (model_name, model_root, providers, det_size) from "
                    "--config when provided.",
    )
    parser.add_argument("--compute-centroid", dest="ref_dirs", nargs="+", required=True,
                        metavar="DIR",
                        help="One or more reference-image directories (averaged together)")
    parser.add_argument("--recursive", action="store_true",
                        help="Also walk subdirectories of each input dir")
    parser.add_argument("--centroid-out", dest="out_file", default=None,
                        help="Optional path to write the centroid JSON to (otherwise stdout)")
    parser.add_argument("--centroid-format", choices=["json", "b64"], default="json",
                        help="json = list of floats; b64 = base64-encoded float32 blob")
    parser.add_argument("--config", default=None,
                        help="Optional config.json sourcing model_name/model_root/"
                             "providers/det_size")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    cfg_root = load_config(Path(args.config) if args.config else None)
    face_cfg = cfg_root.get("face_recognition", {}) or {}
    providers = face_cfg.get("providers") or ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in providers:
        preload_cuda12_libs(face_cfg.get("cuda12_search_dirs"))

    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis

    model_name = face_cfg.get("model_name", "buffalo_l")
    det_size = tuple(face_cfg.get("det_size", [640, 640]))
    kwargs = {"name": model_name, "providers": providers}
    if face_cfg.get("model_root"):
        kwargs["root"] = face_cfg["model_root"]
    print(f"[centroid] loading insightface model={model_name} providers={providers}")
    app = FaceAnalysis(**kwargs)
    app.prepare(ctx_id=0, det_size=det_size)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_embeds = []
    per_dir_counts = []

    for d_str in args.ref_dirs:
        d = Path(d_str).expanduser().resolve()
        if not d.is_dir():
            print(f"[skip] not a directory: {d}")
            continue
        print(f"\n[dir] {d}")
        iterator = d.rglob("*") if args.recursive else sorted(d.iterdir())
        n_dir = 0
        for img_path in iterator:
            if not img_path.is_file() or img_path.suffix.lower() not in exts:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [skip] cannot read {img_path.name}")
                continue
            faces = app.get(img)
            if not faces:
                print(f"  [skip] no face in {img_path.name}")
                continue
            # Largest face = most likely the subject.
            f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            e = f.embedding / np.linalg.norm(f.embedding)
            all_embeds.append(e.astype(np.float32))
            n_dir += 1
            print(f"  [ok]   {img_path.name}")
        per_dir_counts.append((str(d), n_dir))

    if not all_embeds:
        sys.exit("[error] no faces extracted from the given directories")

    centroid = np.mean(np.stack(all_embeds, axis=0), axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    centroid = centroid.astype(np.float32)

    print("\n[centroid] per-dir face counts:")
    for d, n in per_dir_counts:
        print(f"  {n:>4}  {d}")
    print(f"[centroid] total {len(all_embeds)} face(s), dim={centroid.shape[0]}")

    if args.centroid_format == "b64":
        payload = base64.b64encode(centroid.tobytes()).decode("ascii")
        out_dict = {"face_recognition": {"centroid_b64": payload}}
    else:
        out_dict = {"face_recognition": {"centroid": centroid.tolist()}}

    snippet = json.dumps(out_dict, indent=2, ensure_ascii=False)
    if args.out_file:
        Path(args.out_file).expanduser().write_text(snippet + "\n", encoding="utf-8")
        print(f"[centroid] wrote {args.out_file}")
    else:
        print("[centroid] paste this into your config.json:")
        print(snippet)
