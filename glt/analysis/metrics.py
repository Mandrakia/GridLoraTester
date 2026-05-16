"""Per-row similarity-score aggregation."""
from __future__ import annotations


def compute_row_metrics(scores: list) -> dict:
    """Aggregate per-image similarity scores into row-level metrics.

    `scores` is a list of (float | None) — None means "no face detected"."""
    import numpy as np
    valid = [s for s in scores if s is not None]
    metrics = {
        "n_total": len(scores),
        "n_faces": len(valid),
    }
    if valid:
        a = np.asarray(valid, dtype=np.float32)
        metrics.update({
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "std": float(a.std()),
            "p20": float(np.percentile(a, 20)),
            "p80": float(np.percentile(a, 80)),
            "min": float(a.min()),
            "max": float(a.max()),
        })
    return metrics
