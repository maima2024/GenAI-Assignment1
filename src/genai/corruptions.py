"""Corruption definitions shared by training, evaluation and the FastAPI backend.

All functions operate on uint8 RGB numpy arrays of shape (H, W, 3).
Every corruption is fully described by a JSON-serialisable parameter dict so that
validation/test manifests are deterministic and reproducible.
"""
from __future__ import annotations

import math
from typing import Dict, List

import cv2
import numpy as np

CLASSES = ["clean", "saltpepper", "blur", "occlusion"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Fixed test severities (low, medium, high) required by the assignment.
TEST_SEVERITIES = {
    "saltpepper": [{"p": 0.03}, {"p": 0.08}, {"p": 0.15}],
    "blur": [{"k": 3, "sigma": 0.7}, {"k": 5, "sigma": 1.5}, {"k": 7, "sigma": 2.5}],
    "occlusion": [{"n": 1, "frac": 0.10}, {"n": 2, "frac": 0.20}, {"n": 3, "frac": 0.35}],
}
SEVERITY_NAMES = ["low", "medium", "high"]


# --------------------------------------------------------------------------- #
# Parameter sampling
# --------------------------------------------------------------------------- #
def make_rects(n: int, frac: float, h: int, w: int, rng: np.random.Generator) -> List[List[int]]:
    """Sample n non-overlapping black rectangles that jointly cover ~frac of the image.

    The total area is split between rectangles with a Dirichlet draw, every rectangle
    gets a random aspect ratio in [0.5, 2] and a random position. Placement is retried
    to avoid overlaps so that the *joint* coverage matches the requested fraction.
    """
    total = frac * h * w
    shares = rng.dirichlet(np.ones(n)) if n > 1 else np.array([1.0])
    shares = 0.5 * shares + 0.5 / n  # avoid degenerate tiny rectangles
    rects: List[List[int]] = []
    occ = np.zeros((h, w), dtype=bool)
    for s in shares:
        area = total * s
        for _attempt in range(60):
            aspect = rng.uniform(0.5, 2.0)
            rh = int(round(min(h, max(4, math.sqrt(area / aspect)))))
            rw = int(round(min(w, max(4, area / max(rh, 1)))))
            y = int(rng.integers(0, h - rh + 1))
            x = int(rng.integers(0, w - rw + 1))
            if not occ[y:y + rh, x:x + rw].any() or _attempt == 59:
                occ[y:y + rh, x:x + rw] = True
                rects.append([x, y, rw, rh])
                break
    return rects


def sample_params(kind: str, rng: np.random.Generator, h: int = 128, w: int = 128) -> Dict:
    """Sample training-distribution parameters for a corruption type."""
    if kind == "clean":
        return {}
    if kind == "saltpepper":
        return {"p": float(rng.uniform(0.02, 0.15)), "seed": int(rng.integers(0, 2**31 - 1))}
    if kind == "blur":
        return {"k": int(rng.choice([3, 5, 7])), "sigma": float(rng.uniform(0.5, 2.5))}
    if kind == "occlusion":
        n = int(rng.integers(1, 4))
        frac = float(rng.uniform(0.10, 0.35))
        return {"n": n, "frac": frac, "rects": make_rects(n, frac, h, w, rng)}
    raise ValueError(kind)


def fixed_params(kind: str, level: int, rng: np.random.Generator, h: int = 128, w: int = 128) -> Dict:
    """Parameters for one of the three fixed test severity levels (0=low,1=med,2=high)."""
    if kind == "clean":
        return {}
    base = dict(TEST_SEVERITIES[kind][level])
    if kind == "saltpepper":
        base["seed"] = int(rng.integers(0, 2**31 - 1))
    if kind == "occlusion":
        base["rects"] = make_rects(base["n"], base["frac"], h, w, rng)
    return base


def severity_of(kind: str, params: Dict) -> str:
    """Bucket a (possibly random) parameter set into low / medium / high."""
    if kind == "clean":
        return "none"
    if kind == "saltpepper":
        v, cuts = params["p"], (0.0567, 0.1033)
    elif kind == "blur":
        v, cuts = params["sigma"], (1.1667, 1.8333)
    else:
        v, cuts = params.get("frac", 0.2), (0.1833, 0.2667)
    return "low" if v < cuts[0] else ("medium" if v < cuts[1] else "high")


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
def apply_corruption(img: np.ndarray, kind: str, params: Dict) -> np.ndarray:
    """Apply a corruption described by (kind, params) to a uint8 RGB image."""
    out = img.copy()
    h, w = out.shape[:2]
    if kind == "clean":
        return out
    if kind == "saltpepper":
        rng = np.random.default_rng(params.get("seed", 0))
        mask = rng.random((h, w)) < params["p"]
        salt = rng.random((h, w)) < 0.5
        out[mask & salt] = 255
        out[mask & ~salt] = 0
        return out
    if kind == "blur":
        k = int(params["k"])
        return cv2.GaussianBlur(out, (k, k), float(params["sigma"]), borderType=cv2.BORDER_REFLECT)
    if kind == "occlusion":
        rects = params.get("rects")
        if rects is None:
            rects = make_rects(params["n"], params["frac"], h, w, np.random.default_rng(params.get("seed", 0)))
        for x, y, rw, rh in rects:
            out[y:y + rh, x:x + rw] = 0
        return out
    raise ValueError(kind)


def occlusion_coverage(params: Dict, h: int = 128, w: int = 128) -> float:
    m = np.zeros((h, w), bool)
    for x, y, rw, rh in params.get("rects", []):
        m[y:y + rh, x:x + rw] = True
    return float(m.mean())
