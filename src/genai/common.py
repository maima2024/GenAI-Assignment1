"""Shared utilities: seeding, device, SSIM/PSNR, MLflow helpers, budgets."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Compute budgets. "smoke" is used for CPU testing, "fast"/"full" on a GPU.
# --------------------------------------------------------------------------- #
BUDGETS = {
    "smoke": dict(subset=64, t1_trials=2, t1_trial_epochs=1, t1_epochs=1,
                  clf_trials=2, clf_trial_epochs=1, clf_epochs=1,
                  spec_trials=2, spec_trial_epochs=1, spec_epochs=1,
                  moe_trials=2, moe_trial_epochs=1, moe_warmup=1, moe_epochs=1,
                  gan_trials=2, gan_trial_epochs=1, gan_epochs=2, test_limit=24, workers=0),
    "fast": dict(subset=None, t1_trials=12, t1_trial_epochs=4, t1_epochs=25,
                 clf_trials=10, clf_trial_epochs=3, clf_epochs=12,
                 spec_trials=10, spec_trial_epochs=3, spec_epochs=20,
                 moe_trials=8, moe_trial_epochs=2, moe_warmup=2, moe_epochs=8,
                 gan_trials=10, gan_trial_epochs=6, gan_epochs=80, test_limit=None, workers=2),
    "full": dict(subset=None, t1_trials=20, t1_trial_epochs=5, t1_epochs=40,
                 clf_trials=15, clf_trial_epochs=4, clf_epochs=15,
                 spec_trials=12, spec_trial_epochs=4, spec_epochs=30,
                 moe_trials=10, moe_trial_epochs=3, moe_warmup=3, moe_epochs=12,
                 gan_trials=12, gan_trial_epochs=8, gan_epochs=120, test_limit=None, workers=2),
}


def get_budget(name: str | None = None) -> dict:
    name = name or os.environ.get("BUDGET", "fast")
    b = dict(BUDGETS[name])
    b["name"] = name
    return b


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def out_dir() -> Path:
    p = Path(os.environ.get("OUT_DIR", "outputs"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Image quality metrics (differentiable SSIM, implemented after Wang et al. 2004)
# --------------------------------------------------------------------------- #
def _gauss_window(size: int = 11, sigma: float = 1.5, ch: int = 3, dev=None):
    g = torch.arange(size, dtype=torch.float32, device=dev) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    w = (g @ g.t()).expand(ch, 1, size, size).contiguous()
    return w


def ssim(x: torch.Tensor, y: torch.Tensor, reduction: str = "mean", data_range: float = 1.0):
    """SSIM for images in [0, data_range]; returns mean or per-image values."""
    ch = x.shape[1]
    w = _gauss_window(ch=ch, dev=x.device).to(x.dtype)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    pad = w.shape[-1] // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=ch)
    mu_y = F.conv2d(y, w, padding=pad, groups=ch)
    sxx = F.conv2d(x * x, w, padding=pad, groups=ch) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=ch) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=ch) - mu_x * mu_y
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    per = m.flatten(1).mean(1)
    return per.mean() if reduction == "mean" else per


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0):
    mse = ((x - y) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(data_range ** 2 / mse)


def recon_loss(xhat, x, alpha: float):
    """L_UDAE = alpha * L1 + (1 - alpha) * (1 - SSIM)."""
    return alpha * F.l1_loss(xhat, x) + (1 - alpha) * (1 - ssim(xhat, x))


# --------------------------------------------------------------------------- #
# MLflow helpers (file store so runs can be zipped and viewed offline)
# --------------------------------------------------------------------------- #
def setup_mlflow(experiment: str):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # file store = easy to zip/merge/serve offline
    import mlflow
    uri = os.environ.get("MLFLOW_TRACKING_URI", (out_dir() / "mlruns").resolve().as_uri())
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return mlflow


@contextmanager
def timer():
    t = {"start": time.time()}
    yield t
    t["elapsed"] = time.time() - t["start"]


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) tensor in [0,1] -> HxWx3 uint8."""
    a = t.detach().clamp(0, 1).cpu().numpy()
    if a.shape[0] == 1:
        a = np.repeat(a, 3, 0)
    return (a.transpose(1, 2, 0) * 255).round().astype(np.uint8)


def log_light_artifacts(mlflow, folder, artifact_path):
    """Log figures / json / csv from a folder (skip large checkpoints)."""
    folder = Path(folder)
    for f in sorted(folder.rglob("*")):
        if f.is_file() and f.suffix in {".png", ".json", ".csv"}:
            rel = f.parent.relative_to(folder)
            mlflow.log_artifact(str(f), artifact_path=str(Path(artifact_path) / rel) if str(rel) != "." else artifact_path)
