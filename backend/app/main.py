"""FastAPI backend serving the four generative models through ONNX Runtime."""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

try:  # inside the Docker image the shared file is copied next to this module
    from .corruptions import CLASSES, TEST_SEVERITIES, apply_corruption, fixed_params, occlusion_coverage
except ImportError:  # local development: use the training package directly
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from genai.corruptions import CLASSES, TEST_SEVERITIES, apply_corruption, fixed_params, occlusion_coverage
from .models import REGISTRY, ensure_models
from .quality import psnr, ssim

MAX_BYTES = 10 * 1024 * 1024
SIZE = 128
SAMPLES_DIR = Path(__file__).parent / "samples"
BRANCHES = ["identity", "saltpepper", "blur", "occlusion"]
EXPERT_NAMES = {"clean": "Identity bypass", "saltpepper": "Salt-and-pepper expert", "blur": "Blur expert",
                "occlusion": "Occlusion expert"}

app = FastAPI(title="GenAI Assignment 1 - Restoration & Sketch API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    ensure_models()
    REGISTRY.load()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def _read_image(file: Optional[UploadFile], sample: Optional[str], mode="RGB") -> np.ndarray:
    if file is not None and file.filename:
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(415, f"Unsupported file type {file.content_type}; please upload an image")
        data = await file.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Image larger than 10 MB")
        try:
            im = Image.open(io.BytesIO(data))
            im = ImageOps.exif_transpose(im)
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "File could not be decoded as an image")
    elif sample:
        p = (SAMPLES_DIR / Path(sample).name)
        if not p.exists():
            raise HTTPException(404, "Unknown sample")
        im = Image.open(p)
    else:
        raise HTTPException(400, "Provide an image file or choose a sample")
    im = im.convert(mode).resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def _prepare(img: np.ndarray, corruption: str, severity: str, seed: int):
    """Optionally corrupt a clean image with one of the fixed test severities."""
    if corruption not in CLASSES:
        raise HTTPException(400, f"corruption must be one of {CLASSES}")
    if corruption == "clean":
        return img.copy(), {"kind": "clean"}
    lvl = {"low": 0, "medium": 1, "high": 2}.get(severity)
    if lvl is None:
        raise HTTPException(400, "severity must be low, medium or high")
    params = fixed_params(corruption, lvl, np.random.default_rng(seed))
    out = apply_corruption(img, corruption, params)
    info = {"kind": corruption, "severity": severity, "seed": seed, **params}
    if corruption == "occlusion":
        info["coverage"] = round(occlusion_coverage(params), 4)
    return out, info


def _to_input(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def _to_img(t: np.ndarray) -> np.ndarray:
    return (np.clip(t[0].transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)


def _metrics(out, clean):
    if clean is None:
        return None
    return {"psnr": round(psnr(out, clean), 2), "ssim": round(ssim(out, clean), 4)}


async def _inputs(file, sample, corruption, severity, seed):
    img = await _read_image(file, sample)
    # An uploaded image with corruption="clean" is treated as an already-corrupted input (no reference)
    if corruption == "clean" and file is not None and file.filename:
        return img, None, {"kind": "uploaded (no synthetic corruption)"}
    corrupted, info = _prepare(img, corruption, severity, seed)
    return corrupted, img, info


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok" if REGISTRY.ready() else "degraded", "models": REGISTRY.status(),
            "onnxruntime": REGISTRY.ort_version, "providers": REGISTRY.providers}


@app.get("/api/samples")
def samples():
    files = sorted(p.name for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    return {"pets": [f for f in files if f.startswith("pet")], "faces": [f for f in files if f.startswith("face")]}


@app.get("/api/samples/{name}")
def sample_image(name: str):
    from fastapi.responses import FileResponse
    p = SAMPLES_DIR / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/corruptions")
def corruptions():
    return {"classes": CLASSES, "test_severities": TEST_SEVERITIES}


@app.post("/api/restore/universal")
async def universal(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                    corruption: str = Form("clean"), severity: str = Form("medium"), seed: int = Form(0)):
    x, clean, info = await _inputs(file, sample, corruption, severity, seed)
    t0 = time.perf_counter()
    out = REGISTRY.run("udae", {"input": _to_input(x)})[0]
    ms = (time.perf_counter() - t0) * 1000
    out = _to_img(out)
    return {"input": _b64(x), "clean": _b64(clean) if clean is not None else None, "output": _b64(out),
            "corruption": info, "inference_ms": round(ms, 2), "metrics": _metrics(out, clean),
            "input_metrics": _metrics(x, clean), "model": "udae.onnx"}


@app.post("/api/restore/hard")
async def hard(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
               corruption: str = Form("clean"), severity: str = Form("medium"), seed: int = Form(0),
               route_mode: str = Form("predicted")):
    x, clean, info = await _inputs(file, sample, corruption, severity, seed)
    inp = _to_input(x)
    t0 = time.perf_counter()
    _, probs = REGISTRY.run("classifier", {"input": inp})
    t_clf = (time.perf_counter() - t0) * 1000
    probs = probs[0]
    pred = CLASSES[int(np.argmax(probs))]
    chosen = pred
    if route_mode == "oracle" and info.get("kind") in CLASSES:
        chosen = info["kind"]
    t1 = time.perf_counter()
    if chosen == "clean":
        out = inp  # identity bypass - no expert executed
    else:
        out = REGISTRY.run(f"specialist_{chosen}", {"input": inp})[0]
    t_exp = (time.perf_counter() - t1) * 1000
    out = _to_img(out)
    return {"input": _b64(x), "clean": _b64(clean) if clean is not None else None, "output": _b64(out),
            "corruption": info, "probabilities": dict(zip(CLASSES, [round(float(p), 5) for p in probs])),
            "predicted": pred, "routed_to": chosen, "expert": EXPERT_NAMES[chosen], "route_mode": route_mode,
            "inference_ms": round(t_clf + t_exp, 2), "classifier_ms": round(t_clf, 2), "expert_ms": round(t_exp, 2),
            "metrics": _metrics(out, clean), "input_metrics": _metrics(x, clean)}


@app.post("/api/restore/soft")
async def soft(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
               corruption: str = Form("clean"), severity: str = Form("medium"), seed: int = Form(0)):
    x, clean, info = await _inputs(file, sample, corruption, severity, seed)
    t0 = time.perf_counter()
    out, w, logits = REGISTRY.run("soft_moe", {"input": _to_input(x)})
    ms = (time.perf_counter() - t0) * 1000
    out = _to_img(out)
    w = w[0].tolist()
    return {"input": _b64(x), "clean": _b64(clean) if clean is not None else None, "output": _b64(out),
            "corruption": info, "weights": dict(zip(BRANCHES, [round(float(v), 5) for v in w])),
            "dominant": BRANCHES[int(np.argmax(w))], "inference_ms": round(ms, 2),
            "metrics": _metrics(out, clean), "input_metrics": _metrics(x, clean)}


@app.post("/api/sketch")
async def sketch(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None), style: int = Form(1)):
    if style not in (1, 2, 3):
        raise HTTPException(400, "style must be 1, 2 or 3")
    img = await _read_image(file, sample)
    photo = (img.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
    t0 = time.perf_counter()
    sk = REGISTRY.run("face2sketch_generator", {"photo": photo, "style": np.array([style - 1], dtype=np.int64)})[0]
    ms = (time.perf_counter() - t0) * 1000
    sk = ((np.clip(sk[0, 0], -1, 1) + 1) * 127.5).round().astype(np.uint8)
    return {"photo": _b64(img), "sketch": _b64(sk), "style": style, "inference_ms": round(ms, 2)}
