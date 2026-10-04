"""ONNX model registry. Models are read from MODELS_DIR; if missing they are downloaded once
from MODELS_URL: either a GitHub release download base (each <name>.onnx is fetched)
or a .zip containing the .onnx files."""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
MODELS_URL = os.environ.get("MODELS_URL", "")
NAMES = ["udae", "classifier", "specialist_saltpepper", "specialist_blur", "specialist_occlusion", "soft_moe",
         "face2sketch_generator"]


def ensure_models():
    missing = [n for n in NAMES if not (MODELS_DIR / f"{n}.onnx").exists()]
    if not missing or not MODELS_URL:
        return
    print(f"Downloading models ({len(missing)} missing) from {MODELS_URL}", flush=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for n in missing:
        try:
            if MODELS_URL.endswith(".zip"):
                with urllib.request.urlopen(MODELS_URL, timeout=900) as r:
                    with zipfile.ZipFile(io.BytesIO(r.read())) as z:
                        for m in z.namelist():
                            if m.endswith(".onnx"):
                                (MODELS_DIR / Path(m).name).write_bytes(z.read(m))
                break
            url = f"{MODELS_URL.rstrip('/')}/{n}.onnx"
            print("  ->", url, flush=True)
            tmp = MODELS_DIR / f"{n}.onnx.part"
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(MODELS_DIR / f"{n}.onnx")
        except Exception as e:  # keep serving; /api/health reports what is missing
            print(f"Model download failed for {n}:", e, flush=True)


class Registry:
    def __init__(self):
        self.sessions = {}
        self.errors = {}
        self.ort_version = ort.__version__
        self.providers = ort.get_available_providers()

    def load(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
        for n in NAMES:
            p = MODELS_DIR / f"{n}.onnx"
            try:
                self.sessions[n] = ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
            except Exception as e:
                self.errors[n] = str(e)[:200]
        print("loaded models:", list(self.sessions), "errors:", self.errors, flush=True)

    def run(self, name, feeds):
        if name not in self.sessions:
            from fastapi import HTTPException
            raise HTTPException(503, f"Model {name}.onnx is not loaded: {self.errors.get(name, 'missing')}")
        return self.sessions[name].run(None, feeds)

    def ready(self):
        return all(n in self.sessions for n in NAMES)

    def status(self):
        out = {}
        for n in NAMES:
            p = MODELS_DIR / f"{n}.onnx"
            out[n] = {"loaded": n in self.sessions, "size_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else None}
            if n in self.sessions:
                s = self.sessions[n]
                out[n]["inputs"] = [f"{i.name}{i.shape}" for i in s.get_inputs()]
                out[n]["outputs"] = [f"{o.name}{o.shape}" for o in s.get_outputs()]
        return out


REGISTRY = Registry()
