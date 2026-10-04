"""FS2K paired photo -> sketch data for Task 4."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .common import load_json, save_json

SIZE = 128


def locate_fs2k(search: List[str], extract_to: str = "data/raw") -> Path:
    for s in search:
        b = Path(s)
        if not b.exists():
            continue
        for p in [b, *b.rglob("anno_train.json")]:
            root = p if p.is_dir() else p.parent
            if (root / "anno_train.json").exists() and (root / "sketch").is_dir():
                return root
    for s in search:
        b = Path(s)
        if not b.exists():
            continue
        zips = [b] if b.suffix == ".zip" else list(b.rglob("FS2K*.zip"))
        for z in zips:
            print("extracting", z)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(extract_to)
            return locate_fs2k([extract_to])
    raise FileNotFoundError(f"FS2K not found in {search}")


def _find(folder: Path, stem: str) -> Path:
    for ext in [".jpg", ".png", ".JPG", ".jpeg", ".PNG"]:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    hits = list(folder.glob(stem + ".*"))
    if hits:
        return hits[0]
    raise FileNotFoundError(folder / stem)


def build_cache(search: List[str], cache_dir="data/cache") -> Path:
    cache = Path(cache_dir) / "fs2k128.npz"
    if cache.exists():
        return cache
    root = locate_fs2k(search)
    out = {}
    for split in ["train", "test"]:
        anno = json.loads((root / f"anno_{split}.json").read_text())
        P, S, Y, N = [], [], [], []
        for a in anno:
            sub, name = a["image_name"].split("/")  # photo1/image0110
            num = name.replace("image", "")
            pp = _find(root / "photo" / sub, name)
            sp = _find(root / "sketch" / sub.replace("photo", "sketch"), f"sketch{num}")
            with Image.open(pp) as im:
                P.append(np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)))
            with Image.open(sp) as im:
                S.append(np.asarray(im.convert("L").resize((SIZE, SIZE), Image.BICUBIC)))
            Y.append(int(a["style"]))
            N.append(a["image_name"])
        out[f"{split}_photo"], out[f"{split}_sketch"] = np.stack(P), np.stack(S)
        out[f"{split}_style"], out[f"{split}_names"] = np.array(Y), np.array(N)
        print(split, out[f"{split}_photo"].shape, np.bincount(out[f"{split}_style"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    # 15% of the official training portion for validation, stratified by style, seed 42
    idx = np.arange(len(out["train_style"]))
    tr, va = train_test_split(idx, test_size=0.15, random_state=42, stratify=out["train_style"])
    save_json({"train": sorted(tr.tolist()), "val": sorted(va.tolist())}, Path(cache_dir) / "fs2k_split.json")
    return cache


def load(cache_dir="data/cache"):
    d = dict(np.load(Path(cache_dir) / "fs2k128.npz"))
    d["split"] = load_json(Path(cache_dir) / "fs2k_split.json")
    return d


class PairedSketchDataset(Dataset):
    """Returns photo in [-1,1] (3ch), sketch in [-1,1] (1ch), style id.

    Augmentation (training only) is applied IDENTICALLY to photo and sketch:
    the pair is concatenated channel-wise before resize -> random crop -> flip, so the
    pixel correspondence is preserved.
    """

    def __init__(self, photos, sketches, styles, indices, augment=False):
        self.p, self.s, self.y = photos, sketches, styles
        self.idx = list(indices)
        self.augment = augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        ph = torch.from_numpy(self.p[j]).permute(2, 0, 1).float() / 127.5 - 1
        sk = torch.from_numpy(self.s[j])[None].float() / 127.5 - 1
        if self.augment:
            pair = torch.cat([ph, sk], 0)[None]
            big = int(np.random.randint(SIZE, 145))
            pair = F.interpolate(pair, size=(big, big), mode="bilinear", align_corners=False)[0]
            y0, x0 = np.random.randint(0, big - SIZE + 1, 2)
            pair = pair[:, y0:y0 + SIZE, x0:x0 + SIZE]
            if np.random.rand() < 0.5:
                pair = pair.flip(2)
            ph, sk = pair[:3], pair[3:]
        return ph, sk, int(self.y[j])
