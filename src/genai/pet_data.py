"""Oxford-IIIT Pet data pipeline for Tasks 1-3.

* Locates/extracts the official archives (or downloads them with torchvision).
* Converts every image to RGB, resizes to 128x128 and caches uint8 arrays.
* Splits the official trainval list 80/20 with seed 42 (same split for Tasks 1-3).
* Applies corruptions *at runtime* inside the Dataset (training).
* Builds deterministic validation and test corruption manifests (stored as JSON).
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .common import load_json, save_json
from .corruptions import CLASSES, CLASS_TO_IDX, SEVERITY_NAMES, apply_corruption, fixed_params, sample_params

SIZE = 128
SEED = 42


# --------------------------------------------------------------------------- #
# Locating and caching the raw data
# --------------------------------------------------------------------------- #
def _find_root(search: List[Path]) -> Optional[Path]:
    for base in search:
        if not base.exists():
            continue
        for p in [base, *base.rglob("*")]:
            if p.is_dir() and (p / "annotations" / "trainval.txt").exists() and (p / "images").is_dir():
                return p
    return None


def locate_pets(raw_dir: str | Path, extra: List[str] | None = None) -> Path:
    raw_dir = Path(raw_dir)
    search = [raw_dir] + [Path(e) for e in (extra or [])]
    root = _find_root(search)
    if root:
        return root
    # Extract archives if present.
    for base in search:
        if not base.exists():
            continue
        for tgz in list(base.rglob("images.tar.gz")) + list(base.rglob("annotations.tar.gz")):
            print("extracting", tgz)
            raw_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tgz) as t:
                t.extractall(raw_dir)
    root = _find_root([raw_dir])
    if root:
        return root
    print("Archives not found - downloading Oxford-IIIT Pet with torchvision")
    from torchvision.datasets import OxfordIIITPet
    OxfordIIITPet(str(raw_dir), split="trainval", download=True)
    root = _find_root([raw_dir])
    assert root is not None, "Could not find or download Oxford-IIIT Pet"
    return root


def _read_list(path: Path) -> List[str]:
    names = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            names.append(line.split()[0])
    return names


def _load_resized(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def build_cache(raw_dir: str | Path, cache_dir: str | Path, extra=None, limit: Optional[int] = None) -> Path:
    """Create pets128.npz (trainval/test arrays + names) and split.json."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz = cache_dir / "pets128.npz"
    if npz.exists():
        return cache_dir
    root = locate_pets(raw_dir, extra)
    out = {}
    for split in ["trainval", "test"]:
        names = _read_list(root / "annotations" / f"{split}.txt")
        names = [n for n in names if (root / "images" / f"{n}.jpg").exists()]
        if limit:
            names = names[:limit]
        arr = np.stack([_load_resized(root / "images" / f"{n}.jpg") for n in names])
        out[split] = arr
        out[f"{split}_names"] = np.array(names)
        print(split, arr.shape)
    np.savez_compressed(npz, **out)
    # 80/20 split of the official trainval collection with seed 42.
    n = len(out["trainval"])
    perm = np.random.default_rng(SEED).permutation(n)
    n_train = int(round(0.8 * n))
    save_json({"train": sorted(perm[:n_train].tolist()), "val": sorted(perm[n_train:].tolist()),
               "seed": SEED, "n_trainval": n, "n_test": int(len(out["test"]))}, cache_dir / "split.json")
    build_manifests(cache_dir)
    return cache_dir


def build_manifests(cache_dir: Path) -> None:
    """Deterministic validation (one random condition/image) and test (10 conditions/image) manifests."""
    cache_dir = Path(cache_dir)
    split = load_json(cache_dir / "split.json")
    val = []
    for j, idx in enumerate(split["val"]):
        seed = SEED * 100000 + j
        rng = np.random.default_rng(seed)
        kind = CLASSES[j % 4]  # balanced, deterministic
        val.append({"idx": idx, "kind": kind, "label": CLASS_TO_IDX[kind], "seed": seed,
                    "params": sample_params(kind, rng)})
    save_json(val, cache_dir / "val_manifest.json")
    test = []
    for idx in range(split["n_test"]):
        seed = SEED * 1000000 + idx
        rng = np.random.default_rng(seed)
        test.append({"idx": idx, "kind": "clean", "label": 0, "severity": "none", "seed": seed, "params": {}})
        for kind in CLASSES[1:]:
            for lvl in range(3):
                test.append({"idx": idx, "kind": kind, "label": CLASS_TO_IDX[kind],
                             "severity": SEVERITY_NAMES[lvl], "seed": seed, "params": fixed_params(kind, lvl, rng)})
    save_json(test, cache_dir / "test_manifest.json")


def load_cache(cache_dir: str | Path) -> Dict:
    cache_dir = Path(cache_dir)
    d = np.load(cache_dir / "pets128.npz")
    split = load_json(cache_dir / "split.json")
    return {"trainval": d["trainval"], "test": d["test"], "split": split,
            "val_manifest": load_json(cache_dir / "val_manifest.json"),
            "test_manifest": load_json(cache_dir / "test_manifest.json")}


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class RuntimeCorruptionDataset(Dataset):
    """Training dataset: a new corruption type + severity is sampled on every load.

    Items are addressed by (image_index, label). The label is supplied by the
    BalancedBatchSampler so that every batch holds the four conditions in equal
    proportion; a plain int index samples the condition uniformly instead.
    `kinds` restricts the allowed conditions (used for the Task 2 specialists).
    """

    def __init__(self, images: np.ndarray, indices: List[int], kinds: List[str] | None = None):
        self.images = images
        self.indices = list(indices)
        self.kinds = kinds or CLASSES
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            i, label = item
            kind = CLASSES[label]
        else:
            i = item
            kind = self.kinds[int(self.rng.integers(len(self.kinds)))]
        clean = self.images[self.indices[i]]
        corrupted = apply_corruption(clean, kind, sample_params(kind, self.rng))
        return to_tensor(corrupted), to_tensor(clean), CLASS_TO_IDX[kind]


def worker_init(worker_id: int):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % (2**32))


class BalancedBatchSampler(Sampler):
    """Yields batches of (image_index, label) with an equal number of each label."""

    def __init__(self, n: int, batch_size: int, labels: List[int] = (0, 1, 2, 3), seed: int = SEED):
        self.n, self.bs, self.labels = n, batch_size, list(labels)
        self.epoch, self.seed = 0, seed

    def __len__(self):
        return self.n // self.bs

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        perm = rng.permutation(self.n)
        for b in range(len(self)):
            idx = perm[b * self.bs:(b + 1) * self.bs]
            labs = np.resize(np.array(self.labels), len(idx))
            rng.shuffle(labs)
            yield [(int(i), int(l)) for i, l in zip(idx, labs)]


class ManifestDataset(Dataset):
    """Deterministic dataset driven by a validation/test manifest."""

    def __init__(self, images: np.ndarray, manifest: List[Dict], kinds: List[str] | None = None):
        self.images = images
        self.items = [m for m in manifest if kinds is None or m["kind"] in kinds]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        m = self.items[i]
        clean = self.images[m["idx"]]
        corrupted = apply_corruption(clean, m["kind"], m["params"])
        return to_tensor(corrupted), to_tensor(clean), m["label"], i


def setup_pets(budget: Dict, raw_dir="data/raw/pets", cache_dir="data/cache", extra=None):
    """Build/load the cache and apply the budget's subset/test limits (smoke tests only)."""
    import os
    extra = (extra or []) + [p for p in os.environ.get("PETS_SEARCH", "").split(os.pathsep) if p]
    limit = None if budget.get("subset") is None else max(budget["subset"] * 2, 120)
    build_cache(raw_dir, cache_dir, extra, limit=limit)
    d = load_cache(cache_dir)
    tr, va = d["split"]["train"], d["split"]["val"]
    if budget.get("subset"):
        tr = tr[: budget["subset"]]
        vset = set(va[: budget["subset"] // 2])
        d["val_manifest"] = [m for m in d["val_manifest"] if m["idx"] in vset]
    if budget.get("test_limit"):
        d["test_manifest"] = [m for m in d["test_manifest"] if m["idx"] < budget["test_limit"]]
    d["train_idx"], d["val_idx"] = tr, va
    return d
