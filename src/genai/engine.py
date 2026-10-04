"""Training / evaluation loops, ONNX export and plotting helpers shared by the task scripts."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import device, psnr, recon_loss, ssim, to_uint8
from .pet_data import BalancedBatchSampler, ManifestDataset, RuntimeCorruptionDataset, worker_init


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def train_loader(images, indices, batch_size, workers, kinds=None, balanced=True):
    ds = RuntimeCorruptionDataset(images, indices, kinds)
    if balanced and kinds is None:
        return DataLoader(ds, batch_sampler=BalancedBatchSampler(len(ds), batch_size), num_workers=workers,
                          worker_init_fn=worker_init if workers else None, pin_memory=torch.cuda.is_available(),
                          persistent_workers=workers > 0)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers,
                      worker_init_fn=worker_init if workers else None, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0)


def manifest_loader(images, manifest, batch_size=64, workers=2, kinds=None):
    return DataLoader(ManifestDataset(images, manifest, kinds), batch_size=batch_size, shuffle=False,
                      num_workers=workers)


# --------------------------------------------------------------------------- #
# Autoencoder training
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_recon(fn: Callable, loader) -> Dict[str, float]:
    dev = device()
    l1s, ss, ps = [], [], []
    for x, y, *_ in loader:
        x, y = x.to(dev), y.to(dev)
        out = fn(x)
        out = out[0] if isinstance(out, tuple) else out
        l1s.append((out - y).abs().flatten(1).mean(1).cpu())
        ss.append(ssim(out, y, reduction="none").cpu())
        ps.append(psnr(out, y).cpu())
    l1, s, p = torch.cat(l1s).mean().item(), torch.cat(ss).mean().item(), torch.cat(ps).mean().item()
    return {"l1": l1, "ssim": s, "psnr": p, "score": l1 + (1 - s)}


def train_autoencoder(model, tl, vl, epochs, lr, alpha, weight_decay=0.0, trial=None, mlflow=None,
                      prefix="", log_every=1) -> List[Dict]:
    """Train with L = alpha*L1 + (1-alpha)*(1-SSIM). Reports val score to Optuna for pruning."""
    import optuna
    dev = device()
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(tl)))
    hist = []
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, _ in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            loss = recon_loss(model(x), y, alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
            n += len(x)
        model.eval()
        v = eval_recon(model, vl)
        rec = {"epoch": ep + 1, "train_loss": tot / max(n, 1), **{f"val_{k}": val for k, val in v.items()},
               "time": time.time() - t0}
        hist.append(rec)
        print(prefix, rec, flush=True)
        if mlflow is not None:
            mlflow.log_metrics({f"{prefix}{k}": float(val) for k, val in rec.items() if k != "epoch"}, step=ep + 1)
        if v["score"] < best:
            best = v["score"]
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if trial is not None:
            trial.report(v["score"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


# --------------------------------------------------------------------------- #
# Per-condition evaluation on the deterministic test manifest
# --------------------------------------------------------------------------- #
@torch.no_grad()
def per_item_restoration(fn: Callable, images, manifest, batch_size=64, workers=2, extra: Callable = None):
    """Returns dict of arrays (psnr, ssim, l1, in_psnr, in_ssim) aligned with manifest order."""
    dev = device()
    ds = ManifestDataset(images, manifest)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    res = {k: [] for k in ["psnr", "ssim", "l1", "in_psnr", "in_ssim"]}
    extras = []
    for x, y, lab, _ in dl:
        x, y = x.to(dev), y.to(dev)
        out = fn(x)
        aux = None
        if isinstance(out, tuple):
            out, aux = out[0], out[1:]
        res["psnr"].append(psnr(out, y).cpu())
        res["ssim"].append(ssim(out, y, reduction="none").cpu())
        res["l1"].append((out - y).abs().flatten(1).mean(1).cpu())
        res["in_psnr"].append(psnr(x, y).cpu())
        res["in_ssim"].append(ssim(x, y, reduction="none").cpu())
        if extra is not None:
            extras.append(extra(x, out, aux, lab))
    res = {k: torch.cat(v).numpy() for k, v in res.items()}
    return res, extras


def summarise(res: Dict[str, np.ndarray], manifest: List[Dict]) -> Dict:
    """Mean metrics by corruption kind and by (kind, severity)."""
    kinds = np.array([m["kind"] for m in manifest])
    sev = np.array([m.get("severity", "none") for m in manifest])
    out = {"overall": {k: float(v.mean()) for k, v in res.items()}, "by_kind": {}, "by_kind_severity": {}}
    for k in ["clean", "saltpepper", "blur", "occlusion"]:
        m = kinds == k
        if m.any():
            out["by_kind"][k] = {n: float(v[m].mean()) for n, v in res.items()}
        for s in ["none", "low", "medium", "high"]:
            mm = m & (sev == s)
            if mm.any():
                out["by_kind_severity"][f"{k}/{s}"] = {n: float(v[mm].mean()) for n, v in res.items()}
    for s in ["low", "medium", "high"]:
        mm = sev == s
        if mm.any():
            out["by_kind_severity"][f"all/{s}"] = {n: float(v[mm].mean()) for n, v in res.items()}
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_curves(hist: List[Dict], keys: List[str], path, title=""):
    fig, ax = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3))
    ax = np.atleast_1d(ax)
    for a, k in zip(ax, keys):
        if isinstance(k, (list, tuple)):
            for kk in k:
                a.plot([h["epoch"] for h in hist], [h[kk] for h in hist], label=kk)
            a.legend(fontsize=7)
            a.set_title(", ".join(k), fontsize=8)
        else:
            a.plot([h["epoch"] for h in hist], [h[k] for h in hist])
            a.set_title(k, fontsize=9)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def restoration_grid(fn: Callable, images, manifest, picks: List[int], path, title="", annot: Callable = None):
    """Rows: clean target | corrupted input | restored | |error| map."""
    dev = device()
    ds = ManifestDataset(images, manifest)
    n = len(picks)
    fig, ax = plt.subplots(4, n, figsize=(1.6 * n, 7))
    ax = np.array(ax).reshape(4, n)
    for j, i in enumerate(picks):
        x, y, _, _ = ds[i]
        out = fn(x[None].to(dev))
        aux = None
        if isinstance(out, tuple):
            out, aux = out[0], out[1:]
        out = out[0].cpu()
        err = (out - y).abs().mean(0).numpy()
        m = ds.items[i]
        p = psnr(out[None], y[None]).item()
        for r, img in enumerate([to_uint8(y), to_uint8(x), to_uint8(out)]):
            ax[r, j].imshow(img)
        ax[3, j].imshow(err, cmap="inferno", vmin=0, vmax=0.5)
        lab = f"{m['kind'][:5]}/{m.get('severity', '')[:3]}\n{p:.1f}dB"
        if annot is not None:
            lab += "\n" + annot(x, out, aux)
        ax[0, j].set_title(lab, fontsize=6)
        for r in range(4):
            ax[r, j].axis("off")
    for r, name in enumerate(["clean", "input", "output", "|error|"]):
        ax[r, 0].text(-8, 64, name, rotation=90, va="center", ha="right", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def pick_examples(manifest, res, per_kind=3, seed=0):
    """Representative examples: for each kind one random item from each severity (clean: 3 random)."""
    rng = np.random.default_rng(seed)
    picks = []
    for k in ["clean", "saltpepper", "blur", "occlusion"]:
        if k == "clean":
            idx = [i for i, m in enumerate(manifest) if m["kind"] == k]
            picks += list(rng.choice(idx, per_kind, replace=False)) if len(idx) >= per_kind else idx
        else:
            for s in ["low", "medium", "high"][:per_kind]:
                idx = [i for i, m in enumerate(manifest) if m["kind"] == k and m.get("severity") == s]
                if idx:
                    picks.append(int(rng.choice(idx)))
    return [int(p) for p in picks]


def pick_failures(manifest, res, n=4):
    """Failure cases: per corruption kind, the item with the lowest SSIM gain (worst restoration)."""
    gain = res["ssim"] - res["in_ssim"]
    picks = []
    for k in ["saltpepper", "blur", "occlusion", "clean"]:
        idx = np.array([i for i, m in enumerate(manifest) if m["kind"] == k])
        if len(idx):
            key = res["ssim"][idx] if k == "clean" else gain[idx]
            picks.append(int(idx[np.argmin(key)]))
    return picks[:n]


# --------------------------------------------------------------------------- #
# ONNX
# --------------------------------------------------------------------------- #
def export_onnx(model, inputs: tuple, path, input_names, output_names, dynamic_batch=True):
    """Export with the legacy TorchScript exporter (opset 17) and verify with onnxruntime."""
    import onnxruntime as ort
    model = model.eval().cpu()
    inputs = tuple(t.cpu() for t in inputs)
    dyn = {n: {0: "batch"} for n in input_names + output_names} if dynamic_batch else None
    kw = dict(input_names=input_names, output_names=output_names, dynamic_axes=dyn, opset_version=17)
    try:
        torch.onnx.export(model, inputs, str(path), dynamo=False, **kw)
    except TypeError:  # older torch without the `dynamo` kwarg
        torch.onnx.export(model, inputs, str(path), **kw)
    with torch.no_grad():
        ref = model(*inputs)
    ref = ref if isinstance(ref, tuple) else (ref,)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {n: t.numpy() for n, t in zip(input_names, inputs)}
    outs = sess.run(None, feeds)
    diffs = {n: float(np.abs(o - r.numpy()).max()) for n, o, r in zip(output_names, outs, ref)}
    return {"file": Path(path).name, "max_abs_diff": diffs, "size_mb": Path(path).stat().st_size / 1e6,
            "consistent": all(d < 1e-3 for d in diffs.values())}


def optuna_plots(study, folder: Path, name: str):
    """Save optimisation history + parameter importance plots and the trial table."""
    folder.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(folder / f"{name}_trials.csv", index=False)
    done = [t for t in study.trials if t.value is not None]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    xs = [t.number for t in done]
    ys = [t.value for t in done]
    ax[0].scatter(xs, ys, s=14, label="trial")
    ax[0].plot(xs, np.minimum.accumulate(ys) if ys else [], color="C1", label="best so far")
    ax[0].set_xlabel("trial")
    ax[0].set_ylabel("objective (lower is better)")
    ax[0].legend(fontsize=7)
    ax[0].set_title(f"{name}: optimisation history", fontsize=9)
    try:
        import optuna
        imp = optuna.importance.get_param_importances(study)
        ax[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1])
        ax[1].set_title("parameter importance (fANOVA)", fontsize=9)
    except Exception as e:  # too few trials etc.
        ax[1].text(0.1, 0.5, f"importance unavailable:\n{e}", fontsize=7)
        ax[1].axis("off")
    fig.tight_layout()
    fig.savefig(folder / f"{name}_optuna.png", dpi=130)
    plt.close(fig)
    states = {}
    for t in study.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    return {"n_trials": len(study.trials), "states": states,
            "best_value": study.best_value if done else None,
            "best_params": study.best_params if done else None,
            "best_trial": study.best_trial.number if done else None}
