"""Task 3: jointly trained soft mixture-of-experts restoration.

Gate initialised from the Task 2 classifier, experts from the Task 2 specialists,
identity branch for clean inputs. Warm-up (experts frozen, gate trained) followed by
joint fine-tuning with a smaller learning rate.

L_MoE = l1*L1 + ls*(1-SSIM) + lc*CE(gate, corruption label) + lb*sum_k (mean_w_k - 1/4)^2
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genai.common import log_light_artifacts
from genai.common import (device, get_budget, load_json, out_dir, psnr, save_json, seed_everything, setup_mlflow, ssim,
                          to_uint8)
from genai.corruptions import CLASSES, apply_corruption, fixed_params
from genai.engine import (eval_recon, export_onnx, manifest_loader, optuna_plots, pick_examples, pick_failures,
                          plot_curves, restoration_grid, summarise, train_loader)
from genai.models import ConvAutoencoder, CorruptionClassifier, SoftMoE, count_params
from genai.pet_data import ManifestDataset, setup_pets, to_tensor

SPECIALISTS = ["saltpepper", "blur", "occlusion"]
CH_CFGS = {"16-32-64-128": (16, 32, 64, 128), "32-64-128-256": (32, 64, 128, 256),
           "32-64-128-128": (32, 64, 128, 128), "24-48-96-192": (24, 48, 96, 192)}
BRANCHES = ["identity", "saltpepper", "blur", "occlusion"]


def build_moe(tau):
    T2 = out_dir() / "task2"
    c = torch.load(T2 / "classifier.pt", map_location="cpu")
    gate = CorruptionClassifier(CH_CFGS[c["config"]["channels"]], c["config"]["dropout"])
    gate.load_state_dict(c["state_dict"])
    experts = []
    for k in SPECIALISTS:
        s = torch.load(T2 / f"specialist_{k}.pt", map_location="cpu")
        e = ConvAutoencoder(s["config"]["base"], s["config"]["latent_ch"], 0.1)
        e.load_state_dict(s["state_dict"])
        experts.append(e)
    return SoftMoE(gate, experts, tau)


def balance_loss(w):
    return ((w.mean(0) - 0.25) ** 2).sum()


def set_mode(model, stage):
    """warm-up: experts frozen (no grads, eval BN). joint: all params trainable, expert BN stats frozen."""
    model.train()
    for e in model.experts:
        e.eval()  # keep specialist BN statistics (they were estimated on their own corruption)
        for p in e.parameters():
            p.requires_grad_(stage == "joint")


@torch.no_grad()
def eval_moe(model, loader):
    dev = device()
    model.eval()
    l1s, ss, W, Y = [], [], [], []
    for x, y, lab, *_ in loader:
        x, y = x.to(dev), y.to(dev)
        xh, w, _ = model(x)
        l1s.append((xh - y).abs().flatten(1).mean(1).cpu())
        ss.append(ssim(xh, y, reduction="none").cpu())
        W.append(w.cpu())
        Y.append(lab)
    l1, s = torch.cat(l1s).mean().item(), torch.cat(ss).mean().item()
    W, Y = torch.cat(W), torch.cat(Y)
    return {"l1": l1, "ssim": s, "score": l1 + 1 - s, "gate_acc": (W.argmax(1) == Y).float().mean().item(),
            "max_mean_weight": W.mean(0).max().item(), "mean_w": W.mean(0).tolist()}


def train_moe(model, tl, vl, p, warmup, epochs, trial=None, mlflow=None):
    dev = device()
    model.to(dev)
    hist, step = [], 0
    best, best_state = 1e9, None
    stages = [("warmup", warmup, p.get("warmup_lr", 1e-3))] + [("joint", epochs, p["lr"])]
    for stage, n_ep, lr in stages:
        set_mode(model, stage)
        params = [q for q in model.parameters() if q.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_ep * len(tl)))
        for ep in range(n_ep):
            set_mode(model, stage)
            t0 = time.time()
            agg = {"loss": 0, "l1": 0, "ssim": 0, "ce": 0, "bal": 0}
            n = 0
            for x, y, lab in tl:
                x, y, lab = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), lab.to(dev)
                xh, w, logits = model(x)
                l1 = F.l1_loss(xh, y)
                s = 1 - ssim(xh, y)
                ce = F.cross_entropy(logits, lab)
                bal = balance_loss(w)
                loss = p["l1"] * l1 + (1 - p["l1"]) * s + p["lc"] * ce + p["lb"] * bal
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                for k, v in zip(agg, [loss, l1, s, ce, bal]):
                    agg[k] += v.item() * len(x)
                n += len(x)
            v = eval_moe(model, vl)
            rec = {"epoch": step + 1, "stage": stage, **{f"train_{k}": a / n for k, a in agg.items()},
                   **{f"val_{k}": val for k, val in v.items() if k != "mean_w"},
                   **{f"val_w_{b}": w_ for b, w_ in zip(BRANCHES, v["mean_w"])}, "time": time.time() - t0}
            hist.append(rec)
            print("moe", rec, flush=True)
            if mlflow:
                mlflow.log_metrics({k: float(x_) for k, x_ in rec.items() if k not in ("epoch", "stage")}, step=step + 1)
            if v["score"] < best:
                best, best_state = v["score"], {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            if trial is not None:
                trial.report(v["score"], step)
                # prune routing collapse (one branch receives > 85% of all weight on a balanced set)
                if v["max_mean_weight"] > 0.85 or trial.should_prune():
                    raise optuna.TrialPruned()
            step += 1
    model.load_state_dict(best_state)
    return hist


@torch.no_grad()
def collect(model, images, manifest, workers):
    dev = device()
    model.eval()
    dl = DataLoader(ManifestDataset(images, manifest), batch_size=64, shuffle=False, num_workers=workers)
    R = {k: [] for k in ["psnr", "ssim", "l1", "in_psnr", "in_ssim", "w"]}
    for x, y, lab, _ in dl:
        x, y = x.to(dev), y.to(dev)
        xh, w, _ = model(x)
        R["psnr"].append(psnr(xh, y).cpu())
        R["ssim"].append(ssim(xh, y, reduction="none").cpu())
        R["l1"].append((xh - y).abs().flatten(1).mean(1).cpu())
        R["in_psnr"].append(psnr(x, y).cpu())
        R["in_ssim"].append(ssim(x, y, reduction="none").cpu())
        R["w"].append(w.cpu())
    return {k: torch.cat(v).numpy() for k, v in R.items()}


def weight_heatmap(rows, labels, path, title):
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(5, 0.38 * len(labels) + 1.4))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), BRANCHES, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if M[i, j] > 0.6 else "white")
    ax.set_xlabel("branch (gate weight)")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


@torch.no_grad()
def weights_examples(model, inputs, cleans, titles, path, suptitle):
    dev = device()
    n = len(inputs)
    fig, ax = plt.subplots(3, n, figsize=(1.9 * n, 6.2), gridspec_kw={"height_ratios": [1, 1, 0.8]})
    ax = np.array(ax).reshape(3, n)
    for j, (x, y, t) in enumerate(zip(inputs, cleans, titles)):
        xh, w, _ = model(x[None].to(dev))
        w = w[0].cpu().numpy()
        ax[0, j].imshow(to_uint8(x))
        ax[1, j].imshow(to_uint8(xh[0]))
        ax[0, j].set_title(f"{t}\n{psnr(x[None], y[None]).item():.1f}->{psnr(xh.cpu(), y[None]).item():.1f}dB",
                           fontsize=6)
        ax[2, j].bar(range(4), w, color=["#888", "#e07b39", "#3b75af", "#519e3e"])
        ax[2, j].set_ylim(0, 1)
        ax[2, j].set_xticks(range(4), ["id", "s&p", "blur", "occ"], fontsize=6)
        ax[2, j].tick_params(axis="y", labelsize=6)
        ax[0, j].axis("off")
        ax[1, j].axis("off")
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default=None)
    args = ap.parse_args()
    B = get_budget(args.budget)
    seed_everything(42)
    O = out_dir() / "task3"
    O.mkdir(parents=True, exist_ok=True)
    MD = out_dir() / "onnx"
    d = setup_pets(B)
    imgs, W = d["trainval"], B["workers"]
    vl = manifest_loader(imgs, d["val_manifest"], 64, W)
    mlflow = setup_mlflow("task3_soft_moe")
    dev = device()
    BS = 32  # batch of 32 = 8 samples per condition (balanced)

    f_best = O / "best_params.json"
    if not f_best.exists():
        storage = f"sqlite:///{(out_dir() / 'optuna.db').resolve()}"
        study = optuna.create_study(study_name="task3_moe", direction="minimize", storage=storage, load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=44),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1))
        study.enqueue_trial({"lr": 1e-4, "tau": 1.0, "lc": 0.1, "lb": 0.01, "l1": 0.8})  # assignment defaults

        def obj(trial):
            p = {"lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
                 "tau": trial.suggest_float("tau", 0.3, 3.0, log=True),
                 "lc": trial.suggest_float("lc", 0.01, 1.0, log=True),
                 "lb": trial.suggest_float("lb", 1e-3, 1.0, log=True),
                 "l1": trial.suggest_float("l1", 0.5, 0.95)}
            seed_everything(42 + trial.number)
            with mlflow.start_run(run_name=f"moe_trial_{trial.number}", nested=True):
                mlflow.log_params(p)
                m = build_moe(p["tau"])
                h = train_moe(m, train_loader(imgs, d["train_idx"], BS, W), vl, p, 1,
                              max(1, B["moe_trial_epochs"] - 1), trial, mlflow)
                v = min(r["val_score"] for r in h)
                mlflow.log_metric("objective", v)
                return v

        with mlflow.start_run(run_name="moe_optuna_search"):
            done = len([t for t in study.trials if t.state.is_finished()])
            study.optimize(obj, n_trials=max(0, B["moe_trials"] - done), gc_after_trial=True)
            s = optuna_plots(study, O, "task3")
            mlflow.log_artifact(str(O / "task3_optuna.png"))
        save_json(s, O / "optuna_summary.json")
        save_json(study.best_params, f_best)
    p = load_json(f_best)

    model = build_moe(p["tau"]).to(dev)
    f_ck = O / "moe.pt"
    if not f_ck.exists():
        seed_everything(42)
        # baseline: initialised (not fine-tuned) MoE, to show what joint training adds
        init_val = eval_moe(model, vl)
        with mlflow.start_run(run_name="moe_final_training"):
            mlflow.log_params({**p, "warmup_epochs": B["moe_warmup"], "joint_epochs": B["moe_epochs"],
                               "params": count_params(model)})
            h = train_moe(model, train_loader(imgs, d["train_idx"], BS, W), vl, p, B["moe_warmup"], B["moe_epochs"],
                          mlflow=mlflow)
            torch.save({"state_dict": model.state_dict(), "config": p}, f_ck)
            save_json({"history": h, "init_val": init_val}, O / "history.json")
            plot_curves(h, ["train_loss", ["train_ce", "train_bal"], "val_ssim", "val_gate_acc",
                            [f"val_w_{b}" for b in BRANCHES]], O / "curves.png", "Task 3 soft MoE (warm-up then joint)")
    model.load_state_dict(torch.load(f_ck, map_location=dev)["state_dict"])
    model.eval()

    test, man = d["test"], d["test_manifest"]
    R = collect(model, test, man, W)
    torch.save(R, O / "test_per_item.pt")
    res = {k: R[k] for k in ["psnr", "ssim", "l1", "in_psnr", "in_ssim"]}
    summary = summarise(res, man)
    kinds = np.array([m["kind"] for m in man])
    sev = np.array([m["severity"] for m in man])
    rows, labels, wt = [], [], {}
    for k in CLASSES:
        for s in ["none", "low", "medium", "high"]:
            mm = (kinds == k) & (sev == s)
            if mm.any():
                mw = R["w"][mm].mean(0)
                rows.append(mw)
                labels.append(f"{k}/{s}")
                wt[f"{k}/{s}"] = dict(zip(BRANCHES, mw.tolist()))
    weight_heatmap(rows, labels, O / "routing_heatmap.png", "Mean gate weight per true corruption / severity")
    ent = -(R["w"] * np.log(R["w"] + 1e-9)).sum(1)
    top = R["w"].argmax(1)
    usage = {"mean_weight": dict(zip(BRANCHES, R["w"].mean(0).tolist())),
             "argmax_share": {b: float((top == i).mean()) for i, b in enumerate(BRANCHES)},
             "mean_entropy": float(ent.mean()), "max_entropy": float(np.log(4)),
             "inactive_branches": [b for i, b in enumerate(BRANCHES) if (top == i).mean() < 0.02],
             "frac_dominated_gt_0.9": float((R["w"].max(1) > 0.9).mean()),
             "frac_distributed_max_lt_0.6": float((R["w"].max(1) < 0.6).mean())}
    # does any expert dominate unrelated inputs? weight given to each expert on inputs of other kinds
    lab = np.array([m["label"] for m in man])
    usage["off_target_weight"] = {b: float(R["w"][lab != i, i].mean()) for i, b in enumerate(BRANCHES)}
    save_json({"summary": summary, "weights_by_kind_severity": wt, "usage": usage, "params": p,
               "val": eval_moe(model, vl)}, O / "test_results.json")

    def annot(x, out, aux):
        w = aux[0][0].cpu().numpy()
        return "w=" + ",".join(f"{v:.2f}" for v in w)

    restoration_grid(model, test, man, pick_examples(man, res), O / "examples.png", "Task 3: soft MoE results", annot)
    restoration_grid(model, test, man, pick_failures(man, res), O / "failures.png", "Task 3: failure cases", annot)
    ds = ManifestDataset(test, man)
    dom = [int(i) for i in np.argsort(ent)[:4]]
    dist = [int(i) for i in np.argsort(-ent)[:4]]
    weights_examples(model, [ds[i][0] for i in dom + dist], [ds[i][1] for i in dom + dist],
                     [f"{ds.items[i]['kind'][:5]}/{ds.items[i]['severity'][:3]}" + (" (dom)" if i in dom else " (dist)")
                      for i in dom + dist], O / "dominant_vs_distributed.png",
                     "Left: one expert dominates (lowest entropy). Right: distributed weights (highest entropy)")
    # mixed corruptions (not seen in training) - motivates soft routing
    rng = np.random.default_rng(7)
    mix_in, mix_clean, mix_t = [], [], []
    for i in range(4):
        clean = test[i * 7]
        a = apply_corruption(clean, "blur", fixed_params("blur", 1, rng))
        a = apply_corruption(a, "saltpepper", fixed_params("saltpepper", 1, rng))
        b = apply_corruption(clean, "saltpepper", fixed_params("saltpepper", 1, rng))
        b = apply_corruption(b, "occlusion", fixed_params("occlusion", 1, rng))
        for img, t in [(a, "blur+s&p"), (b, "s&p+occ")]:
            mix_in.append(to_tensor(img)); mix_clean.append(to_tensor(clean)); mix_t.append(t)
    weights_examples(model, mix_in, mix_clean, mix_t, O / "mixed_corruptions.png",
                     "Mixed corruptions (not in training): gate distributes weight across experts")

    info = export_onnx(model, (torch.rand(1, 3, 128, 128),), MD / "soft_moe.onnx", ["input"],
                       ["output", "weights", "logits"])
    save_json(info, O / "onnx_check.json")
    model.to(dev)

    # cross-task comparison table
    comp = {"task3_soft_moe": summary}
    try:
        comp["task1_universal"] = load_json(out_dir() / "task1" / "test_results.json")
        rr = load_json(out_dir() / "task2" / "routing_results.json")
        comp["task2_oracle"], comp["task2_predicted"] = rr["oracle"], rr["predicted"]
    except FileNotFoundError:
        pass
    save_json(comp, out_dir() / "comparison.json")
    with mlflow.start_run(run_name="t3_evaluation"):
        mlflow.log_metrics({f"test_{k}": v for k, v in summary["overall"].items()})
        mlflow.log_metrics({f"mean_w_{k}": v for k, v in usage["mean_weight"].items()})
        log_light_artifacts(mlflow, O, "task3")
    print("TASK 3 DONE", summary["by_kind"], usage)


if __name__ == "__main__":
    main()
