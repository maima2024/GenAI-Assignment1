"""Task 4: style-conditioned face-to-sketch generation with a conditional GAN (pix2pix-style).

G: U-Net with learned style embedding.  D: conditional PatchGAN on (photo, sketch, style).
L_D = BCE(D(x,y,s),1) + BCE(D(x,G(x,s),s),0)      L_G = BCE(D(x,G(x,s),s),1) + lambda_L1 * |y - G(x,s)|
"""
import argparse
import os
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
from genai.common import device, get_budget, load_json, out_dir, psnr, save_json, seed_everything, setup_mlflow, ssim
from genai.engine import export_onnx, optuna_plots, plot_curves
from genai.fs2k_data import PairedSketchDataset, build_cache, load
from genai.models import PatchDiscriminator, UNetGenerator, count_params, init_gan_weights

STYLE_NAMES = ["Style 1", "Style 2", "Style 3"]


def to01(t):
    return (t.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def evaluate(G, loader):
    dev = device()
    G.eval()
    L, S, P = [], [], []
    for x, y, s in loader:
        x, y, s = x.to(dev), y.to(dev), s.to(dev)
        g = G(x, s)
        L.append((g - y).abs().flatten(1).mean(1).cpu() / 2)  # L1 on [0,1] scale
        S.append(ssim(to01(g), to01(y), reduction="none").cpu())
        P.append(psnr(to01(g), to01(y)).cpu())
    G.train()
    l, s_, p = torch.cat(L).mean().item(), torch.cat(S).mean().item(), torch.cat(P).mean().item()
    return {"l1": l, "ssim": s_, "psnr": p, "score": l + (1 - s_)}, (torch.cat(L), torch.cat(S), torch.cat(P))


@torch.no_grad()
def sample_grid(G, fixed, path, title):
    dev = device()
    G.eval()
    x, y, s = fixed
    n = len(x)
    fig, ax = plt.subplots(3, n, figsize=(1.5 * n, 4.8))
    g = G(x.to(dev), s.to(dev)).cpu()
    for j in range(n):
        ax[0, j].imshow(to01(x[j]).permute(1, 2, 0).numpy())
        ax[1, j].imshow(to01(y[j, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
        ax[2, j].imshow(to01(g[j, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
        ax[0, j].set_title(STYLE_NAMES[int(s[j])], fontsize=7)
        for r in range(3):
            ax[r, j].axis("off")
    for r, name in enumerate(["photo", "real sketch", "generated"]):
        ax[r, 0].text(-8, 64, name, rotation=90, va="center", ha="right", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    G.train()


def train_gan(p, data, epochs, trial=None, mlflow=None, sample_dir=None, sample_every=10):
    dev = device()
    G = UNetGenerator(p["base"], p["emb_dim"], dropout=p["dropout"]).to(dev)
    D = PatchDiscriminator(64, p["emb_dim"]).to(dev)
    G.apply(init_gan_weights)
    D.apply(init_gan_weights)
    oG = torch.optim.Adam(G.parameters(), lr=p["lr_g"], betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr=p["lr_d"], betas=(0.5, 0.999))
    # linear decay over the second half (pix2pix schedule)
    lam = lambda e: 1.0 if e < epochs // 2 else max(0.0, 1 - (e - epochs // 2) / max(1, epochs - epochs // 2))
    sG, sD = torch.optim.lr_scheduler.LambdaLR(oG, lam), torch.optim.lr_scheduler.LambdaLR(oD, lam)
    bce = nn.BCEWithLogitsLoss()
    tl = DataLoader(data["train_ds"], batch_size=p["batch_size"], shuffle=True, drop_last=True,
                    num_workers=data["workers"])
    hist, best, best_state = [], 1e9, None
    for ep in range(epochs):
        t0 = time.time()
        agg = {"d_real": 0, "d_fake": 0, "g_adv": 0, "g_l1": 0}
        n = 0
        for x, y, s in tl:
            x, y, s = x.to(dev), y.to(dev), s.to(dev)
            fake = G(x, s)
            # --- discriminator ---
            pr = D(x, y, s)
            pf = D(x, fake.detach(), s)
            d_real = bce(pr, torch.full_like(pr, 0.9))  # one-sided label smoothing
            d_fake = bce(pf, torch.zeros_like(pf))
            oD.zero_grad(set_to_none=True)
            (0.5 * (d_real + d_fake)).backward()
            oD.step()
            # --- generator ---
            pf = D(x, fake, s)
            g_adv = bce(pf, torch.ones_like(pf))
            g_l1 = F.l1_loss(fake, y)
            oG.zero_grad(set_to_none=True)
            (g_adv + p["lambda_l1"] * g_l1).backward()
            oG.step()
            for k, v in zip(agg, [d_real, d_fake, g_adv, g_l1]):
                agg[k] += v.item() * len(x)
            n += len(x)
        sG.step()
        sD.step()
        v, _ = evaluate(G, data["val_dl"])
        rec = {"epoch": ep + 1, **{k: a / n for k, a in agg.items()}, **{f"val_{k}": x_ for k, x_ in v.items()},
               "time": time.time() - t0}
        hist.append(rec)
        print("gan", rec, flush=True)
        if mlflow:
            mlflow.log_metrics({k: float(x_) for k, x_ in rec.items() if k != "epoch"}, step=ep + 1)
        if sample_dir is not None and ((ep + 1) % sample_every == 0 or ep == 0 or ep + 1 == epochs):
            f = Path(sample_dir) / f"samples_epoch_{ep + 1:03d}.png"
            sample_grid(G, data["fixed"], f, f"Fixed validation photos - epoch {ep + 1}")
            if mlflow:
                mlflow.log_artifact(str(f), artifact_path="samples")
        if v["score"] < best:
            best, best_state = v["score"], {k: t.detach().cpu().clone() for k, t in G.state_dict().items()}
        if trial is not None:
            trial.report(v["score"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    G.load_state_dict(best_state)
    return G, D, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default=None)
    args = ap.parse_args()
    B = get_budget(args.budget)
    seed_everything(42)
    O = out_dir() / "task4"
    (O / "samples").mkdir(parents=True, exist_ok=True)
    MD = out_dir() / "onnx"
    MD.mkdir(exist_ok=True)
    search = [p for p in os.environ.get("FS2K_SEARCH", "").split(os.pathsep) if p] + ["data/raw", "data", "/kaggle/input"]
    build_cache(search)
    d = load()
    tr, va = d["split"]["train"], d["split"]["val"]
    if B.get("subset"):
        tr, va = tr[:B["subset"]], va[:B["subset"] // 2]
    W = B["workers"]
    train_ds = PairedSketchDataset(d["train_photo"], d["train_sketch"], d["train_style"], tr, augment=True)
    val_ds = PairedSketchDataset(d["train_photo"], d["train_sketch"], d["train_style"], va)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=W)
    # fixed validation photos (2 per style) for progress grids
    fx = []
    for st in range(3):
        fx += [i for i in range(len(val_ds)) if val_ds.y[val_ds.idx[i]] == st][:2]
    fixed = [torch.stack([val_ds[i][k] for i in fx]) if k < 2 else torch.tensor([val_ds[i][2] for i in fx])
             for k in range(3)]
    data = {"train_ds": train_ds, "val_dl": val_dl, "fixed": fixed, "workers": W}
    mlflow = setup_mlflow("task4_face2sketch_cgan")
    save_json({"n_train": len(tr), "n_val": len(va), "n_test": int(len(d["test_style"])),
               "train_style_counts": np.bincount(d["train_style"][tr], minlength=3).tolist(),
               "val_style_counts": np.bincount(d["train_style"][va], minlength=3).tolist(),
               "test_style_counts": np.bincount(d["test_style"], minlength=3).tolist()}, O / "data_split.json")

    f_best = O / "best_params.json"
    if not f_best.exists():
        storage = f"sqlite:///{(out_dir() / 'optuna_t4.db').resolve()}"
        study = optuna.create_study(study_name="task4_cgan", direction="minimize", storage=storage,
                                    load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=45),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2))
        study.enqueue_trial({"lr_g": 2e-4, "lr_d": 2e-4, "batch_size": 8, "base": 48, "dropout": 0.5,
                             "emb_dim": 16, "lambda_l1": 100.0})  # pix2pix defaults

        def obj(trial):
            p = {"lr_g": trial.suggest_float("lr_g", 5e-5, 5e-4, log=True),
                 "lr_d": trial.suggest_float("lr_d", 5e-5, 5e-4, log=True),
                 "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
                 "base": trial.suggest_categorical("base", [32, 48, 64]),
                 "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                 "emb_dim": trial.suggest_categorical("emb_dim", [8, 16, 32]),
                 "lambda_l1": trial.suggest_float("lambda_l1", 10.0, 200.0, log=True)}
            seed_everything(42 + trial.number)
            with mlflow.start_run(run_name=f"gan_trial_{trial.number}", nested=True):
                mlflow.log_params(p)
                _, _, h = train_gan(p, data, B["gan_trial_epochs"], trial, mlflow)
                v = min(r["val_score"] for r in h)
                mlflow.log_metric("objective", v)
                return v

        with mlflow.start_run(run_name="gan_optuna_search"):
            done = len([t for t in study.trials if t.state.is_finished()])
            study.optimize(obj, n_trials=max(0, B["gan_trials"] - done), gc_after_trial=True)
            s = optuna_plots(study, O, "task4")
            mlflow.log_artifact(str(O / "task4_optuna.png"))
        save_json(s, O / "optuna_summary.json")
        save_json(study.best_params, f_best)
    p = load_json(f_best)
    p["batch_size"] = int(p["batch_size"])

    f_ck = O / "generator.pt"
    dev = device()
    if not f_ck.exists():
        seed_everything(42)
        with mlflow.start_run(run_name="gan_final_training"):
            mlflow.log_params({**p, "epochs": B["gan_epochs"]})
            G, D, h = train_gan(p, data, B["gan_epochs"], mlflow=mlflow, sample_dir=O / "samples",
                                sample_every=max(1, B["gan_epochs"] // 8))
            mlflow.log_params({"g_params": count_params(G), "d_params": count_params(D)})
            torch.save({"state_dict": G.state_dict(), "config": p}, f_ck)
            torch.save({"state_dict": D.state_dict(), "config": p}, O / "discriminator.pt")
            save_json(h, O / "history.json")
            plot_curves(h, [["d_real", "d_fake"], "g_adv", "g_l1", ["val_l1"], "val_ssim"], O / "curves.png",
                        "Task 4 cGAN losses")
    G = UNetGenerator(p["base"], p["emb_dim"], dropout=p["dropout"]).to(dev)
    G.load_state_dict(torch.load(f_ck, map_location=dev)["state_dict"])
    G.eval()

    # ---- test evaluation (official test set, used only here) ----
    test_ds = PairedSketchDataset(d["test_photo"], d["test_sketch"], d["test_style"], range(len(d["test_style"])))
    if B.get("test_limit"):
        test_ds.idx = test_ds.idx[:B["test_limit"]]
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=W)
    overall, (L, S, P) = evaluate(G, test_dl)
    G.eval()
    st = d["test_style"][test_ds.idx]
    res = {"overall": overall, "by_style": {STYLE_NAMES[k]: {"l1": float(L[st == k].mean()), "ssim": float(S[st == k].mean()),
                                                             "psnr": float(P[st == k].mean()), "n": int((st == k).sum())}
                                            for k in range(3) if (st == k).any()},
           "params": p, "g_params": count_params(G)}
    # style-swap consistency: the same photo rendered in all three styles
    with torch.no_grad():
        idx = [int(i) for i in np.random.default_rng(1).choice(len(test_ds), 6, replace=False)]
        fig, ax = plt.subplots(len(idx), 5, figsize=(7.5, 1.6 * len(idx)))
        for r, i in enumerate(idx):
            x, y, s = test_ds[i]
            ax[r, 0].imshow(to01(x).permute(1, 2, 0).numpy())
            ax[r, 1].imshow(to01(y[0]).numpy(), cmap="gray")
            for k in range(3):
                g = G(x[None].to(dev), torch.tensor([k], device=dev))[0, 0].cpu()
                ax[r, 2 + k].imshow(to01(g).numpy(), cmap="gray", vmin=0, vmax=1)
            for c in range(5):
                ax[r, c].axis("off")
            ax[r, 1].set_title(f"GT ({STYLE_NAMES[s]})" if r == 0 else STYLE_NAMES[s], fontsize=7)
        for c, t in enumerate(["photo", "", "gen Style 1", "gen Style 2", "gen Style 3"]):
            if t:
                ax[0, c].set_title(t, fontsize=7)
        fig.tight_layout()
        fig.savefig(O / "style_swap_test.png", dpi=130)
        plt.close(fig)
        # best / worst test examples
        order = np.argsort(S.numpy())
        picks = [int(i) for i in list(order[-6:][::-1]) + list(order[:6])]
        x = torch.stack([test_ds[i][0] for i in picks]); y = torch.stack([test_ds[i][1] for i in picks])
        s = torch.tensor([test_ds[i][2] for i in picks])
        sample_grid(G, (x, y, s), O / "test_best_worst.png", "Test: 6 best (left) and 6 worst (right) by SSIM")
    res["worst_examples"] = [d["test_names"][test_ds.idx[i]] for i in picks[6:]]
    save_json(res, O / "test_results.json")
    info = export_onnx(G, (torch.rand(1, 3, 128, 128) * 2 - 1, torch.tensor([1])), MD / "face2sketch_generator.onnx",
                       ["photo", "style"], ["sketch"])
    save_json(info, O / "onnx_check.json")
    with mlflow.start_run(run_name="t4_evaluation"):
        mlflow.log_metrics({f"test_{k}": v for k, v in overall.items()})
        log_light_artifacts(mlflow, O, "task4")
    print("TASK 4 DONE", res["overall"], res["by_style"], info)


if __name__ == "__main__":
    main()
