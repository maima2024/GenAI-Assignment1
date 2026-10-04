"""Task 2: corruption classifier + hard-routed specialist autoencoders.

Stages (resumable):
  1. Optuna search for the classifier (lr, batch size, channels, dropout, weight decay).
  2. Final classifier training + evaluation (accuracy, macro P/R/F1, per-class, confusion matrix).
  3. Shared Optuna search for the specialist architecture (lr, bottleneck, channels, batch, alpha).
  4. Independent training of the salt-and-pepper, blur and occlusion specialists.
  5. Oracle-routing vs predicted-routing evaluation, misrouting analysis, figures.
  6. ONNX export of the classifier and all specialists.
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
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from genai.common import log_light_artifacts
from genai.common import (device, get_budget, load_json, out_dir, psnr, save_json, seed_everything, setup_mlflow, ssim,
                          to_uint8)
from genai.corruptions import CLASSES
from genai.engine import (export_onnx, manifest_loader, optuna_plots, plot_curves, summarise, train_autoencoder,
                          train_loader)
from genai.models import ConvAutoencoder, CorruptionClassifier, count_params
from genai.pet_data import ManifestDataset, setup_pets

CH_CFGS = {"16-32-64-128": (16, 32, 64, 128), "32-64-128-256": (32, 64, 128, 256),
           "32-64-128-128": (32, 64, 128, 128), "24-48-96-192": (24, 48, 96, 192)}
SPECIALISTS = ["saltpepper", "blur", "occlusion"]


# --------------------------------------------------------------------------- #
# Classifier helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(clf, loader):
    dev = device()
    clf.eval()
    P, Y = [], []
    for x, _, y, *_ in loader:
        P.append(torch.softmax(clf(x.to(dev)), 1).cpu())
        Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def train_classifier(clf, tl, vl, epochs, lr, wd, trial=None, mlflow=None):
    dev = device()
    clf.to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, epochs * len(tl)))
    hist, best, best_state = [], -1, None
    for ep in range(epochs):
        clf.train()
        tot, n, correct = 0.0, 0, 0
        t0 = time.time()
        for x, _, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            logits = clf(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
            n += len(x)
            correct += (logits.argmax(1) == y).sum().item()
        P, Y = predict(clf, vl)
        vloss = float(-np.log(P[np.arange(len(Y)), Y] + 1e-9).mean())
        _, _, f1, _ = precision_recall_fscore_support(Y, P.argmax(1), average="macro", zero_division=0)
        rec = {"epoch": ep + 1, "train_loss": tot / n, "train_acc": correct / n, "val_loss": vloss,
               "val_acc": float((P.argmax(1) == Y).mean()), "val_macro_f1": float(f1), "time": time.time() - t0}
        hist.append(rec)
        print("clf", rec, flush=True)
        if mlflow:
            mlflow.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=ep + 1)
        if f1 > best:
            best, best_state = f1, {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
        if trial is not None:
            trial.report(1 - f1, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    clf.load_state_dict(best_state)
    return hist


def confusion_plot(y, p, path, title):
    cm = confusion_matrix(y, p, labels=range(4), normalize="true")
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), CLASSES, rotation=30, fontsize=8)
    ax.set_yticks(range(4), CLASSES, fontsize=8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return cm.tolist()


def clf_metrics(y, P, manifest=None):
    p = P.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    rep = classification_report(y, p, labels=range(4), target_names=CLASSES, output_dict=True, zero_division=0)
    out = {"accuracy": float((p == y).mean()), "macro_precision": float(pr), "macro_recall": float(rc),
           "macro_f1": float(f1), "per_class": {c: rep[c] for c in CLASSES}}
    if manifest is not None:
        sev = np.array([m.get("severity", "none") for m in manifest])
        kinds = np.array([m["kind"] for m in manifest])
        out["accuracy_by_kind_severity"] = {}
        for k in CLASSES:
            for s in ["none", "low", "medium", "high"]:
                mm = (kinds == k) & (sev == s)
                if mm.any():
                    out["accuracy_by_kind_severity"][f"{k}/{s}"] = float((p[mm] == y[mm]).mean())
    return out


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def route(x, r, specialists):
    """Hard routing: r==0 identity bypass, r==k -> specialist k."""
    out = x.clone()
    for k, name in enumerate(SPECIALISTS, start=1):
        m = r == k
        if m.any():
            out[m] = specialists[name](x[m])
    return out


@torch.no_grad()
def eval_routing(clf, specialists, images, manifest, workers):
    dev = device()
    dl = DataLoader(ManifestDataset(images, manifest), batch_size=64, shuffle=False, num_workers=workers)
    keys = ["or_psnr", "or_ssim", "pr_psnr", "pr_ssim", "in_psnr", "in_ssim"]
    R = {k: [] for k in keys}
    R["pred"], R["true"], R["probs"] = [], [], []
    for x, y, lab, _ in dl:
        x, y, lab = x.to(dev), y.to(dev), lab.to(dev)
        probs = torch.softmax(clf(x), 1)
        pred = probs.argmax(1)
        o_or, o_pr = route(x, lab, specialists), route(x, pred, specialists)
        R["or_psnr"].append(psnr(o_or, y).cpu())
        R["or_ssim"].append(ssim(o_or, y, reduction="none").cpu())
        R["pr_psnr"].append(psnr(o_pr, y).cpu())
        R["pr_ssim"].append(ssim(o_pr, y, reduction="none").cpu())
        R["in_psnr"].append(psnr(x, y).cpu())
        R["in_ssim"].append(ssim(x, y, reduction="none").cpu())
        R["pred"].append(pred.cpu())
        R["true"].append(lab.cpu())
        R["probs"].append(probs.cpu())
    return {k: torch.cat(v).numpy() for k, v in R.items()}


@torch.no_grad()
def misroute_grid(clf, specialists, images, manifest, picks, path):
    dev = device()
    ds = ManifestDataset(images, manifest)
    n = len(picks)
    fig, ax = plt.subplots(4, n, figsize=(1.9 * n, 7.5))
    ax = np.array(ax).reshape(4, n)
    for j, i in enumerate(picks):
        x, y, lab, _ = ds[i]
        x1, y1 = x[None].to(dev), y[None].to(dev)
        probs = torch.softmax(clf(x1), 1)[0]
        pred = int(probs.argmax())
        o_pr = route(x1, torch.tensor([pred], device=dev), specialists)
        o_or = route(x1, torch.tensor([lab], device=dev), specialists)
        for r, img in enumerate([y, x, o_or[0].cpu(), o_pr[0].cpu()]):
            ax[r, j].imshow(to_uint8(img))
            ax[r, j].axis("off")
        m = ds.items[i]
        ax[0, j].set_title(f"true {m['kind'][:5]}/{m['severity'][:3]}\npred {CLASSES[pred][:5]} ({probs[pred]:.2f})\n"
                           f"oracle {psnr(o_or, y1).item():.1f}dB / pred {psnr(o_pr, y1).item():.1f}dB", fontsize=6)
    for r, name in enumerate(["clean", "input", "oracle route", "predicted route"]):
        ax[r, 0].text(-8, 64, name, rotation=90, va="center", ha="right", fontsize=7)
    fig.suptitle("Task 2: restoration failures caused by classifier errors", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default=None)
    args = ap.parse_args()
    B = get_budget(args.budget)
    seed_everything(42)
    O = out_dir() / "task2"
    O.mkdir(parents=True, exist_ok=True)
    MD = out_dir() / "onnx"
    MD.mkdir(exist_ok=True)
    d = setup_pets(B)
    imgs, W = d["trainval"], B["workers"]
    vl = manifest_loader(imgs, d["val_manifest"], 128, W)
    mlflow = setup_mlflow("task2_hard_routing")
    storage = f"sqlite:///{(out_dir() / 'optuna.db').resolve()}"
    dev = device()

    # ---------------- 1. classifier Optuna ----------------
    f_clf = O / "clf_best_params.json"
    if not f_clf.exists():
        study = optuna.create_study(study_name="task2_classifier", direction="minimize", storage=storage,
                                    load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=42),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))

        def obj(trial):
            p = {"lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                 "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                 "channels": trial.suggest_categorical("channels", list(CH_CFGS)),
                 "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                 "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)}
            seed_everything(42 + trial.number)
            with mlflow.start_run(run_name=f"clf_trial_{trial.number}", nested=True):
                mlflow.log_params(p)
                clf = CorruptionClassifier(CH_CFGS[p["channels"]], p["dropout"])
                h = train_classifier(clf, train_loader(imgs, d["train_idx"], p["batch_size"], W), vl,
                                     B["clf_trial_epochs"], p["lr"], p["weight_decay"], trial, mlflow)
                v = 1 - max(r["val_macro_f1"] for r in h)
                mlflow.log_metric("objective", v)
                return v

        with mlflow.start_run(run_name="clf_optuna_search"):
            done = len([t for t in study.trials if t.state.is_finished()])
            study.optimize(obj, n_trials=max(0, B["clf_trials"] - done), gc_after_trial=True)
            s = optuna_plots(study, O, "task2_classifier")
            mlflow.log_artifact(str(O / "task2_classifier_optuna.png"))
        save_json(s, O / "clf_optuna_summary.json")
        save_json(study.best_params, f_clf)
    pc = load_json(f_clf)

    # ---------------- 2. final classifier ----------------
    clf = CorruptionClassifier(CH_CFGS[pc["channels"]], pc["dropout"]).to(dev)
    f_ck = O / "classifier.pt"
    if not f_ck.exists():
        seed_everything(42)
        with mlflow.start_run(run_name="clf_final_training"):
            mlflow.log_params({**pc, "epochs": B["clf_epochs"], "params": count_params(clf)})
            h = train_classifier(clf, train_loader(imgs, d["train_idx"], pc["batch_size"], W), vl, B["clf_epochs"],
                                 pc["lr"], pc["weight_decay"], mlflow=mlflow)
            torch.save({"state_dict": clf.state_dict(), "config": pc}, f_ck)
            save_json(h, O / "clf_history.json")
            plot_curves(h, [["train_loss", "val_loss"], ["train_acc", "val_acc"], "val_macro_f1"],
                        O / "clf_curves.png", "Task 2 classifier")
    clf.load_state_dict(torch.load(f_ck, map_location=dev)["state_dict"])
    clf.eval()
    test, man = d["test"], d["test_manifest"]
    tl_test = manifest_loader(test, man, 128, W)
    P, Y = predict(clf, tl_test)
    Pv, Yv = predict(clf, vl)
    cres = {"test": clf_metrics(Y, P, man), "val": clf_metrics(Yv, Pv)}
    cres["test"]["confusion_normalized"] = confusion_plot(Y, P.argmax(1), O / "confusion_test.png",
                                                          "Classifier - test (row-normalised)")
    cres["val"]["confusion_normalized"] = confusion_plot(Yv, Pv.argmax(1), O / "confusion_val.png",
                                                         "Classifier - validation (row-normalised)")
    save_json(cres, O / "classifier_results.json")

    # ---------------- 3. shared specialist Optuna ----------------
    f_sp = O / "spec_best_params.json"
    vl_spec = {k: manifest_loader(imgs, d["val_manifest"], 64, W, kinds=[k]) for k in SPECIALISTS}
    vl_corr = manifest_loader(imgs, d["val_manifest"], 64, W, kinds=SPECIALISTS)
    if not f_sp.exists():
        study = optuna.create_study(study_name="task2_specialists", direction="minimize", storage=storage,
                                    load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=43),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))

        def obj(trial):
            p = {"lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                 "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                 "latent_ch": trial.suggest_categorical("latent_ch", [16, 32, 64, 128]),
                 "base": trial.suggest_categorical("base", [16, 32, 48]),
                 "alpha": trial.suggest_float("alpha", 0.5, 0.95)}
            seed_everything(42 + trial.number)
            with mlflow.start_run(run_name=f"spec_trial_{trial.number}", nested=True):
                mlflow.log_params(p)
                m = ConvAutoencoder(p["base"], p["latent_ch"], 0.1)
                tl = train_loader(imgs, d["train_idx"], p["batch_size"], W, kinds=SPECIALISTS)
                h = train_autoencoder(m, tl, vl_corr, B["spec_trial_epochs"], p["lr"], p["alpha"], trial=trial,
                                      mlflow=mlflow)
                v = min(r["val_score"] for r in h)
                mlflow.log_metric("objective", v)
                return v

        with mlflow.start_run(run_name="spec_optuna_search"):
            done = len([t for t in study.trials if t.state.is_finished()])
            study.optimize(obj, n_trials=max(0, B["spec_trials"] - done), gc_after_trial=True)
            s = optuna_plots(study, O, "task2_specialists")
            mlflow.log_artifact(str(O / "task2_specialists_optuna.png"))
        save_json(s, O / "spec_optuna_summary.json")
        save_json(study.best_params, f_sp)
    ps = load_json(f_sp)

    # ---------------- 4. specialists ----------------
    specialists = {}
    for k in SPECIALISTS:
        m = ConvAutoencoder(ps["base"], ps["latent_ch"], 0.1).to(dev)
        f = O / f"specialist_{k}.pt"
        if not f.exists():
            seed_everything(42)
            with mlflow.start_run(run_name=f"specialist_{k}"):
                mlflow.log_params({**ps, "corruption": k, "epochs": B["spec_epochs"], "params": count_params(m)})
                tl = train_loader(imgs, d["train_idx"], ps["batch_size"], W, kinds=[k])
                h = train_autoencoder(m, tl, vl_spec[k], B["spec_epochs"], ps["lr"], ps["alpha"], mlflow=mlflow,
                                      prefix=f"{k}_")
                torch.save({"state_dict": m.state_dict(), "config": ps}, f)
                save_json(h, O / f"history_{k}.json")
                plot_curves(h, ["train_loss", "val_l1", "val_ssim", "val_psnr"], O / f"curves_{k}.png",
                            f"Specialist: {k}")
        m.load_state_dict(torch.load(f, map_location=dev)["state_dict"])
        specialists[k] = m.eval()

    # ---------------- 5. routing evaluation ----------------
    R = eval_routing(clf, specialists, test, man, W)
    torch.save(R, O / "routing_per_item.pt")
    res_or = {"psnr": R["or_psnr"], "ssim": R["or_ssim"], "in_psnr": R["in_psnr"], "in_ssim": R["in_ssim"]}
    res_pr = {"psnr": R["pr_psnr"], "ssim": R["pr_ssim"], "in_psnr": R["in_psnr"], "in_ssim": R["in_ssim"]}
    wrong = R["pred"] != R["true"]
    drop = R["or_psnr"] - R["pr_psnr"]
    routing = {"oracle": summarise(res_or, man), "predicted": summarise(res_pr, man),
               "n_items": int(len(wrong)), "n_misrouted": int(wrong.sum()),
               "misrouted_mean_psnr_drop": float(drop[wrong].mean()) if wrong.any() else 0.0,
               "misrouted_by_true_pred": {}}
    for t in range(4):
        for p_ in range(4):
            mm = wrong & (R["true"] == t) & (R["pred"] == p_)
            if mm.any():
                routing["misrouted_by_true_pred"][f"{CLASSES[t]}->{CLASSES[p_]}"] = {
                    "count": int(mm.sum()), "mean_psnr_drop": float(drop[mm].mean())}
    save_json(routing, O / "routing_results.json")
    if wrong.any():
        order = np.argsort(-drop * wrong)[:6]
        misroute_grid(clf, specialists, test, man, [int(i) for i in order if wrong[i]], O / "misrouting_failures.png")

    # ---------------- 6. ONNX ----------------
    class ClfProbs(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c

        def forward(self, x):
            lg = self.c(x)
            return lg, torch.softmax(lg, 1)

    info = {"classifier": export_onnx(ClfProbs(clf), (torch.rand(1, 3, 128, 128),), MD / "classifier.onnx",
                                      ["input"], ["logits", "probs"])}
    for k, m in specialists.items():
        info[k] = export_onnx(m, (torch.rand(1, 3, 128, 128),), MD / f"specialist_{k}.onnx", ["input"], ["output"])
    save_json(info, O / "onnx_check.json")
    clf.to(dev)
    for m in specialists.values():
        m.to(dev)
    with mlflow.start_run(run_name="t2_evaluation"):
        mlflow.log_metrics({"test_acc": cres["test"]["accuracy"], "test_macro_f1": cres["test"]["macro_f1"],
                            "oracle_psnr": routing["oracle"]["overall"]["psnr"],
                            "pred_psnr": routing["predicted"]["overall"]["psnr"],
                            "n_misrouted": routing["n_misrouted"]})
        log_light_artifacts(mlflow, O, "task2")
    print("TASK 2 DONE", cres["test"]["accuracy"], routing["oracle"]["overall"], routing["predicted"]["overall"])


if __name__ == "__main__":
    main()
