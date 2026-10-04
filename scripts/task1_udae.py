"""Task 1: Universal multi-corruption denoising autoencoder.

Stages (each is skipped if its output already exists, so the script can resume):
  1. Optuna search over lr, batch size, bottleneck size, encoder channels, dropout, alpha.
  2. Final training of the best configuration.
  3. Evaluation on the deterministic test manifest (by corruption and severity) + figures.
  4. ONNX export and PyTorch/ONNX consistency check.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import optuna
import torch

from genai.common import log_light_artifacts
from genai.common import device, get_budget, out_dir, save_json, seed_everything, setup_mlflow, load_json
from genai.engine import (eval_recon, export_onnx, manifest_loader, optuna_plots, per_item_restoration, pick_examples,
                          pick_failures, plot_curves, restoration_grid, summarise, train_autoencoder, train_loader)
from genai.models import ConvAutoencoder, count_params
from genai.pet_data import setup_pets

SPACE = {"lr": (1e-4, 3e-3), "batch_size": [16, 32, 64], "latent_ch": [16, 32, 64, 128],
         "base": [16, 32, 48], "dropout": (0.0, 0.3), "alpha": (0.5, 0.95)}


def suggest(trial):
    return {"lr": trial.suggest_float("lr", *SPACE["lr"], log=True),
            "batch_size": trial.suggest_categorical("batch_size", SPACE["batch_size"]),
            "latent_ch": trial.suggest_categorical("latent_ch", SPACE["latent_ch"]),
            "base": trial.suggest_categorical("base", SPACE["base"]),
            "dropout": trial.suggest_float("dropout", *SPACE["dropout"]),
            "alpha": trial.suggest_float("alpha", *SPACE["alpha"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default=None)
    args = ap.parse_args()
    B = get_budget(args.budget)
    seed_everything(42)
    O = out_dir() / "task1"
    O.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir() / "onnx"
    models_dir.mkdir(exist_ok=True)
    d = setup_pets(B)
    imgs = d["trainval"]
    W = B["workers"]
    vl = manifest_loader(imgs, d["val_manifest"], 64, W)
    mlflow = setup_mlflow("task1_universal_dae")

    # ---------------- 1. Optuna ----------------
    best_file = O / "best_params.json"
    if not best_file.exists():
        storage = f"sqlite:///{(out_dir() / 'optuna.db').resolve()}"
        study = optuna.create_study(study_name="task1_udae", direction="minimize", storage=storage, load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=42),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))

        def objective(trial):
            p = suggest(trial)
            seed_everything(42 + trial.number)
            with mlflow.start_run(run_name=f"t1_trial_{trial.number}", nested=True):
                mlflow.log_params({**p, "stage": "optuna", "trial": trial.number})
                m = ConvAutoencoder(p["base"], p["latent_ch"], p["dropout"])
                tl = train_loader(imgs, d["train_idx"], p["batch_size"], W)
                hist = train_autoencoder(m, tl, vl, B["t1_trial_epochs"], p["lr"], p["alpha"], trial=trial,
                                         mlflow=mlflow, prefix="")
                best = min(h["val_score"] for h in hist)
                mlflow.log_metric("objective", best)
                return best

        done = len([t for t in study.trials if t.state.is_finished()])
        with mlflow.start_run(run_name="t1_optuna_search"):
            mlflow.log_params({"n_trials": B["t1_trials"], "trial_epochs": B["t1_trial_epochs"],
                               "objective": "val L1 + (1 - val SSIM)", "space": str(SPACE)})
            study.optimize(objective, n_trials=max(0, B["t1_trials"] - done), gc_after_trial=True)
            summ = optuna_plots(study, O, "task1")
            summ["search_space"] = {k: str(v) for k, v in SPACE.items()}
            mlflow.log_metric("best_objective", summ["best_value"])
            mlflow.log_artifact(str(O / "task1_optuna.png"))
        save_json(summ, O / "optuna_summary.json")
        save_json(study.best_params, best_file)
    p = load_json(best_file)
    print("best params", p)

    # ---------------- 2. Final training ----------------
    ckpt = O / "udae.pt"
    model = ConvAutoencoder(p["base"], p["latent_ch"], p["dropout"]).to(device())
    if not ckpt.exists():
        seed_everything(42)
        with mlflow.start_run(run_name="t1_final_training"):
            mlflow.log_params({**p, "epochs": B["t1_epochs"], "stage": "final", "params": count_params(model),
                               "latent_dim": model.latent_dim})
            tl = train_loader(imgs, d["train_idx"], p["batch_size"], W)
            hist = train_autoencoder(model, tl, vl, B["t1_epochs"], p["lr"], p["alpha"], mlflow=mlflow)
            torch.save({"state_dict": model.state_dict(), "config": p}, ckpt)
            save_json(hist, O / "history.json")
            plot_curves(hist, ["train_loss", ["val_l1"], "val_ssim", "val_psnr"], O / "curves.png", "Task 1 UDAE")
            mlflow.log_artifact(str(O / "curves.png"))
    model.load_state_dict(torch.load(ckpt, map_location=device())["state_dict"])
    model.eval()

    # ---------------- 3. Evaluation ----------------
    test, man = d["test"], d["test_manifest"]
    res, _ = per_item_restoration(model, test, man, workers=W)
    summary = summarise(res, man)
    summary["params"] = count_params(model)
    summary["latent_dim"] = model.latent_dim
    summary["val"] = eval_recon(model, vl)
    save_json(summary, O / "test_results.json")
    torch.save(res, O / "test_per_item.pt")
    restoration_grid(model, test, man, pick_examples(man, res), O / "examples.png", "Task 1: representative results")
    restoration_grid(model, test, man, pick_failures(man, res), O / "failures.png", "Task 1: failure cases")

    # ---------------- 4. ONNX ----------------
    onnx_info = export_onnx(model, (torch.rand(1, 3, 128, 128),), models_dir / "udae.onnx", ["input"], ["output"])
    save_json(onnx_info, O / "onnx_check.json")
    with mlflow.start_run(run_name="t1_evaluation"):
        mlflow.log_metrics({f"test_{k}": v for k, v in summary["overall"].items()})
        for k, v in summary["by_kind"].items():
            mlflow.log_metrics({f"test_{k}_psnr": v["psnr"], f"test_{k}_ssim": v["ssim"]})
        log_light_artifacts(mlflow, O, "task1")
    print("TASK 1 DONE", summary["by_kind"], onnx_info)


if __name__ == "__main__":
    main()
