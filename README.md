# GenAI Assignment 1 – Restoration Experts & Face-to-Sketch cGAN

**Umaima Zeeshan · 23i-0790 · Generative AI**

Four generative models, trained with PyTorch, tuned with **Optuna**, tracked with **MLflow**, exported to **ONNX**, and served by one **FastAPI + React/Tailwind** application through **Docker Compose**.

| Task | Workspace in the app | Model |
|---|---|---|
| 1 | Universal Restoration | Bottlenecked convolutional denoising autoencoder (no skip connections) |
| 2 | Hard-Routed Restoration | CNN corruption classifier → identity bypass or one of 3 specialist autoencoders |
| 3 | Soft Mixture-of-Experts Restoration | Gate (init. from classifier) blends identity + 3 experts, jointly fine-tuned |
| 4 | Face-to-Sketch Generator | Style-embedding U-Net generator + conditional PatchGAN (pix2pix) on FS2K |

Report: [`report/main.pdf`](report/main.pdf) · Demo video: https://youtu.be/VIDEO_ID

---

## 1. Run the application (one command)

Requirements: Docker Desktop (or Docker Engine + Compose v2).

```bash
git clone https://github.com/maima2024/GenAI-Assignment1.git
cd GenAI-Assignment1
docker compose up --build
```

| Service | URL |
|---|---|
| Web application (4 workspaces) | http://localhost:3000 |
| FastAPI interactive docs | http://localhost:8000/docs |
| MLflow experiment tracking UI | http://localhost:5000 |

**Model files.** On first start the backend downloads `onnx_models.zip` from the
[v1.0 GitHub release](https://github.com/maima2024/GenAI-Assignment1/releases/tag/v1.0) into `./models/`
(≈ 150–250 MB, one time). To do it manually instead: download the zip from the release page, extract the
`.onnx` files into `models/`, then run `docker compose up --build`. The *System Status* page in the app shows
which models are loaded.

Stop with `Ctrl+C` / `docker compose down`.

## 2. Repository layout

```
src/genai/            training package
  corruptions.py      corruption definitions (shared with the backend - single source of truth)
  pet_data.py         Oxford-IIIT Pet caching, 80/20 split (seed 42), runtime corruption Dataset,
                      balanced batch sampler, deterministic val/test manifests
  fs2k_data.py        FS2K pairs, style-stratified 15% validation split, paired augmentation
  models.py           autoencoder, classifier, SoftMoE, U-Net generator, PatchGAN
  engine.py           training loops, evaluation, figures, ONNX export + verification
  common.py           SSIM/PSNR, seeds, budgets, MLflow setup
scripts/
  task1_udae.py           Optuna → final training → per-corruption/severity eval → ONNX
  task2_hard_routing.py   classifier study + training, shared specialist study, 3 specialists,
                          oracle vs predicted routing, misrouting analysis, ONNX
  task3_soft_moe.py       warm-started soft MoE, Optuna (with collapse pruning), gate analysis, ONNX
  task4_cgan.py           cGAN Optuna study, full retrain, fixed-sample grids, test eval, ONNX
  collect_results.py      merges the Kaggle/Colab result zips into models/, report/, experiment-tracking/
  make_diagrams.py        architecture diagrams for the report
notebooks/                Kaggle/Colab notebooks that run the scripts on a free GPU
backend/                  FastAPI app (ONNX Runtime) + Dockerfile
frontend/                 React + Tailwind (Vite) app + nginx Dockerfile
experiment-tracking/      MLflow runs (file store) + MLflow UI Dockerfile
report/                   IEEE LaTeX report (main.tex) and figures
docs/                     Google Stitch prompts/design evidence, demo-video script
docker-compose.yml
```

## 3. Reproduce training

Training was run on free GPUs (Kaggle for Tasks 1–3, Colab for Task 4) with the notebooks in `notebooks/`:

* `A_tasks1-3_restoration.ipynb` – downloads Oxford-IIIT Pet automatically and runs Tasks 1 → 2 → 3.
* `B_task4_face2sketch.ipynb` – reads `FS2K.zip` from Google Drive and runs Task 4.

Locally (GPU recommended):

```bash
pip install -r requirements.txt
export BUDGET=fast            # smoke | fast | full  (see src/genai/common.py)
python scripts/task1_udae.py  # Pets are downloaded to data/raw automatically
python scripts/task2_hard_routing.py
python scripts/task3_soft_moe.py
FS2K_SEARCH=/path/to/FS2K.zip python scripts/task4_cgan.py
python scripts/collect_results.py      # copy models / figures / tables / mlruns into place
```

Every stage is resumable (Optuna studies live in `outputs/optuna*.db`; finished stages are skipped).
`BUDGET=smoke` runs the full pipeline on a tiny subset in a few minutes on CPU, which is how the code was tested.

Datasets are **not** stored in the repository:
* Oxford-IIIT Pet – https://www.robots.ox.ac.uk/~vgg/data/pets/ (auto-downloaded by torchvision)
* FS2K – https://github.com/DengPingFan/FS2K

## 4. Data protocol (Tasks 1–3)

* Official trainval → 80 % train / 20 % validation (seed 42); official test untouched until final evaluation.
* RGB, 128×128. Corruptions are generated **at load time**; each batch contains the four conditions in equal numbers.
* Validation manifest (`data/cache/val_manifest.json`) and test manifest (`test_manifest.json`) store corruption
  type, severity, parameters, rectangle coordinates, blur settings and seed. The test manifest has 10 inputs per test
  image: clean + 3 corruptions × fixed low/medium/high severities.

## 5. API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, loaded ONNX models with input/output signatures |
| POST | `/api/restore/universal` | Task 1 |
| POST | `/api/restore/hard` | Task 2 (`route_mode=predicted|oracle`) – probabilities, predicted class, expert |
| POST | `/api/restore/soft` | Task 3 – four routing weights, dominant expert |
| POST | `/api/sketch` | Task 4 (`style=1|2|3`) |

Restoration endpoints accept `file` (upload) **or** `sample` (bundled clean image), plus
`corruption` (`clean|saltpepper|blur|occlusion`), `severity` (`low|medium|high`) and `seed`.

## 6. AI assistance

AI tools were used for coding assistance and drafting; see the AI-use appendix in the report for details and verification steps.
