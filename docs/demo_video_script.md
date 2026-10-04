# Demo video script (target 6 minutes, allowed 5–7)

Record your screen with OBS, Xbox Game Bar (Win + Alt + R) or Loom, and talk over it.
Upload to YouTube as **Unlisted**, then put the link in `report/main.tex` (title block) and `README.md`.
Each **[REQUIRED]** tag marks something the assignment explicitly says the video must show.

| Time | On screen | What to say (roughly) |
|---|---|---|
| 0:00–0:20 | GitHub repo page | "Hi, I'm Umaima Zeeshan, 23i-0790. This is my Generative AI Assignment 1: four generative models served from one web app. Everything is in this repository." |
| 0:20–1:00 | Terminal: `git clone …`, `cd GenAI-Assignment1`, `docker compose up --build`; then Docker Desktop showing 3 containers running | **[REQUIRED startup]** "An evaluator only needs one command. Compose builds the FastAPI backend, the React frontend served by nginx, and an MLflow server. On first start the backend downloads the ONNX models from the GitHub release." |
| 1:00–1:15 | Browser `localhost:3000` → System Status page | "Status page: all seven ONNX models are loaded, with their input and output shapes." |
| 1:15–2:05 | **Universal Restoration** | **[REQUIRED upload + runtime corruption]** Pick a sample → Salt & pepper, High → *Restore image*. Then Blur, Medium. Then Occlusion, High. "The same autoencoder handles every corruption without being told which one it is. You can see input/output PSNR and SSIM, the exact corruption parameters, and the inference time." Then **upload** one of your own pet photos with "None (already corrupted)" or with a corruption. **[REQUIRED universal restoration]** Click **Download** on the output. **[REQUIRED downloading]** |
| 2:05–3:05 | **Hard-Routed Restoration** | **[REQUIRED hard routing]** Blur, Low → *Classify & route*. "The classifier gives four probabilities; blur is predicted, so only the blur expert runs and the others are skipped." Try Clean: "clean images go through the identity bypass, no network runs." Then find a case it gets wrong (low-severity blur or a small occlusion often works) and switch **Predicted ↔ Oracle** to show the restoration difference. "This is the weakness of hard routing: a classifier mistake sends the image to the wrong expert." |
| 3:05–4:05 | **Soft Mixture-of-Experts** | **[REQUIRED soft expert weights]** Run S&P medium, then Occlusion high. "Here the gate gives a continuous weight to every branch; the bars show how much each expert contributes to the output." Upload or create a borderline case (low blur) and point out weights shared between identity and blur. |
| 4:05–5:05 | **Face-to-Sketch Generator** | **[REQUIRED face-to-sketch]** Pick a face sample → Style 1 → *Generate*. Switch to Style 2 and Style 3 (it regenerates automatically). "The style is a learned embedding fed into both generator and discriminator, so the same photo gives different sketch styles." Use **Webcam** to capture your own face and generate. Click **Download**. **[REQUIRED downloading]** |
| 5:05–5:50 | `localhost:5000` MLflow UI | **[REQUIRED experiment tracking]** Open experiment *task1_universal_dae*: show the Optuna trial runs (nested), sort by `objective`, open the final run and show the loss curves / metrics charts. Open *task4_face2sketch_cgan* → final run → Artifacts → samples over epochs. "Every Optuna trial, the final training, metrics and figures are tracked in MLflow." |
| 5:50–6:10 | `localhost:8000/docs` (optional) then back to app | "The backend validates uploads, preprocesses, runs ONNX Runtime and returns routing and timing information. Thanks for watching." |

Tips
* Do a full dry run once before recording so the models are already downloaded (startup is then fast).
* Keep browser zoom at 90% so the whole workspace fits on screen.
* If something is slow, cut the waiting in YouTube's editor or just keep talking.
