# Google Stitch – design prompts and evidence

The assignment requires the UI to be designed in **Google Stitch** first (https://stitch.withgoogle.com),
with evidence in the report. Use the prompts below, then **screenshot** the generated designs and save them as:

* `report/figures/stitch_design.png` – the main screenshot used in the report (e.g. the Hard-Routed screen)
* `docs/stitch/*.png` – the other screens (optional, for extra evidence)

Choose **Web** (desktop) as the platform.

---

## Prompt 1 – overall app + Universal Restoration screen

> Design a dark-themed desktop web app called **"RestoreLab – Generative AI Studio"** for an AI image-restoration
> research project. Layout: a fixed left sidebar (dark navy #111827) with the logo (purple-to-pink gradient square
> with letter "R"), and five navigation items, each with a small purple caption above the title:
> "TASK 1 – Universal Restoration", "TASK 2 – Hard-Routed Restoration", "TASK 3 – Soft Mixture-of-Experts Restoration",
> "TASK 4 – Face-to-Sketch Generator", "OPS – System Status". At the bottom of the sidebar a small card with a green
> dot "Backend ok – 7/7 ONNX models loaded".
> Main area (near-black #0b0f17 background, indigo #6366f1 accent, rounded-2xl cards, Inter font):
> page title "Universal Restoration" with subtitle "One denoising autoencoder for every corruption".
> Left control card (340 px wide): "Upload image" button, a 4×2 grid of clean sample thumbnails (cats and dogs),
> a "Runtime corruption" section with four selectable tiles (None, Salt & pepper, Gaussian blur, Occlusion) each with a
> coloured dot (grey, amber, sky blue, emerald), a segmented control "Low / Medium / High" severity, a numeric seed
> input with "randomise" link, and a full-width indigo "Restore image" button.
> Right side: a card with three equal square image panels labelled "Clean reference", "Model input" (with a chip
> "occlusion · high") and "Restored output" (with a chip "38 ms"), each with a Download button underneath.
> Below: a stats card with tiles "Inference time 38 ms", "PSNR (in → out) 12.6 → 24.1", "SSIM (in → out) 0.60 → 0.86",
> and a code block "Applied corruption settings" showing JSON.

## Prompt 2 – Hard-Routed Restoration screen

> Same app and style. Screen "Hard-Routed Restoration": same left control card plus a segmented toggle
> "Predicted (classifier) / Oracle (true label)" and button "Classify & route". Right side: the three image panels,
> a stats card (inference time split "classifier 6 ms + expert 30 ms", PSNR, SSIM, "Selected expert: Blur expert"),
> and a card "Classifier probabilities" with four horizontal progress bars (Clean 2%, Salt & pepper 1%,
> Gaussian blur 95% highlighted with a star, Occlusion 2%) in grey/amber/sky/emerald, plus a "Hard route" list of
> four rows (Identity bypass, A_saltpepper, A_blur, A_occlusion) where only the A_blur row is highlighted
> "● executed" and the others say "skipped".

## Prompt 3 – Soft Mixture-of-Experts screen

> Same app and style. Screen "Soft Mixture-of-Experts Restoration": same layout; the right card shows
> "Gating weights (softmax / τ)" with four horizontal bars (Identity 5%, Salt & pepper 61% ★, Blur 30%,
> Occlusion 4%) and underneath a single stacked horizontal bar "Contribution to x̂ = Σ wₖ·branchₖ(x̃)"
> split proportionally in the four colours. A stat tile "Strongest contributor: Salt & pepper – 61% of the output".

## Prompt 4 – Face-to-Sketch Generator screen

> Same app and style. Screen "Face-to-Sketch Generator": left card with "Upload image" and "Webcam" buttons,
> four face-photo sample thumbnails, a segmented control "Style 1 / Style 2 / Style 3" and a button
> "Generate sketch". Right side: two large square panels side by side, "Original photograph" and
> "Generated sketch · Style 2" (a grey-scale pencil sketch of the face) with a Download button and an inference-time
> chip. Below, three stat tiles: "Generator – U-Net + style embedding", "Inference time 45 ms", "Condition Style 2".

## Prompt 5 – System Status screen (optional)

> Same app and style. Screen "System Status": three cards on top (API status "ok", ONNX Runtime "1.20.1 –
> CPUExecutionProvider", Experiment tracking "Open MLflow UI ↗"), and a table "Loaded models" with columns
> Model, Status, Size, Inputs, Outputs and seven rows: udae.onnx, classifier.onnx, specialist_saltpepper.onnx,
> specialist_blur.onnx, specialist_occlusion.onnx, soft_moe.onnx, face2sketch_generator.onnx (all "loaded" in green).

---

**How the design was used:** the React/Tailwind implementation follows the Stitch layout (sidebar navigation,
340 px control card + results area, three-panel image comparison, stat tiles, coloured expert bars). Colours were
mapped to Tailwind tokens in `frontend/tailwind.config.js`.
