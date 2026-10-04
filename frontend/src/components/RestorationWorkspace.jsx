import React, { useState } from "react";
import { postForm } from "../api";
import {
  Bars, CorruptionControls, ErrorBox, ImagePanel, ImageSource, PRETTY, RunButton, Segmented, Settings, Stat, dl,
} from "./ui";

const ENDPOINT = { universal: "/api/restore/universal", hard: "/api/restore/hard", soft: "/api/restore/soft" };

/** Shared workspace for Task 1 (universal), Task 2 (hard routing) and Task 3 (soft MoE). */
export default function RestorationWorkspace({ mode }) {
  const [img, setImg] = useState(null);
  const [corruption, setCorruption] = useState("saltpepper");
  const [severity, setSeverity] = useState("medium");
  const [seed, setSeed] = useState(0);
  const [routeMode, setRouteMode] = useState("predicted");
  const [res, setRes] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function run() {
    setLoading(true);
    setError("");
    try {
      const r = await postForm(ENDPOINT[mode], {
        file: img?.file, sample: img?.sample, corruption, severity, seed,
        ...(mode === "hard" ? { route_mode: routeMode } : {}),
      });
      setRes(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const m = res?.metrics, im = res?.input_metrics;
  const fmt = (v, d = 2) => (v === undefined || v === null ? "—" : !isFinite(v) ? "∞" : v.toFixed(d));

  return (
    <div className="grid gap-6 xl:grid-cols-[340px_1fr]">
      <div className="card space-y-6 self-start">
        <ImageSource kind="pets" value={img} onChange={(v) => { setImg(v); if (v.file) setCorruption("clean"); }} />
        <CorruptionControls corruption={corruption} setCorruption={setCorruption} severity={severity}
          setSeverity={setSeverity} seed={seed} setSeed={setSeed} isUpload={!!img?.file} />
        {mode === "hard" && (
          <div className="space-y-2">
            <div className="label">Routing mode</div>
            <Segmented value={routeMode} onChange={setRouteMode}
              options={[{ value: "predicted", label: "Predicted (classifier)" }, { value: "oracle", label: "Oracle (true label)" }]} />
          </div>
        )}
        <RunButton loading={loading} disabled={!img} onClick={run}>
          {mode === "universal" ? "Restore image" : mode === "hard" ? "Classify & route" : "Run mixture of experts"}
        </RunButton>
        <ErrorBox error={error} />
      </div>

      <div className="space-y-6">
        <div className="card">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <ImagePanel title="Clean reference" src={res?.clean} />
            <ImagePanel title="Model input" src={res?.input}
              badge={res && <span className="chip">{res.corruption.kind}{res.corruption.severity ? ` · ${res.corruption.severity}` : ""}</span>}
              onDownload={() => dl(res.input, `input_${mode}.png`)} />
            <ImagePanel title="Restored output" src={res?.output}
              badge={res && <span className="chip text-brand-300">{res.inference_ms} ms</span>}
              onDownload={() => dl(res.output, `restored_${mode}.png`)} />
          </div>
        </div>

        {res && (
          <div className="grid gap-6 2xl:grid-cols-[1.4fr_1fr]">
            <div className="card space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <Stat label="Inference time" value={`${res.inference_ms} ms`}
                  sub={mode === "hard" ? `classifier ${res.classifier_ms} ms + expert ${res.expert_ms} ms` : "ONNX Runtime · CPU"} />
                <Stat label="PSNR (in → out)" value={m ? `${fmt(im.psnr, 1)} → ${fmt(m.psnr, 1)}` : "n/a"} sub={m ? "dB vs clean reference" : "no reference for uploads"} />
                <Stat label="SSIM (in → out)" value={m ? `${fmt(im.ssim, 3)} → ${fmt(m.ssim, 3)}` : "n/a"} />
                {mode === "hard" && <Stat label="Selected expert" value={res.expert} sub={`predicted: ${PRETTY[res.predicted]} · mode: ${res.route_mode}`} />}
                {mode === "soft" && <Stat label="Strongest contributor" value={PRETTY[res.dominant]} sub={`${(res.weights[res.dominant] * 100).toFixed(1)}% of the output`} />}
                {mode === "universal" && <Stat label="Model" value="Universal DAE" sub="one network, corruption not given" />}
              </div>
              <Settings info={res.corruption} />
            </div>
            <div className="card space-y-5">
              {mode === "hard" && (
                <>
                  <Bars title="Classifier probabilities" values={res.probabilities} highlight={res.predicted} />
                  <RouteDiagram chosen={res.routed_to} />
                </>
              )}
              {mode === "soft" && (
                <>
                  <Bars title="Gating weights (softmax / τ)" values={res.weights} highlight={res.dominant} />
                  <MixBar weights={res.weights} />
                </>
              )}
              {mode === "universal" && (
                <div className="space-y-3 text-sm text-slate-400">
                  <div className="label">How it works</div>
                  <p>A single convolutional autoencoder (encoder → 8×8 spatial bottleneck → decoder, no skip connections) reconstructs the clean image from any of the four input conditions without being told which corruption was applied.</p>
                  <p>Trained with α·L1 + (1−α)·(1−SSIM); α and the architecture were selected with Optuna.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function RouteDiagram({ chosen }) {
  const rows = ["clean", "saltpepper", "blur", "occlusion"];
  return (
    <div>
      <div className="label mb-2">Hard route</div>
      <div className="space-y-1.5">
        {rows.map((r) => (
          <div key={r} className={`flex items-center justify-between rounded-lg border px-3 py-2 text-xs transition ${r === chosen ? "border-brand-400 bg-brand-500/15 text-white" : "border-ink-700 text-slate-500"}`}>
            <span>{r === "clean" ? "Identity bypass (no expert)" : `A_${r}`}</span>
            <span>{r === chosen ? "● executed" : "skipped"}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function MixBar({ weights }) {
  return (
    <div>
      <div className="label mb-2">Contribution to x̂ = Σ wₖ·branchₖ(x̃)</div>
      <div className="flex h-6 overflow-hidden rounded-lg">
        {Object.entries(weights).map(([k, v]) => (
          <div key={k} title={`${k}: ${(v * 100).toFixed(1)}%`} style={{ width: `${v * 100}%`, background: { identity: "#94a3b8", saltpepper: "#f59e0b", blur: "#38bdf8", occlusion: "#34d399" }[k] }} />
        ))}
      </div>
    </div>
  );
}
