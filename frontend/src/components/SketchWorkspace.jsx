import React, { useState } from "react";
import { postForm } from "../api";
import { ErrorBox, ImagePanel, ImageSource, RunButton, Segmented, Stat, dl } from "./ui";

export default function SketchWorkspace() {
  const [img, setImg] = useState(null);
  const [style, setStyle] = useState(1);
  const [res, setRes] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function run(s = style) {
    setLoading(true);
    setError("");
    try {
      setRes(await postForm("/api/sketch", { file: img?.file, sample: img?.sample, style: s }));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[340px_1fr]">
      <div className="card space-y-6 self-start">
        <ImageSource kind="faces" value={img} onChange={setImg} webcam />
        <div className="space-y-2">
          <div className="label">Sketch style (learned embedding)</div>
          <Segmented value={style} onChange={(s) => { setStyle(s); if (res) run(s); }}
            options={[{ value: 1, label: "Style 1" }, { value: 2, label: "Style 2" }, { value: 3, label: "Style 3" }]} />
        </div>
        <RunButton loading={loading} disabled={!img} onClick={() => run()}>Generate sketch</RunButton>
        <ErrorBox error={error} />
        <p className="text-[11px] leading-relaxed text-slate-500">Tip: a front-facing, well-lit face filling most of the frame works best (FS2K photos are tightly cropped portraits).</p>
      </div>
      <div className="space-y-6">
        <div className="card">
          <div className="grid gap-6 sm:grid-cols-2">
            <ImagePanel title="Original photograph (128×128)" src={res?.photo} pixelated={false} />
            <ImagePanel title={`Generated sketch${res ? ` · Style ${res.style}` : ""}`} src={res?.sketch} pixelated={false}
              onDownload={() => dl(res.sketch, `sketch_style${res.style}.png`)}
              badge={res && <span className="chip text-brand-300">{res.inference_ms} ms</span>} />
          </div>
        </div>
        {res && (
          <div className="grid gap-3 sm:grid-cols-3">
            <Stat label="Generator" value="U-Net + style emb." sub="conditional GAN (PatchGAN-trained)" />
            <Stat label="Inference time" value={`${res.inference_ms} ms`} sub="ONNX Runtime · CPU" />
            <Stat label="Condition" value={`Style ${res.style}`} sub="FS2K style category" />
          </div>
        )}
      </div>
    </div>
  );
}
