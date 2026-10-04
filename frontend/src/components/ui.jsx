import React, { useEffect, useRef, useState } from "react";
import { download, getJSON, sampleUrl } from "../api";

export const EXPERT_COLORS = {
  clean: "#94a3b8", identity: "#94a3b8", saltpepper: "#f59e0b", blur: "#38bdf8", occlusion: "#34d399",
};
export const PRETTY = {
  clean: "Clean", identity: "Identity", saltpepper: "Salt & pepper", blur: "Gaussian blur", occlusion: "Occlusion",
};

export function Segmented({ options, value, onChange }) {
  return (
    <div className="seg">
      {options.map((o) => (
        <button key={o.value} className={value === o.value ? "on" : ""} onClick={() => onChange(o.value)} type="button">
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Image source: upload a file or pick a bundled sample (optionally a webcam capture). */
export function ImageSource({ kind = "pets", value, onChange, webcam = false }) {
  const [samples, setSamples] = useState([]);
  const [camOn, setCamOn] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    getJSON("/api/samples").then((s) => setSamples(s[kind] || [])).catch(() => setSamples([]));
  }, [kind]);
  useEffect(() => () => stopCam(), []);

  function onFile(e) {
    const f = e.target.files?.[0];
    if (f) onChange({ file: f, preview: URL.createObjectURL(f), name: f.name });
  }
  async function startCam() {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 480 } });
      streamRef.current = s;
      setCamOn(true);
      setTimeout(() => { if (videoRef.current) videoRef.current.srcObject = s; }, 50);
    } catch (e) {
      alert("Webcam unavailable: " + e.message);
    }
  }
  function stopCam() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCamOn(false);
  }
  function capture() {
    const v = videoRef.current;
    const side = Math.min(v.videoWidth, v.videoHeight);
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    c.getContext("2d").drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, 256, 256);
    c.toBlob((b) => {
      const f = new File([b], "webcam.png", { type: "image/png" });
      onChange({ file: f, preview: URL.createObjectURL(f), name: "webcam capture" });
      stopCam();
    }, "image/png");
  }

  return (
    <div className="space-y-3">
      <div className="label">Input image</div>
      <div className="flex gap-2">
        <label className="btn-ghost flex-1 cursor-pointer">
          <UploadIcon /> Upload image
          <input type="file" accept="image/*" className="hidden" onChange={onFile} />
        </label>
        {webcam && (
          <button className="btn-ghost" type="button" onClick={camOn ? stopCam : startCam}>
            <CamIcon /> {camOn ? "Close" : "Webcam"}
          </button>
        )}
      </div>
      {camOn && (
        <div className="space-y-2">
          <video ref={videoRef} autoPlay playsInline muted className="aspect-square w-full rounded-xl bg-black object-cover" />
          <button className="btn-primary w-full" onClick={capture} type="button">Capture photo</button>
        </div>
      )}
      {samples.length > 0 && (
        <>
          <div className="text-xs text-slate-500">or pick a clean sample</div>
          <div className="grid grid-cols-4 gap-2">
            {samples.map((s) => (
              <button key={s} type="button" onClick={() => onChange({ sample: s, preview: sampleUrl(s), name: s })}
                className={`overflow-hidden rounded-lg border-2 transition ${value?.sample === s ? "border-brand-400" : "border-transparent hover:border-ink-600"}`}>
                <img src={sampleUrl(s)} alt={s} className="aspect-square w-full object-cover" />
              </button>
            ))}
          </div>
        </>
      )}
      {value?.preview && (
        <div className="flex items-center gap-3 rounded-xl border border-ink-700 bg-ink-800 p-2">
          <img src={value.preview} className="h-12 w-12 rounded-lg object-cover" alt="selected" />
          <div className="min-w-0 text-xs">
            <div className="truncate font-medium text-slate-200">{value.name}</div>
            <div className="text-slate-500">{value.file ? "uploaded file" : "bundled clean sample"}</div>
          </div>
        </div>
      )}
    </div>
  );
}

export function CorruptionControls({ corruption, setCorruption, severity, setSeverity, seed, setSeed, isUpload }) {
  return (
    <div className="space-y-3">
      <div className="label">Runtime corruption</div>
      <div className="grid grid-cols-2 gap-2">
        {["clean", "saltpepper", "blur", "occlusion"].map((c) => (
          <button key={c} type="button" onClick={() => setCorruption(c)}
            className={`rounded-xl border px-3 py-2 text-left text-xs font-medium transition ${corruption === c ? "border-brand-400 bg-brand-500/15 text-white" : "border-ink-700 bg-ink-800 text-slate-400 hover:border-ink-600"}`}>
            <span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: EXPERT_COLORS[c] }} />
            {c === "clean" ? (isUpload ? "None (already corrupted)" : "None (clean)") : PRETTY[c]}
          </button>
        ))}
      </div>
      {corruption !== "clean" && (
        <>
          <div className="label">Severity</div>
          <Segmented value={severity} onChange={setSeverity}
            options={[{ value: "low", label: "Low" }, { value: "medium", label: "Medium" }, { value: "high", label: "High" }]} />
          <div className="text-[11px] leading-relaxed text-slate-500">{SEV_HELP[corruption][severity]}</div>
          <div className="flex items-center gap-2">
            <span className="label">Seed</span>
            <input type="number" value={seed} onChange={(e) => setSeed(parseInt(e.target.value || "0", 10))}
              className="w-24 rounded-lg border border-ink-700 bg-ink-800 px-2 py-1 text-xs" />
            <button className="text-xs text-brand-300 hover:underline" type="button" onClick={() => setSeed(Math.floor(Math.random() * 10000))}>randomise</button>
          </div>
        </>
      )}
    </div>
  );
}

const SEV_HELP = {
  saltpepper: { low: "p = 0.03 of pixels replaced", medium: "p = 0.08 of pixels replaced", high: "p = 0.15 of pixels replaced" },
  blur: { low: "kernel 3, sigma 0.7", medium: "kernel 5, sigma 1.5", high: "kernel 7, sigma 2.5" },
  occlusion: { low: "1 rectangle, ~10% area", medium: "2 rectangles, ~20% area", high: "3 rectangles, ~35% area" },
};

export function ImagePanel({ title, src, badge, onDownload, pixelated = true }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="label">{title}</span>
        {badge}
      </div>
      <div className="relative aspect-square overflow-hidden rounded-xl border border-ink-700 bg-ink-800">
        {src ? (
          <img src={src} alt={title} className="h-full w-full object-contain" style={pixelated ? { imageRendering: "pixelated" } : {}} />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-slate-600">—</div>
        )}
      </div>
      {onDownload && src && (
        <button className="btn-ghost py-1.5 text-xs" onClick={onDownload} type="button"><DownloadIcon /> Download</button>
      )}
    </div>
  );
}

export function Stat({ label, value, sub }) {
  return (
    <div className="rounded-xl border border-ink-700 bg-ink-800 px-4 py-3">
      <div className="label">{label}</div>
      <div className="mt-1 font-mono text-base font-semibold text-white">{value}</div>
      {sub && <div className="text-[11px] text-slate-500">{sub}</div>}
    </div>
  );
}

export function Bars({ values, highlight, title }) {
  const entries = Object.entries(values);
  return (
    <div className="space-y-2.5">
      <div className="label">{title}</div>
      {entries.map(([k, v]) => (
        <div key={k}>
          <div className="mb-1 flex justify-between text-xs">
            <span className={k === highlight ? "font-semibold text-white" : "text-slate-400"}>
              <span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: EXPERT_COLORS[k] }} />
              {PRETTY[k]} {k === highlight && "★"}
            </span>
            <span className="font-mono text-slate-300">{(v * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2.5 overflow-hidden rounded-full bg-ink-700">
            <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.max(v * 100, 0.5)}%`, background: EXPERT_COLORS[k], opacity: k === highlight ? 1 : 0.55 }} />
          </div>
        </div>
      ))}
    </div>
  );
}

export function Settings({ info }) {
  if (!info) return null;
  return (
    <div>
      <div className="label mb-2">Applied corruption settings</div>
      <pre className="max-h-40 overflow-auto rounded-xl border border-ink-700 bg-ink-950 p-3 font-mono text-[11px] text-slate-300">{JSON.stringify(info, null, 2)}</pre>
    </div>
  );
}

export function ErrorBox({ error }) {
  if (!error) return null;
  return <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">{error}</div>;
}

export function RunButton({ loading, disabled, onClick, children }) {
  return (
    <button className="btn-primary w-full" onClick={onClick} disabled={loading || disabled} type="button">
      {loading ? <Spinner /> : <PlayIcon />} {loading ? "Running inference…" : children}
    </button>
  );
}

export function dl(src, name) { download(src, name); }

export const Spinner = () => <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />;
const I = (d) => () => (
  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">{d}</svg>
);
export const UploadIcon = I(<><path d="M12 16V4" /><path d="m6 10 6-6 6 6" /><path d="M4 20h16" /></>);
export const DownloadIcon = I(<><path d="M12 4v12" /><path d="m6 10 6 6 6-6" /><path d="M4 20h16" /></>);
export const PlayIcon = I(<path d="M7 4v16l13-8z" />);
export const CamIcon = I(<><rect x="3" y="7" width="13" height="10" rx="2" /><path d="m16 11 5-3v8l-5-3" /></>);
