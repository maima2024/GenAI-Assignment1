import React, { useEffect, useState } from "react";
import { getJSON } from "./api";
import RestorationWorkspace from "./components/RestorationWorkspace";
import SketchWorkspace from "./components/SketchWorkspace";

const WORKSPACES = [
  { id: "universal", name: "Universal Restoration", task: "Task 1", desc: "One denoising autoencoder for every corruption." },
  { id: "hard", name: "Hard-Routed Restoration", task: "Task 2", desc: "Classifier selects a single specialist autoencoder." },
  { id: "soft", name: "Soft Mixture-of-Experts Restoration", task: "Task 3", desc: "Gate blends identity + three experts." },
  { id: "sketch", name: "Face-to-Sketch Generator", task: "Task 4", desc: "Style-conditioned pix2pix cGAN on FS2K." },
  { id: "system", name: "System Status", task: "Ops", desc: "Backend health, loaded ONNX models, tracking." },
];

export default function App() {
  const [active, setActive] = useState(() => window.location.hash.slice(1) || "universal");
  const [health, setHealth] = useState(null);

  useEffect(() => { window.location.hash = active; }, [active]);
  useEffect(() => {
    const f = () => setActive(window.location.hash.slice(1) || "universal");
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);
  useEffect(() => {
    const load = () => getJSON("/api/health").then(setHealth).catch(() => setHealth({ status: "offline" }));
    load();
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, []);

  const ws = WORKSPACES.find((w) => w.id === active) || WORKSPACES[0];
  const ok = health?.status === "ok";

  return (
    <div className="flex min-h-screen flex-col lg:flex-row">
      <aside className="border-b border-ink-800 bg-ink-900/60 lg:w-72 lg:shrink-0 lg:border-b-0 lg:border-r">
        <div className="flex items-center gap-3 px-6 py-6">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-brand-400 to-fuchsia-500 font-bold text-white">R</div>
          <div>
            <div className="font-semibold text-white">RestoreLab</div>
            <div className="text-xs text-slate-500">Generative AI Studio</div>
          </div>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3 lg:flex-col lg:pb-0">
          {WORKSPACES.map((w) => (
            <button key={w.id} onClick={() => setActive(w.id)}
              className={`group shrink-0 rounded-xl px-4 py-3 text-left transition ${active === w.id ? "bg-brand-500/15 ring-1 ring-brand-400/50" : "hover:bg-ink-800"}`}>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-brand-300">{w.task}</div>
              <div className={`text-sm font-medium ${active === w.id ? "text-white" : "text-slate-300"}`}>{w.name}</div>
            </button>
          ))}
        </nav>
        <div className="mx-6 mt-6 hidden rounded-xl border border-ink-700 bg-ink-800 p-4 text-xs lg:block">
          <div className="flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full ${ok ? "bg-emerald-400" : health?.status === "degraded" ? "bg-amber-400" : "bg-rose-500"}`} />
            <span className="font-medium text-slate-300">Backend {health?.status || "…"}</span>
          </div>
          {health?.models && (
            <div className="mt-1 text-slate-500">
              {Object.values(health.models).filter((m) => m.loaded).length}/{Object.keys(health.models).length} ONNX models loaded
            </div>
          )}
        </div>
      </aside>

      <main className="flex-1 px-4 py-6 sm:px-8 lg:py-10">
        <header className="mb-8">
          <div className="text-xs font-semibold uppercase tracking-wider text-brand-300">{ws.task}</div>
          <h1 className="mt-1 text-2xl font-bold text-white sm:text-3xl">{ws.name}</h1>
          <p className="mt-1 text-sm text-slate-400">{ws.desc}</p>
        </header>
        {active === "sketch" ? <SketchWorkspace /> : active === "system" ? <SystemStatus health={health} /> :
          <RestorationWorkspace key={active} mode={active} />}
        <footer className="mt-12 text-xs text-slate-600">Umaima Zeeshan · 23i-0790 · Generative AI Assignment 1 · React + Tailwind · FastAPI · ONNX Runtime</footer>
      </main>
    </div>
  );
}

function SystemStatus({ health }) {
  const mlflow = `${window.location.protocol}//${window.location.hostname}:5000`;
  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <div className="card"><div className="label">API status</div><div className="mt-2 text-xl font-semibold text-white">{health?.status || "…"}</div></div>
        <div className="card"><div className="label">ONNX Runtime</div><div className="mt-2 font-mono text-xl text-white">{health?.onnxruntime || "—"}</div><div className="text-xs text-slate-500">{health?.providers?.join(", ")}</div></div>
        <div className="card"><div className="label">Experiment tracking</div><a className="mt-2 inline-block text-brand-300 hover:underline" href={mlflow} target="_blank" rel="noreferrer">Open MLflow UI ↗</a><div className="text-xs text-slate-500">Optuna trials, losses, metrics, artifacts</div></div>
      </div>
      <div className="card overflow-x-auto">
        <div className="label mb-3">Loaded models</div>
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-slate-500"><tr><th className="py-2">Model</th><th>Status</th><th>Size</th><th>Inputs</th><th>Outputs</th></tr></thead>
          <tbody>
            {health?.models && Object.entries(health.models).map(([n, m]) => (
              <tr key={n} className="border-t border-ink-700">
                <td className="py-2 font-mono text-xs text-white">{n}.onnx</td>
                <td>{m.loaded ? <span className="text-emerald-400">loaded</span> : <span className="text-rose-400">missing</span>}</td>
                <td className="font-mono text-xs">{m.size_mb ? `${m.size_mb} MB` : "—"}</td>
                <td className="font-mono text-[11px] text-slate-400">{m.inputs?.join(", ")}</td>
                <td className="font-mono text-[11px] text-slate-400">{m.outputs?.join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
