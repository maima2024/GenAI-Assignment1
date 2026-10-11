"""Merge the Kaggle/Colab result zips into the repository.

  python scripts/collect_results.py outputs_A.zip outputs_B.zip

* extracts both into ./outputs
* copies ONNX models to ./models and builds models/onnx_models.zip (for the GitHub release)
* copies MLflow runs to ./experiment-tracking/mlruns and rewrites artifact paths to /mlruns
* copies figures to ./report/figures and writes LaTeX tables to ./report/generated
"""
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "report" / "figures"
GEN = ROOT / "report" / "generated"


def lj(p):
    p = OUT / p
    return json.loads(p.read_text()) if p.exists() else None


def f2(v, d=2):
    if v is None:
        return "--"
    if v >= 99.99 and d <= 2:
        return r"$\infty$"
    return f"{v:.{d}f}"


def extract(zips):
    OUT.mkdir(exist_ok=True)
    for z in zips:
        print("extracting", z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(OUT)


def models():
    md = ROOT / "models"
    md.mkdir(exist_ok=True)
    onnx = sorted((OUT / "onnx").glob("*.onnx"))
    for f in onnx:
        shutil.copy2(f, md / f.name)
    with zipfile.ZipFile(md / "onnx_models.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in onnx:
            z.write(f, f.name)
    print("models:", [f.name for f in onnx])


def mlruns():
    src, dst = OUT / "mlruns", ROOT / "experiment-tracking" / "mlruns"
    dst.mkdir(parents=True, exist_ok=True)
    for exp in src.iterdir():
        if exp.is_dir() and exp.name not in ("0", ".trash"):
            shutil.copytree(exp, dst / exp.name, dirs_exist_ok=True)
    pat = re.compile(r"file://[^\s'\"]*?/mlruns")
    for meta in dst.rglob("meta.yaml"):
        t = meta.read_text()
        meta.write_text(pat.sub("file:///mlruns", t))
    print("mlruns experiments:", [p.name for p in dst.iterdir()])


def figures():
    FIG.mkdir(parents=True, exist_ok=True)
    for t in ["task1", "task2", "task3", "task4"]:
        for f in (OUT / t).glob("*.png"):
            shutil.copy2(f, FIG / f"{t}_{f.name}")
    s = sorted((OUT / "task4" / "samples").glob("*.png"))
    for f in s:
        shutil.copy2(f, FIG / f"task4_{f.name}")
    if s:
        shutil.copy2(s[len(s) // 2], FIG / "task4_samples_mid.png")
        shutil.copy2(s[-1], FIG / "task4_samples_last.png")


def write(name, body):
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / name).write_text(body)


ROWS = [("clean/none", "Clean", "--")] + [(f"{k}/{s}", n, s) for k, n in
                                          [("saltpepper", "S\\&P"), ("blur", "Blur"), ("occlusion", "Occl.")]
                                          for s in ["low", "medium", "high"]]


def restoration_table(summary, label, caption):
    b = summary["by_kind_severity"]
    lines = [r"\begin{table}[t]\centering\caption{" + caption + r"}\label{" + label + "}",
             r"\setlength{\tabcolsep}{4pt}\begin{tabular}{llcccc}\toprule",
             r"Cond. & Sev. & In PSNR & Out PSNR & In SSIM & Out SSIM\\\midrule"]
    for key, name, sev in ROWS:
        if key in b:
            r = b[key]
            lines.append(f"{name} & {sev} & {f2(r['in_psnr'])} & \\textbf{{{f2(r['psnr'])}}} & {f2(r['in_ssim'], 3)} & \\textbf{{{f2(r['ssim'], 3)}}}\\\\")
    o = summary["overall"]
    lines.append(r"\midrule All & & -- & " + f"{f2(o['psnr'])} & -- & {f2(o['ssim'], 3)}" + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(lines)


def optuna_table():
    studies = [("T1 UDAE", "task1/optuna_summary.json"), ("T2 classifier", "task2/clf_optuna_summary.json"),
               ("T2 specialists", "task2/spec_optuna_summary.json"), ("T3 soft MoE", "task3/optuna_summary.json"),
               ("T4 cGAN", "task4/optuna_summary.json")]
    lines = [r"\begin{table*}[t]\centering\caption{Optuna studies (TPE sampler, median pruner). Objective is minimised: "
             r"val.~L1\,+\,(1$-$SSIM) for reconstruction tasks, 1$-$macro-F1 for the classifier.}\label{tab:optuna}",
             r"\small\begin{tabular}{lcccp{10.2cm}}\toprule",
             r"Study & Trials & Complete/Pruned & Best obj. & Best configuration\\\midrule"]
    for name, f in studies:
        s = lj(f)
        if not s:
            continue
        st = s.get("states", {})
        bp = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in (s["best_params"] or {}).items())
        bp = bp.replace("_", r"\_")
        lines.append(f"{name} & {s['n_trials']} & {st.get('COMPLETE', 0)}/{st.get('PRUNED', 0)} & {f2(s['best_value'], 4)} & {bp}\\\\")
    lines += [r"\bottomrule\end{tabular}\end{table*}"]
    return "\n".join(lines)


def classifier_table(c):
    t = c["test"]
    lines = [r"\begin{table}[t]\centering\caption{Corruption classifier on the deterministic test manifest "
             f"(accuracy {100 * t['accuracy']:.2f}\\%, macro-F1 {t['macro_f1']:.4f}).}}\\label{{tab:clf}}",
             r"\begin{tabular}{lcccc}\toprule Class & Precision & Recall & F1 & Support\\\midrule"]
    for k, n in [("clean", "Clean"), ("saltpepper", "Salt \\& pepper"), ("blur", "Blur"), ("occlusion", "Occlusion")]:
        r = t["per_class"][k]
        lines.append(f"{n} & {r['precision']:.4f} & {r['recall']:.4f} & {r['f1-score']:.4f} & {int(r['support'])}\\\\")
    lines.append(r"\midrule Macro avg & " + f"{t['macro_precision']:.4f} & {t['macro_recall']:.4f} & {t['macro_f1']:.4f} & --\\\\")
    v = c["val"]
    lines.append(r"Validation & " + f"{v['macro_precision']:.4f} & {v['macro_recall']:.4f} & {v['macro_f1']:.4f} & acc {100 * v['accuracy']:.2f}\\%\\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    acc = t.get("accuracy_by_kind_severity", {})
    lines2 = [r"\begin{table}[t]\centering\caption{Classifier accuracy per corruption and severity (test).}\label{tab:clfsev}",
              r"\begin{tabular}{lccc}\toprule & Low & Medium & High\\\midrule",
              f"Clean & \\multicolumn{{3}}{{c}}{{{100 * acc.get('clean/none', 0):.2f}\\%}}\\\\"]
    for k, n in [("saltpepper", "Salt \\& pepper"), ("blur", "Blur"), ("occlusion", "Occlusion")]:
        lines2.append(n + " & " + " & ".join(f"{100 * acc.get(f'{k}/{s}', 0):.2f}\\%" for s in ["low", "medium", "high"]) + r"\\")
    lines2 += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(lines) + "\n\n" + "\n".join(lines2)


def routing_table(r):
    o, p = r["oracle"]["by_kind_severity"], r["predicted"]["by_kind_severity"]
    lines = [r"\begin{table}[t]\centering\caption{Hard routing: oracle vs.\ predicted routing (test PSNR/SSIM). "
             f"{r['n_misrouted']} of {r['n_items']} test inputs were misrouted.}}\\label{{tab:routing}}",
             r"\setlength{\tabcolsep}{3.5pt}\begin{tabular}{llcccc}\toprule",
             r"& & \multicolumn{2}{c}{Oracle} & \multicolumn{2}{c}{Predicted}\\ Cond. & Sev. & PSNR & SSIM & PSNR & SSIM\\\midrule"]
    for key, name, sev in ROWS:
        if key in o:
            lines.append(f"{name} & {sev} & {f2(o[key]['psnr'])} & {f2(o[key]['ssim'], 3)} & {f2(p[key]['psnr'])} & {f2(p[key]['ssim'], 3)}\\\\")
    oo, pp = r["oracle"]["overall"], r["predicted"]["overall"]
    lines.append(r"\midrule All & & " + f"{f2(oo['psnr'])} & {f2(oo['ssim'], 3)} & {f2(pp['psnr'])} & {f2(pp['ssim'], 3)}\\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    mis = r.get("misrouted_by_true_pred", {})
    if mis:
        lines += [r"\begin{table}[t]\centering\caption{Misrouted test inputs (true$\to$predicted) and the PSNR lost relative to oracle routing.}\label{tab:misroute}",
                  r"\begin{tabular}{lcc}\toprule Error & Count & Mean PSNR drop (dB)\\\midrule"]
        for k, v in sorted(mis.items(), key=lambda kv: -kv[1]["count"]):
            lines.append(f"{k.replace('saltpepper', 'S&P').replace('&', chr(92) + '&').replace('->', '$\\rightarrow$')} & {v['count']} & {v['mean_psnr_drop']:.2f}\\\\")
        lines += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(lines)


def cavg(b, pre=""):
    ks = ["saltpepper", "blur", "occlusion"]
    p = sum(b[k][pre + "psnr"] for k in ks) / 3
    q = sum(b[k][pre + "ssim"] for k in ks) / 3
    return f"{p:.2f} / {q:.3f}"


def sev(s, pre=""):
    b = s["by_kind_severity"]
    return " / ".join(f"{b[f'all/{x}'][pre + 'psnr']:.2f}" for x in ["low", "medium", "high"])


def comparison_table():
    t1 = lj("task1/test_results.json")
    r2 = lj("task2/routing_results.json")
    t3 = lj("task3/test_results.json")
    rows = []
    if t1: rows.append(("T1 universal DAE", t1))
    if r2: rows += [("T2 hard (oracle)", r2["oracle"]), ("T2 hard (predicted)", r2["predicted"])]
    if t3: rows.append(("T3 soft MoE", t3["summary"]))
    if not rows:
        return ""
    lines = [r"\begin{table*}[t]\centering\caption{Cross-task comparison on the identical deterministic test manifest (PSNR dB / SSIM). PSNR of an output identical to the target is capped at 100 dB (shown as $\infty$); averages that include such values are therefore not meaningful, so the corrupted-input average excludes the clean condition.}\label{tab:compare}",
             r"\begin{tabular}{lcccccc}\toprule System & Clean & Salt \& pepper & Blur & Occlusion & Corrupted avg. & Low / Med / High PSNR\\\midrule"]
    base = rows[0][1]["by_kind"]
    lines.append("Corrupted input & " + " & ".join(
        (f"{f2(base[k]['in_psnr'])} / {f2(base[k]['in_ssim'], 3)}") for k in ["clean", "saltpepper", "blur", "occlusion"]) + " & " + cavg(base, "in_") + " & " + sev(rows[0][1], "in_") + r" \\")
    for n, s in rows:
        b = s["by_kind"]
        lines.append(n + " & " + " & ".join(f"{f2(b[k]['psnr'])} / {f2(b[k]['ssim'], 3)}" for k in ["clean", "saltpepper", "blur", "occlusion"])
                     + " & " + cavg(b) + " & " + sev(s) + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table*}"]
    return "\n".join(lines)


def moe_tables(t3):
    u = t3["usage"]
    lines = [r"\begin{table}[t]\centering\caption{Gate behaviour on the test manifest.}\label{tab:usage}",
             r"\begin{tabular}{lcccc}\toprule & Identity & S\&P & Blur & Occl.\\\midrule"]
    order = ["identity", "saltpepper", "blur", "occlusion"]
    lines.append("Mean weight & " + " & ".join(f"{u['mean_weight'][k]:.3f}" for k in order) + r"\\")
    lines.append("Argmax share & " + " & ".join(f"{u['argmax_share'][k]:.3f}" for k in order) + r"\\")
    lines.append("Off-target weight & " + " & ".join(f"{u['off_target_weight'][k]:.3f}" for k in order) + r"\\")
    lines += [r"\bottomrule\end{tabular}", r"\par\vspace{2pt}{\footnotesize Mean entropy " + f"{u['mean_entropy']:.3f} (max {u['max_entropy']:.3f}); "
              f"{100 * u['frac_dominated_gt_0.9']:.1f}\\% of inputs have $\\max_k w_k>0.9$, {100 * u['frac_distributed_max_lt_0.6']:.1f}\\% have $\\max_k w_k<0.6$.}}",
              r"\end{table}"]
    return "\n".join(lines)


def gan_table(t4, split):
    lines = [r"\begin{table}[t]\centering\caption{Face-to-sketch generator on the official FS2K test set.}\label{tab:gan}",
             r"\begin{tabular}{lcccc}\toprule Style & $n$ & L1 $\downarrow$ & SSIM $\uparrow$ & PSNR $\uparrow$\\\midrule"]
    for k, v in t4["by_style"].items():
        lines.append(f"{k} & {v['n']} & {v['l1']:.4f} & {v['ssim']:.4f} & {v['psnr']:.2f}\\\\")
    o = t4["overall"]
    lines.append(r"\midrule All & " + f"{sum(v['n'] for v in t4['by_style'].values())} & {o['l1']:.4f} & {o['ssim']:.4f} & {o['psnr']:.2f}\\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(lines)


def onnx_table():
    lines = [r"\begin{table}[t]\centering\caption{ONNX export check: maximum absolute difference between PyTorch and ONNX Runtime outputs on a random input.}\label{tab:onnx}",
             r"\setlength{\tabcolsep}{3pt}\begin{tabular}{lcc}\toprule Model & Size (MB) & max $|\Delta|$\\\midrule"]
    for f in ["task1/onnx_check.json", "task2/onnx_check.json", "task3/onnx_check.json", "task4/onnx_check.json"]:
        j = lj(f)
        if not j:
            continue
        items = j.values() if "file" not in j else [j]
        for it in items:
            lines.append(f"{it['file'].replace('_', chr(92) + '_')} & {it['size_mb']:.1f} & {max(it['max_abs_diff'].values()):.1e}\\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(lines)


def tables():
    t1, c2, r2, t3, t4 = (lj("task1/test_results.json"), lj("task2/classifier_results.json"),
                          lj("task2/routing_results.json"), lj("task3/test_results.json"), lj("task4/test_results.json"))
    write("optuna.tex", optuna_table())
    write("onnx.tex", onnx_table())
    write("compare.tex", comparison_table())
    if t1: write("t1_results.tex", restoration_table(t1, "tab:t1", "Task 1 universal DAE: test results per corruption and severity."))
    if c2: write("t2_classifier.tex", classifier_table(c2))
    if r2: write("t2_routing.tex", routing_table(r2))
    if t3:
        write("t3_results.tex", restoration_table(t3["summary"], "tab:t3", "Task 3 soft MoE: test results per corruption and severity."))
        write("t3_usage.tex", moe_tables(t3))
    if t4: write("t4_results.tex", gan_table(t4, lj("task4/data_split.json")))
    print("tables written to", GEN)


if __name__ == "__main__":
    zips = sys.argv[1:]
    if zips:
        extract(zips)
    models()
    mlruns()
    figures()
    tables()
