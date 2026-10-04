"""Architecture diagrams for the report (matplotlib, no extra dependencies)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
C = {"enc": "#c7d2fe", "lat": "#fde68a", "dec": "#bbf7d0", "io": "#e5e7eb", "gate": "#fbcfe8", "exp": "#bae6fd",
     "d": "#fecaca"}


def box(ax, x, y, w, h, text, color, fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=color, ec="#334155", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#334155", **kw))


def fig(w, h):
    f, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 10)
    ax.set_ylim(0, h * 10)
    ax.axis("off")
    return f, ax


def ae():
    f, ax = fig(7.2, 2.2)
    stages = [("input\n3x128x128", C["io"], 12), ("conv 3x3\nC/2x128²", C["enc"], 11), ("down+res\nCx64²", C["enc"], 10),
              ("down+res\n2Cx32²", C["enc"], 9), ("down+res\n4Cx16²", C["enc"], 8), ("down+res\n8Cx8²", C["enc"], 7),
              ("1x1 conv\nlatent\nLx8x8", C["lat"], 6), ("up+res\n4Cx16²", C["dec"], 8), ("up+res\n2Cx32²", C["dec"], 9),
              ("up+res\nCx64²", C["dec"], 10), ("up+res\nC/2x128²", C["dec"], 11), ("conv+\nsigmoid\n3x128²", C["io"], 12)]
    x = 1
    for i, (t, c, h) in enumerate(stages):
        box(ax, x, 11 - h / 2, 5.4, h, t, c, 4.8)
        if i:
            arrow(ax, (x - 0.6, 11), (x, 11))
        x += 6
    ax.text(36, 2.2, "Encoder E: strided 4x4 conv + BN + LeakyReLU + residual block  |  Bottleneck: L x 8 x 8 code "
            "(no skip connections)  |  Decoder D: bilinear upsample + conv + dropout + residual", ha="center", fontsize=5.5)
    f.savefig(OUT / "arch_autoencoder.png", dpi=200, bbox_inches="tight")
    plt.close(f)


def hard():
    f, ax = fig(7.2, 2.4)
    box(ax, 1, 9, 9, 6, "corrupted\nimage x̃", C["io"])
    box(ax, 14, 9, 12, 6, "CNN classifier C\n4 conv stages\navg+max pool, FC", C["gate"])
    box(ax, 30, 9, 9, 6, "p = softmax\nr = argmax p", C["gate"])
    names = [("identity (bypass)", C["io"]), ("A_salt", C["exp"]), ("A_blur", C["exp"]), ("A_occlusion", C["exp"])]
    for i, (n, c) in enumerate(names):
        y = 19 - i * 5.3
        box(ax, 45, y, 12, 4, n, c)
        arrow(ax, (39, 12), (45, y + 2))
        arrow(ax, (57, y + 2), (62, 12))
    box(ax, 62, 9, 9, 6, "restored\nx̂", C["io"])
    arrow(ax, (10, 12), (14, 12))
    arrow(ax, (26, 12), (30, 12))
    ax.text(51, 0.6, "exactly ONE branch is executed", ha="center", fontsize=6, style="italic")
    f.savefig(OUT / "arch_hard_routing.png", dpi=200, bbox_inches="tight")
    plt.close(f)


def soft():
    f, ax = fig(7.2, 2.6)
    box(ax, 1, 10, 9, 6, "corrupted\nimage x̃", C["io"])
    box(ax, 14, 19, 14, 6, "gate G (init. from\nTask-2 classifier)", C["gate"])
    box(ax, 32, 19, 12, 6, "w = softmax(G/τ)", C["gate"])
    names = [("identity", C["io"]), ("A_salt", C["exp"]), ("A_blur", C["exp"]), ("A_occlusion", C["exp"])]
    for i, (n, c) in enumerate(names):
        y = 13 - i * 4
        box(ax, 20, y, 12, 3.2, n, c, 6)
        arrow(ax, (10, 13), (20, y + 1.6))
        arrow(ax, (32, y + 1.6), (50, 13))
    box(ax, 50, 10, 11, 6, "x̂ = Σ wₖ·branchₖ(x̃)", "#ddd6fe", 6.5)
    arrow(ax, (28, 22), (32, 22))
    arrow(ax, (44, 22), (55.5, 16))
    arrow(ax, (10, 14), (14, 22))
    box(ax, 64, 10, 7, 6, "x̂", C["io"])
    arrow(ax, (61, 13), (64, 13))
    ax.text(36, 0.3, "L = λ1·L1 + λs·(1-SSIM) + λc·CE(G(x̃), y) + λb·Σ(w̄ₖ - 1/4)²   |   warm-up: experts frozen  →  joint fine-tuning",
            ha="center", fontsize=6)
    f.savefig(OUT / "arch_soft_moe.png", dpi=200, bbox_inches="tight")
    plt.close(f)


def gan():
    f, ax = fig(8.0, 3.0)
    box(ax, 0.5, 17, 8, 5, "photo x\n3x128²", C["io"], 6)
    box(ax, 0.5, 9, 8, 5, "style s\nEmbedding(3,E)", C["lat"], 6)
    enc = [("down\n64²", 12), ("down\n32²", 10.5), ("down\n16²", 9), ("...\n2²", 7.5), ("1²", 6)]
    dec = [("up\n2²", 6), ("...\n16²", 7.5), ("up\n32²", 9), ("up\n64²", 10.5), ("tanh\n128²", 12)]
    x, cy, ex = 12, 17, []
    for t, h in enc:
        box(ax, x, cy - h / 2, 4, h, t, C["enc"], 5.5); ex.append(x); x += 5.2
    box(ax, x, cy - 2.5, 4.5, 5, "+style\nemb", C["lat"], 5.5); bx = x; x += 5.7
    dx = []
    for t, h in dec:
        box(ax, x, cy - h / 2, 4, h, t, C["dec"], 5.5); dx.append(x); x += 5.2
    for i in range(4):
        a, b = ex[i] + 2, dx[4 - i] + 2
        ax.annotate("", xy=(b, cy + 6.3 + i * 0.2), xytext=(a, cy + 6.3 + i * 0.2), arrowprops=dict(
            arrowstyle="-|>", color="#6366f1", lw=0.6, connectionstyle=f"arc3,rad=-{0.25 - i * 0.04}"))
    ax.text((ex[0] + dx[-1]) / 2 + 2, 29.2, "U-Net skip connections (concatenation)", fontsize=6, color="#6366f1", ha="center")
    box(ax, x + 0.5, cy - 2.5, 7, 5, "sketch ŷ\n1x128²", C["io"], 6)
    arrow(ax, (8.5, 19.5), (12, 18)); arrow(ax, (8.5, 11.5), (12, 15)); arrow(ax, (8.5, 11), (bx + 2, cy - 2.5))
    box(ax, 44, 1, 26, 6, "conditional PatchGAN D(x, y | G(x,s), s)\n4 conv blocks → 14x14 real/fake logits", C["d"], 6)
    arrow(ax, (x + 4, cy - 2.5), (60, 7))
    ax.text(20, 3.5, "L_G = BCE(D(x,G(x,s),s),1) + λ_L1·|y - G(x,s)|\nL_D = BCE(D(x,y,s),0.9) + BCE(D(x,G,s),0)",
            fontsize=6, ha="center")
    f.savefig(OUT / "arch_cgan.png", dpi=200, bbox_inches="tight")
    plt.close(f)


def system():
    f, ax = fig(7.2, 2.2)
    box(ax, 1, 7, 13, 8, "Browser\nReact + Tailwind\n(4 workspaces)", C["io"], 6.5)
    box(ax, 20, 7, 13, 8, "nginx container\nstatic build +\n/api reverse proxy", C["enc"], 6.5)
    box(ax, 39, 7, 14, 8, "FastAPI container\nvalidation, preprocessing,\ncorruption, routing", C["gate"], 6.5)
    box(ax, 59, 12, 12, 7, "ONNX Runtime\n7 .onnx models", C["exp"], 6.5)
    box(ax, 59, 2, 12, 7, "MLflow UI\ncontainer :5000", C["lat"], 6.5)
    arrow(ax, (14, 11), (20, 11))
    arrow(ax, (33, 11), (39, 11))
    arrow(ax, (53, 12), (59, 15))
    ax.text(36, 19.5, "docker compose up --build  →  http://localhost:3000", ha="center", fontsize=7, weight="bold")
    f.savefig(OUT / "arch_system.png", dpi=200, bbox_inches="tight")
    plt.close(f)


if __name__ == "__main__":
    ae(); hard(); soft(); gan(); system()
    print("diagrams written to", OUT)
