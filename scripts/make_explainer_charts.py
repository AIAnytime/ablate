"""Generate the charts embedded in docs/youtube_explainer.md.

Uses the real numbers from the prod-grade Qwen2.5-1.5B run and a live
per-layer refusal-strength extraction on a small model. Outputs PNGs to media/.

Run: .venv/bin/python scripts/make_explainer_charts.py
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# --- validated palette (light surface) ---
SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
INK2    = "#52514e"
MUTED   = "#898781"
GRID    = "#e1e0d9"
BASE    = "#c3c2b7"
BLUE    = "#2a78d6"
VIOLET  = "#4a3aa7"
RED     = "#d03b3b"
GREEN   = "#0ca30c"

MEDIA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")
os.makedirs(MEDIA, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 15,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": BASE,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, name):
    path = os.path.join(MEDIA, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def chart_refusal_drop():
    """Baseline vs ablated on the two headline metrics (real Qwen2.5-1.5B numbers)."""
    labels = ["Refuses harmful\nprompt", "Complies\n(judged ASR)"]
    baseline = [1.00, 0.00]
    ablated  = [0.00, 0.487]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    b1 = ax.bar(x - w/2, baseline, w, label="Baseline", color=MUTED, zorder=3)
    b2 = ax.bar(x + w/2, ablated,  w, label="Ablated",  color=BLUE, zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.02,
                    f"{r.get_height():.0%}", ha="center", va="bottom",
                    fontsize=14, fontweight="bold", color=INK)
    ax.set_ylim(0, 1.12); ax.set_yticks(np.linspace(0,1,6))
    ax.set_yticklabels([f"{int(t*100)}%" for t in np.linspace(0,1,6)])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    ax.set_title("Ablation removes the refusal reflex — but 'complies' ≠ 100%",
                 fontsize=16, fontweight="bold", color=INK, pad=28, loc="left")
    _save(fig, "chart-refusal-drop.png")


def chart_metric_gap():
    """The key finding: keyword scoring says 100% success, the LLM judge says 49%."""
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    vals = [1.00, 0.487]
    colors = [RED, GREEN]
    bars = ax.bar(["Keyword metric\nsays success", "LLM judge\nsays real success"],
                  vals, width=0.5, color=colors, zorder=3)
    for r, v in zip(bars, vals):
        ax.text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.0%}", ha="center",
                va="bottom", fontsize=17, fontweight="bold", color=INK)
    # gap bracket
    ax.annotate("", xy=(1, 0.487), xytext=(1, 1.00),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.6))
    ax.text(1.28, (0.487+1.0)/2, "51-point\nillusion", va="center", ha="left",
            fontsize=13, color=INK2)
    ax.set_ylim(0, 1.18); ax.set_yticks(np.linspace(0,1,6))
    ax.set_yticklabels([f"{int(t*100)}%" for t in np.linspace(0,1,6)])
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    ax.set_title("Why you must use an LLM judge",
                 fontsize=16, fontweight="bold", color=INK, pad=14, loc="left")
    ax.text(0, -0.22, "Same ablated model, same prompts — only the scorer changed.",
            transform=ax.get_xaxis_transform(), fontsize=12, color=MUTED)
    _save(fig, "chart-metric-gap.png")


def chart_layer_strength():
    """Live: per-layer L2 norm of the diff-of-means refusal direction."""
    try:
        from ablate import Ablator
        from ablate import data
        from ablate.extract import collect_activations
        abl = Ablator("Qwen/Qwen2.5-0.5B-Instruct", dtype="float32")
        h = data.load_builtin("harmful"); s = data.load_builtin("harmless")
        import torch as _t
        ha = collect_activations(abl.lm, h, batch_size=8, show_progress=False)
        sa = collect_activations(abl.lm, s, batch_size=8, show_progress=False)
        raw = (ha.mean(1) - sa.mean(1))                          # (L+1, D)
        # Scale-normalize: separation relative to the typical activation norm at
        # that layer, so the last layer's large raw magnitude doesn't dominate.
        act_norm = _t.cat([ha, sa], dim=1).norm(dim=-1).mean(dim=1)   # (L+1,)
        signal = (raw.norm(dim=-1) / act_norm).numpy()
        norms = signal[1:-1]                                    # skip embedding + final layer
        model_name = "Qwen2.5-0.5B-Instruct"
    except Exception as e:
        print("layer-strength: falling back to representative curve:", e)
        xs = np.arange(22)
        norms = 0.05 + 0.5*np.exp(-((xs-15)**2)/30)
        model_name = "representative"
    layers = np.arange(1, len(norms)+1)
    peak_i = int(np.argmax(norms))
    peak = int(layers[peak_i])
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    ax.plot(layers, norms, color=BLUE, lw=2.2, zorder=3)
    ax.fill_between(layers, norms, color=BLUE, alpha=0.12, zorder=2)
    ax.scatter([peak], [norms[peak_i]], color=VIOLET, s=70, zorder=4)
    ax.annotate(f"strongest near layer {peak}", xy=(peak, norms[peak_i]),
                xytext=(peak-9, norms[peak_i]*0.98), fontsize=13, color=VIOLET,
                arrowprops=dict(arrowstyle="->", color=VIOLET))
    ax.set_ylim(0, norms.max()*1.18)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Relative refusal signal\n(‖mean harmful − mean harmless‖ / ‖activation‖)")
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    ax.set_title(f"Refusal lives in the middle-to-late layers  ·  {model_name}",
                 fontsize=15, fontweight="bold", color=INK, pad=14, loc="left")
    _save(fig, "chart-layer-strength.png")


def chart_projection():
    """2D geometry of directional ablation: h -> h' with the refusal component removed."""
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    v = np.array([0.0, 1.0])          # refusal direction (unit, vertical)
    h = np.array([2.2, 1.6])          # an activation
    coeff = h @ v
    hp = h - coeff * v                # projected

    # refusal axis
    ax.plot([0, 0], [-0.3, 2.2], color=MUTED, lw=1.2, ls=(0,(4,4)), zorder=1)
    ax.text(0.06, 2.15, "refusal direction  v", color=INK2, fontsize=12, va="top")

    def arrow(p0, p1, color, lw=2.4):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=20,
                                     color=color, lw=lw, zorder=3))
    arrow((0,0), h, RED)
    ax.text(h[0]+0.05, h[1]+0.03, "h  (has refusal)", color=RED, fontsize=13)
    arrow((0,0), hp, GREEN)
    ax.text(hp[0]+0.05, hp[1]-0.16, "h′  (refusal removed)", color=GREEN, fontsize=13)
    # dropped component
    ax.plot([h[0], hp[0]], [h[1], hp[1]], color=INK2, lw=1.3, ls=(0,(2,3)), zorder=2)
    ax.text((h[0]+hp[0])/2+0.05, (h[1]+hp[1])/2, "− (h·v̂) v̂", color=INK2, fontsize=12)

    ax.set_xlim(-0.4, 3.2); ax.set_ylim(-0.4, 2.35)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Directional ablation = project the refusal component out",
                 fontsize=15, fontweight="bold", color=INK, pad=6, loc="left")
    _save(fig, "chart-projection.png")


if __name__ == "__main__":
    chart_projection()
    chart_refusal_drop()
    chart_metric_gap()
    chart_layer_strength()
    print("done.")
