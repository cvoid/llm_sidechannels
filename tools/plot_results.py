"""Generate publication figures from experiment 1 results.

Produces figures saved to analysis/:
  fig1_tpq_classifiers.png   -- accuracy vs TPQ, three classifiers at temp=0.3
  fig2_tpq_temperatures.png  -- accuracy vs TPQ, RF across four temperatures
  fig3_defense_comparison.png -- defense accuracy and bandwidth overhead
  fig8_confusion_matrix.png  -- LightGBM confusion matrix at tpq=25, temp=0.3
                                (requires --manifest and --window-ms)

Example:
    uv run python tools/plot_results.py
    uv run python tools/plot_results.py \\
        --manifest data/raw_clean/manifest_all.jsonl --window-ms 3.5
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# --- style -----------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
})

CHANCE = 1 / 50  # 50-class classification


# ---------------------------------------------------------------------------
# Figure 1: accuracy vs TPQ -- three classifiers at temp=0.3
# ---------------------------------------------------------------------------

def fig1_tpq_classifiers(out: Path) -> None:
    rf   = pd.read_csv("analysis/exp1_tpq_sweep_clean.csv")
    lgbm = pd.read_csv("analysis/exp1_tpq_sweep_clean_lgbm.csv")
    lstm = pd.read_csv("analysis/exp1_tpq_sweep_clean_bilstm.csv")

    rf03   = rf[rf["temperature"]   == 0.3].sort_values("tpq")
    lgbm03 = lgbm[lgbm["temperature"] == 0.3].sort_values("tpq")
    lstm03 = lstm[lstm["temperature"] == 0.3].sort_values("tpq")

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    ax.plot(rf03["tpq"],   rf03["accuracy"],   marker="o", label="Random Forest")
    ax.plot(lgbm03["tpq"], lgbm03["accuracy"], marker="s", label="LightGBM")
    ax.plot(lstm03["tpq"], lstm03["accuracy"], marker="^", label="BiLSTM")
    ax.axhline(CHANCE, color="#aaaaaa", linewidth=0.8, linestyle="--", label="Chance (2%)")

    ax.set_xlabel("Traces per query (TPQ)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Query fingerprinting accuracy vs TPQ\n(50 classes, temp=0.3)")
    ax.set_xticks(rf03["tpq"].tolist())
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 2: accuracy vs TPQ -- RF across temperatures
# ---------------------------------------------------------------------------

def fig2_tpq_temperatures(out: Path) -> None:
    rf = pd.read_csv("analysis/exp1_tpq_sweep_clean.csv")

    temps = sorted(rf["temperature"].unique())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(temps)))  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for temp, color in zip(temps, colors):
        sub = rf[rf["temperature"] == temp].sort_values("tpq")
        ax.plot(sub["tpq"], sub["accuracy"], marker="o", color=color,
                label=f"temp={temp}")

    ax.axhline(CHANCE, color="#aaaaaa", linewidth=0.8, linestyle="--", label="Chance (2%)")

    ax.set_xlabel("Traces per query (TPQ)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Random Forest accuracy vs TPQ\nacross sampling temperatures")
    tpqs = sorted(rf["tpq"].unique())
    ax.set_xticks(tpqs)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 3: defense comparison
# ---------------------------------------------------------------------------

_DEFENSE_GROUPS = {
    "undefended":   "baseline",
    "agg batch=2":  "aggregation",
    "agg batch=4":  "aggregation",
    "agg batch=8":  "aggregation",
    "pad rand=128": "padding",
    "pad rand=256": "padding",
    "pad rand=512": "padding",
    "pad fixed=1500": "padding",
    "pad fixed=2048": "padding",
    "cbr burst":    "CBR",
    "cbr 512/20ms": "CBR",
}

_GROUP_COLORS = {
    "baseline":    "#555555",
    "aggregation": "#4878cf",
    "padding":     "#e87d22",
    "CBR":         "#2ca02c",
}


def fig3_defense_comparison(out: Path) -> None:
    df = pd.read_csv("analysis/defense_comparison.csv")

    labels   = df["defense"].tolist()
    accuracy = df["accuracy"].tolist()
    overhead = df["overhead_x"].tolist()

    colors = [_GROUP_COLORS[_DEFENSE_GROUPS.get(l, "baseline")] for l in labels]
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    bars = ax1.bar(x, accuracy, color=colors, alpha=0.85, zorder=3)
    ax2.plot(x, overhead, color="#d62728", marker="D", markersize=5,
             linewidth=1.2, zorder=4, label="Overhead (right axis)")

    ax1.axhline(CHANCE, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Top-1 accuracy")
    ax1.set_ylim(0, 1.1)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax1.set_title("Defense evaluation: accuracy and bandwidth overhead\n(50 classes, temp=0.3, TPQ=30, RF classifier)")

    ax2.set_ylabel("Bandwidth overhead (x baseline)", color="#d62728")
    ax2.tick_params(axis="y", colors="#d62728")
    ax2.set_ylim(0, 1.6)
    ax2.grid(False)

    # legend patches
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.artist import Artist
    legend_handles: list[Artist] = [Patch(facecolor=c, alpha=0.85, label=g)
                                    for g, c in _GROUP_COLORS.items()]
    legend_handles.append(
        Line2D([0], [0], color="#d62728", marker="D", markersize=5,
               linewidth=1.2, label="Bandwidth overhead")
    )
    ax1.legend(handles=legend_handles, loc="upper right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 8: LightGBM confusion matrix at tpq=25, temp=0.3
# ---------------------------------------------------------------------------

_LABEL_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"What to expect if I have (.+?) \(Outlook/Prognosis\)\?"), 1),
    (re.compile(r"What are the (?:symptoms|causes) of (.+?)\?"), 1),
    (re.compile(r"What causes (.+?)\?"), 1),
    (re.compile(r"Who is at highest risk for (.+?)\?"), 1),
    (re.compile(r"When to seek urgent medical care when I have (.+?)\?"), 1),
    (re.compile(r"How many children have (.+?)\?"), 1),
]


def _condition_name(text: str) -> str:
    for pat, grp in _LABEL_PATTERNS:
        m = pat.match(text)
        if m:
            return m.group(grp)
    return text[:25]


def fig8_confusion_matrix(manifest_path: Path, window_ms: float, out: Path) -> None:
    from features.build import build_dataset
    from attack.dataset import split
    from attack.train import fit_lgbm
    from sklearn.metrics import confusion_matrix

    prompts: dict[int, str] = {}
    prompts_file = Path("collect/data/exp1_prompts.jsonl")
    with open(prompts_file) as f:
        for line in f:
            e: dict[str, object] = json.loads(line)
            prompts[int(str(e["id"]))] = str(e["text"])

    X, y = build_dataset(manifest_path, trace_length=100, window_ms=window_ms,
                         server_port=8443, temperature=0.3)
    X_train, X_test, y_train, y_test = split(X, y, train_tpq=25)
    clf = fit_lgbm(X_train, y_train)
    y_pred = clf.predict(X_test)

    classes = sorted(int(c) for c in np.unique(y).tolist())
    labels = [_condition_name(prompts[c]) for c in classes]
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(14, 12))
    # Show only cells with non-zero values; diagonal in grey, errors in red.
    diag_mask = np.eye(len(classes), dtype=bool)
    off_diag = cm.copy()
    off_diag[diag_mask] = 0

    # Background: diagonal (correct) in light blue, zeros white
    bg = np.zeros_like(cm, dtype=float)
    bg[diag_mask] = cm[diag_mask].astype(float) / max(cm[diag_mask].max(), 1)
    ax.imshow(bg, cmap="Blues", vmin=0, vmax=1, aspect="equal", interpolation="nearest")

    # Overlay errors in red
    if off_diag.max() > 0:
        err_img = np.ma.masked_where(off_diag == 0, off_diag.astype(float))
        ax.imshow(err_img, cmap="Reds", vmin=0, vmax=off_diag.max(),
                  aspect="equal", interpolation="nearest")

    # Annotate non-zero cells
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            if val > 0:
                color = "white" if val > cm[diag_mask].max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=6.5, color=color, fontweight="bold" if i != j else "normal")

    n = len(classes)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    total = int(cm.sum())
    errors = int(off_diag.sum())
    acc = (total - errors) / total
    ax.set_title(
        f"LightGBM confusion matrix (temp=0.3, tpq=25)\n"
        f"{total} test samples, {errors} errors, {acc:.1%} accuracy",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved -> {out}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--manifest", type=Path, default=None,
                        help="manifest.jsonl for fig8 confusion matrix")
    parser.add_argument("--window-ms", type=float, default=None,
                        help="iteration window for fig8 confusion matrix")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig1_tpq_classifiers(args.out_dir / "fig1_tpq_classifiers.png")
    fig2_tpq_temperatures(args.out_dir / "fig2_tpq_temperatures.png")
    fig3_defense_comparison(args.out_dir / "fig3_defense_comparison.png")

    if args.manifest and args.window_ms:
        fig8_confusion_matrix(args.manifest, args.window_ms,
                              args.out_dir / "fig8_confusion_matrix.png")


if __name__ == "__main__":
    main()
