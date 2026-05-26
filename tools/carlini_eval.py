"""Reproduce Carlini & Nasr (2410.17175) timing-based binary disambiguation.

For every pair of prompts, trains one GMM on timing features (inter-packet
gaps) for each hypothesis and evaluates binary classification via log-
likelihood ratio. Reports AUPRC for all pairs and precision-recall curves
for a representative sample.

Features: first 100 inter-packet gaps (server->client, ms) per trace.
Training: first train_n traces per prompt. Testing: remaining traces.
GMM: diagonal covariance, n_components components per hypothesis.

Example:
    uv run python tools/carlini_eval.py --window-ms 3.5
"""
from __future__ import annotations

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_recall_curve

from attack.gmm import GMMBinaryClassifier
from features.timing import extract_gaps

# Suppress GMM convergence warnings for small training sets.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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


def load_features(
    manifest_path: Path,
    server_port: int,
    n_gaps: int,
    temperature: float,
) -> dict[int, list[np.ndarray]]:
    """Return {prompt_id: [gap_array, ...]} for all traces that yield n_gaps."""
    data: dict[int, list[np.ndarray]] = {}
    with open(manifest_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if float(str(entry["temperature"])) != temperature:
                continue
            pid = int(entry["prompt_id"])
            gaps = extract_gaps(
                Path(str(entry["pcap"])), server_port=server_port, n_gaps=n_gaps
            )
            if gaps is None:
                continue
            data.setdefault(pid, []).append(gaps)
    return data


def evaluate_all_pairs(
    features: dict[int, list[np.ndarray]],
    train_n: int,
    n_components: int,
) -> pd.DataFrame:
    """Evaluate GMM binary classifier on every prompt pair."""
    prompt_ids = sorted(features.keys())
    rows = []
    for pid_a, pid_b in combinations(prompt_ids, 2):
        traces_a = features[pid_a]
        traces_b = features[pid_b]
        if len(traces_a) <= train_n or len(traces_b) <= train_n:
            continue

        X_train_a = np.stack(traces_a[:train_n])
        X_train_b = np.stack(traces_b[:train_n])
        X_test_a  = np.stack(traces_a[train_n:])
        X_test_b  = np.stack(traces_b[train_n:])

        X_test = np.vstack([X_test_a, X_test_b])
        y_test = np.array([0] * len(X_test_a) + [1] * len(X_test_b))

        clf = GMMBinaryClassifier(n_components=n_components)
        clf.fit(X_train_a, X_train_b)
        auprc = clf.auprc(X_test, y_test)
        rows.append({"prompt_a": pid_a, "prompt_b": pid_b, "auprc": round(auprc, 4)})

    return pd.DataFrame(rows)


def plot_pr_curves(
    features: dict[int, list[np.ndarray]],
    train_n: int,
    n_components: int,
    auprc_df: pd.DataFrame,
    out: Path,
) -> None:
    """Plot PR curves for worst, median, and best pairs."""
    df = auprc_df.sort_values("auprc").reset_index(drop=True)
    n = len(df)
    indices = {
        "worst":  df.iloc[0],
        "median": df.iloc[n // 2],
        "best":   df.iloc[-1],
    }

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    for ax, (label, row) in zip(axes, indices.items()):
        pid_a, pid_b = int(row["prompt_a"]), int(row["prompt_b"])
        traces_a = features[pid_a]
        traces_b = features[pid_b]

        X_train_a = np.stack(traces_a[:train_n])
        X_train_b = np.stack(traces_b[:train_n])
        X_test_a  = np.stack(traces_a[train_n:])
        X_test_b  = np.stack(traces_b[train_n:])

        X_test = np.vstack([X_test_a, X_test_b])
        y_test = np.array([0] * len(X_test_a) + [1] * len(X_test_b))

        clf = GMMBinaryClassifier(n_components=n_components)
        clf.fit(X_train_a, X_train_b)
        scores = clf.log_ratio(X_test)

        prec, rec, _ = precision_recall_curve(y_test, scores)
        chance = 0.5

        ax.plot(rec, prec, linewidth=1.8)
        ax.axhline(chance, color="#aaaaaa", linewidth=0.8, linestyle="--",
                   label="Chance (50%)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{label.capitalize()} pair\nprompts {pid_a} vs {pid_b}"
                     f"\nAUPRC={row['auprc']:.3f}")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.suptitle(
        "Timing-based binary disambiguation (Carlini & Nasr)\n"
        "Precision-recall curves: worst / median / best prompt pair",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"saved -> {out}")


def plot_auprc_distribution(auprc_df: pd.DataFrame, out: Path) -> None:
    """Histogram of AUPRC values across all prompt pairs."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.hist(auprc_df["auprc"], bins=20, color="#4878cf", alpha=0.85, edgecolor="white")
    ax.axvline(0.5, color="#aaaaaa", linewidth=0.8, linestyle="--", label="Chance (0.50)")
    ax.axvline(auprc_df["auprc"].median(), color="#d62728", linewidth=1.2,
               linestyle="-", label=f"Median ({auprc_df['auprc'].median():.3f})")
    ax.set_xlabel("AUPRC")
    ax.set_ylabel("Number of prompt pairs")
    ax.set_title(
        f"Binary disambiguation AUPRC distribution\n"
        f"({len(auprc_df)} pairs, {int(auprc_df['auprc'].ge(0.9).sum())} pairs ≥0.90)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"saved -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/raw_clean/temp_0.3/manifest.jsonl"))
    parser.add_argument("--server-port", type=int, default=8443)
    parser.add_argument("--n-gaps", type=int, default=50,
                        help="inter-packet gaps per trace (default: 50)")
    parser.add_argument("--train-n", type=int, default=20,
                        help="training traces per prompt (default: 20)")
    parser.add_argument("--n-components", type=int, default=4,
                        help="GMM mixture components per hypothesis (default: 4)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    print(f"loading timing features from {args.manifest}...")
    features = load_features(
        args.manifest, args.server_port, args.n_gaps, args.temperature
    )
    n_prompts = len(features)
    n_traces  = {pid: len(v) for pid, v in features.items()}
    usable    = sum(1 for c in n_traces.values() if c > args.train_n)
    print(f"  {n_prompts} prompts, {usable} with >{args.train_n} traces")
    print(f"  traces per prompt: min={min(n_traces.values())} "
          f"max={max(n_traces.values())}")

    print(f"evaluating all pairs (train_n={args.train_n}, "
          f"n_components={args.n_components})...")
    df = evaluate_all_pairs(features, args.train_n, args.n_components)
    print(f"  {len(df)} pairs evaluated")
    print(f"  AUPRC: median={df['auprc'].median():.3f}  "
          f"mean={df['auprc'].mean():.3f}  "
          f"min={df['auprc'].min():.3f}  "
          f"max={df['auprc'].max():.3f}")
    print(f"  pairs >= 0.90 AUPRC: {(df['auprc'] >= 0.90).sum()} / {len(df)}")
    print(f"  pairs >= 0.75 AUPRC: {(df['auprc'] >= 0.75).sum()} / {len(df)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "exp2_carlini_auprc.csv"
    df.to_csv(csv_path, index=False)
    print(f"saved -> {csv_path}")

    print("generating figures...")
    plot_pr_curves(features, args.train_n, args.n_components, df,
                   args.out_dir / "fig4_carlini_pr_curves.png")
    plot_auprc_distribution(df, args.out_dir / "fig5_carlini_auprc_dist.png")


if __name__ == "__main__":
    main()
