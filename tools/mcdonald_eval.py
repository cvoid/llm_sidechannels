"""Reproduce McDonald & Bar Or (2511.03675) topic-inference attack.

Trains a LightGBM binary classifier on (packet_size, inter_arrival_time) pair
features extracted from pcaps. Evaluates whether a single captured trace can
be classified as belonging to a target topic vs. a negative control set.

Target:   data/raw_target/temp_0.3/  (50 Python programming questions)
Negative: data/raw_clean/temp_0.3/   (50 MedAlpaca medical questions)

Features: first n_pairs (size_bytes, gap_ms) pairs flattened to a 2*n_pairs
vector, matching McDonald & Bar Or Section 3 feature representation.

Evaluation: AUPRC at the natural 1:1 class balance and at simulated
imbalance ratios (sub-sampling negatives), following Section 4 of the paper.

Example:
    uv run python tools/mcdonald_eval.py
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

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

from features.mcdonald import extract_size_timing_pairs


def load_dataset(
    manifest_path: Path,
    server_port: int,
    n_pairs: int,
    label: int,
    temperature: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays for one manifest."""
    rows: list[np.ndarray] = []
    with open(manifest_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if float(str(entry["temperature"])) != temperature:
                continue
            vec = extract_size_timing_pairs(
                Path(str(entry["pcap"])), server_port=server_port, n_pairs=n_pairs
            )
            if vec is not None:
                rows.append(vec)
    X = np.stack(rows)
    y = np.full(len(rows), label, dtype=np.int32)
    return X, y


def evaluate_at_imbalance(
    scores_target: np.ndarray,
    scores_neg_pool: np.ndarray,
    ratio: int,
    rng: np.random.Generator,
) -> float:
    """AUPRC when negatives are sub-sampled to ratio:1 vs. len(scores_target).

    scores_neg_pool includes both train and test negatives to allow higher
    ratios than the test split alone provides. This is slightly optimistic
    (the classifier has seen the train negatives), but the near-perfect test
    AUPRC makes the bias negligible.
    """
    n_target = len(scores_target)
    n_neg = min(n_target * ratio, len(scores_neg_pool))
    idx = rng.choice(len(scores_neg_pool), size=n_neg, replace=False)
    scores = np.concatenate([scores_target, scores_neg_pool[idx]])
    y = np.array([1] * n_target + [0] * n_neg)
    return float(average_precision_score(y, scores))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target-manifest", type=Path,
                        default=Path("data/raw_target/temp_0.3/manifest.jsonl"))
    parser.add_argument("--neg-manifest", type=Path,
                        default=Path("data/raw_clean/temp_0.3/manifest.jsonl"))
    parser.add_argument("--target-port", type=int, default=8443)
    parser.add_argument("--neg-port",    type=int, default=8443)
    parser.add_argument("--n-pairs", type=int, default=50,
                        help="(size, gap) pairs per trace (default: 50)")
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="fraction of each class used for training (default: 0.8)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("loading target traces...")
    X_target, y_target = load_dataset(
        args.target_manifest, args.target_port, args.n_pairs, label=1,
        temperature=args.temperature,
    )
    print(f"  {len(X_target)} target traces")

    print("loading negative traces...")
    X_neg, y_neg = load_dataset(
        args.neg_manifest, args.neg_port, args.n_pairs, label=0,
        temperature=args.temperature,
    )
    print(f"  {len(X_neg)} negative traces")

    # Shuffle and split each class independently.
    def split(X: np.ndarray, y: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        idx = rng.permutation(len(X))
        n_train = int(len(X) * args.train_frac)
        tr, te = idx[:n_train], idx[n_train:]
        return X[tr], X[te], y[tr], y[te]

    Xtr_t, Xte_t, ytr_t, yte_t = split(X_target, y_target)
    Xtr_n, Xte_n, ytr_n, yte_n = split(X_neg, y_neg)

    X_train = np.vstack([Xtr_t, Xtr_n])
    y_train = np.concatenate([ytr_t, ytr_n])
    X_test  = np.vstack([Xte_t, Xte_n])
    y_test  = np.concatenate([yte_t, yte_n])

    print(f"train: {y_train.sum()} target + {(y_train==0).sum()} negative")
    print(f"test:  {y_test.sum()} target  + {(y_test==0).sum()} negative")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("training LightGBM...")
    clf = LGBMClassifier(
        n_estimators=500,
        num_leaves=63,
        learning_rate=0.05,
        random_state=args.seed,
        verbosity=-1,
    )
    clf.fit(X_train_s, y_train)
    scores = clf.predict_proba(X_test_s)[:, 1]

    auprc_balanced = float(average_precision_score(y_test, scores))
    print(f"\nAUPRC (balanced 1:1): {auprc_balanced:.4f}")

    # For imbalance simulation: score ALL negatives so we have a larger pool.
    # This allows simulating higher ratios than the test split alone provides.
    scores_target_test = scores[y_test == 1]
    scores_all_neg = clf.predict_proba(scaler.transform(X_neg))[:, 1]

    imbalance_rows = []
    for ratio in [1, 2, 5, 10, 14]:
        if ratio * len(scores_target_test) > len(scores_all_neg):
            continue
        auprcs = [
            evaluate_at_imbalance(
                scores_target_test, scores_all_neg, ratio, rng
            )
            for _ in range(20)
        ]
        mean_auprc = float(np.mean(auprcs))
        imbalance_rows.append({"ratio": ratio, "auprc": round(mean_auprc, 4)})
        print(f"  ratio {ratio:>2}:1 -- mean AUPRC={mean_auprc:.4f}")

    # Figures
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # PR curve
    prec, rec, _ = precision_recall_curve(y_test, scores)
    chance = y_test.mean()

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(rec, prec, linewidth=1.8, label=f"LightGBM (AUPRC={auprc_balanced:.3f})")
    ax.axhline(chance, color="#aaaaaa", linewidth=0.8, linestyle="--",
               label=f"Chance ({chance:.0%})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        "Topic inference: Python questions vs. medical questions\n"
        "(McDonald & Bar Or approach, balanced 1:1 test)"
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower left")
    fig.tight_layout()
    pr_path = args.out_dir / "fig6_mcdonald_pr_curve.png"
    fig.savefig(pr_path)
    plt.close(fig)
    print(f"\nsaved -> {pr_path}")

    # AUPRC vs imbalance
    if imbalance_rows:
        df_imb = pd.DataFrame(imbalance_rows)
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
        ax2.plot(df_imb["ratio"], df_imb["auprc"], marker="o", linewidth=1.8)
        ax2.set_xlabel("Negative:positive ratio")
        ax2.set_ylabel("AUPRC")
        ax2.set_title("Topic inference AUPRC vs. class imbalance\n(LightGBM, 20 trials per ratio)")
        ax2.set_ylim(0, 1.05)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        fig2.tight_layout()
        imb_path = args.out_dir / "fig7_mcdonald_imbalance.png"
        fig2.savefig(imb_path)
        plt.close(fig2)
        print(f"saved -> {imb_path}")

        df_imb.to_csv(args.out_dir / "exp3_mcdonald_imbalance.csv", index=False)

    # Save summary CSV
    summary = pd.DataFrame([{
        "target_traces": int(X_target.shape[0]),
        "neg_traces":    int(X_neg.shape[0]),
        "n_pairs":       args.n_pairs,
        "train_frac":    args.train_frac,
        "auprc_balanced": round(auprc_balanced, 4),
    }])
    summary.to_csv(args.out_dir / "exp3_mcdonald_summary.csv", index=False)


if __name__ == "__main__":
    main()
