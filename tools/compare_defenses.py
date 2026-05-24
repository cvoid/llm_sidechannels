"""Generate a defense comparison table from profiling runs.

Compares accuracy and payload overhead for each defense configuration
against the undefended baseline, reproducing the structure of Wei et al.
Table 2 / Figure 6.

Overhead is the ratio of mean server-to-client bytes per response relative
to the undefended baseline, measured from the captured pcaps.

Example:
    uv run python tools/compare_defenses.py --window-ms 3.5 --tpq 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from attack.dataset import split
from attack.train import fit
from attack.evaluate import score
from features.build import build_dataset
from features.parse import extract_records


# (label, data_dir, server_port)
# server_port is the nginx port the client connects to, used to filter
# server->client packets in the pcaps.
CONFIGS: list[tuple[str, str, int]] = [
    ("undefended",      "data/raw_clean",                 8443),
    ("agg batch=2",     "data/raw_defend/agg_batch2",     8444),
    ("agg batch=4",     "data/raw_defend/agg_batch4",     8444),
    ("agg batch=8",     "data/raw_defend/agg_batch8",     8444),
    ("pad rand=128",    "data/raw_defend/pad_rand128",     8445),
    ("pad rand=256",    "data/raw_defend/pad_rand256",     8445),
    ("pad rand=512",    "data/raw_defend/pad_rand512",     8445),
    ("pad fixed=1500",  "data/raw_defend/pad_fixed1500",   8445),
]


def _mean_response_bytes(
    manifest_path: Path,
    temperature: float,
    server_port: int,
) -> float:
    totals: list[int] = []
    with open(manifest_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if float(str(entry["temperature"])) != temperature:
                continue
            pcap = Path(str(entry["pcap"]))
            records = extract_records(pcap, server_port=server_port)
            totals.append(sum(b for _, b in records))
    return float(np.mean(totals)) if totals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--window-ms", type=float, required=True)
    parser.add_argument("--tpq", type=int, default=30,
                        help="total traces per query collected (default: 30)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--trace-length", type=int, default=100)
    parser.add_argument("--out", type=Path,
                        default=Path("analysis/defense_comparison.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    baseline_bytes: float | None = None

    for label, data_dir, server_port in CONFIGS:
        manifest = Path(data_dir) / f"temp_{args.temperature}" / "manifest.jsonl"
        if not manifest.exists():
            print(f"  skip {label}: {manifest} not found")
            continue

        print(f"  evaluating {label}...")

        X, y = build_dataset(
            manifest, args.trace_length, args.window_ms,
            server_port=server_port, temperature=args.temperature,
        )
        X_train, X_test, y_train, y_test = split(
            X, y, train_tpq=args.tpq - 5, test_tpq=5,
        )
        clf = fit(X_train, y_train)
        metrics = score(clf, X_test, y_test)

        mean_bytes = _mean_response_bytes(manifest, args.temperature, server_port)
        if baseline_bytes is None:
            baseline_bytes = mean_bytes

        overhead = (mean_bytes / baseline_bytes) if baseline_bytes else 1.0

        rows.append({
            "defense":        label,
            "accuracy":       round(metrics["accuracy"], 3),
            "f1_macro":       round(metrics["f1_macro"], 3),
            "mean_bytes":     round(mean_bytes),
            "overhead_x":     round(overhead, 2),
        })

    if not rows:
        print("No completed configs found. Run tools/eval_defenses.sh first.")
        return

    df = pd.DataFrame(rows)
    baseline_acc = float(df.loc[df["defense"] == "undefended", "accuracy"].iloc[0])
    df["acc_reduction"] = (baseline_acc - df["accuracy"]).round(3)

    col_order = ["defense", "accuracy", "acc_reduction", "f1_macro",
                 "mean_bytes", "overhead_x"]
    df = df[col_order]

    print()
    print(df.to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
