"""TPQ sweep — replicates paper Figure 3.

Trains a Random Forest at each (temperature, TPQ) combination and reports
accuracy and F1. Saves results to analysis/.

Example:
    uv run python tools/tpq_sweep.py \\
        --manifest data/raw/temp_0.3/manifest.jsonl \\
        --window-ms 2.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

from attack.evaluate import tpq_sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True,
                        help="path to manifest.jsonl produced by tools/profile.py")
    parser.add_argument("--window-ms", type=float, required=True,
                        help="iteration grouping window from tools/calibrate.sh")
    parser.add_argument("--trace-length", type=int, default=100,
                        help="fixed trace length in iterations (default: 100)")
    parser.add_argument("--tpq", type=int, nargs="+", default=[5, 10, 20, 30],
                        help="TPQ values to sweep (default: 5 10 20 30)")
    parser.add_argument("--temperatures", type=float, nargs="+", default=None,
                        help="temperatures to evaluate (default: all in manifest)")
    parser.add_argument("--out", type=Path, default=None,
                        help="CSV output path (default: analysis/exp1_tpq_sweep_<dir>.csv)")
    args = parser.parse_args()

    temperatures = args.temperatures or [0.3, 0.6, 0.8, 1.0]

    df = tpq_sweep(
        manifest_path=args.manifest,
        tpq_values=args.tpq,
        temperatures=temperatures,
        trace_length=args.trace_length,
        window_ms=args.window_ms,
    )
    print(df.to_string(index=False))

    out = args.out or Path("analysis") / f"exp1_tpq_sweep_{args.manifest.parent.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
