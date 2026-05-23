"""3-prompt × 5-run sanity check.

Profiles a small subset, builds features, trains a classifier, and checks
that accuracy is well above chance. Run this before committing to a full
1500-query profiling session.

Expected: accuracy > 0.50 with 3 classes. If you see ~0.33, traces are
likely empty — check pcap sizes and re-examine window_ms.

Example:
    uv run python tools/smoke_test.py --window-ms 2.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from attack.dataset import split
from attack.evaluate import score
from attack.train import fit
from collect.prompts import load_prompts
from collect.run import MEDICAL_SYSTEM_PROMPT, profile_all
from features.build import build_dataset

CHANCE = 1.0 / 3.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--window-ms", type=float, required=True,
                        help="from tools/calibrate.sh")
    parser.add_argument("--trace-length", type=int, default=100)
    parser.add_argument("--host", default="server.local")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--iface", default="lo")
    parser.add_argument("--system-prompt", default=MEDICAL_SYSTEM_PROMPT,
                        help="system prompt sent with each query (default: medical assistant)")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="send no system prompt")
    args = parser.parse_args()

    system_prompt = "" if args.no_system_prompt else args.system_prompt
    out_dir = Path("data/smoke")
    print("profiling 3 prompts × 5 runs...")
    manifest = profile_all(
        prompts=load_prompts()[:3],
        tpq=5,
        temperature=0.3,
        out_dir=out_dir,
        host=args.host,
        port=args.port,
        iface=args.iface,
        system_prompt=system_prompt,
    )

    print("building features...")
    X, y = build_dataset(manifest, trace_length=args.trace_length, window_ms=args.window_ms)
    X_train, X_test, y_train, y_test = split(X, y, train_tpq=3, test_tpq=2)

    print("training classifier...")
    clf = fit(X_train, y_train)
    result = score(clf, X_test, y_test)

    print()
    print(f"accuracy : {result['accuracy']:.2f}  (chance = {CHANCE:.2f})")
    print(f"f1_macro : {result['f1_macro']:.2f}")

    if result["accuracy"] < CHANCE + 0.10:
        print()
        print("FAIL: accuracy near chance.")
        print("  - Check pcap sizes: ls -lh data/smoke/*.pcap")
        print("  - Re-run tools/calibrate.sh and use the new window_ms")
        print("  - Confirm speculative decoding: grep -i draft logs/llama.log")
        sys.exit(1)

    print()
    print("OK — proceed to full profiling with tools/profile.py")


if __name__ == "__main__":
    main()
