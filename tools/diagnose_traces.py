"""Diagnose whether captured traces contain a usable fingerprinting signal.

Reads pcaps from a manifest, prints raw iteration-byte sequences, and checks:
  - whether traces are non-trivial (not all identical iteration sizes)
  - whether same-prompt traces are more similar to each other than to other prompts
  - whether packet counts are plausible for speculative decoding

Run after tools/smoke_test.py fails to understand what the captures contain.

Example:
    uv run python tools/diagnose_traces.py \\
        --manifest data/smoke/manifest.jsonl \\
        --window-ms 2.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from features.parse import extract_records, group_iterations, trace_from_pcap
from features.build import cosine_similarity


def _stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"min={min(values):.3f}  max={max(values):.3f}  mean={sum(values)/len(values):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/smoke/manifest.jsonl"))
    parser.add_argument("--window-ms", type=float, required=True)
    parser.add_argument("--server-port", type=int, default=8443)
    parser.add_argument("--max-traces", type=int, default=6,
                        help="max traces to print in full (default: 6)")
    args = parser.parse_args()

    entries: list[dict[str, object]] = []
    with open(args.manifest) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    traces_by_prompt: dict[int, list[list[int]]] = {}
    all_traces: list[list[int]] = []

    print(f"{'─'*60}")
    print(f"manifest : {args.manifest}  ({len(entries)} entries)")
    print(f"window_ms: {args.window_ms}")
    print(f"{'─'*60}\n")

    for i, entry in enumerate(entries):
        pcap = Path(str(entry["pcap"]))
        pid = int(str(entry["prompt_id"]))
        records = extract_records(pcap, args.server_port)
        trace = group_iterations(records, args.window_ms)

        traces_by_prompt.setdefault(pid, []).append(trace)
        all_traces.append(trace)

        if i < args.max_traces:
            sizes = [r[1] for r in records]
            unique_sizes = len(set(sizes))
            print(f"prompt={pid:02d} run={entry['run_id']:02d}  "
                  f"packets={len(records):4d}  iterations={len(trace):4d}  "
                  f"unique_pkt_sizes={unique_sizes}")
            if trace:
                print(f"  iter bytes: {trace[:20]}{'...' if len(trace) > 20 else ''}")
            else:
                print("  iter bytes: EMPTY")
            print()

    if len(entries) > args.max_traces:
        print(f"  ... ({len(entries) - args.max_traces} more entries not shown)\n")

    print(f"{'─'*60}")
    print("SIGNAL DIAGNOSTICS")
    print(f"{'─'*60}\n")

    # Check 1: are traces non-empty?
    empty = sum(1 for t in all_traces if len(t) == 0)
    print(f"Empty traces    : {empty}/{len(all_traces)}")
    if empty > 0:
        print("  → FIX: packets aren't being captured. Check tcpdump interface and BPF filter.")
        return

    # Check 2: are traces flat (all iterations same size)?
    flat = sum(1 for t in all_traces if len(set(t)) == 1)
    print(f"Flat traces     : {flat}/{len(all_traces)}  (all iterations same byte count)")
    if flat == len(all_traces):
        print("  → FIX: no per-iteration variation. Speculative decoding is likely not active.")
        print("         Check: grep -i 'draft\\|spec' logs/llama.log")
        print("         Check: does llama-server log show the draft model loading?")

    # Check 3: within-prompt vs across-prompt cosine similarity
    max_len = max(len(t) for t in all_traces)
    padded = [t + [0] * (max_len - len(t)) for t in all_traces]

    within_sims: list[float] = []
    across_sims: list[float] = []

    prompt_ids = list(traces_by_prompt.keys())
    for pid, group in traces_by_prompt.items():
        padded_group = [t + [0] * (max_len - len(t)) for t in group]
        for i in range(len(padded_group)):
            for j in range(i + 1, len(padded_group)):
                within_sims.append(cosine_similarity(padded_group[i], padded_group[j]))

    for i, pid_a in enumerate(prompt_ids):
        for pid_b in prompt_ids[i + 1:]:
            for ta in traces_by_prompt[pid_a]:
                for tb in traces_by_prompt[pid_b]:
                    pa = ta + [0] * (max_len - len(ta))
                    pb = tb + [0] * (max_len - len(tb))
                    across_sims.append(cosine_similarity(pa, pb))

    print(f"\nCosine similarity (paper target: within~0.9-1.0, across~0.4-0.8):")
    print(f"  within-prompt : {_stats(within_sims)}")
    print(f"  across-prompt : {_stats(across_sims)}")

    if within_sims and across_sims:
        mean_within = sum(within_sims) / len(within_sims)
        mean_across = sum(across_sims) / len(across_sims)
        sep = mean_within - mean_across
        print(f"  separation    : {sep:+.3f}  (need > 0.1 for classifier to work)")
        print()
        if sep < 0.05:
            print("DIAGNOSIS: traces are not prompt-specific.")
            if flat == len(all_traces):
                print("  Most likely cause: speculative decoding not active — all iterations")
                print("  produce 1 token so packet sizes are uniform across all prompts.")
            else:
                print("  Most likely cause: nginx is buffering chunks, collapsing per-iteration")
                print("  boundaries into larger packets. Check proxy_buffering off in nginx.conf")
                print("  and that nginx was reloaded after the config was installed.")
        elif sep < 0.2:
            print("DIAGNOSIS: weak signal. Classifier may need more TPQ or longer trace_length.")
        else:
            print("DIAGNOSIS: signal looks healthy. Try more TPQ in smoke test.")


if __name__ == "__main__":
    main()
