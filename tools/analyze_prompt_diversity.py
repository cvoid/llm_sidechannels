"""Measure response-length diversity across the 50-prompt dataset.

For each prompt, compute mean trace length (iterations) and mean total bytes
from existing pcap data. Then measure within-template vs across-template
cosine similarity to identify which question templates produce confusable
traces.

The concern: structurally identical templates (e.g. 14 "When to seek urgent
medical care" questions) may generate responses of similar length, collapsing
those classes together in feature space.

Example:
    uv run python tools/analyze_prompt_diversity.py \\
        --manifest data/raw/temp_0.3/manifest.jsonl \\
        --window-ms 2.5
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from features.build import cosine_similarity, pad_or_truncate
from features.parse import trace_from_pcap


TEMPLATES = [
    ("outlook",  re.compile(r"What to expect if I have")),
    ("symptoms", re.compile(r"What are the symptoms of")),
    ("causes",   re.compile(r"What causes|What are the causes of")),
    ("urgent",   re.compile(r"When to seek urgent medical care")),
    ("risk",     re.compile(r"Who is at highest risk")),
    ("stats",    re.compile(r"How many")),
]


def classify_template(text: str) -> str:
    for name, pat in TEMPLATES:
        if pat.search(text):
            return name
    return "other"


def _stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    arr = np.array(values)
    return f"mean={arr.mean():.3f}  std={arr.std():.3f}  min={arr.min():.3f}  max={arr.max():.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--window-ms", type=float, required=True)
    parser.add_argument("--server-port", type=int, default=8443)
    parser.add_argument("--max-empty-warn", type=int, default=5)
    args = parser.parse_args()

    entries: list[dict[str, object]] = []
    with open(args.manifest) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # Build per-prompt trace lists and metadata.
    traces_by_prompt: dict[int, list[list[int]]] = defaultdict(list)
    prompt_text: dict[int, str] = {}
    empty_count = 0

    for entry in entries:
        pid = int(str(entry["prompt_id"]))
        pcap = Path(str(entry["pcap"]))
        prompt_text[pid] = str(entry["prompt"])

        trace = trace_from_pcap(pcap, args.server_port, args.window_ms)
        if not trace:
            empty_count += 1
            if empty_count <= args.max_empty_warn:
                print(f"  WARNING: empty trace: {pcap}")
            continue
        traces_by_prompt[pid].append(trace)

    if empty_count > args.max_empty_warn:
        print(f"  WARNING: {empty_count} empty traces total (showing first {args.max_empty_warn})")

    print()

    # Per-prompt summary: mean iterations and mean total bytes.
    print(f"{'─'*72}")
    print(f"{'PID':>4}  {'tmpl':8}  {'runs':>4}  {'iters(mean)':>11}  "
          f"{'bytes(mean)':>11}  {'bytes(std)':>10}  prompt")
    print(f"{'─'*72}")

    per_prompt_mean_iters: dict[int, float] = {}
    per_prompt_mean_bytes: dict[int, float] = {}
    per_prompt_template: dict[int, str] = {}

    for pid in sorted(traces_by_prompt.keys()):
        traces = traces_by_prompt[pid]
        tmpl = classify_template(prompt_text[pid])
        iters = [len(t) for t in traces]
        totals = [sum(t) for t in traces]
        mean_iters = float(np.mean(iters))
        mean_bytes = float(np.mean(totals))
        std_bytes = float(np.std(totals))
        per_prompt_mean_iters[pid] = mean_iters
        per_prompt_mean_bytes[pid] = mean_bytes
        per_prompt_template[pid] = tmpl
        text_short = prompt_text[pid][:45]
        print(f"{pid:>4}  {tmpl:8}  {len(traces):>4}  {mean_iters:>11.1f}  "
              f"{mean_bytes:>11.0f}  {std_bytes:>10.0f}  {text_short}")

    # Sort by mean bytes to show length distribution.
    sorted_by_bytes = sorted(per_prompt_mean_bytes.items(), key=lambda x: x[1])
    print()
    print(f"{'─'*72}")
    print("Response length ranking (shortest to longest, by mean bytes):")
    print(f"{'─'*72}")
    for rank, (pid, mb) in enumerate(sorted_by_bytes, 1):
        tmpl = per_prompt_template[pid]
        text_short = prompt_text[pid][:55]
        print(f"  {rank:>2}.  pid={pid:>2}  {mb:>9.0f} B  [{tmpl:8}]  {text_short}")

    # Template-level cosine similarity analysis.
    print()
    print(f"{'─'*72}")
    print("Within-template vs across-template cosine similarity:")
    print(f"{'─'*72}")

    max_len = max(len(t) for traces in traces_by_prompt.values() for t in traces)

    templates_present: dict[str, list[int]] = defaultdict(list)
    for pid, tmpl in per_prompt_template.items():
        templates_present[tmpl].append(pid)

    # Within-template: pairs from same template.
    within_by_template: dict[str, list[float]] = defaultdict(list)
    for tmpl, pids in templates_present.items():
        for i, pid_a in enumerate(pids):
            for pid_b in pids[i + 1:]:
                for ta in traces_by_prompt[pid_a]:
                    for tb in traces_by_prompt[pid_b]:
                        pa = pad_or_truncate(ta, max_len)
                        pb = pad_or_truncate(tb, max_len)
                        within_by_template[tmpl].append(cosine_similarity(pa, pb))

    # Across-template: random sample to keep it fast.
    rng = np.random.default_rng(0)
    all_pids = sorted(traces_by_prompt.keys())
    across_sims: list[float] = []
    for i, pid_a in enumerate(all_pids):
        for pid_b in all_pids[i + 1:]:
            if per_prompt_template[pid_a] == per_prompt_template[pid_b]:
                continue
            # Sample one trace pair per cross-template prompt pair.
            ta = traces_by_prompt[pid_a][int(rng.integers(len(traces_by_prompt[pid_a])))]
            tb = traces_by_prompt[pid_b][int(rng.integers(len(traces_by_prompt[pid_b])))]
            pa = pad_or_truncate(ta, max_len)
            pb = pad_or_truncate(tb, max_len)
            across_sims.append(cosine_similarity(pa, pb))

    print(f"\n  across-template (all pairs): {_stats(across_sims)}")
    print()
    for tmpl in sorted(within_by_template.keys()):
        sims = within_by_template[tmpl]
        n_pids = len(templates_present[tmpl])
        print(f"  {tmpl:8} ({n_pids:>2} prompts):  {_stats(sims)}")

    print()
    all_within = [s for sims in within_by_template.values() for s in sims]
    if all_within and across_sims:
        sep = float(np.mean(all_within)) - float(np.mean(across_sims))
        print(f"  overall within mean: {np.mean(all_within):.3f}")
        print(f"  overall across mean: {np.mean(across_sims):.3f}")
        print(f"  template separation: {sep:+.3f}  "
              f"({'problem' if sep > 0.1 else 'ok'})")
        print()
        if sep > 0.1:
            print("  WARNING: templates cluster in feature space.")
            print("  Prompts within the same template are more similar to each other")
            print("  than to prompts in other templates. The classifier will be unable")
            print("  to distinguish prompts within each cluster.")
            print()
            print("  The 'urgent' and 'symptoms' templates have the most prompts.")
            print("  Consider replacing some with prompts from underrepresented templates,")
            print("  or accept lower accuracy on within-template pairs and focus evaluation")
            print("  on cross-template accuracy.")

    # Most confusable pairs (highest across-prompt cosine similarity within same template).
    print(f"{'─'*72}")
    print("Top 10 most confusable prompt pairs (highest within-template cosine sim):")
    print(f"{'─'*72}")

    confusable: list[tuple[float, int, int]] = []
    for tmpl, pids in templates_present.items():
        for i, pid_a in enumerate(pids):
            for pid_b in pids[i + 1:]:
                pair_sims = []
                for ta in traces_by_prompt[pid_a][:5]:
                    for tb in traces_by_prompt[pid_b][:5]:
                        pa = pad_or_truncate(ta, max_len)
                        pb = pad_or_truncate(tb, max_len)
                        pair_sims.append(cosine_similarity(pa, pb))
                if pair_sims:
                    confusable.append((float(np.mean(pair_sims)), pid_a, pid_b))

    confusable.sort(reverse=True)
    for sim, pid_a, pid_b in confusable[:10]:
        text_a = prompt_text[pid_a][:40]
        text_b = prompt_text[pid_b][:40]
        print(f"  sim={sim:.3f}  pid={pid_a:>2} '{text_a}'")
        print(f"              pid={pid_b:>2} '{text_b}'")
        print()


if __name__ == "__main__":
    main()
