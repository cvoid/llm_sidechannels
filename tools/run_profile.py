"""Offline profiling phase for Experiment 1.

Captures one pcap per (prompt, run) pair and writes a manifest.jsonl.
Run from the repo root with llama-server already running.

Examples:
    uv run python tools/run_profile.py --temperature 0.3 --tpq 30
    uv run python tools/run_profile.py --temperature 0.6 --tpq 30
    uv run python tools/run_profile.py --temperature 0.3 --tpq 5 --prompts 3  # quick test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from collect.prompts import load_prompts
from collect.run import MEDICAL_SYSTEM_PROMPT, profile_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="sampling temperature (default: 0.3)")
    parser.add_argument("--tpq", type=int, default=30,
                        help="traces per query (default: 30)")
    parser.add_argument("--prompts", type=int, default=None,
                        help="limit to first N prompts (default: all 50)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: data/raw/temp_<T>)")
    parser.add_argument("--host", default="server.local")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--iface", default="lo")
    parser.add_argument("--prompts-file", type=Path, default=None,
                        help="JSONL file with 'text' field per line (default: exp1_prompts.jsonl)")
    parser.add_argument("--system-prompt", default=MEDICAL_SYSTEM_PROMPT,
                        help="system prompt sent with each query (default: medical assistant)")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="send no system prompt")
    args = parser.parse_args()

    system_prompt = "" if args.no_system_prompt else args.system_prompt
    out_dir = args.out_dir or Path(f"data/raw/temp_{args.temperature}")
    prompts = load_prompts(args.prompts_file) if args.prompts_file else load_prompts()
    if args.prompts is not None:
        prompts = prompts[: args.prompts]

    n_queries = len(prompts) * args.tpq
    print(f"profiling : {len(prompts)} prompts × {args.tpq} runs = {n_queries} queries")
    print(f"temperature: {args.temperature}")
    print(f"output    : {out_dir}")
    print(f"system prompt: {'(none)' if not system_prompt else system_prompt[:60] + '...'}")
    print(f"est. time : {n_queries * 5 / 3600:.1f} h at ~5 s/query")
    print()

    manifest = profile_all(
        prompts=prompts,
        tpq=args.tpq,
        temperature=args.temperature,
        out_dir=out_dir,
        host=args.host,
        port=args.port,
        iface=args.iface,
        system_prompt=system_prompt,
    )
    print(f"done: {manifest}")


if __name__ == "__main__":
    main()
