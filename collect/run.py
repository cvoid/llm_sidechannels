from __future__ import annotations

import json
import time
from pathlib import Path

from . import capture, query

MEDICAL_SYSTEM_PROMPT = (
    "You are a medical AI assistant. Answer patient questions about diseases, "
    "symptoms, and medical conditions concisely and accurately."
)


def profile_one(
    prompt_id: int,
    prompt: str,
    run_id: int,
    temperature: float,
    out_dir: Path,
    host: str,
    port: int = 8443,
    iface: str = "lo",
    bpf_filter: str = "tcp port 8443",
    model: str = "qwen2.5:7b-instruct-q4_K_M",
    system_prompt: str = "",
) -> Path:
    pcap_path = out_dir / f"prompt_{prompt_id:03d}_run_{run_id:03d}.pcap"
    proc = capture.start(iface, pcap_path, bpf_filter)
    time.sleep(0.1)  # let tcpdump bind before the request goes out
    try:
        query.send(prompt, host, port, temperature, system_prompt=system_prompt, model=model)
    finally:
        time.sleep(0.1)  # let tcpdump flush the last packets
        capture.stop(proc)
    return pcap_path


def profile_all(
    prompts: list[str],
    tpq: int,
    temperature: float,
    out_dir: Path,
    host: str,
    port: int = 8443,
    iface: str = "lo",
    bpf_filter: str = "tcp port 8443",
    model: str = "qwen2.5:7b-instruct-q4_K_M",
    system_prompt: str = "",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for prompt_id, prompt in enumerate(prompts):
        for run_id in range(tpq):
            pcap_path = profile_one(
                prompt_id, prompt, run_id, temperature,
                out_dir, host, port, iface, bpf_filter, model, system_prompt,
            )
            manifest.append({
                "prompt_id": prompt_id,
                "run_id": run_id,
                "prompt": prompt,
                "temperature": temperature,
                "system_prompt": system_prompt,
                "pcap": str(pcap_path),
            })
    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")
    return manifest_path
