from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import parse


def pad_or_truncate(trace: list[int], length: int) -> list[int]:
    if len(trace) >= length:
        return trace[:length]
    return trace + [0] * (length - len(trace))


def cosine_similarity(a: list[int], b: list[int]) -> float:
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def build_dataset(
    manifest_path: Path,
    trace_length: int,
    window_ms: float,
    server_port: int = 8443,
    temperature: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    entries: list[dict[str, object]] = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                entry: dict[str, object] = json.loads(line)
                if temperature is None or float(str(entry["temperature"])) == temperature:
                    entries.append(entry)

    X: list[list[int]] = []
    y: list[int] = []
    for entry in entries:
        trace = parse.trace_from_pcap(
            Path(str(entry["pcap"])),
            server_port=server_port,
            window_ms=window_ms,
        )
        X.append(pad_or_truncate(trace, trace_length))
        y.append(int(str(entry["prompt_id"])))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
