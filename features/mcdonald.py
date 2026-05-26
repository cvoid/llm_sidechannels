"""(size, timing) pair feature extraction for topic-inference attacks.

Extracts the first N consecutive (payload_bytes, inter_arrival_ms) pairs
from a pcap. Used by the McDonald & Bar Or topic-detection pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from features.parse import extract_records


def extract_size_timing_pairs(
    pcap_path: Path,
    server_port: int = 8443,
    n_pairs: int = 50,
) -> np.ndarray | None:
    """Return a 2*n_pairs feature vector, or None if the trace is too short.

    Feature layout: [size_0, gap_0, size_1, gap_1, ..., size_{n-1}, gap_{n-1}]
    where gap_0 = 0.0 (no predecessor for the first packet).
    """
    records = extract_records(pcap_path, server_port=server_port)
    if len(records) < n_pairs:
        return None

    sizes = np.array([b for _, b in records[:n_pairs]], dtype=np.float64)
    times = np.array([t for t, _ in records[:n_pairs]], dtype=np.float64)
    gaps  = np.empty(n_pairs, dtype=np.float64)
    gaps[0] = 0.0
    gaps[1:] = (times[1:] - times[:-1]) * 1000.0

    # Interleave: [s0, g0, s1, g1, ...]
    out = np.empty(2 * n_pairs, dtype=np.float64)
    out[0::2] = sizes
    out[1::2] = gaps
    return out
