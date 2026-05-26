"""Inter-packet gap feature extraction for timing-based attacks.

Extracts the first N consecutive inter-packet delays (server->client, ms)
from a pcap. Used by the Carlini & Nasr GMM-based binary disambiguation attack.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from features.parse import extract_records


def extract_gaps(
    pcap_path: Path,
    server_port: int = 8443,
    n_gaps: int = 100,
) -> np.ndarray | None:
    """Return first n_gaps inter-packet delays in milliseconds, or None.

    Returns None if the trace has fewer than n_gaps+1 server->client packets
    (after excluding zero-payload packets).
    """
    records = extract_records(pcap_path, server_port=server_port)
    times = [t for t, b in records]
    if len(times) < n_gaps + 1:
        return None
    gaps = np.array(
        [(times[i + 1] - times[i]) * 1000.0 for i in range(len(times) - 1)],
        dtype=np.float64,
    )
    return gaps[:n_gaps]
