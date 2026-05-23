from __future__ import annotations

from pathlib import Path

import numpy as np
from scapy.all import rdpcap, IP, TCP  # type: ignore[import-untyped]


def extract_records(
    pcap_path: Path,
    server_port: int = 8443,
) -> list[tuple[float, int]]:
    packets = rdpcap(str(pcap_path))
    records: list[tuple[float, int]] = []
    for pkt in packets:
        if not (pkt.haslayer(TCP) and pkt.haslayer(IP)):
            continue
        if pkt[TCP].sport != server_port:
            continue
        payload_len: int = len(pkt[TCP].payload)
        if payload_len == 0:
            continue
        records.append((float(pkt.time), payload_len))
    return records


def group_iterations(
    records: list[tuple[float, int]],
    window_ms: float,
) -> list[int]:
    if not records:
        return []
    window_s = window_ms / 1000.0
    iterations: list[int] = []
    group_bytes = records[0][1]
    prev_ts = records[0][0]
    for ts, nbytes in records[1:]:
        if ts - prev_ts < window_s:
            group_bytes += nbytes
        else:
            iterations.append(group_bytes)
            group_bytes = nbytes
        prev_ts = ts
    iterations.append(group_bytes)
    return iterations


def trace_from_pcap(
    pcap_path: Path,
    server_port: int = 8443,
    window_ms: float = 50.0,
    skip_leading: int = 2,
) -> list[int]:
    records = extract_records(pcap_path, server_port)
    # First two iterations are the TLS handshake record and the HTTP response
    # headers -- constant across all requests, zero discriminating signal.
    return group_iterations(records, window_ms)[skip_leading:]


def calibrate_window(
    pcap_path: Path,
    server_port: int = 8443,
) -> float:
    """Estimate window_ms from a single response pcap.

    Finds the valley between the two gap clusters (within-iteration TCP
    fragmentation vs between-iteration inference pauses) by locating the
    largest jump in sorted gaps and returning its geometric midpoint.

    If unimodal (no clear bimodal split), returns half the minimum gap so
    every packet is treated as a separate iteration.
    """
    records = extract_records(pcap_path, server_port)
    if len(records) < 2:
        raise ValueError(f"need >= 2 records to calibrate, got {len(records)}")
    gaps_ms = sorted(
        (records[i + 1][0] - records[i][0]) * 1000.0
        for i in range(len(records) - 1)
    )
    min_gap = gaps_ms[0]
    max_gap = gaps_ms[-1]
    if max_gap <= 100.0 * (min_gap + 1e-6):
        return float(min_gap / 2.0)
    # Find the valley: largest ratio between consecutive sorted gaps.
    # Ratio (not absolute diff) correctly locates the split between the two
    # log-scale clusters regardless of the magnitude of either cluster.
    max_ratio = 0.0
    valley_lo = min_gap
    valley_hi = min_gap
    for i in range(len(gaps_ms) - 1):
        lo = gaps_ms[i] + 1e-9
        hi = gaps_ms[i + 1]
        ratio = hi / lo
        if ratio > max_ratio:
            max_ratio = ratio
            valley_lo = gaps_ms[i]
            valley_hi = hi
    # Geometric mean sits at the midpoint on a log scale.
    return float(np.sqrt(valley_lo * valley_hi))
