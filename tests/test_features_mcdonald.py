from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scapy.all import IP, TCP, wrpcap  # type: ignore[import-untyped]

from features.mcdonald import extract_size_timing_pairs

SERVER_PORT = 8443
CLIENT_PORT = 12345


def _pkt(t: float, size: int) -> object:
    p = IP(src="127.0.0.1", dst="127.0.0.2") / TCP(sport=SERVER_PORT, dport=CLIENT_PORT, flags="PA") / (b"X" * size)
    p.time = t
    return p


@pytest.fixture
def mcd_pcap(tmp_path: Path) -> Path:
    """Four server->client packets with known sizes and inter-arrival times."""
    pkts = [
        _pkt(0.000, 50),
        _pkt(0.010, 30),  # gap 10ms
        _pkt(0.030, 20),  # gap 20ms
        _pkt(0.060, 40),  # gap 30ms
    ]
    path = tmp_path / "mcd.pcap"
    wrpcap(str(path), pkts)
    return path


def test_returns_2n_dim_vector(mcd_pcap: Path) -> None:
    result = extract_size_timing_pairs(mcd_pcap, server_port=SERVER_PORT, n_pairs=3)
    assert result is not None
    assert result.shape == (6,)


def test_first_gap_is_zero(mcd_pcap: Path) -> None:
    result = extract_size_timing_pairs(mcd_pcap, server_port=SERVER_PORT, n_pairs=3)
    assert result is not None
    assert result[1] == 0.0


def test_sizes_and_gaps_correct(mcd_pcap: Path) -> None:
    result = extract_size_timing_pairs(mcd_pcap, server_port=SERVER_PORT, n_pairs=3)
    assert result is not None
    # even indices: sizes; odd indices: gaps
    np.testing.assert_allclose(result[0::2], [50.0, 30.0, 20.0])
    np.testing.assert_allclose(result[1::2], [0.0, 10.0, 20.0], atol=0.5)


def test_too_short_returns_none(mcd_pcap: Path) -> None:
    # 4 packets available; requesting 5 pairs should return None
    result = extract_size_timing_pairs(mcd_pcap, server_port=SERVER_PORT, n_pairs=5)
    assert result is None
