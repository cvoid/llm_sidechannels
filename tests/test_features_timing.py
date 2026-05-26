from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scapy.all import IP, TCP, wrpcap  # type: ignore[import-untyped]

from features.timing import extract_gaps

SERVER_PORT = 8443
CLIENT_PORT = 12345


def _pkt(t: float, size: int) -> object:
    p = IP(src="127.0.0.1", dst="127.0.0.2") / TCP(sport=SERVER_PORT, dport=CLIENT_PORT, flags="PA") / (b"X" * size)
    p.time = t
    return p


@pytest.fixture
def timing_pcap(tmp_path: Path) -> Path:
    """Five server->client packets with known inter-packet gaps."""
    pkts = [
        _pkt(0.000, 100),
        _pkt(0.010, 100),  # gap 10ms
        _pkt(0.025, 100),  # gap 15ms
        _pkt(0.050, 100),  # gap 25ms
        _pkt(0.100, 100),  # gap 50ms
    ]
    path = tmp_path / "timing.pcap"
    wrpcap(str(path), pkts)
    return path


def test_extract_gaps_returns_n_gaps(timing_pcap: Path) -> None:
    result = extract_gaps(timing_pcap, server_port=SERVER_PORT, n_gaps=3)
    assert result is not None
    assert result.shape == (3,)
    assert result.dtype == np.float64


def test_extract_gaps_values(timing_pcap: Path) -> None:
    result = extract_gaps(timing_pcap, server_port=SERVER_PORT, n_gaps=4)
    assert result is not None
    np.testing.assert_allclose(result, [10.0, 15.0, 25.0, 50.0], atol=0.5)


def test_extract_gaps_too_short_returns_none(timing_pcap: Path) -> None:
    # 5 packets -> 4 gaps; requesting 5 gaps should return None
    result = extract_gaps(timing_pcap, server_port=SERVER_PORT, n_gaps=5)
    assert result is None
