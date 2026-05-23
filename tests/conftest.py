from __future__ import annotations

from pathlib import Path

import pytest
from scapy.all import IP, TCP, wrpcap  # type: ignore[import-untyped]

SERVER_PORT = 8443
CLIENT_PORT = 12345
SERVER_IP = "127.0.0.1"
CLIENT_IP = "127.0.0.2"


def _pkt(sport: int, dport: int, payload: bytes, t: float) -> object:
    p = IP(src=SERVER_IP, dst=CLIENT_IP) / TCP(sport=sport, dport=dport, flags="PA") / payload
    p.time = t
    return p


@pytest.fixture
def synthetic_pcap(tmp_path: Path) -> Path:
    """Three decode iterations: [80, 20, 40] bytes.

    Iteration 1: two packets at t=0.000 and t=0.005 (gap 5ms < 50ms window).
    Iteration 2: one packet at t=0.100 (gap 95ms).
    Iteration 3: one packet at t=0.200 (gap 100ms).
    Also includes one client->server packet that must be filtered out.
    """
    pkts = [
        _pkt(SERVER_PORT, CLIENT_PORT, b"A" * 50, 0.000),
        _pkt(SERVER_PORT, CLIENT_PORT, b"B" * 30, 0.005),
        _pkt(SERVER_PORT, CLIENT_PORT, b"C" * 20, 0.100),
        _pkt(CLIENT_PORT, SERVER_PORT, b"D" * 10, 0.150),  # filtered out
        _pkt(SERVER_PORT, CLIENT_PORT, b"E" * 40, 0.200),
    ]
    pcap_path = tmp_path / "test.pcap"
    wrpcap(str(pcap_path), pkts)
    return pcap_path


@pytest.fixture
def single_packet_pcap(tmp_path: Path) -> Path:
    """One server->client packet; used to test edge cases."""
    pkts = [_pkt(SERVER_PORT, CLIENT_PORT, b"X" * 100, 0.0)]
    pcap_path = tmp_path / "single.pcap"
    wrpcap(str(pcap_path), pkts)
    return pcap_path
