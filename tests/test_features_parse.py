from __future__ import annotations

from pathlib import Path

import pytest

from features.parse import calibrate_window, extract_records, group_iterations, trace_from_pcap

SERVER_PORT = 8443


def test_extract_records_filters_client_to_server(synthetic_pcap: Path) -> None:
    records = extract_records(synthetic_pcap, SERVER_PORT)
    # Only server->client packets have sport=8443; client->server packet filtered out.
    assert len(records) == 4


def test_extract_records_returns_timestamps_and_sizes(synthetic_pcap: Path) -> None:
    records = extract_records(synthetic_pcap, SERVER_PORT)
    timestamps = [r[0] for r in records]
    sizes = [r[1] for r in records]
    assert timestamps == sorted(timestamps), "records must be in time order"
    assert all(s > 0 for s in sizes)


def test_group_iterations_single_window() -> None:
    # Two packets 5ms apart, window=50ms -> one iteration.
    records = [(0.0, 50), (0.005, 30)]
    result = group_iterations(records, window_ms=50.0)
    assert result == [80]


def test_group_iterations_separate_packets() -> None:
    records = [(0.0, 50), (0.1, 30), (0.2, 20)]
    result = group_iterations(records, window_ms=50.0)
    assert result == [50, 30, 20]


def test_group_iterations_empty() -> None:
    assert group_iterations([], window_ms=50.0) == []


def test_trace_from_pcap_iteration_bytes(synthetic_pcap: Path) -> None:
    # With window_ms=50: iter1=80 (50+30), iter2=20, iter3=40.
    # skip_leading=0 to test grouping logic independently from the handshake skip.
    trace = trace_from_pcap(synthetic_pcap, SERVER_PORT, window_ms=50.0, skip_leading=0)
    assert trace == [80, 20, 40]


def test_trace_from_pcap_tight_window(synthetic_pcap: Path) -> None:
    # With window_ms=1ms, the 5ms gap also becomes a boundary -> 4 iterations.
    # skip_leading=0 to test grouping logic independently from the handshake skip.
    trace = trace_from_pcap(synthetic_pcap, SERVER_PORT, window_ms=1.0, skip_leading=0)
    assert trace == [50, 30, 20, 40]


def test_calibrate_window_raises_on_single_packet(single_packet_pcap: Path) -> None:
    with pytest.raises(ValueError, match="need >= 2 records"):
        calibrate_window(single_packet_pcap, SERVER_PORT)


def test_calibrate_window_unimodal_returns_half_min_gap(synthetic_pcap: Path) -> None:
    # Gaps: 5ms, 95ms, 100ms. Max/min = 100/5 = 20 < 100 -> unimodal path.
    result = calibrate_window(synthetic_pcap, SERVER_PORT)
    assert result == pytest.approx(5.0 / 2.0, rel=1e-3)


def test_calibrate_window_returns_float(synthetic_pcap: Path) -> None:
    result = calibrate_window(synthetic_pcap, SERVER_PORT)
    assert isinstance(result, float)
    assert result > 0.0
