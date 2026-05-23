from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from collect.capture import start, stop


def test_start_creates_parent_dir(tmp_path: Path) -> None:
    pcap = tmp_path / "sub" / "capture.pcap"
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        start("lo", pcap)
    assert pcap.parent.exists()


def test_start_passes_pcap_path(tmp_path: Path) -> None:
    pcap = tmp_path / "out.pcap"
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        start("lo", pcap)
        cmd = mock_popen.call_args[0][0]
    assert str(pcap) in cmd


def test_start_uses_custom_filter(tmp_path: Path) -> None:
    pcap = tmp_path / "out.pcap"
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        start("eth0", pcap, bpf_filter="tcp port 9000")
        cmd = mock_popen.call_args[0][0]
    assert "tcp port 9000" in cmd


def test_stop_terminates(tmp_path: Path) -> None:
    proc: subprocess.Popen[bytes] = MagicMock(spec=subprocess.Popen)
    proc.wait = MagicMock(return_value=0)
    stop(proc)
    proc.terminate.assert_called_once()


def test_stop_kills_on_timeout() -> None:
    proc: subprocess.Popen[bytes] = MagicMock(spec=subprocess.Popen)
    proc.wait = MagicMock(side_effect=[subprocess.TimeoutExpired("tcpdump", 5), 0])
    stop(proc)
    proc.kill.assert_called_once()
