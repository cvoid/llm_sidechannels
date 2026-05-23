from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from collect.run import profile_all, profile_one


def _mock_capture_and_query(pcap_path: Path) -> tuple[object, object, object]:
    mock_proc = MagicMock()
    mock_proc.wait = MagicMock(return_value=0)

    mock_start = MagicMock(return_value=mock_proc)
    mock_stop = MagicMock()
    mock_send = MagicMock(return_value="fake response")
    return mock_start, mock_stop, mock_send


def test_profile_one_returns_pcap_path(tmp_path: Path) -> None:
    mock_proc = MagicMock()
    with (
        patch("collect.run.capture.start", return_value=mock_proc),
        patch("collect.run.capture.stop"),
        patch("collect.run.query.send", return_value=""),
        patch("time.sleep"),
    ):
        result = profile_one(0, "prompt", 0, 0.3, tmp_path, "localhost")
    assert result == tmp_path / "prompt_000_run_000.pcap"


def test_profile_one_zero_pads_ids(tmp_path: Path) -> None:
    mock_proc = MagicMock()
    with (
        patch("collect.run.capture.start", return_value=mock_proc),
        patch("collect.run.capture.stop"),
        patch("collect.run.query.send", return_value=""),
        patch("time.sleep"),
    ):
        result = profile_one(3, "p", 12, 0.3, tmp_path, "localhost")
    assert result.name == "prompt_003_run_012.pcap"


def test_profile_all_creates_manifest(tmp_path: Path) -> None:
    mock_proc = MagicMock()
    with (
        patch("collect.run.capture.start", return_value=mock_proc),
        patch("collect.run.capture.stop"),
        patch("collect.run.query.send", return_value=""),
        patch("time.sleep"),
    ):
        manifest = profile_all(["q1", "q2"], tpq=2, temperature=0.3, out_dir=tmp_path, host="localhost")
    assert manifest.exists()


def test_profile_all_manifest_entry_count(tmp_path: Path) -> None:
    mock_proc = MagicMock()
    prompts = ["p1", "p2", "p3"]
    tpq = 3
    with (
        patch("collect.run.capture.start", return_value=mock_proc),
        patch("collect.run.capture.stop"),
        patch("collect.run.query.send", return_value=""),
        patch("time.sleep"),
    ):
        manifest = profile_all(prompts, tpq=tpq, temperature=0.3, out_dir=tmp_path, host="localhost")
    entries = [json.loads(l) for l in manifest.read_text().splitlines() if l]
    assert len(entries) == len(prompts) * tpq


def test_profile_all_manifest_fields(tmp_path: Path) -> None:
    mock_proc = MagicMock()
    with (
        patch("collect.run.capture.start", return_value=mock_proc),
        patch("collect.run.capture.stop"),
        patch("collect.run.query.send", return_value=""),
        patch("time.sleep"),
    ):
        manifest = profile_all(["only"], tpq=1, temperature=0.6, out_dir=tmp_path, host="localhost")
    entry = json.loads(manifest.read_text().strip())
    assert set(entry.keys()) == {"prompt_id", "run_id", "prompt", "temperature", "pcap", "system_prompt"}
    assert entry["temperature"] == 0.6
