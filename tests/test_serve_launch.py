from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from serve.launch import build_cmd, start, stop, wait_ready


def test_build_cmd_contains_model_flags(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    draft = tmp_path / "draft.gguf"
    cmd = build_cmd(model, draft, "127.0.0.1", 8080, n_gpu_layers=33, n_draft=5, ctx_size=4096)
    assert "--model" in cmd
    assert str(model) in cmd
    assert "--model-draft" in cmd
    assert str(draft) in cmd
    assert "--port" in cmd
    assert "8080" in cmd
    assert "--spec-draft-n-max" in cmd
    assert "5" in cmd


def test_build_cmd_is_list_of_strings(tmp_path: Path) -> None:
    cmd = build_cmd(tmp_path / "m", tmp_path / "d", "localhost", 8080, 0, 3, 2048)
    assert all(isinstance(s, str) for s in cmd)


def test_wait_ready_raises_on_timeout() -> None:
    with patch("urllib.request.urlopen", side_effect=OSError("refused")):
        with pytest.raises(TimeoutError):
            wait_ready("127.0.0.1", 9999, timeout=0.1)


def test_wait_ready_succeeds_on_first_try() -> None:
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_ctx):
        wait_ready("127.0.0.1", 8080, timeout=5.0)  # should not raise


def test_stop_terminates_process() -> None:
    proc: subprocess.Popen[bytes] = MagicMock(spec=subprocess.Popen)
    proc.wait = MagicMock(return_value=0)
    stop(proc)
    proc.terminate.assert_called_once()


def test_stop_kills_on_timeout() -> None:
    proc: subprocess.Popen[bytes] = MagicMock(spec=subprocess.Popen)
    proc.wait = MagicMock(side_effect=[subprocess.TimeoutExpired("cmd", 10), 0])
    stop(proc)
    proc.kill.assert_called_once()


def test_start_creates_log_dir(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "server.log"
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        start(["echo", "hi"], log_path)
    assert log_path.parent.exists()
