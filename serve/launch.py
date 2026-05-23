from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def build_cmd(
    model_path: Path,
    draft_model_path: Path,
    host: str,
    port: int,
    n_gpu_layers: int,
    n_draft: int,
    ctx_size: int,
) -> list[str]:
    return [
        "llama-server",
        "--model", str(model_path),
        "--model-draft", str(draft_model_path),
        "--host", host,
        "--port", str(port),
        "--n-gpu-layers", str(n_gpu_layers),
        "--spec-draft-n-max", str(n_draft),
        "--ctx-size", str(ctx_size),
        "--cont-batching",
        "--threads-http", "4",
    ]


def start(cmd: list[str], log_path: Path) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "wb")
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        cmd, stdout=log_file, stderr=log_file
    )
    log_file.close()
    return proc


def wait_ready(host: str, port: int, timeout: float = 30.0) -> None:
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise TimeoutError(f"llama.cpp not ready at {url} after {timeout}s")


def stop(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
