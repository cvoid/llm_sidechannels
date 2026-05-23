from __future__ import annotations

import subprocess
from pathlib import Path


def start(
    iface: str,
    pcap_path: Path,
    bpf_filter: str = "tcp port 8443",
) -> subprocess.Popen[bytes]:
    pcap_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        ["tcpdump", "-i", iface, "-w", str(pcap_path), bpf_filter],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
