from __future__ import annotations

import subprocess
import threading
from pathlib import Path


def _drain(proc: subprocess.Popen[bytes]) -> None:
    # libpcap 1.10.4 (TPACKET_V3) requires stderr to be actively consumed
    # line-by-line during capture. Without a live reader the BPF filter counts
    # packets but tcpdump's capture loop never writes them to the pcap file.
    try:
        if proc.stderr:
            for _ in proc.stderr:
                pass
    except Exception:
        pass


def start(
    iface: str,
    pcap_path: Path,
    bpf_filter: str = "tcp port 8443",
) -> subprocess.Popen[bytes]:
    pcap_path.parent.mkdir(parents=True, exist_ok=True)
    # --immediate-mode: bypass TPACKET_V3 block batching. Without it, libpcap
    # accumulates packets into ring-buffer blocks and only delivers them when a
    # block fills or its retirement timeout fires (~64ms). Short responses can
    # complete before any block retires, yielding "N received, 0 captured".
    proc = subprocess.Popen(
        ["tcpdump", "--immediate-mode", "-i", iface, "-w", str(pcap_path), bpf_filter],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    threading.Thread(target=_drain, args=(proc,), daemon=True).start()
    return proc


def stop(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
