#!/usr/bin/env bash
# Capture one query and compute window_ms for the current hardware setup.
# Run from the repo root with llama-server already running.
# Prints window_ms — pass it to tools/profile.py and tools/tpq_sweep.py.
set -euo pipefail
cd "$(dirname "$0")/.."

PCAP=data/calibration/cal.pcap
mkdir -p data/calibration

echo "==> Checking llama-server health (:8080)..."
curl -sf http://127.0.0.1:8080/health > /dev/null \
    || { echo "ERROR: llama-server not responding at :8080 — is it running?"; exit 1; }

echo "==> Checking nginx TLS proxy (:8443)..."
curl -sk https://server.local:8443/ > /dev/null \
    || { echo "ERROR: cannot connect to nginx at :8443."; \
         echo "  sudo systemctl status nginx"; \
         echo "  sudo systemctl start nginx"; \
         echo "  sudo nginx -t"; \
         exit 1; }

echo "==> Starting capture on lo:8443..."
tcpdump -i lo -w "$PCAP" 'tcp port 8443' &
TCPDUMP_PID=$!

echo "==> Sending calibration query..."
uv run python -c "
from collect.query import send
from collect.run import MEDICAL_SYSTEM_PROMPT
send('What are the symptoms of Common cold?', 'server.local', system_prompt=MEDICAL_SYSTEM_PROMPT)
print('query complete')
"

sleep 0.5
kill "$TCPDUMP_PID"
wait "$TCPDUMP_PID" 2>/dev/null || true

echo ""
echo "==> Analysing capture..."
uv run python -c "
from pathlib import Path
from features.parse import calibrate_window, extract_records

pcap = Path('$PCAP')
records = extract_records(pcap)

if len(records) < 2:
    print('ERROR: fewer than 2 packets captured.')
    print('  - Is nginx running and proxying to :8080?')
    print('  - Did the query reach the server?')
    raise SystemExit(1)

gaps = [(records[i+1][0] - records[i][0]) * 1000 for i in range(len(records) - 1)]
window_ms = calibrate_window(pcap)

print(f'packets  : {len(records)}')
print(f'gaps ms  : min={min(gaps):.2f}  max={max(gaps):.2f}  mean={sum(gaps)/len(gaps):.2f}')
print(f'window_ms: {window_ms:.2f}')
print()
if len(records) < 10:
    print('WARNING: low packet count — speculative decoding may not be active.')
    print('  Check: grep -i draft logs/llama.log')
else:
    print('Use --window-ms', round(window_ms, 2), 'with tools/smoke_test.py and tools/profile.py')
"
