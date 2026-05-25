#!/usr/bin/env bash
# Defense evaluation: profile each defense configuration then generate
# a comparison table against the clean undefended baseline.
#
# Prerequisites:
#   - llama-server running on :8080
#   - nginx with serve/nginx_defend.conf loaded (:8444 and :8445 active)
#   - Clean undefended data already in data/raw_clean/
#
# Usage:
#   bash tools/eval_defenses.sh [--tpq N] [--window-ms W]
#
# Defaults: --tpq 30 --window-ms 3.5
set -euo pipefail
cd "$(dirname "$0")/.."

TPQ=30
WINDOW_MS=3.5
TEMP=0.3

while [[ $# -gt 0 ]]; do
    case $1 in
        --tpq)       TPQ="$2";       shift 2 ;;
        --window-ms) WINDOW_MS="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

echo "==> Defense evaluation  tpq=$TPQ  window_ms=$WINDOW_MS  temp=$TEMP"
echo ""

# Verify prerequisites.
curl -sf http://127.0.0.1:8080/health > /dev/null \
    || { echo "ERROR: llama-server not running on :8080"; exit 1; }
curl -sk https://server.local:8444/ > /dev/null \
    || { echo "ERROR: nginx :8444 not responding -- is nginx_defend.conf installed?"; exit 1; }
curl -sk https://server.local:8445/ > /dev/null \
    || { echo "ERROR: nginx :8445 not responding -- is nginx_defend.conf installed?"; exit 1; }
curl -sk https://server.local:8446/ > /dev/null \
    || { echo "ERROR: nginx :8446 not responding -- is nginx_defend.conf installed?"; exit 1; }

kill_port() {
    # Kill any process currently listening on the given TCP port.
    local port="$1"
    local pid
    pid=$(ss -tlnp "sport = :$port" 2>/dev/null \
          | awk 'NR>1 && /LISTEN/ {match($0,/pid=([0-9]+)/,a); if(a[1]) print a[1]}' \
          | head -1)
    if [[ -n "$pid" ]]; then
        echo "==> killing stale process $pid on port $port"
        kill "$pid" 2>/dev/null || true
        sleep 0.5
    fi
}

run_config() {
    local name="$1"
    local port="$2"
    local proxy_cmd="$3"
    local out_dir="data/raw_defend/$name/temp_$TEMP"

    if [[ -f "$out_dir/manifest.jsonl" ]]; then
        local n
        n=$(wc -l < "$out_dir/manifest.jsonl")
        local expected=$(( 50 * TPQ ))
        if [[ "$n" -ge "$expected" ]]; then
            echo "==> $name: already complete ($n entries), skipping"
            return
        fi
    fi

    # Kill any stale proxy on the upstream port before starting a fresh one.
    # The upstream port is always proxy_port = nginx_port - 362 (8444->8082,
    # 8445->8083, 8446->8084). Derive it from the nginx port.
    local proxy_port=$(( port - 362 ))
    kill_port "$proxy_port"

    echo "==> $name: starting proxy (port $proxy_port, nginx port $port)..."
    # Run without 2>/dev/null so bind errors are visible.
    eval "$proxy_cmd" &
    local proxy_pid=$!
    # Wait for aiohttp to bind; fail loud if the proxy exited immediately.
    sleep 1
    if ! kill -0 "$proxy_pid" 2>/dev/null; then
        echo "ERROR: proxy for $name failed to start -- aborting"
        exit 1
    fi
    # Verify the proxy is actually accepting connections.
    if ! ss -tlnp "sport = :$proxy_port" 2>/dev/null | grep -q LISTEN; then
        echo "ERROR: proxy for $name not listening on :$proxy_port -- aborting"
        kill "$proxy_pid" 2>/dev/null || true
        exit 1
    fi

    echo "==> $name: profiling 50 prompts × $TPQ runs..."
    .venv/bin/python tools/run_profile.py \
        --temperature "$TEMP" \
        --tpq "$TPQ" \
        --port "$port" \
        --out-dir "$out_dir" \
        2>&1

    echo "==> $name: stopping proxy..."
    kill "$proxy_pid" 2>/dev/null || true
    wait "$proxy_pid" 2>/dev/null || true
    echo "==> $name: done"
    echo ""
}

# Aggregation sweep
run_config "agg_batch2"     8444 ".venv/bin/python -m defend.aggregate --batch-size 2  "
run_config "agg_batch4"     8444 ".venv/bin/python -m defend.aggregate --batch-size 4  "
run_config "agg_batch8"     8444 ".venv/bin/python -m defend.aggregate --batch-size 8  "

# Random padding sweep
run_config "pad_rand128"    8445 ".venv/bin/python -m defend.pad --mode random --max-pad 128  "
run_config "pad_rand256"    8445 ".venv/bin/python -m defend.pad --mode random --max-pad 256  "
run_config "pad_rand512"    8445 ".venv/bin/python -m defend.pad --mode random --max-pad 512  "

# Fixed padding -- 1500 and 2048 (2048 covers 100% of observed event sizes)
run_config "pad_fixed1500"  8445 ".venv/bin/python -m defend.pad --mode fixed --fixed-size 1500  "
run_config "pad_fixed2048"  8445 ".venv/bin/python -m defend.pad --mode fixed --fixed-size 2048  "

# CBR chunk streaming -- burst (all-at-once) and fixed rate
run_config "cbr_burst"      8446 ".venv/bin/python -m defend.cbr --chunk-size 512 --interval-ms 0   "
run_config "cbr_512_20ms"   8446 ".venv/bin/python -m defend.cbr --chunk-size 512 --interval-ms 20  "

echo "==> All profiling complete. Generating comparison table..."
.venv/bin/python tools/compare_defenses.py \
    --window-ms "$WINDOW_MS" \
    --tpq "$TPQ" \
    --out analysis/defense_comparison.csv
