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

    echo "==> $name: starting proxy (port $port)..."
    eval "$proxy_cmd" &
    local proxy_pid=$!
    # Give aiohttp time to bind before the first request.
    sleep 1

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
run_config "agg_batch2"     8444 ".venv/bin/python -m defend.aggregate --batch-size 2  2>/dev/null"
run_config "agg_batch4"     8444 ".venv/bin/python -m defend.aggregate --batch-size 4  2>/dev/null"
run_config "agg_batch8"     8444 ".venv/bin/python -m defend.aggregate --batch-size 8  2>/dev/null"

# Random padding sweep
run_config "pad_rand128"    8445 ".venv/bin/python -m defend.pad --mode random --max-pad 128  2>/dev/null"
run_config "pad_rand256"    8445 ".venv/bin/python -m defend.pad --mode random --max-pad 256  2>/dev/null"
run_config "pad_rand512"    8445 ".venv/bin/python -m defend.pad --mode random --max-pad 512  2>/dev/null"

# Fixed padding -- 1500 and 2048 (2048 covers 100% of observed event sizes)
run_config "pad_fixed1500"  8445 ".venv/bin/python -m defend.pad --mode fixed --fixed-size 1500  2>/dev/null"
run_config "pad_fixed2048"  8445 ".venv/bin/python -m defend.pad --mode fixed --fixed-size 2048  2>/dev/null"

# CBR chunk streaming -- burst (all-at-once) and fixed rate
run_config "cbr_burst"      8446 ".venv/bin/python -m defend.cbr --chunk-size 512 --interval-ms 0   2>/dev/null"
run_config "cbr_512_20ms"   8446 ".venv/bin/python -m defend.cbr --chunk-size 512 --interval-ms 20  2>/dev/null"

echo "==> All profiling complete. Generating comparison table..."
.venv/bin/python tools/compare_defenses.py \
    --window-ms "$WINDOW_MS" \
    --tpq "$TPQ" \
    --out analysis/defense_comparison.csv
