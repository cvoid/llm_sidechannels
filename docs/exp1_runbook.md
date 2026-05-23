# Experiment 1 — Runbook
## Wei et al. query fingerprinting on llama.cpp

Model management uses ollama for downloads. Inference uses llama-server
directly — ollama doesn't expose the `--model-draft` flag needed for
speculative decoding, which is the source of the attack signal.

---

## One-time setup

```bash
bash tools/setup_once.sh
```

This installs Python deps, pulls both models via ollama, generates the
self-signed TLS cert, adds `server.local` to `/etc/hosts`, installs the
nginx config, and grants tcpdump the network capture capability.

At the end it prints the GGUF blob paths. Copy them into `tools/start_llama.sh`
before continuing.

---

## Per-experiment run

**1. Start llama-server** (leave this terminal open)
```bash
bash tools/start_llama.sh
```

Confirm it's up:
```bash
curl -s http://127.0.0.1:8080/health
# expect: {"status":"ok"}
```

**2. Calibrate** (one-time per hardware setup)
```bash
bash tools/calibrate.sh
```

Note the printed `window_ms`. Pass it to every subsequent step.

**3. Smoke test** — run before committing to full profiling
```bash
uv run python tools/smoke_test.py --window-ms <value>
```

Expect accuracy > 0.50. If it's near 0.33 (chance), traces are empty —
check pcap sizes and re-run calibration.

**4. Offline profiling**

Primary temperature (paper §4.4):
```bash
uv run python tools/profile.py --temperature 0.3 --tpq 30
```

Ablation temperatures:
```bash
for T in 0.6 0.8 1.0; do
    uv run python tools/profile.py --temperature $T --tpq 30
done
```

> **Time estimate:** 50 prompts × 30 runs = 1500 queries per temperature.
> At ~5 s/query expect ~2 h per temperature.

**5. TPQ sweep** — replicates paper Figure 3
```bash
uv run python tools/tpq_sweep.py \
    --manifest  data/raw/temp_0.3/manifest.jsonl \
    --window-ms <value>
```

Output columns: `temperature`, `tpq`, `accuracy`, `f1_macro`. Results saved
to `analysis/exp1_tpq_sweep_temp_0.3.csv`. The paper reports ~70–100%
accuracy for REST at TPQ=30; our llama.cpp + draft model setup should land
in the LADE/BiLD range (~65–100% depending on temperature).
