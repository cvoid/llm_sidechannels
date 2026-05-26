# Experiment Runbook

Covers all three experiments. Run them in order -- each builds on data from
the previous one.

Model management uses ollama for downloads. Inference uses llama-server
directly; ollama does not expose the `--model-draft` flag needed for
speculative decoding.

---

## One-time setup

```bash
sudo bash tools/setup_once.sh
```

Installs tcpdump, nginx, generates the self-signed TLS cert for server.local,
adds your user to the pcap group, and installs the nginx config (four virtual
servers on ports 8443-8446).

At the end it prints the GGUF blob paths. Copy them into `tools/start_llama.sh`
before continuing.

---

## Before every session

**Start llama-server** (leave this terminal open):
```bash
bash tools/start_llama.sh
```

Confirm speculative decoding is active:
```bash
grep -i draft logs/llama.log
curl -s http://127.0.0.1:8080/health
# expect: {"status":"ok"}
```

**Start nginx** (if not running):
```bash
sudo nginx
# or reload after config changes:
sudo nginx -s reload
```

---

## Calibration (one-time per hardware setup)

```bash
bash tools/calibrate.sh
```

Captures one query on the undefended port (8443) and prints `window_ms`.
The value separates within-iteration sub-ms TCP fragmentation gaps from
between-iteration 10-100ms inference pauses. Pass it to all subsequent steps.

---

## Smoke test

```bash
uv run python tools/smoke_test.py --window-ms <value>
```

Profiles 3 prompts x 5 runs, trains a RandomForest, checks accuracy > 10pp
above chance (0.33 for 3 classes). If it fails, check pcap sizes
(`ls -lh data/smoke/*.pcap`) and recalibrate.

---

## Experiment 1 -- Wei et al. query fingerprinting

### Profiling

```bash
uv run python tools/run_profile.py --temperature 0.3 --tpq 30
uv run python tools/run_profile.py --temperature 0.6 --tpq 30
uv run python tools/run_profile.py --temperature 0.8 --tpq 30
uv run python tools/run_profile.py --temperature 1.0 --tpq 30
```

~5 s/query at temp=0.3 with speculative decoding active. Estimated time: ~2 h
per temperature, ~8 h total.

### TPQ sweep

Combine per-temperature manifests, then run all three classifiers:

```bash
cat data/raw/temp_*/manifest.jsonl > data/raw/all_temps/manifest.jsonl

uv run python tools/tpq_sweep.py \
    --manifest data/raw/all_temps/manifest.jsonl \
    --window-ms <value> \
    --out analysis/exp1_tpq_sweep_rf.csv

uv run python tools/tpq_sweep.py \
    --manifest data/raw/all_temps/manifest.jsonl \
    --window-ms <value> \
    --classifier lgbm \
    --out analysis/exp1_tpq_sweep_lgbm.csv

uv run python tools/tpq_sweep.py \
    --manifest data/raw/all_temps/manifest.jsonl \
    --window-ms <value> \
    --classifier bilstm \
    --out analysis/exp1_tpq_sweep_bilstm.csv
```

### Defense comparison

Start each defense proxy in a separate terminal, then profile against it:

```bash
# Token aggregation (port 8444):
uv run python -m defend.aggregate --batch 4

# Padding (port 8445):
uv run python -m defend.pad --mode random --max-pad 512

# CBR (port 8446):
uv run python -m defend.cbr --mode burst
```

Profile against a defense:
```bash
uv run python tools/run_profile.py \
    --port 8445 \
    --out-dir data/raw_defend/pad512/temp_0.3 \
    --temperature 0.3 --tpq 30
```

Run all 11 defense configurations via the batch script:
```bash
bash tools/eval_defenses.sh --window-ms <value>
```

### Prompt diversity analysis

```bash
uv run python tools/analyze_prompt_diversity.py \
    --manifest data/raw_clean/manifest_all.jsonl \
    --window-ms <value>
```

Reports template breakdown, cosine similarity stats, and the 10 most
confusable prompt pairs.

### Expected results (temp=0.3, tpq=30)

| Classifier | Accuracy |
|------------|----------|
| RF | 0.956 |
| LightGBM | 0.968 |
| BiLSTM | 0.704 |

---

## Experiment 2 -- Carlini & Nasr timing attack

Uses the timing features from Experiment 1's profiling data (no additional
captures needed).

```bash
uv run python tools/carlini_eval.py \
    --manifest data/raw_clean/manifest_all.jsonl \
    --temperature 0.3 \
    --n-gaps 50 \
    --train-n 20
```

Evaluates all C(50,2)=1225 prompt pairs. Writes per-pair AUPRC to
`analysis/exp2_carlini_auprc.csv` and saves PR curve figures to `analysis/`.

### Expected results

| Metric | Value |
|--------|-------|
| Pairs evaluated | 1,176 |
| Median AUPRC | 0.881 |
| Pairs >= 0.90 | 545 / 1,176 |

---

## Experiment 3 -- McDonald & Bar Or topic inference

Requires separate profiling of the target topic (Python programming questions).
The MedAlpaca data from Experiment 1 serves as the negative set.

### Profile target topic

```bash
uv run python tools/run_profile.py \
    --prompts-file collect/data/exp3_target_prompts.jsonl \
    --out-dir data/raw_exp3/temp_0.3 \
    --temperature 0.3 --tpq 10
```

### Run evaluation

```bash
uv run python tools/mcdonald_eval.py \
    --target-manifest data/raw_exp3/temp_0.3/manifest.jsonl \
    --neg-manifest data/raw_clean/manifest_all.jsonl \
    --n-pairs 50 \
    --temperature 0.3
```

Writes results to `analysis/exp3_mcdonald_summary.csv` and
`analysis/exp3_mcdonald_imbalance.csv`. Saves PR curve and imbalance figures.

### Expected results

| Metric | Value |
|--------|-------|
| AUPRC (balanced) | 0.986 |
| AUPRC at 14:1 imbalance | 0.984 |

---

## Generate plots

```bash
uv run python tools/plot_results.py
```

Writes fig1-fig7 to `analysis/`.

---

## Verification

```bash
uv run pytest            # 71 tests
uv run mypy --strict .   # 0 errors
```
