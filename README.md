# LLM Side-Channel Experiment Rig

Passive network side-channel attacks on streaming LLMs. This repo reproduces
the core experiments from three papers listed below, targeting speculative
decoding as the signal source. A passive adversary monitoring encrypted TLS
traffic can fingerprint user queries with high accuracy by observing
per-iteration packet sizes and inter-arrival gaps, without decrypting anything.

## Papers

**[1] Carlini & Nasr (2024). Remote Timing Attacks on Efficient Language Model Inference.**
arXiv:2410.17175. Google DeepMind.

The founding paper. Shows that speculative decoding and other efficient
inference techniques introduce data-dependent timing characteristics: a model
runs faster or slower depending on the difficulty of each token. A passive
network adversary monitoring encrypted traffic can learn the topic of a user's
conversation (e.g., medical advice vs. coding assistance) with 90%+ precision
against production systems including ChatGPT and Claude. With black-box access
to open-source systems, an active adversary can recover PII (phone numbers,
credit card numbers) placed in prompts.

**[2] Wei et al. (2025). When Speculation Spills Secrets: Side Channels via Speculative Decoding in LLMs.**
arXiv:2411.01076. University of Toronto.

Focuses specifically on speculative decoding as the signal source. Correct
speculations generate multiple tokens per iteration (larger packets); incorrect
speculations fall back to one token (smaller packets). By observing the
sequence of packet sizes, an adversary can fingerprint queries from a set of
50 prompts with over 75% accuracy across four speculative decoding schemes
(REST 100%, LADE 91.6%, BiLD 95.2%, EAGLE 77.6%). Also demonstrates leaking
confidential data-store contents at rates exceeding 25 tokens per second.

**[3] McDonald & Bar Or (2025). Whisper Leak: A Side-Channel Attack on Large Language Models.**
arXiv:2511.03675. Microsoft.

Broadest evaluation: 28 production LLMs from major providers, up to 21,716
queries per model. Trains LightGBM, LSTM, and BERT classifiers on packet size
and timing sequences to detect whether a conversation matches a sensitive
target topic. Achieves greater than 98% AUPRC for most models, with 100%
precision at 5-20% recall under a 10,000:1 noise-to-target imbalance. Proposes
random padding, token batching, and packet injection as partial mitigations.

## Hardware

- AMD Threadripper (48 logical cores), 64 GB RAM
- 2x RTX 3070 8 GB (16 GB VRAM total)
- Single-machine setup: network "remote" is simulated via nginx reverse proxy
  and tcpdump on loopback

## Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| LLM server | llama.cpp | Cleaner SSE chunking and speculative decoding at 16 GB VRAM |
| Target model | Qwen2.5-7B-Instruct Q4_K_M | Fits in VRAM with headroom for draft |
| Draft model | Qwen2.5-0.5B-Instruct Q8 | Fast enough to sustain useful speculation rate |
| TLS termination | nginx with self-signed cert | Realistic TLS boundary on server.local:8443 |
| Packet capture | tcpdump + scapy | tcpdump for raw capture, scapy for offline parsing |
| Classifier | RandomForest (sklearn), LightGBM | Matches paper hyperparameters; BiLSTM next |
| Environment | Python 3.11+, uv | Reproducible, fast |

**nginx is configured with `tcp_nodelay on` and `proxy_buffering off`** so each
SSE chunk from llama.cpp becomes its own TLS record without Nagle coalescing.

**tcpdump is invoked with `--immediate-mode`** to bypass TPACKET_V3 ring-buffer
block retirement. Without this flag, libpcap batches packets into 64ms blocks
and short responses complete before any block retires, yielding zero captures.

## Repo Layout

```
serve/          nginx config, llama.cpp launch helpers
collect/        capture orchestration, prompt datasets, query client
  data/         prompt JSONL files (committed; pcap data is gitignored)
features/       pcap parsing, TLS record extraction, feature builders
attack/         classifiers, training pipeline, evaluation metrics
analysis/       CSVs from tpq sweeps and accuracy comparisons
docs/           experiment plans, runbooks, paper PDFs
tools/          scripts for setup, calibration, smoke test, profiling
data/           pcap files and parquet feature files (gitignored)
logs/           llama.cpp and nginx logs (gitignored)
```

## Setup

**One-time system setup** (requires root for tcpdump group and nginx TLS cert):

```bash
sudo bash tools/setup_once.sh
```

This installs tcpdump, nginx, creates the self-signed cert for server.local,
and adds your user to the pcap group.

**Python environment:**

```bash
uv sync
```

## Running an Experiment

### 1. Start the LLM server

```bash
bash tools/start_llama.sh
```

This starts llama.cpp with speculative decoding (Qwen2.5-7B target,
Qwen2.5-0.5B draft), GPU offload, and 4 HTTP threads. Logs go to
`logs/llama.log`. Confirm speculative decoding is active:

```bash
grep -i draft logs/llama.log
```

### 2. Calibrate window_ms

```bash
bash tools/calibrate.sh
```

Captures one query and computes `window_ms`, the gap threshold that separates
within-iteration TCP fragmentation from between-iteration inference pauses.
The calibration finds the valley between the two gap clusters using the largest
ratio between consecutive sorted gaps, then returns the geometric mean of the
valley bounds. Copy the printed `window_ms` value for the next steps.

### 3. Smoke test

```bash
uv run python tools/smoke_test.py --window-ms <value>
```

Profiles 3 prompts x 5 runs, builds features, trains a RandomForest, and
checks that accuracy is at least 10 percentage points above chance (0.33 for 3
classes). A passing smoke test confirms the full capture-to-classifier pipeline
is working before committing to a 1500-query profiling session.

### 4. Full profiling

```bash
uv run python tools/profile.py --temperature 0.3 --tpq 30
uv run python tools/profile.py --temperature 0.6 --tpq 30
```

Captures `n_prompts * tpq` pcap files and writes a `manifest.jsonl`.
Estimated time is printed before the run starts (~5 s per query at temperature
0.3 with speculative decoding active).

### 5. TPQ sweep

```bash
uv run python tools/tpq_sweep.py --manifest data/raw/temp_0.3/manifest.jsonl \
    --window-ms <value>
```

Evaluates accuracy as a function of traces-per-query across multiple
temperatures. Results are written to `analysis/`.

## Key Implementation Notes

**Prompt ordering matters.** The first prompts in `collect/data/exp1_prompts.jsonl`
are ordered for response-length diversity (short causal, medium symptoms list,
long prognosis). Structurally similar prompts that generate similar-length
responses produce low cosine separation between traces and degrade classifier
accuracy toward chance.

**Skip leading iterations.** `features/parse.py:trace_from_pcap` drops the
first two grouped iterations by default. These correspond to the TLS handshake
record and the HTTP response headers, which are constant across all requests
and carry no discriminating signal.

**min_samples_split.** The RandomForest in `attack/train.py` uses
`min_samples_split=2` rather than the paper's value of 10. The paper's value
was tuned for datasets with thousands of training samples; at small scale it
prevents any tree from splitting, collapsing accuracy to chance.

## Tests

```bash
uv run pytest
uv run mypy --strict .
```

## References

- Carlini, N. & Nasr, M. (2024). Remote Timing Attacks on Efficient Language
  Model Inference. arXiv:2410.17175.
- Wei, J., Abdulrazzag, A., Zhang, T., Muursepp, A. & Saileshwar, G. (2025).
  When Speculation Spills Secrets: Side Channels via Speculative Decoding in
  LLMs. arXiv:2411.01076.
- McDonald, G. & Bar Or, J. (2025). Whisper Leak: A Side-Channel Attack on
  Large Language Models. arXiv:2511.03675.
