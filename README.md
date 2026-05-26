# LLM Side-Channels

Validation of side-channel attacks against LLMs and the corresponding defenses.

## LLM Side-Channel Experiment Rig

Passive network side-channel attacks on streaming LLMs. This repo reproduces
the core query-fingerprinting experiment from Wei et al. (arXiv:2411.01076)
and related work, targeting speculative decoding as the signal source. A
passive adversary monitoring encrypted TLS traffic can fingerprint which of 50
medical queries a user sent -- with 96.8% accuracy (LightGBM) at temperature
0.3 -- by observing per-iteration packet sizes, without decrypting anything.

## Papers

**[1] Carlini & Nasr (2024). Remote Timing Attacks on Efficient Language Model Inference.**
[arXiv:2410.17175](https://arxiv.org/abs/2410.17175). Google DeepMind.

The founding paper. Shows that speculative decoding and other efficient
inference techniques introduce data-dependent timing characteristics: a model
runs faster or slower depending on the difficulty of each token. A passive
network adversary monitoring encrypted traffic can learn the topic of a user's
conversation (e.g., medical advice vs. coding assistance) with 90%+ precision
against production systems including ChatGPT and Claude. With black-box access
to open-source systems, an active adversary can recover PII (phone numbers,
credit card numbers) placed in prompts.

**[2] Wei et al. (2025). When Speculation Spills Secrets: Side Channels via Speculative Decoding in LLMs.**
[arXiv:2411.01076](https://arxiv.org/abs/2411.01076). University of Toronto.

The primary paper this repo reproduces. Correct speculations generate multiple
tokens per iteration (larger TLS records); incorrect speculations fall back to
one token (smaller records). By observing the sequence of per-iteration packet
sizes, an adversary can fingerprint queries from a set of 50 prompts with over
75% accuracy across four speculative decoding schemes (REST 100%, LADE 91.6%,
BiLD 95.2%, EAGLE 77.6%) at temperature 0.3. Also demonstrates leaking
confidential data-store contents at rates exceeding 25 tokens per second.

**[3] McDonald & Bar Or (2025). Whisper Leak: A Side-Channel Attack on Large Language Models.**
[arXiv:2511.03675](https://arxiv.org/abs/2511.03675). Microsoft.

Broadest evaluation: 28 production LLMs from major providers, up to 21,716
queries per model. Trains LightGBM, LSTM, and BERT classifiers on packet size
and timing sequences to detect whether a conversation matches a sensitive
target topic. Achieves greater than 98% AUPRC for most models, with 100%
precision at 5-20% recall under a 10,000:1 noise-to-target imbalance. Proposes
random padding, token batching, and packet injection as partial mitigations.

## Results

### Experiment 1 -- Wei et al. query fingerprinting

Reproduces the Wei et al. query-fingerprinting attack (paper Figure 3) on our
local llama.cpp setup. 50 MedAlpaca prompts, 30 traces per query, 25 train /
5 test traces per class.

Three classifiers have been evaluated: Random Forest (paper baseline), LightGBM
(primary per McDonald & Bar Or), and BiLSTM (200 epochs, 2-layer bidirectional,
hidden=128 per direction).

### TPQ sweep -- accuracy vs traces per query

![TPQ sweep -- three classifiers](analysis/fig1_tpq_classifiers.png)
![TPQ sweep -- RF across temperatures](analysis/fig2_tpq_temperatures.png)

**Random Forest**

| temp | tpq=5 | tpq=10 | tpq=20 | tpq=30 |
|------|-------|--------|--------|--------|
| 0.3  | 0.892 | 0.940  | 0.940  | **0.956** |
| 0.6  | 0.692 | 0.784  | 0.852  | 0.860  |
| 0.8  | 0.596 | 0.704  | 0.808  | 0.804  |
| 1.0  | 0.544 | 0.644  | 0.712  | 0.744  |

**LightGBM**

| temp | tpq=5 | tpq=10 | tpq=20 | tpq=30 |
|------|-------|--------|--------|--------|
| 0.3  | 0.804 | 0.912  | 0.936  | **0.968** |
| 0.6  | 0.580 | 0.792  | 0.884  | 0.920  |
| 0.8  | 0.584 | 0.724  | 0.868  | 0.884  |
| 1.0  | 0.488 | 0.624  | 0.768  | 0.812  |

**BiLSTM** (2-layer bidirectional, hidden=128, 200 epochs, Adam lr=1e-3)

| temp | tpq=5 | tpq=10 | tpq=20 | tpq=30 |
|------|-------|--------|--------|--------|
| 0.3  | 0.088 | 0.412  | 0.612  | **0.704** |
| 0.6  | 0.076 | 0.156  | 0.396  | 0.564  |
| 0.8  | 0.076 | 0.136  | 0.380  | 0.416  |
| 1.0  | 0.064 | 0.188  | 0.416  | 0.524  |

Paper reports ~100% for REST-style speculative decoding at temperature 0.3.
Our RF result of 95.6% and LightGBM result of 96.8% at tpq=30 are consistent
after accounting for hardware differences (A100 vs dual RTX 3070, remote server
vs loopback).

### Classifier comparison

At tpq=30, temp=0.3: LightGBM (0.968) > RF (0.956) > BiLSTM (0.704).

LightGBM outperforms RF by 1-6% across all temperatures; the gap widens at
higher temperatures where the signal is noisier. BiLSTM peaks at 0.704 and
performs near-randomly at tpq=5 (0.088). This mirrors the sample-efficiency
literature: with ~25 training traces per class, tree ensembles' axis-aligned
splits on individual features outperform recurrent networks' temporal modeling.
The BiLSTM trains to near-zero training loss by epoch 200 but generalizes
poorly, indicating that at this dataset size the model overfits rather than
learning a generalizable representation. The BiLSTM advantage reported in
McDonald & Bar Or likely requires larger training sets (they used 21,716
queries per model).

### Per-class breakdown at temp=0.3, tpq=30

43 of 50 prompts classify perfectly. The 7 errors are concentrated in two
structurally similar pairs:

- **Antisocial personality disorder** (urgent) confused with **Avoidant
  personality disorder** (urgent) -- 3 of 5 test traces misclassified. Both
  are short psychiatric "when to seek care" responses with nearly identical
  response lengths and iteration profiles.
- Three other prompts each produce one error against a prompt of similar
  response length in the same question template.

The accuracy degradation at higher temperatures is expected: at temperature
1.0, the draft model's speculative tokens are accepted less consistently,
making per-iteration byte counts noisier across repeated captures of the same
prompt.

### Defense evaluation

![Defense comparison](analysis/fig3_defense_comparison.png)

All defenses evaluated at temp=0.3, tpq=30, RF classifier. Overhead is
mean server-to-client bytes relative to undefended baseline.

| Defense | Accuracy | Reduction | Overhead |
|---------|----------|-----------|----------|
| undefended | 0.956 | -- | 1.00x |
| agg batch=2 | 0.940 | 0.016 | 0.96x |
| agg batch=4 | 0.936 | 0.020 | 0.96x |
| agg batch=8 | 0.944 | 0.012 | 0.96x |
| pad rand=128 | 0.912 | 0.044 | 1.17x |
| pad rand=256 | 0.932 | 0.024 | 1.18x |
| pad rand=512 | 0.936 | 0.020 | 1.18x |
| pad fixed=1500 | 0.932 | 0.024 | 1.17x |
| pad fixed=2048 | 0.936 | 0.020 | 1.17x |
| **cbr burst** | **0.020** | **0.936** | **0.89x** |
| **cbr 512/20ms** | **0.020** | **0.936** | **0.91x** |

**Token aggregation** (batching N SSE events) is nearly useless: max 2%
reduction at any batch size. Our byte-size signal is additive -- batching N
iterations produces one packet whose total bytes still correlate with the
sum of the N per-iteration token counts.

**Packet padding** (random or fixed size) is also ineffective: max 4.4%
reduction even at max_pad=128. Padding is applied at the SSE event level,
but each decode iteration produces N SSE events (one per accepted speculative
token). The feature extractor sums all bytes within window_ms=3.5ms into a
single per-iteration observation, so the measured value is N x padding_size.
This still encodes the acceptance count N -- padding scales the signal but
does not destroy it. Fixed=2048 covers the full observed event-size range yet
only reduces accuracy by 2% for exactly this reason.

**CBR (constant-bit-rate) streaming** is the only effective defense: both
variants reduce accuracy from 95.6% to 2.0% (chance for 50 classes).
`cbr burst` buffers the complete response and sends everything at once;
the feature extractor collapses all bytes into a single iteration and the
only remaining signal is total response length, which is not sufficient for
50-class classification. `cbr 512/20ms` sends 512-byte chunks at 20ms
intervals, producing a flat constant-value feature vector. Both destroy the
per-iteration structure that the classifier depends on.

CBR costs no extra bandwidth (0.89-0.91x overhead vs 1.17-1.18x for
padding) but adds latency equal to full generation time -- the client
receives nothing until the model finishes generating. For a 200-token
response at ~15 tokens/sec that is roughly 13 seconds of added latency.

### Experiment 2 -- Carlini & Nasr timing-based binary disambiguation

Reproduces the Carlini & Nasr GMM attack on our local llama.cpp setup.
Features: first 50 inter-packet gaps (server->client, ms) per trace. Two
GMMs trained on 20 traces per prompt (one per hypothesis), classified by
log-likelihood ratio.

![AUPRC distribution](analysis/fig5_carlini_auprc_dist.png)
![Precision-recall curves](analysis/fig4_carlini_pr_curves.png)

Evaluated across all 1,176 prompt pairs from the 50-prompt dataset
(temp=0.3, 30 traces per prompt, 20 train / 10 test):

| Metric | Value |
|--------|-------|
| Pairs evaluated | 1,176 |
| Median AUPRC | 0.881 |
| Mean AUPRC | 0.840 |
| Pairs ≥ 0.90 AUPRC | 545 / 1,176 (46%) |
| Pairs ≥ 0.75 AUPRC | 875 / 1,176 (74%) |
| Best pair AUPRC | 1.000 |
| Worst pair AUPRC | 0.200 |

The distribution is heavily right-skewed: the majority of pairs cluster near
1.0, with a tail of hard pairs where the two prompts produce similar response
lengths and iteration patterns. The worst pair (prompts 12 vs 29) has nearly
identical timing profiles; the best pairs (AUPRC=1.0) involve prompts with
very different response lengths or speculative acceptance rates.

Note: the paper used 100 inter-packet gaps and 100 training traces per
hypothesis on remote commercial APIs. We use 50 gaps (shorter local responses
saturate 100-gap requirement) and 20 training traces. Despite smaller training
sets, the inter-packet timing signal is clearly present on llama.cpp with
speculative decoding.

### Experiment 3 -- McDonald & Bar Or topic inference

Reproduces the McDonald & Bar Or LightGBM topic-detection pipeline on our
local llama.cpp setup. Target topic: 50 Python programming questions (10
traces each). Negative set: 50 MedAlpaca medical questions (30 traces each).
Features: first 50 (packet_size, inter_arrival_ms) pairs flattened to a
100-dim vector. 80/20 train/test split per class.

![Topic inference PR curve](analysis/fig6_mcdonald_pr_curve.png)
![AUPRC vs imbalance](analysis/fig7_mcdonald_imbalance.png)

| Metric | Value |
|--------|-------|
| Target traces | 495 (50 prompts × ~10 runs) |
| Negative traces | 1,467 (50 prompts × 30 runs) |
| AUPRC (balanced 1:1) | **0.986** |
| AUPRC at 5:1 imbalance | 0.990 |
| AUPRC at 10:1 imbalance | 0.987 |
| AUPRC at 14:1 imbalance | 0.984 |

The (size, timing) feature representation cleanly separates the two topics:
Python programming responses contain code blocks and structured explanations,
producing a characteristic pattern of large initial packets (code) followed
by smaller continuation packets. Medical responses have different pacing.
AUPRC remains above 0.984 up to 14:1 negative:positive imbalance, consistent
with the paper's finding of >0.98 AUPRC across 28 production LLMs.

Note: the paper trained on 100 prompt variants and evaluated against 11,716
Quora question negatives (117:1 imbalance). Our dataset is smaller; the
topic boundary (programming vs. medical) is also more distinct than the
paper's "money laundering legality" vs. general Quora questions, which
explains the high AUPRC even at moderate training set sizes.

## How It Works

### The signal source

llama.cpp runs speculative decoding: a small draft model (Qwen2.5-0.5B)
proposes multiple candidate tokens per iteration, and the large target model
(Qwen2.5-7B) verifies them in a single forward pass. When the draft's
predictions are correct, multiple tokens are accepted and streamed together in
one SSE chunk -- producing a large TLS record. When the draft is wrong, only
one token is accepted -- producing a small record. The number of tokens
accepted per iteration is input-dependent: "easy" content (common medical
phrases, predictable continuations) accepts more speculative tokens than
"hard" content (unusual terminology, complex reasoning).

This means the sequence of TLS record sizes for a given prompt is a
fingerprint: it encodes which parts of the response were easy vs hard for the
draft to predict, which is a function of the specific content generated.

### The feature vector

For each response, we capture the raw TLS traffic with tcpdump on the loopback
interface. Scapy parses the pcap and extracts server-to-client payload sizes
with timestamps. We group packets into decode iterations using a timing window
(`window_ms`, calibrated per-machine), sum the bytes within each window, drop
the first two iterations (constant TLS handshake and HTTP header overhead),
and pad or truncate to a fixed length of 100 iterations. The result is a
100-dimensional vector of bytes-per-iteration.

### The attack

**Offline profiling phase:** capture 25-30 traces per prompt across the 50
target queries. Build a 100-dimension feature matrix and train a Random Forest
classifier (150 trees, max depth 15).

**Online phase:** capture one trace for an unknown query. Extract its feature
vector and classify against the trained model.

### Why nginx and tcpdump matter

Two configuration details are critical for the signal to be observable:

**nginx must not buffer or compress.** With `proxy_buffering on` or `gzip on`,
nginx accumulates SSE chunks before forwarding them, collapsing multiple
per-iteration records into a single packet. The classifier sees a flat signal
and accuracy collapses to chance. The config uses `proxy_buffering off`,
`gzip off`, and `tcp_nodelay on`.

**tcpdump must use `--immediate-mode`.** Without it, libpcap (TPACKET_V3)
accumulates packets into ring-buffer blocks and only delivers them when a
block fills or a ~64ms retirement timer fires. Short responses can complete
before any block retires, yielding zero captured packets. `--immediate-mode`
forces per-packet delivery.

## Dependencies

**System packages** (installed by `tools/setup_once.sh`):
- `tcpdump` -- packet capture; requires the running user to be in the `pcap`
  group or have `CAP_NET_RAW`
- `nginx` -- TLS termination reverse proxy
- `llama-server` -- from [llama.cpp](https://github.com/ggerganov/llama.cpp);
  must be on `$PATH`

**Models** (pulled via ollama, paths copied into `tools/start_llama.sh`):
- `qwen2.5:7b-instruct-q4_K_M` -- target model (~4.7 GB)
- `qwen2.5:0.5b-instruct-q8_0` -- draft model for speculative decoding (~530 MB)

**Python packages** (managed by [uv](https://github.com/astral-sh/uv), declared in `pyproject.toml`):

| Package | Use |
|---------|-----|
| `scapy` | Offline pcap parsing |
| `httpx` | HTTPS streaming client for LLM queries |
| `aiohttp` | Async HTTP server/client for the defense proxy |
| `scikit-learn` | RandomForest classifier |
| `numpy` | Feature array construction |
| `pandas` | TPQ sweep result tables |
| `lightgbm` | LightGBM classifier |
| `torch` | BiLSTM classifier (CUDA 12.1 build) |

Dev extras (`uv sync --group dev`): `pytest`, `mypy`, `pytest-mock`.

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
| Classifier | RandomForest, LightGBM, BiLSTM | RF matches paper §4.4; LightGBM and BiLSTM follow McDonald & Bar Or |
| Environment | Python 3.11+, uv | Reproducible, fast |

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

Starts llama.cpp with speculative decoding (Qwen2.5-7B target, Qwen2.5-0.5B
draft), full GPU offload, and 4 HTTP threads. Logs go to `logs/llama.log`.
Confirm speculative decoding is active before proceeding:

```bash
grep -i draft logs/llama.log
```

### 2. Calibrate window_ms

```bash
bash tools/calibrate.sh
```

Captures one query and computes `window_ms`: the gap threshold that separates
within-iteration inter-event gaps (sub-millisecond) from between-iteration
inference pauses (tens of milliseconds). The algorithm finds the largest ratio
between consecutive sorted inter-packet gaps, then returns the geometric mean
of the bounding gaps. The value is hardware-dependent and should be
recalibrated if the server or models change.

### 3. Smoke test

```bash
uv run python tools/smoke_test.py --window-ms <value>
```

Profiles 3 prompts x 5 runs, builds features, trains a RandomForest, and
checks that accuracy is at least 10 percentage points above chance (0.33 for 3
classes). A passing smoke test confirms the full capture-to-classifier pipeline
is working before committing to a multi-hour profiling session. If accuracy is
near chance, check pcap sizes (`ls -lh data/smoke/*.pcap`) and recalibrate.

### 4. Full profiling

```bash
uv run python tools/run_profile.py --temperature 0.3 --tpq 30
uv run python tools/run_profile.py --temperature 0.6 --tpq 30
uv run python tools/run_profile.py --temperature 0.8 --tpq 30
uv run python tools/run_profile.py --temperature 1.0 --tpq 30
```

Captures `n_prompts * tpq` pcap files and writes a `manifest.jsonl` per
temperature. Estimated time is printed before each run starts (~5 s per query
at temperature 0.3 with speculative decoding active, ~8 hours for all four
temperatures at tpq=30).

### 5. TPQ sweep

Merge the per-temperature manifests into a combined manifest, then run the
sweep. The combined manifest is required because `tpq_sweep.py` filters by
temperature internally and each per-temperature manifest only contains one
temperature's data.

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

Trains and evaluates each classifier at each (temperature, TPQ) combination,
following the paper's protocol: train on the first `tpq` traces per class,
test on the remaining 5. Results are written to `analysis/`.

### 6. Prompt diversity analysis

```bash
uv run python tools/analyze_prompt_diversity.py \
    --manifest data/raw_clean/manifest_all.jsonl \
    --window-ms <value>
```

Reports per-prompt response length statistics, groups prompts by question
template, and computes within-template vs across-template cosine similarity.
Use this to check whether the prompt set has enough response-length diversity
before committing to a full profiling run. The concern is that structurally
identical templates (e.g. 14 "When to seek urgent medical care" questions)
may generate responses of similar length, collapsing those classes in feature
space. For this dataset, template separation is +0.030 -- well within
acceptable range.

## Key Implementation Notes

**Prompt ordering matters.** The first prompts in `collect/data/exp1_prompts.jsonl`
are ordered for response-length diversity: a short causal question, a medium
symptoms-list question, and a long prognosis question. Structurally similar
prompts in the first few positions produce low cosine separation between traces
and degrade the smoke test toward chance.

**Skip leading iterations.** `features/parse.py:trace_from_pcap` drops the
first two grouped iterations by default. These correspond to the TLS handshake
record and the HTTP response headers, which are constant across all requests
and carry no discriminating signal.

**min_samples_split.** The RandomForest in `attack/train.py` uses
`min_samples_split=2` rather than the paper's value of 10. The paper's value
was tuned for datasets with thousands of training samples; with fewer training
samples per class it prevents any tree node from splitting, collapsing all
predictions to the majority class and accuracy to chance.

**Thread count.** `serve/launch.py` passes `--threads-http 4` to llama-server.
The default of n_cpu-1 (47 threads on this machine) combined with 24 inference
threads caused CCD scheduling thrash on the Threadripper and locked up the
machine during early testing.

## Tests

```bash
uv run pytest
uv run mypy --strict .
```

## References

- Carlini, N. & Nasr, M. (2024). Remote Timing Attacks on Efficient Language
  Model Inference. [arXiv:2410.17175](https://arxiv.org/abs/2410.17175).
- Wei, J., Abdulrazzag, A., Zhang, T., Muursepp, A. & Saileshwar, G. (2025).
  When Speculation Spills Secrets: Side Channels via Speculative Decoding in
  LLMs. [arXiv:2411.01076](https://arxiv.org/abs/2411.01076).
- McDonald, G. & Bar Or, J. (2025). Whisper Leak: A Side-Channel Attack on
  Large Language Models. [arXiv:2511.03675](https://arxiv.org/abs/2511.03675).
