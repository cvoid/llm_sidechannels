# Design Decisions and Deviations from Source Papers

This document records the key design choices made when adapting the three
source papers to this local llama.cpp setup, and the places where the
implementation deviates from the papers' descriptions.

---

## Adaptation context

The three papers target different infrastructure:

| Paper | Target system | Models | Scale |
|-------|--------------|--------|-------|
| Wei et al. (2411.01076) | vLLM + EAGLE on A100, remote server | GPT-4, Llama-2-13B | 50 prompts, 30 TPQ |
| Carlini & Nasr (2410.17175) | ChatGPT, Claude, Gemini (commercial API) | Production LLMs | 50+ prompts, 100 TPQ |
| McDonald & Bar Or (2511.03675) | 28 commercial LLM APIs | GPT-4o, Claude 3, etc. | 100+ prompts, 21716 queries/model |

Our setup: llama.cpp speculative decoding on loopback, Qwen2.5-7B-Instruct
Q4_K_M target + Qwen2.5-0.5B-Instruct Q8 draft, nginx TLS termination on
server.local:8443, tcpdump on loopback.

---

## Deviations from paper specifications

### Wei et al. -- feature extraction

**Paper:** groups TLS records by timing window; does not describe how to handle
the TLS handshake or HTTP headers.

**Our implementation:** `features/parse.py:trace_from_pcap` drops the first
two grouped iterations (`skip_leading=2`). These correspond to the TLS
handshake record and the HTTP response headers, which are constant across all
requests. This is inferred from the paper's Figure 2 which shows a distinctive
large initial record; including it would add a constant feature to every trace.

**Paper:** min_samples_split=10 for the RandomForest.

**Our implementation:** `attack/train.py:build_rf` uses `min_samples_split=2`
(sklearn default). The paper's value of 10 was tuned for thousands of training
samples; with 25 samples per class it prevents all tree splits and collapses
accuracy to chance. LightGBM is the primary classifier per McDonald & Bar Or.

**Paper:** `calibrate_window` described as "percentile-based gap threshold."

**Our implementation:** `features/parse.py:calibrate_window` uses the largest
ratio between consecutive sorted gaps (geometric mean as the threshold). This
correctly handles the log-scale gap between within-iteration sub-millisecond
gaps and between-iteration 10-100ms pauses, whereas a fixed percentile breaks
if the gap distribution is uneven.

### Carlini & Nasr -- timing features

**Paper:** uses n_gaps=100 inter-packet delays as the feature vector.

**Our implementation:** `features/timing.py:extract_gaps` defaults to
n_gaps=100 but we use n_gaps=50 in practice (see `tools/carlini_eval.py`).
Local responses are shorter than commercial API responses; at n_gaps=100 only
~40% of prompts produce enough packets. At n_gaps=50, all 50 prompts are
usable.

**Paper:** trains on 100 traces per hypothesis.

**Our implementation:** 20 training traces per prompt (from 30-per-class
budget with 10 held for test). Despite the smaller training set, the timing
signal is clearly present.

### McDonald & Bar Or -- dataset

**Paper:** uses money laundering legality as the target topic; general Quora
questions as the negative set; 21,716 queries per model.

**Our implementation:** Python programming questions as the target topic;
MedAlpaca medical questions as negatives. The topic boundary is more distinct
(natural language vs code structure), which may explain the high AUPRC
(0.986) despite the much smaller dataset. Results would be more conservative
with semantically adjacent topics.

---

## Infrastructure deviations

**nginx buffering:** must be disabled (`proxy_buffering off`, `gzip off`,
`tcp_nodelay on`). With buffering on, multiple SSE chunks are batched before
delivery, collapsing the per-iteration signal. This is the single most common
failure mode during development.

**tcpdump immediate mode:** `--immediate-mode` required. Without it, libpcap's
TPACKET_V3 block retirement timer (~64ms) can delay delivery until after
short responses complete, yielding zero captured packets.

**Thread count:** llama-server uses `--threads-http 4`. The default
(n_cpu - 1 = 47 on this Threadripper) combined with 24 inference threads
caused CCD scheduling thrash that locked up the machine.

---

## Module map (as built)

```
serve/launch.py          -- start/stop/wait_ready for llama-server
serve/nginx.conf         -- TLS proxy config for all four ports
collect/capture.py       -- tcpdump subprocess wrapper
collect/query.py         -- HTTPS streaming client
collect/run.py           -- profiling orchestration loop
collect/prompts.py       -- JSONL prompt loader
features/parse.py        -- pcap parsing, iteration grouping, calibration
features/build.py        -- dataset assembly (pad/truncate, cosine sim)
features/timing.py       -- inter-packet gap extraction (Carlini & Nasr)
features/mcdonald.py     -- (size, timing) pair extraction (McDonald & Bar Or)
attack/dataset.py        -- train/test split by TPQ
attack/train.py          -- RandomForest and LightGBM builders + save/load
attack/bilstm.py         -- BiLSTM classifier (sklearn-compatible wrapper)
attack/evaluate.py       -- accuracy/F1 scoring and TPQ sweep driver
attack/gmm.py            -- GMM binary classifier (Carlini & Nasr)
defend/aggregate.py      -- SSE event batching proxy
defend/pad.py            -- SSE comment-line padding proxy
defend/cbr.py            -- constant-bit-rate streaming proxy
```
