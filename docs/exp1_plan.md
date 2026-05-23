# Experiment 1 — File Plan
## Wei et al. query fingerprinting on llama.cpp (Exact Knowledge)

**Attack pipeline:** offline profiling phase (run each of 50 prompts 5–30 times,
capture pcap per trial) → feature extraction (group TLS payload bytes by
decode-iteration timing windows) → Random Forest classifier (paper §4.4 params)
→ online phase (capture one trace for unknown prompt, classify).

**Adaptation from paper:** paper used vLLM + EAGLE on a remote A100; we use
llama.cpp + Qwen2.5-0.5B Q8 draft on loopback, nginx TLS termination on port
8443 (proxies to llama.cpp on plain HTTP internally), tcpdump on the veth/loopback.

---

## serve/

### `serve/launch.py`
Starts and stops the llama.cpp server process with the correct
speculative-decoding flags, and blocks until the server is accepting requests.

```python
def build_cmd(
    model_path: Path,
    draft_model_path: Path,
    host: str,
    port: int,
    n_gpu_layers: int,
    n_draft: int,
    ctx_size: int,
) -> list[str]: ...

def start(cmd: list[str], log_path: Path) -> subprocess.Popen: ...

def wait_ready(host: str, port: int, timeout: float = 30.0) -> None: ...

def stop(proc: subprocess.Popen) -> None: ...
```

Deps: none internal.

---

### `serve/nginx.conf`
TLS-terminating reverse proxy: self-signed cert at `server.local`, listens on
8443, forwards to llama.cpp on localhost internal port. Not a Python file; no
signatures. Included because `collect/query.py` and `collect/capture.py` both
assume TLS on a named interface.

Deps: none.

---

## collect/

### `collect/data/exp1_prompts.jsonl`
The 50 MedAlpaca prompts from paper Appendix A.1, one JSON object per line:
`{"id": 0, "text": "..."}`. Not a Python file. Populated before profiling begins.

---

### `collect/prompts.py`
Loads prompts from a JSONL file. Default path points to
`collect/data/exp1_prompts.jsonl` so callers need not pass it unless overriding.

```python
def load_prompts(path: Path = Path(__file__).parent / "data" / "exp1_prompts.jsonl") -> list[str]: ...
```

Deps: none.

---

### `collect/capture.py`
Thin wrapper around `tcpdump` that starts a capture subprocess against a specific
interface and BPF filter, writes a `.pcap` file, then stops cleanly.

```python
def start(
    iface: str,
    pcap_path: Path,
    bpf_filter: str = "tcp port 8443",
) -> subprocess.Popen: ...

def stop(proc: subprocess.Popen) -> None: ...
```

Deps: none internal.

---

### `collect/query.py`
Sends a single HTTPS prompt to the nginx-fronted llama.cpp endpoint and returns
the full SSE response text. Uses a custom CA bundle for the self-signed cert.

```python
def send(
    prompt: str,
    host: str,
    port: int = 8443,
    temperature: float = 0.3,
    system_prompt: str = "",
    timeout: float = 120.0,
) -> str: ...
```

Deps: none internal. Uses `httpx` or `requests`.

---

### `collect/run.py`
Main orchestration loop for the offline profiling phase: for each prompt × run,
starts a capture, fires the query, stops the capture, saves metadata. Produces
one `.pcap` per trial and a manifest JSONL.

```python
def profile_all(
    prompts: list[str],
    tpq: int,
    temperature: float,
    out_dir: Path,
    host: str,
    port: int = 8443,
    iface: str = "lo",
    bpf_filter: str = "tcp port 8443",
) -> Path: ...  # returns manifest path

def profile_one(
    prompt_id: int,
    prompt: str,
    run_id: int,
    temperature: float,
    out_dir: Path,
    host: str,
    port: int = 8443,
    iface: str = "lo",
    bpf_filter: str = "tcp port 8443",
) -> Path: ...  # returns pcap path
```

Deps: `collect/prompts.py`, `collect/capture.py`, `collect/query.py`.

---

## features/

### `features/parse.py`
Reads a `.pcap`, extracts TLS record sizes and timestamps for the server→client
direction, then groups records into decode iterations using a timing window.
Returns the raw trace — a list of aggregate byte counts, one per iteration —
which is the packet-size proxy for tokens-per-iteration (paper §4.2, r=0.747).

Also provides a calibration function to determine `window_ms` empirically from
a single response pcap by fitting the distribution of inter-record gaps.

```python
def extract_records(
    pcap_path: Path,
    server_port: int = 8443,
) -> list[tuple[float, int]]: ...  # (timestamp_s, payload_bytes)

def group_iterations(
    records: list[tuple[float, int]],
    window_ms: float,
) -> list[int]: ...  # bytes per decode iteration

def trace_from_pcap(
    pcap_path: Path,
    server_port: int = 8443,
    window_ms: float = 50.0,
) -> list[int]: ...

def calibrate_window(
    pcap_path: Path,
    server_port: int = 8443,
    percentile: float = 95.0,
) -> float: ...  # returns window_ms
# Looks at inter-record gap distribution in a single response pcap.
# Within-iteration gaps are small; across-iteration gaps are large.
# Returns the percentile-th gap as the iteration boundary.
```

Deps: none internal. Uses `scapy` for parsing.

---

### `features/build.py`
Loads all pcaps listed in a manifest, builds per-trial traces, pads or truncates
to a fixed length, and assembles the `(X, y)` arrays the classifier consumes.
Also exposes cosine similarity for the reproducibility sanity check (paper
Appendix B: expected ~0.9–1.0 within-prompt, 0.4–0.8 across-prompt).

```python
def pad_or_truncate(trace: list[int], length: int) -> list[int]: ...

def cosine_similarity(a: list[int], b: list[int]) -> float: ...

def build_dataset(
    manifest_path: Path,
    trace_length: int,
    window_ms: float,
    server_port: int = 8443,
) -> tuple[np.ndarray, np.ndarray]: ...  # X shape (n_trials, trace_length), y shape (n_trials,)
```

Deps: `features/parse.py`.

---

## attack/

### `attack/dataset.py`
Splits the built dataset into train and test sets following the paper's protocol:
train on TPQ traces per prompt (varying TPQ for the sweep), hold out 5 traces
per prompt for testing.

```python
def split(
    X: np.ndarray,
    y: np.ndarray,
    train_tpq: int,
    test_tpq: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
# returns X_train, X_test, y_train, y_test
```

Deps: `features/build.py` (upstream; consumes arrays it produces).

---

### `attack/train.py`
Instantiates the Random Forest classifier with the exact hyperparameters from
paper §4.4 and fits it.

```python
def build_rf() -> RandomForestClassifier: ...
# 150 estimators, max_depth=15, min_samples_split=10,
# min_samples_leaf=1, criterion="squared_error"

def fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier: ...

def save(clf: RandomForestClassifier, path: Path) -> None: ...

def load(path: Path) -> RandomForestClassifier: ...
```

Deps: `attack/dataset.py`.

---

### `attack/evaluate.py`
Runs accuracy and F1 evaluation on a fitted classifier, and produces the
TPQ-sweep table replicating paper Figure 3 for our llama.cpp setup.

```python
def score(
    clf: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]: ...  # keys: accuracy, f1_macro

def tpq_sweep(
    manifest_path: Path,
    tpq_values: list[int],
    temperatures: list[float],
    trace_length: int,
    window_ms: float,
    server_port: int = 8443,
) -> pd.DataFrame: ...
```

Deps: `attack/train.py`, `attack/dataset.py`, `features/build.py`.

---

## Summary table

| # | file | group | key external dep |
|---|------|-------|-----------------|
| 1 | `serve/launch.py` | serve | `subprocess` |
| 2 | `serve/nginx.conf` | serve | — |
| 3 | `collect/data/exp1_prompts.jsonl` | collect | — |
| 4 | `collect/prompts.py` | collect | — |
| 5 | `collect/capture.py` | collect | `tcpdump` via `subprocess` |
| 6 | `collect/query.py` | collect | `httpx` |
| 7 | `collect/run.py` | collect | 4, 5, 6 |
| 8 | `features/parse.py` | features | `scapy` |
| 9 | `features/build.py` | features | 8, `numpy` |
| 10 | `attack/dataset.py` | attack | 9, `numpy` |
| 11 | `attack/train.py` | attack | 10, `scikit-learn` |
| 12 | `attack/evaluate.py` | attack | 11, 10, 9, `pandas` |
