from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from features.build import build_dataset, cosine_similarity, pad_or_truncate


def test_pad_or_truncate_pads_short() -> None:
    assert pad_or_truncate([1, 2, 3], 5) == [1, 2, 3, 0, 0]


def test_pad_or_truncate_truncates_long() -> None:
    assert pad_or_truncate([1, 2, 3, 4, 5], 3) == [1, 2, 3]


def test_pad_or_truncate_exact_length() -> None:
    assert pad_or_truncate([1, 2, 3], 3) == [1, 2, 3]


def test_pad_or_truncate_empty() -> None:
    assert pad_or_truncate([], 4) == [0, 0, 0, 0]


def test_cosine_similarity_identical() -> None:
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal() -> None:
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector() -> None:
    assert cosine_similarity([0, 0], [1, 2]) == 0.0


def test_build_dataset_shape(tmp_path: Path, synthetic_pcap: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    entries = [
        {"prompt_id": 0, "run_id": 0, "prompt": "p0", "temperature": 0.3, "pcap": str(synthetic_pcap)},
        {"prompt_id": 1, "run_id": 0, "prompt": "p1", "temperature": 0.3, "pcap": str(synthetic_pcap)},
        {"prompt_id": 0, "run_id": 1, "prompt": "p0", "temperature": 0.3, "pcap": str(synthetic_pcap)},
    ]
    manifest_path.write_text("\n".join(json.dumps(e) for e in entries))
    X, y = build_dataset(manifest_path, trace_length=10, window_ms=50.0)
    assert X.shape == (3, 10)
    assert y.shape == (3,)


def test_build_dataset_temperature_filter(tmp_path: Path, synthetic_pcap: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    entries = [
        {"prompt_id": 0, "run_id": 0, "prompt": "p", "temperature": 0.3, "pcap": str(synthetic_pcap)},
        {"prompt_id": 0, "run_id": 1, "prompt": "p", "temperature": 0.6, "pcap": str(synthetic_pcap)},
    ]
    manifest_path.write_text("\n".join(json.dumps(e) for e in entries))
    X, y = build_dataset(manifest_path, trace_length=10, window_ms=50.0, temperature=0.3)
    assert X.shape[0] == 1


def test_build_dataset_labels(tmp_path: Path, synthetic_pcap: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    entries = [
        {"prompt_id": 7, "run_id": 0, "prompt": "p7", "temperature": 0.3, "pcap": str(synthetic_pcap)},
    ]
    manifest_path.write_text(json.dumps(entries[0]))
    _, y = build_dataset(manifest_path, trace_length=5, window_ms=50.0)
    assert y[0] == 7
