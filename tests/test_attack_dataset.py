from __future__ import annotations

import numpy as np
import pytest

from attack.dataset import split


def _make_xy(n_classes: int, samples_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.random((n_classes * samples_per_class, 10)).astype(np.float32)
    y = np.repeat(np.arange(n_classes), samples_per_class).astype(np.int32)
    return X, y


def test_split_train_size() -> None:
    X, y = _make_xy(n_classes=5, samples_per_class=20)
    X_train, _, y_train, _ = split(X, y, train_tpq=10, test_tpq=5)
    assert len(X_train) == 5 * 10
    assert len(y_train) == 5 * 10


def test_split_test_size() -> None:
    X, y = _make_xy(n_classes=5, samples_per_class=20)
    _, X_test, _, y_test = split(X, y, train_tpq=10, test_tpq=5)
    assert len(X_test) == 5 * 5
    assert len(y_test) == 5 * 5


def test_split_no_overlap() -> None:
    X, y = _make_xy(n_classes=3, samples_per_class=15)
    X_train, X_test, _, _ = split(X, y, train_tpq=5, test_tpq=5)
    # Compare rows; none should be identical (indices are disjoint).
    for row in X_test:
        assert not any(np.array_equal(row, r) for r in X_train)


def test_split_reproducible() -> None:
    X, y = _make_xy(n_classes=4, samples_per_class=12)
    r1 = split(X, y, train_tpq=5, test_tpq=3, seed=0)
    r2 = split(X, y, train_tpq=5, test_tpq=3, seed=0)
    for a, b in zip(r1, r2):
        np.testing.assert_array_equal(a, b)


def test_split_different_seeds_differ() -> None:
    X, y = _make_xy(n_classes=4, samples_per_class=12)
    X_train_a, _, _, _ = split(X, y, train_tpq=5, test_tpq=3, seed=0)
    X_train_b, _, _, _ = split(X, y, train_tpq=5, test_tpq=3, seed=1)
    assert not np.array_equal(X_train_a, X_train_b)


def test_split_all_classes_in_train() -> None:
    X, y = _make_xy(n_classes=5, samples_per_class=15)
    _, _, y_train, _ = split(X, y, train_tpq=5, test_tpq=5)
    assert set(np.unique(y_train).tolist()) == set(range(5))
