from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from attack.train import build_rf, fit, load, save


def _tiny_dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.random((60, 20)).astype(np.float32)
    y = np.repeat(np.arange(3), 20).astype(np.int32)
    return X, y


def test_build_rf_params() -> None:
    clf = build_rf()
    assert clf.n_estimators == 150
    assert clf.max_depth == 15
    assert clf.min_samples_split == 10
    assert clf.min_samples_leaf == 1


def test_fit_returns_classifier() -> None:
    X, y = _tiny_dataset()
    clf = fit(X, y)
    assert isinstance(clf, RandomForestClassifier)
    assert hasattr(clf, "estimators_")


def test_fit_predicts_training_labels() -> None:
    X, y = _tiny_dataset()
    clf = fit(X, y)
    preds = clf.predict(X)
    accuracy = float(np.mean(preds == y))
    assert accuracy > 0.9


def test_save_load_roundtrip(tmp_path: Path) -> None:
    X, y = _tiny_dataset()
    clf = fit(X, y)
    path = tmp_path / "model.pkl"
    save(clf, path)
    loaded = load(path)
    np.testing.assert_array_equal(clf.predict(X), loaded.predict(X))


def test_save_creates_parent_dir(tmp_path: Path) -> None:
    X, y = _tiny_dataset()
    clf = fit(X, y)
    path = tmp_path / "nested" / "dir" / "model.pkl"
    save(clf, path)
    assert path.exists()
