from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from attack.evaluate import score


def _perfect_clf(n_classes: int) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.random((n_classes * 10, 20)).astype(np.float32)
    y = np.repeat(np.arange(n_classes), 10).astype(np.int32)
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    return clf, X, y


def test_score_returns_expected_keys() -> None:
    clf, X, y = _perfect_clf(3)
    result = score(clf, X, y)
    assert "accuracy" in result
    assert "f1_macro" in result


def test_score_perfect_accuracy() -> None:
    # Train and test on same data to get a near-perfect score.
    clf, X, y = _perfect_clf(3)
    result = score(clf, X, y)
    assert result["accuracy"] > 0.9


def test_score_values_in_range() -> None:
    clf, X, y = _perfect_clf(4)
    result = score(clf, X, y)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["f1_macro"] <= 1.0


def test_score_random_predictions_low_accuracy() -> None:
    rng = np.random.default_rng(0)
    X = rng.random((100, 10)).astype(np.float32)
    y = np.repeat(np.arange(50), 2).astype(np.int32)
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    clf.fit(X, y)
    # Evaluate on shuffled labels -> accuracy should be low.
    y_shuffled = rng.permutation(y)
    result = score(clf, X, y_shuffled)
    assert result["accuracy"] < 0.5
