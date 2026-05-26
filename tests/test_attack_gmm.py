from __future__ import annotations

import numpy as np
import pytest

from attack.gmm import GMMBinaryClassifier, fit_gmm


def _two_clusters(
    rng: np.random.Generator,
    n: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two well-separated 1-D Gaussian clusters as column vectors."""
    X_a = rng.normal(loc=0.0, scale=0.1, size=(n, 1))
    X_b = rng.normal(loc=10.0, scale=0.1, size=(n, 1))
    return X_a, X_b


def test_fit_returns_self() -> None:
    rng = np.random.default_rng(0)
    X_a, X_b = _two_clusters(rng)
    clf = GMMBinaryClassifier(n_components=1)
    result = clf.fit(X_a, X_b)
    assert result is clf


def test_log_ratio_shape() -> None:
    rng = np.random.default_rng(0)
    X_a, X_b = _two_clusters(rng)
    clf = GMMBinaryClassifier(n_components=1).fit(X_a, X_b)
    X_test = rng.normal(loc=10.0, scale=0.1, size=(50, 1))
    scores = clf.log_ratio(X_test)
    assert scores.shape == (50,)


def test_log_ratio_direction() -> None:
    rng = np.random.default_rng(0)
    X_a, X_b = _two_clusters(rng)
    clf = GMMBinaryClassifier(n_components=1).fit(X_a, X_b)
    X_b_test = rng.normal(loc=10.0, scale=0.1, size=(50, 1))
    # Samples from cluster B should have positive log-ratio (more likely under GMM_B)
    assert np.all(clf.log_ratio(X_b_test) > 0)


def test_auprc_separable() -> None:
    rng = np.random.default_rng(0)
    X_a_train, X_b_train = _two_clusters(rng, n=100)
    clf = GMMBinaryClassifier(n_components=1).fit(X_a_train, X_b_train)
    X_a_test = rng.normal(loc=0.0, scale=0.1, size=(50, 1))
    X_b_test = rng.normal(loc=10.0, scale=0.1, size=(50, 1))
    X_test = np.vstack([X_a_test, X_b_test])
    y_test = np.array([0] * 50 + [1] * 50)
    assert clf.auprc(X_test, y_test) > 0.95


def test_auprc_random() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(loc=0.0, scale=1.0, size=(200, 1))
    # Train and test on identical distributions -- no discriminating signal
    clf = GMMBinaryClassifier(n_components=2).fit(X[:100], X[100:])
    X_test = rng.normal(loc=0.0, scale=1.0, size=(100, 1))
    y_test = np.array([0] * 50 + [1] * 50)
    assert abs(clf.auprc(X_test, y_test) - 0.5) < 0.2
