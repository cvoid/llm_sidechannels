"""GMM-based binary query disambiguation (Carlini & Nasr 2410.17175).

Trains one GMM per hypothesis on timing feature vectors. At test time,
classifies by log-likelihood ratio: score = log P(x | GMM_B) - log P(x | GMM_A).
Positive score -> predict B; negative -> predict A. Sweeping the decision
threshold produces a precision-recall curve.
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import average_precision_score


class GMMBinaryClassifier:
    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "diag",
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self._gmm_a: GaussianMixture | None = None
        self._gmm_b: GaussianMixture | None = None

    def fit(self, X_a: np.ndarray, X_b: np.ndarray) -> "GMMBinaryClassifier":
        kwargs = dict(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=200,
        )
        self._gmm_a = GaussianMixture(**kwargs).fit(X_a)
        self._gmm_b = GaussianMixture(**kwargs).fit(X_b)
        return self

    def log_ratio(self, X: np.ndarray) -> np.ndarray:
        """Log-likelihood ratio score: positive -> B, negative -> A."""
        assert self._gmm_a is not None and self._gmm_b is not None
        return self._gmm_b.score_samples(X) - self._gmm_a.score_samples(X)

    def auprc(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Average precision (area under precision-recall curve). y=1 means B."""
        scores = self.log_ratio(X_test)
        return float(average_precision_score(y_test, scores))


def fit_gmm(
    X_a: np.ndarray,
    X_b: np.ndarray,
    n_components: int = 4,
    covariance_type: str = "diag",
) -> GMMBinaryClassifier:
    return GMMBinaryClassifier(
        n_components=n_components,
        covariance_type=covariance_type,
    ).fit(X_a, X_b)
