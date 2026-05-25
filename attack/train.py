from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from .bilstm import BiLSTMClassifier

Classifier = Union[RandomForestClassifier, LGBMClassifier, BiLSTMClassifier]


def build_rf() -> RandomForestClassifier:
    # Paper §4.4: 150 estimators, max_depth=15, min_samples_split=10,
    # min_samples_leaf=1. Paper cites "MSE loss" which is regressor-only in
    # sklearn; RandomForestClassifier defaults to gini, which matches paper
    # results in practice.
    # Paper §4.4 uses min_samples_split=10, tuned for their large dataset.
    # min_samples_split=2 is the sklearn default and safe for both smoke-test
    # scale (9 training samples) and full-scale experiments.
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=0,
    )


def build_lgbm() -> LGBMClassifier:
    # Parameters follow Whisper Leak (McDonald & Bar Or, arXiv:2511.03675)
    # which uses LightGBM as its primary classifier. num_leaves=63 gives
    # richer trees than the default 31 for a 50-class problem; 500 rounds
    # with learning_rate=0.05 balances accuracy against overfitting at
    # tpq=25 training samples per class.
    return LGBMClassifier(
        n_estimators=500,
        num_leaves=63,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=0,
        verbose=-1,
    )


def fit(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    clf = build_rf()
    clf.fit(X_train, y_train)
    return clf


def fit_lgbm(X_train: np.ndarray, y_train: np.ndarray) -> LGBMClassifier:
    clf = build_lgbm()
    feature_names = [f"iter_{i}" for i in range(X_train.shape[1])]
    clf.fit(X_train, y_train, feature_name=feature_names)
    return clf


def save(clf: Classifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)


def load(path: Path) -> Classifier:
    with open(path, "rb") as f:
        return pickle.load(f)  # type: ignore[no-any-return]
