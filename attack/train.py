from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build_rf() -> RandomForestClassifier:
    # Paper §4.4: 150 estimators, max_depth=15, min_samples_split=10,
    # min_samples_leaf=1. Paper cites "MSE loss" which is regressor-only in
    # sklearn; RandomForestClassifier defaults to gini, which matches paper
    # results in practice.
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=0,
    )


def fit(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    clf = build_rf()
    clf.fit(X_train, y_train)
    return clf


def save(clf: RandomForestClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)


def load(path: Path) -> RandomForestClassifier:
    with open(path, "rb") as f:
        return pickle.load(f)  # type: ignore[no-any-return]
