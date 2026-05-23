from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from features import build as _build
from . import dataset as _dataset, train as _train


def score(
    clf: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    y_pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }


def tpq_sweep(
    manifest_path: Path,
    tpq_values: list[int],
    temperatures: list[float],
    trace_length: int,
    window_ms: float,
    server_port: int = 8443,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for temperature in temperatures:
        X, y = _build.build_dataset(
            manifest_path, trace_length, window_ms, server_port, temperature
        )
        for tpq in tpq_values:
            X_train, X_test, y_train, y_test = _dataset.split(X, y, train_tpq=tpq)
            clf = _train.fit(X_train, y_train)
            metrics = score(clf, X_test, y_test)
            rows.append({"temperature": temperature, "tpq": tpq, **metrics})
    return pd.DataFrame(rows)
