from __future__ import annotations

import numpy as np
from numpy.random import default_rng


def split(
    X: np.ndarray,
    y: np.ndarray,
    train_tpq: int,
    test_tpq: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[0] == 0:
        raise ValueError("dataset is empty -- check manifest path and temperature filter")
    rng = default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        shuffled = rng.permutation(idx)
        test_idx.extend(shuffled[:test_tpq].tolist())
        train_idx.extend(shuffled[test_tpq: test_tpq + train_tpq].tolist())
    t = np.array(train_idx, dtype=np.intp)
    v = np.array(test_idx, dtype=np.intp)
    return X[t], X[v], y[t], y[v]
