"""BiLSTM classifier for token-timing traces.

Follows McDonald & Bar Or (arXiv:2511.03675) Section 4.5: bidirectional LSTM
with two stacked layers, trained with cross-entropy on per-iteration byte-size
sequences. Input is (batch, seq_len) float32; labels are integer class indices.

The fit_bilstm function matches the (X_train, y_train) -> Classifier signature
used by tpq_sweep so it drops into the existing evaluation pipeline unchanged.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class _BiLSTM(nn.Module):
    def __init__(self, n_classes: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) -> (B, T, 1)
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(out[:, -1, :])


class BiLSTMClassifier:
    """Sklearn-compatible wrapper so BiLSTM fits the Classifier Union type."""

    def __init__(
        self,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.3,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self._model: _BiLSTM | None = None
        self._classes: np.ndarray | None = None
        self._scaler: StandardScaler | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BiLSTMClassifier":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        self._scaler = scaler

        classes = np.unique(y)
        self._classes = classes
        n_classes = len(classes)
        label_map = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([label_map[c] for c in y], dtype=np.int64)

        model = _BiLSTM(n_classes, self.hidden, self.layers, self.dropout)
        model.to(self.device)
        self._model = model

        Xt = torch.tensor(X_scaled, dtype=torch.float32)
        yt = torch.tensor(y_idx, dtype=torch.long)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1}/{self.epochs}  loss={epoch_loss/len(loader):.4f}",
                      file=sys.stderr)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None and self._classes is not None and self._scaler is not None
        X_scaled = self._scaler.transform(X).astype(np.float32)
        self._model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            logits = self._model(Xt)
        idx = logits.argmax(dim=1).cpu().numpy()
        return self._classes[idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None and self._scaler is not None
        X_scaled = self._scaler.transform(X).astype(np.float32)
        self._model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            probs = torch.softmax(self._model(Xt), dim=1)
        return probs.cpu().numpy()


def fit_bilstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden: int = 128,
    layers: int = 2,
    dropout: float = 0.3,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
    verbose: bool = False,
) -> BiLSTMClassifier:
    clf = BiLSTMClassifier(
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        verbose=verbose,
    )
    clf.fit(X_train, y_train)
    return clf


def save(clf: BiLSTMClassifier, path: Path) -> None:
    assert clf._model is not None and clf._scaler is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(clf._model.state_dict(), buf)
    data = {
        "state_dict_bytes": buf.getvalue(),
        "classes": clf._classes,
        "scaler": clf._scaler,
        "hidden": clf.hidden,
        "layers": clf.layers,
        "dropout": clf.dropout,
    }
    import pickle
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load(path: Path, device: str | None = None) -> BiLSTMClassifier:
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    clf = BiLSTMClassifier(hidden=data["hidden"], layers=data["layers"],
                           dropout=data["dropout"], device=device)
    clf._classes = data["classes"]
    clf._scaler = data["scaler"]
    n_classes = len(clf._classes)
    model = _BiLSTM(n_classes, clf.hidden, clf.layers, clf.dropout)
    model.load_state_dict(torch.load(io.BytesIO(data["state_dict_bytes"]), weights_only=True))
    model.to(clf.device)
    clf._model = model
    return clf
