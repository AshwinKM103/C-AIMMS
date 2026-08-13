"""fluxmem/selector.py -- format-adaptive selector f_theta (COLM Eq. 6, FluxMem Alg. 3 / Eq. 8).

Two-layer MLP: `Linear(FEATURE_DIM, hidden) -> ReLU -> Linear(hidden, 3)`,
trained by minimizing cross-entropy (COLM Eq. 6) against `fluxmem.supervision`'s
offline-derived labels. `StandardScaler -> softmax -> cross-entropy -> Adam ->
early stop` per FluxMem Alg. 3; `predict` uses `argmax softmax` (Eq. 8).

PyTorch, not sklearn's `MLPClassifier`: torch is already a dependency (frozen
Qwen3 backbone, HETREP encoders, WORKMEM's OSAM all require it downstream),
and an explicit training loop makes Eq. 6's cross-entropy literal rather than
hidden inside a library call. Cost is slower, seed-sensitive tests --
mitigated by a tiny CPU-only net, fixed `torch.manual_seed`, and
`torch.use_deterministic_algorithms(True, warn_only=True)` (see tests/conftest.py).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn

from fluxmem.config import SelectorConfig
from fluxmem.features import FEATURE_DIM
from fluxmem.interfaces import MemoryFormat
from fluxmem.supervision import FORMAT_ORDER


class _SelectorNet(nn.Module):
    """The two-layer MLP itself: `Linear -> ReLU -> Linear`."""

    def __init__(self, feature_dim: int, hidden_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainingHistory:
    """Per-epoch losses, for tests and `/track` logging."""

    train_loss: list[float]
    val_loss: list[float]
    stopped_epoch: int


class FormatSelector:
    """Trains and serves `f_theta: feature vector -> MemoryFormat`.

    `StandardScaler` is fit on train features only, never on validation --
    FluxMem Alg. 3 line 1 names the scaler but doesn't spell this out; fitting
    on validation would leak its statistics into training, the standard
    reading of "fit" in this context.
    """

    def __init__(self, config: SelectorConfig | None = None) -> None:
        self._config = config or SelectorConfig()
        self._scaler = StandardScaler()
        # Seed before constructing the net: its weight init consumes the
        # global torch RNG, so seeding only in fit() would leave two
        # identically-configured selectors with different initial weights
        # if anything else drew from the RNG between construction and fit.
        torch.manual_seed(self._config.seed)
        self._net = _SelectorNet(FEATURE_DIM, self._config.hidden_dim, len(FORMAT_ORDER))
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainingHistory:
        """Adam + `F.cross_entropy` (Eq. 6 literally), minibatches, early stop on val loss."""
        torch.manual_seed(self._config.seed)
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_val_scaled = self._scaler.transform(X_val)

        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._config.learning_rate)

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        stopped_epoch = self._config.max_epochs

        n = X_train_t.shape[0]
        batch_size = min(self._config.batch_size, n)

        for epoch in range(self._config.max_epochs):
            self._net.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                logits = self._net(X_train_t[idx])
                loss = F.cross_entropy(logits, y_train_t[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            train_losses.append(epoch_loss / n)

            self._net.eval()
            with torch.no_grad():
                val_loss = F.cross_entropy(self._net(X_val_t), y_val_t).item()
            val_losses.append(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self._config.patience:
                    stopped_epoch = epoch + 1
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._fitted = True
        return TrainingHistory(train_losses, val_losses, stopped_epoch)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Returns the softmax distribution over `FORMAT_ORDER` for a single feature vector."""
        if not self._fitted:
            raise RuntimeError("FormatSelector.predict_proba called before fit()")
        self._net.eval()
        x_scaled = self._scaler.transform(x.reshape(1, -1))
        x_t = torch.tensor(x_scaled, dtype=torch.float32)
        with torch.no_grad():
            probs = F.softmax(self._net(x_t), dim=-1)
        return probs.squeeze(0).numpy()

    def predict(self, x: np.ndarray) -> MemoryFormat:
        """`argmax softmax(f_theta(x))` (FluxMem Eq. 8)."""
        probs = self.predict_proba(x)
        return FORMAT_ORDER[int(np.argmax(probs))]

    def save(self, path: str | Path) -> None:
        """Persists scaler + state_dict together -- loading one without the
        other yields silently wrong predictions, so they serialize as one artifact."""
        payload = {
            "config": self._config.model_dump(),
            "scaler": self._scaler,
            "state_dict": self._net.state_dict(),
            "fitted": self._fitted,
        }
        with Path(path).open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> FormatSelector:
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        selector = cls(SelectorConfig(**payload["config"]))
        selector._scaler = payload["scaler"]
        selector._net.load_state_dict(payload["state_dict"])
        selector._fitted = payload["fitted"]
        return selector
