"""Tests for fluxmem/selector.py -- format-adaptive MLP selector."""

from __future__ import annotations

import numpy as np
import pytest

from fluxmem.config import SelectorConfig
from fluxmem.features import FEATURE_DIM
from fluxmem.interfaces import MemoryFormat
from fluxmem.selector import FormatSelector
from fluxmem.supervision import FORMAT_ORDER


def _separable_dataset(n_per_class: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A trivially linearly-separable synthetic dataset: one-hot-ish clusters per class."""
    rng = np.random.default_rng(seed)
    X_parts = []
    y_parts = []
    for class_idx in range(len(FORMAT_ORDER)):
        center = np.zeros(FEATURE_DIM)
        center[class_idx % FEATURE_DIM] = 5.0
        cluster = center + rng.normal(scale=0.05, size=(n_per_class, FEATURE_DIM))
        X_parts.append(cluster)
        y_parts.append(np.full(n_per_class, class_idx, dtype=np.int64))
    X = np.concatenate(X_parts).astype(np.float64)
    y = np.concatenate(y_parts)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


class TestTrainingOverfitsSeparableData:
    def test_train_accuracy_near_one(self) -> None:
        X, y = _separable_dataset()
        config = SelectorConfig(max_epochs=300, patience=300, seed=0)
        selector = FormatSelector(config)
        selector.fit(X, y, X, y)

        correct = sum(1 for xi, yi in zip(X, y) if selector.predict(xi) == FORMAT_ORDER[yi])
        accuracy = correct / len(y)
        assert accuracy >= 0.95

    def test_loss_decreases_over_training(self) -> None:
        X, y = _separable_dataset()
        config = SelectorConfig(max_epochs=100, patience=100, seed=0)
        selector = FormatSelector(config)
        history = selector.fit(X, y, X, y)
        # Not strictly monotonic every epoch (minibatch noise), but the trend
        # over the run should be a clear decrease.
        assert history.train_loss[-1] < history.train_loss[0]
        first_quarter = history.train_loss[: max(1, len(history.train_loss) // 4)]
        last_quarter = history.train_loss[-max(1, len(history.train_loss) // 4) :]
        assert np.mean(last_quarter) < np.mean(first_quarter)


class TestEarlyStopping:
    def test_fires_on_non_improving_validation(self) -> None:
        X_train, y_train = _separable_dataset(n_per_class=20, seed=0)
        # Validation set that shares no signal with train -> val loss won't improve.
        rng = np.random.default_rng(99)
        X_val = rng.normal(size=(10, FEATURE_DIM))
        y_val = rng.integers(0, len(FORMAT_ORDER), size=10).astype(np.int64)

        config = SelectorConfig(max_epochs=500, patience=3, seed=0)
        selector = FormatSelector(config)
        history = selector.fit(X_train, y_train, X_val, y_val)

        assert history.stopped_epoch < 500


class TestSaveLoadRoundTrip:
    def test_predictions_identical_after_round_trip(self, tmp_path) -> None:
        X, y = _separable_dataset()
        config = SelectorConfig(max_epochs=50, patience=50, seed=0)
        selector = FormatSelector(config)
        selector.fit(X, y, X, y)

        path = tmp_path / "selector.pkl"
        selector.save(path)
        loaded = FormatSelector.load(path)

        for xi in X[:10]:
            assert selector.predict(xi) == loaded.predict(xi)
            np.testing.assert_allclose(selector.predict_proba(xi), loaded.predict_proba(xi))


class TestScalerNotFitOnValidation:
    def test_scaler_stats_come_from_train_only(self) -> None:
        X_train = np.ones((10, FEATURE_DIM)) * 2.0
        y_train = np.zeros(10, dtype=np.int64)
        X_val = np.ones((10, FEATURE_DIM)) * 1000.0
        y_val = np.zeros(10, dtype=np.int64)

        config = SelectorConfig(max_epochs=1, patience=1, seed=0)
        selector = FormatSelector(config)
        selector.fit(X_train, y_train, X_val, y_val)

        # StandardScaler fit on train-only constant data: mean == 2.0 (constant fallback).
        np.testing.assert_allclose(selector._scaler.mean_, np.full(FEATURE_DIM, 2.0))


class TestSeededReproducibility:
    def test_same_seed_reproduces_identical_predictions(self) -> None:
        X, y = _separable_dataset()
        config = SelectorConfig(max_epochs=30, patience=30, seed=42)

        selector_a = FormatSelector(SelectorConfig(**config.model_dump()))
        history_a = selector_a.fit(X, y, X, y)

        selector_b = FormatSelector(SelectorConfig(**config.model_dump()))
        history_b = selector_b.fit(X, y, X, y)

        assert history_a.train_loss == pytest.approx(history_b.train_loss)
        for xi in X[:5]:
            np.testing.assert_allclose(selector_a.predict_proba(xi), selector_b.predict_proba(xi))


class TestPredictBeforeFit:
    def test_predict_before_fit_raises(self) -> None:
        selector = FormatSelector()
        with pytest.raises(RuntimeError):
            selector.predict(np.zeros(FEATURE_DIM))


class TestPredictReturnsMemoryFormat:
    def test_predict_returns_valid_format(self) -> None:
        X, y = _separable_dataset()
        selector = FormatSelector(SelectorConfig(max_epochs=20, patience=20, seed=0))
        selector.fit(X, y, X, y)
        result = selector.predict(X[0])
        assert isinstance(result, MemoryFormat)
