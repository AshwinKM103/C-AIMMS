"""Tests for fluxmem/config.py -- AdaStoreConfig assembly and load_config."""

from __future__ import annotations

import pathlib

import pytest
from pydantic import ValidationError

from fluxmem.config import AdaStoreConfig, PromotionThresholds, UtilityWeights, load_config

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class TestLoadConfig:
    def test_loads_default_yaml(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        assert isinstance(config, AdaStoreConfig)
        assert config.stim.capacity == 4
        assert config.utility_weights.w1 == pytest.approx(0.4)
        assert config.promotion.tau_u == pytest.approx(0.5)
        assert config.fusion.tau == pytest.approx(0.7)
        assert config.selector.hidden_dim == 16
        assert config.reward.lambda_judge == pytest.approx(0.7)

    def test_missing_required_fields_rejected(self, tmp_path) -> None:
        incomplete = tmp_path / "incomplete.yaml"
        incomplete.write_text("stim:\n  capacity: 4\n")
        with pytest.raises(ValidationError):
            load_config(incomplete)


class TestNoTauC:
    def test_config_has_no_tau_c_field(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        assert not hasattr(config.promotion, "tau_c")
        assert "tau_c" not in AdaStoreConfig.model_fields
        assert "tau_c" not in PromotionThresholds.model_fields

    def test_default_yaml_has_no_tau_c_key(self) -> None:
        # The prose comment mentions "tau_c" by name (explaining why it's
        # absent); check no YAML *key* named tau_c exists, not the prose.
        lines = DEFAULT_CONFIG_PATH.read_text().splitlines()
        keys = [line.split(":")[0].strip() for line in lines if not line.strip().startswith("#")]
        assert "tau_c" not in keys


class TestValidationBoundaries:
    def test_negative_utility_weight_rejected(self) -> None:
        with pytest.raises(ValidationError):
            UtilityWeights(w1=-0.1, w2=0.5, w3=0.5)

    def test_tau_r_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PromotionThresholds(tau_u=0.5, tau_r=1.5, max_age=10.0)

    def test_zero_max_age_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PromotionThresholds(tau_u=0.5, tau_r=0.5, max_age=0.0)
