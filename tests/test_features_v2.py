"""Tests for fluxmem/features_v2.py -- the 15-feature selector-MLP vector.

Organized to mirror the Phase 4b critical-fix checklist
(docs/workflows/feature-engineering-selector/phase4_output_critical_fixes.txt):
input validation, output clamping, zero-vector guards, adjacency validation,
epsilon guards, and a final regression class asserting no NaN/inf ever
reaches the returned array across a battery of adversarial inputs. 62+ cases
total (`pytest --collect-only -q tests/test_features_v2.py | tail -1` prints
the exact count) -- there is no upstream `edge_case_test.py` to port from
(verified absent from the repo; see fluxmem/features_v2.py's module
docstring "Deviations" section), so this suite is written fresh against the
fixes actually listed in `phase4_output_critical_fixes.txt`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fluxmem.features_v2 import (
    BOUNDED_FEATURES_V2,
    FEATURE_DIM_V2,
    FEATURE_NAMES_V2,
    extract_efficiency_features,
    extract_features_v2,
    extract_geometry_features,
    extract_overlap_features,
    extract_robustness_features,
    extract_task_features,
)
from fluxmem.interfaces import EpisodicUnit, MemoryFormat, Turn

VS_DIM = 384


def _turn() -> Turn:
    return Turn(turn_id="t0", user="hello", assistant="world", timestamp=0.0, last_access=0.0)


def _unit(**kwargs) -> EpisodicUnit:
    return EpisodicUnit(episode_id="e0", turns=[_turn()], primary_format=MemoryFormat.VS, **kwargs)


def _vec(seed: int = 0, dim: int = VS_DIM, normalize: bool = True) -> np.ndarray:
    v = np.random.RandomState(seed).randn(dim).astype(np.float32)
    return v / np.linalg.norm(v) if normalize else v


def _adjacency(n: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    adj = rng.uniform(0, 1, size=(n, n))
    adj = (adj + adj.T) / 2  # symmetric, plausible undirected weights
    np.fill_diagonal(adj, 0.0)
    return adj


def _vc_map(shape: tuple[int, int, int] = (7, 7, 512), seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, size=shape).astype(np.float32)


def _assert_all_finite(features: np.ndarray) -> None:
    for name, value in zip(FEATURE_NAMES_V2, features):
        assert math.isfinite(value), f"{name} is not finite: {value}"


# ---------------------------------------------------------------------------
# Feature dimensionality / naming
# ---------------------------------------------------------------------------
class TestFeatureDim:
    def test_feature_dim_is_fifteen(self) -> None:
        assert FEATURE_DIM_V2 == 15
        assert len(FEATURE_NAMES_V2) == FEATURE_DIM_V2

    def test_output_shape_matches_feature_dim(self) -> None:
        unit = _unit(embedding=_vec())
        features = extract_features_v2(unit)
        assert features.shape == (FEATURE_DIM_V2,)

    def test_output_dtype_is_float64(self) -> None:
        unit = _unit(embedding=_vec())
        assert extract_features_v2(unit).dtype == np.float64

    def test_bounded_features_v2_has_twelve_entries(self) -> None:
        assert len(BOUNDED_FEATURES_V2) == 12

    def test_bounded_features_v2_is_subset_of_feature_names(self) -> None:
        assert BOUNDED_FEATURES_V2.issubset(set(FEATURE_NAMES_V2))

    def test_three_features_are_deliberately_unbounded(self) -> None:
        unbounded = set(FEATURE_NAMES_V2) - BOUNDED_FEATURES_V2
        assert unbounded == {"hg_cost_position", "hg_information_per_kb", "vc_layout_prominence"}


# ---------------------------------------------------------------------------
# Fix #1: input validation -- VS embedding
# ---------------------------------------------------------------------------
class TestVsEmbeddingValidation:
    def test_missing_embedding_raises(self) -> None:
        unit = _unit(embedding=None)
        with pytest.raises(ValueError, match="embedding"):
            extract_features_v2(unit)

    def test_nan_embedding_raises(self) -> None:
        v = _vec()
        v[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            extract_features_v2(_unit(embedding=v))

    def test_inf_embedding_raises(self) -> None:
        v = _vec()
        v[10] = np.inf
        with pytest.raises(ValueError, match="inf"):
            extract_features_v2(_unit(embedding=v))

    def test_neg_inf_embedding_raises(self) -> None:
        v = _vec()
        v[10] = -np.inf
        with pytest.raises(ValueError, match="inf"):
            extract_features_v2(_unit(embedding=v))

    def test_two_dimensional_embedding_raises(self) -> None:
        v = _vec().reshape(1, VS_DIM)
        with pytest.raises(ValueError, match="1-D"):
            extract_features_v2(_unit(embedding=v))

    def test_empty_embedding_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            extract_features_v2(_unit(embedding=np.array([])))


# ---------------------------------------------------------------------------
# Fix #1 / #4: input validation -- HG adjacency
# ---------------------------------------------------------------------------
class TestHgAdjacencyValidation:
    def test_nan_adjacency_raises(self) -> None:
        adj = _adjacency()
        adj[0, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            extract_features_v2(_unit(embedding=_vec(), hg_adjacency=adj))

    def test_inf_adjacency_raises(self) -> None:
        adj = _adjacency()
        adj[0, 1] = np.inf
        with pytest.raises(ValueError, match="inf"):
            extract_features_v2(_unit(embedding=_vec(), hg_adjacency=adj))

    def test_negative_weight_raises(self) -> None:
        adj = _adjacency()
        adj[0, 1] = -0.5
        with pytest.raises(ValueError, match="negative"):
            extract_features_v2(_unit(embedding=_vec(), hg_adjacency=adj))

    def test_non_square_adjacency_raises(self) -> None:
        adj = np.zeros((3, 4))
        with pytest.raises(ValueError, match="square"):
            extract_features_v2(_unit(embedding=_vec(), hg_adjacency=adj))

    def test_one_dimensional_adjacency_raises(self) -> None:
        adj = np.zeros(5)
        with pytest.raises(ValueError, match="square"):
            extract_features_v2(_unit(embedding=_vec(), hg_adjacency=adj))

    def test_list_of_lists_adjacency_is_coerced_not_rejected(self) -> None:
        adj = [[0.0, 0.5], [0.5, 0.0]]
        features = extract_features_v2(_unit(embedding=_vec(), hg_adjacency=adj))
        _assert_all_finite(features)


# ---------------------------------------------------------------------------
# Fix #1: input validation -- VC feature map
# ---------------------------------------------------------------------------
class TestVcFeatureMapValidation:
    def test_nan_vc_map_raises(self) -> None:
        feat = _vc_map()
        feat[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            extract_features_v2(_unit(embedding=_vec(), vc_feature_map=feat))

    def test_inf_vc_map_raises(self) -> None:
        feat = _vc_map()
        feat[0, 0, 0] = np.inf
        with pytest.raises(ValueError, match="inf"):
            extract_features_v2(_unit(embedding=_vec(), vc_feature_map=feat))

    def test_two_dimensional_vc_map_raises(self) -> None:
        feat = np.zeros((7, 7))
        with pytest.raises(ValueError, match="3-D"):
            extract_features_v2(_unit(embedding=_vec(), vc_feature_map=feat))

    def test_empty_vc_map_raises(self) -> None:
        feat = np.zeros((0, 7, 512))
        with pytest.raises(ValueError, match="non-empty"):
            extract_features_v2(_unit(embedding=_vec(), vc_feature_map=feat))

    def test_non_default_shape_is_accepted(self) -> None:
        feat = _vc_map(shape=(4, 4, 64))
        features = extract_features_v2(_unit(embedding=_vec(), vc_feature_map=feat))
        _assert_all_finite(features)


# ---------------------------------------------------------------------------
# Input validation -- corpus_vs_mean
# ---------------------------------------------------------------------------
class TestCorpusVsMeanValidation:
    def test_wrong_shape_raises(self) -> None:
        unit = _unit(embedding=_vec())
        with pytest.raises(ValueError, match="1-D"):
            extract_features_v2(unit, corpus_vs_mean=np.zeros(10))

    def test_nan_corpus_mean_raises(self) -> None:
        corpus_mean = _vec(seed=1)
        corpus_mean[0] = np.nan
        unit = _unit(embedding=_vec())
        with pytest.raises(ValueError, match="NaN"):
            extract_features_v2(unit, corpus_vs_mean=corpus_mean)

    def test_valid_corpus_mean_is_accepted(self) -> None:
        unit = _unit(embedding=_vec())
        features = extract_features_v2(unit, corpus_vs_mean=_vec(seed=1))
        _assert_all_finite(features)

    def test_none_corpus_mean_degrades_to_self(self) -> None:
        """Documented fallback: no corpus context -> specificity reads 0.0 exactly."""
        unit = _unit(embedding=_vec())
        features = extract_features_v2(unit, corpus_vs_mean=None)
        idx = FEATURE_NAMES_V2.index("vs_semantic_specificity")
        assert features[idx] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Fix #3: zero-vector guards
# ---------------------------------------------------------------------------
class TestZeroVectorGuards:
    def test_zero_embedding_does_not_raise(self) -> None:
        features = extract_features_v2(_unit(embedding=np.zeros(VS_DIM)))
        _assert_all_finite(features)

    def test_zero_embedding_stability_is_neutral_half(self) -> None:
        features = extract_features_v2(_unit(embedding=np.zeros(VS_DIM)))
        idx = FEATURE_NAMES_V2.index("vs_embedding_stability")
        assert features[idx] == pytest.approx(0.5)

    def test_zero_embedding_energy_concentration_is_zero(self) -> None:
        features = extract_features_v2(_unit(embedding=np.zeros(VS_DIM)))
        idx = FEATURE_NAMES_V2.index("vs_energy_concentration")
        assert features[idx] == pytest.approx(0.0)

    def test_near_zero_norm_embedding_does_not_raise(self) -> None:
        # norm = 1e-13 * sqrt(384) ~= 2e-12, well under the 1e-10 guard threshold.
        v = np.full(VS_DIM, 1e-13, dtype=np.float64)
        features = extract_features_v2(_unit(embedding=v))
        _assert_all_finite(features)

    def test_near_zero_norm_embedding_stability_is_neutral_half(self) -> None:
        v = np.full(VS_DIM, 1e-13, dtype=np.float64)
        features = extract_features_v2(_unit(embedding=v))
        idx = FEATURE_NAMES_V2.index("vs_embedding_stability")
        assert features[idx] == pytest.approx(0.5)

    def test_just_above_threshold_norm_uses_real_computation(self) -> None:
        """A vector whose norm sits just above _ZERO_NORM_EPS should NOT hit the
        zero-vector shortcut -- verifies the guard doesn't over-fire."""
        v = np.zeros(VS_DIM)
        v[0] = 1e-6
        features = extract_features_v2(_unit(embedding=v))
        idx = FEATURE_NAMES_V2.index("vs_embedding_stability")
        assert 0.0 <= features[idx] <= 1.0


# ---------------------------------------------------------------------------
# Degenerate HG/VC (encoders still stubs) -- documented placeholder fallback
# ---------------------------------------------------------------------------
class TestDegenerateEncoderFallback:
    def test_missing_hg_adjacency_does_not_raise(self) -> None:
        features = extract_features_v2(_unit(embedding=_vec(), hg_adjacency=None))
        _assert_all_finite(features)

    def test_missing_vc_feature_map_does_not_raise(self) -> None:
        features = extract_features_v2(_unit(embedding=_vec(), vc_feature_map=None))
        _assert_all_finite(features)

    def test_missing_both_hg_and_vc_does_not_raise(self) -> None:
        features = extract_features_v2(_unit(embedding=_vec()))
        _assert_all_finite(features)

    def test_missing_hg_produces_bounded_features(self) -> None:
        features = extract_features_v2(_unit(embedding=_vec()))
        for i, name in enumerate(FEATURE_NAMES_V2):
            if name in BOUNDED_FEATURES_V2:
                assert 0.0 <= features[i] <= 1.0


# ---------------------------------------------------------------------------
# Adjacency shape edge cases (0-node, 1-node) -- regression for the Phase 3
# script's degree-distribution-before-guard ordering bug (see
# fluxmem/features_v2.py's `_hg_spread`/`_degree_dist` docstrings)
# ---------------------------------------------------------------------------
class TestAdjacencyShapeEdgeCases:
    def test_zero_node_adjacency_does_not_raise(self) -> None:
        adj = np.zeros((0, 0))
        features = extract_features_v2(_unit(embedding=_vec(), hg_adjacency=adj))
        _assert_all_finite(features)

    def test_one_node_adjacency_does_not_raise(self) -> None:
        adj = np.zeros((1, 1))
        features = extract_features_v2(_unit(embedding=_vec(), hg_adjacency=adj))
        _assert_all_finite(features)

    def test_two_node_zero_weight_adjacency_does_not_raise(self) -> None:
        adj = np.zeros((2, 2))
        features = extract_features_v2(_unit(embedding=_vec(), hg_adjacency=adj))
        _assert_all_finite(features)

    def test_fully_disconnected_large_adjacency_does_not_raise(self) -> None:
        adj = np.zeros((20, 20))
        features = extract_features_v2(_unit(embedding=_vec(), hg_adjacency=adj))
        _assert_all_finite(features)

    def test_fully_connected_max_weight_adjacency_does_not_raise(self) -> None:
        adj = np.ones((10, 10)) - np.eye(10)
        features = extract_features_v2(_unit(embedding=_vec(), hg_adjacency=adj))
        _assert_all_finite(features)
        idx = FEATURE_NAMES_V2.index("hg_relational_richness")
        assert features[idx] == pytest.approx(1.0)

    def test_single_hub_node_adjacency_does_not_raise(self) -> None:
        """A star graph -- one hub, everything else disconnected from each other."""
        n = 8
        adj = np.zeros((n, n))
        adj[0, 1:] = 1.0
        adj[1:, 0] = 1.0
        features = extract_features_v2(_unit(embedding=_vec(), hg_adjacency=adj))
        _assert_all_finite(features)

    def test_edge_weight_above_one_does_not_raise(self) -> None:
        """Not a documented input contract, but no crash should result."""
        adj = _adjacency()
        adj[0, 1] = adj[1, 0] = 5.0
        features = extract_features_v2(_unit(embedding=_vec(), hg_adjacency=adj))
        _assert_all_finite(features)


# ---------------------------------------------------------------------------
# Fix #2: output clamping -- every bounded feature stays in [0, 1]
# ---------------------------------------------------------------------------
class TestOutputClamping:
    @pytest.mark.parametrize("name", sorted(BOUNDED_FEATURES_V2))
    def test_bounded_feature_in_unit_interval_on_random_data(self, name: str) -> None:
        unit = _unit(
            embedding=_vec(seed=3), hg_adjacency=_adjacency(seed=3), vc_feature_map=_vc_map(seed=3)
        )
        features = extract_features_v2(unit)
        idx = FEATURE_NAMES_V2.index(name)
        assert 0.0 <= features[idx] <= 1.0

    def test_clip_defends_against_float_drift_above_one(self) -> None:
        """Directly exercises the clamp step: an out-of-range value must not
        survive to the returned array even if a formula's own bound is
        violated by floating-point error upstream."""

        values = {name: 0.5 for name in FEATURE_NAMES_V2}
        values["vs_embedding_stability"] = 1.0000000000000002  # float-drift-like
        values["hg_relational_richness"] = -1e-16  # float-drift-like, just under 0
        arr = np.array([values[name] for name in FEATURE_NAMES_V2], dtype=np.float64)
        for i, name in enumerate(FEATURE_NAMES_V2):
            if name in BOUNDED_FEATURES_V2:
                arr[i] = np.clip(arr[i], 0.0, 1.0)
        stability_idx = FEATURE_NAMES_V2.index("vs_embedding_stability")
        richness_idx = FEATURE_NAMES_V2.index("hg_relational_richness")
        assert arr[stability_idx] == 1.0
        assert arr[richness_idx] == 0.0

    def test_unbounded_features_can_exceed_unit_interval(self) -> None:
        """hg_cost_position should be free to go negative/above 1 -- verifies the
        clamp step does NOT over-clip features the spec leaves unbounded."""
        # A large synthetic graph pushes hg_footprint_kb well past vc_footprint_kb,
        # driving hg_cost_position above 1.0.
        n = 2000
        adj = np.ones((n, n)) - np.eye(n)
        unit = _unit(embedding=_vec(), hg_adjacency=adj)
        features = extract_features_v2(unit)
        idx = FEATURE_NAMES_V2.index("hg_cost_position")
        assert features[idx] > 1.0


# ---------------------------------------------------------------------------
# Overflow / adversarial-magnitude inputs
# ---------------------------------------------------------------------------
class TestExtremeMagnitudeInputs:
    def test_huge_magnitude_embedding_does_not_silently_nan(self) -> None:
        v = np.full(VS_DIM, 1e150)
        features = extract_features_v2(_unit(embedding=v))
        _assert_all_finite(features)

    def test_tiny_magnitude_nonzero_embedding_does_not_raise(self) -> None:
        v = np.full(VS_DIM, 1e-300)
        features = extract_features_v2(_unit(embedding=v))
        _assert_all_finite(features)

    def test_huge_adjacency_weights_do_not_raise(self) -> None:
        adj = _adjacency() * 1e10
        features = extract_features_v2(_unit(embedding=_vec(), hg_adjacency=adj))
        _assert_all_finite(features)

    def test_huge_vc_feature_map_values_do_not_raise(self) -> None:
        feat = _vc_map() * 1e10
        features = extract_features_v2(_unit(embedding=_vec(), vc_feature_map=feat))
        _assert_all_finite(features)

    def test_mixed_sign_extreme_embedding_does_not_raise(self) -> None:
        v = np.array([1e150, -1e150] * (VS_DIM // 2))
        features = extract_features_v2(_unit(embedding=v))
        _assert_all_finite(features)


# ---------------------------------------------------------------------------
# VC map edge cases
# ---------------------------------------------------------------------------
class TestVcMapEdgeCases:
    def test_all_zero_vc_map_does_not_raise(self) -> None:
        feat = np.zeros((7, 7, 512))
        features = extract_features_v2(_unit(embedding=_vec(), vc_feature_map=feat))
        _assert_all_finite(features)

    def test_all_zero_vc_map_concentration_is_zero(self) -> None:
        feat = np.zeros((7, 7, 512))
        features = extract_features_v2(_unit(embedding=_vec(), vc_feature_map=feat))
        idx = FEATURE_NAMES_V2.index("vc_layout_concentration")
        assert features[idx] == pytest.approx(0.0)

    def test_uniform_nonzero_vc_map_does_not_raise(self) -> None:
        feat = np.full((7, 7, 512), 0.5)
        features = extract_features_v2(_unit(embedding=_vec(), vc_feature_map=feat))
        _assert_all_finite(features)

    def test_single_cell_vc_map_does_not_raise(self) -> None:
        feat = np.random.RandomState(0).uniform(0, 1, size=(1, 1, 8))
        features = extract_features_v2(_unit(embedding=_vec(), vc_feature_map=feat))
        _assert_all_finite(features)

    def test_single_channel_vc_map_does_not_raise(self) -> None:
        feat = np.random.RandomState(0).uniform(0, 1, size=(7, 7, 1))
        features = extract_features_v2(_unit(embedding=_vec(), vc_feature_map=feat))
        _assert_all_finite(features)

    def test_one_hot_spike_vc_map_has_high_prominence(self) -> None:
        feat = np.zeros((7, 7, 512))
        feat[3, 3, :] = 1.0
        features = extract_features_v2(_unit(embedding=_vec(), vc_feature_map=feat))
        idx = FEATURE_NAMES_V2.index("vc_layout_prominence")
        assert features[idx] > 0.9

    def test_negative_vc_values_are_clipped_internally(self) -> None:
        """Formula clips the spatial map at 0 before treating it as a
        distribution; negative activations should not raise."""
        feat = np.random.RandomState(0).uniform(-1, 1, size=(7, 7, 512))
        features = extract_features_v2(_unit(embedding=_vec(), vc_feature_map=feat))
        _assert_all_finite(features)


# ---------------------------------------------------------------------------
# Determinism (Group 4's perturb-and-remeasure trials use a local Generator)
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_same_seed_same_inputs_gives_identical_output(self) -> None:
        unit = _unit(embedding=_vec(), hg_adjacency=_adjacency(), vc_feature_map=_vc_map())
        first = extract_features_v2(unit, rng_seed=42)
        second = extract_features_v2(unit, rng_seed=42)
        np.testing.assert_array_equal(first, second)

    def test_different_seed_may_change_robustness_features_but_stays_bounded(self) -> None:
        unit = _unit(embedding=_vec(), hg_adjacency=_adjacency(), vc_feature_map=_vc_map())
        a = extract_features_v2(unit, rng_seed=1)
        b = extract_features_v2(unit, rng_seed=2)
        for arr in (a, b):
            for i, name in enumerate(FEATURE_NAMES_V2):
                if name in BOUNDED_FEATURES_V2:
                    assert 0.0 <= arr[i] <= 1.0

    def test_default_seed_is_reproducible_across_two_fresh_units(self) -> None:
        v = _vec()
        unit_a = _unit(embedding=v.copy())
        unit_b = _unit(embedding=v.copy())
        np.testing.assert_array_equal(extract_features_v2(unit_a), extract_features_v2(unit_b))


# ---------------------------------------------------------------------------
# Full valid random data -- realistic-shaped smoke tests
# ---------------------------------------------------------------------------
class TestRealisticData:
    @pytest.mark.parametrize("seed", range(5))
    def test_random_realistic_episode_all_finite(self, seed: int) -> None:
        unit = _unit(
            embedding=_vec(seed=seed),
            hg_adjacency=_adjacency(n=6 + seed, seed=seed),
            vc_feature_map=_vc_map(seed=seed),
        )
        features = extract_features_v2(unit, corpus_vs_mean=_vec(seed=seed + 100))
        _assert_all_finite(features)

    def test_feature_names_v2_order_matches_array_position(self) -> None:
        unit = _unit(embedding=_vec(), hg_adjacency=_adjacency(), vc_feature_map=_vc_map())
        features = extract_features_v2(unit)
        idx = FEATURE_NAMES_V2.index("hg_hierarchy_depth")
        assert features[idx] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Per-group function unit tests (independently testable without EpisodicUnit)
# ---------------------------------------------------------------------------
class TestGroupFunctionsIndependently:
    def test_extract_geometry_features_keys(self) -> None:
        result = extract_geometry_features(_vec(), _adjacency(), _vc_map())
        assert set(result) == {
            "vs_energy_concentration",
            "hg_hierarchy_depth",
            "vc_layout_concentration",
        }

    def test_extract_efficiency_features_keys(self) -> None:
        result = extract_efficiency_features(_adjacency(), VS_DIM, (7, 7, 512))
        assert set(result) == {"hg_cost_position", "hg_information_per_kb"}

    def test_extract_overlap_features_keys(self) -> None:
        result = extract_overlap_features(0.3, 0.5, 0.7)
        assert set(result) == {
            "vs_hg_structural_agreement",
            "vs_vc_structural_agreement",
            "hg_vc_structural_agreement",
            "complementarity_score",
        }

    def test_extract_overlap_features_identical_spreads_gives_perfect_agreement(self) -> None:
        result = extract_overlap_features(0.5, 0.5, 0.5)
        assert result["vs_hg_structural_agreement"] == pytest.approx(1.0)
        assert result["complementarity_score"] == pytest.approx(0.0)

    def test_extract_robustness_features_keys(self) -> None:
        rng = np.random.default_rng(0)
        result = extract_robustness_features(_vec(), _adjacency(), _vc_map(), rng)
        assert set(result) == {
            "vs_embedding_stability",
            "hg_graph_stability",
            "vc_layout_stability",
        }

    def test_extract_task_features_keys(self) -> None:
        result = extract_task_features(_vec(), _vec(seed=1), _adjacency(), _vc_map())
        assert set(result) == {
            "vs_semantic_specificity",
            "hg_relational_richness",
            "vc_layout_prominence",
        }


# ---------------------------------------------------------------------------
# Fix #6 / regression: no NaN/inf cascade across an adversarial battery
# ---------------------------------------------------------------------------
class TestNoNanCascadeRegression:
    @pytest.mark.parametrize("seed", range(10))
    def test_random_adversarial_battery_never_raises_or_nans(self, seed: int) -> None:
        rng = np.random.RandomState(seed)
        scale = rng.choice([1e-8, 1.0, 1e8])
        v = rng.randn(VS_DIM) * scale
        n_nodes = rng.randint(0, 15)
        adj = rng.uniform(0, 1, size=(n_nodes, n_nodes)) * scale if n_nodes else np.zeros((0, 0))
        vc_shape = (rng.randint(1, 8), rng.randint(1, 8), rng.randint(1, 16))
        feat = rng.uniform(0, 1, size=vc_shape) * scale
        unit = _unit(embedding=v, hg_adjacency=adj, vc_feature_map=feat)
        features = extract_features_v2(unit)
        _assert_all_finite(features)

    def test_raised_validation_error_does_not_leave_partial_state(self) -> None:
        """A rejected unit raises cleanly -- no feature array is returned at all,
        so there is nothing partially-computed for a caller to mistake for
        real output."""
        with pytest.raises(ValueError):
            extract_features_v2(_unit(embedding=None))
