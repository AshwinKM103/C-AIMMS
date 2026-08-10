"""Tests for fluxmem/fusion.py -- BMM-gated fusion (highest logic risk)."""

from __future__ import annotations

import numpy as np
import pytest

from fluxmem.config import FusionConfig
from fluxmem.fusion import BetaMixtureModel, normalize_scores, select_merge_target
from fluxmem.interfaces import EpisodicUnit, MemoryFormat, Turn


def _episode(episode_id: str) -> EpisodicUnit:
    turn = Turn(turn_id=f"{episode_id}-t0", user="x", assistant="x", timestamp=0.0, last_access=0.0)
    return EpisodicUnit(episode_id=episode_id, turns=[turn], primary_format=MemoryFormat.HG)


class TestNormalizeScores:
    def test_basic_min_max(self) -> None:
        scores = np.array([0.0, 5.0, 10.0])
        x = normalize_scores(scores, eps=0.0)
        np.testing.assert_allclose(x, [0.0, 0.5, 1.0])

    def test_eps_inset_bounds(self) -> None:
        scores = np.array([0.0, 10.0])
        x = normalize_scores(scores, eps=0.1)
        np.testing.assert_allclose(x, [0.1, 0.9])

    def test_degenerate_all_identical_scores_is_half(self) -> None:
        scores = np.array([3.0, 3.0, 3.0])
        x = normalize_scores(scores, eps=0.1)
        np.testing.assert_allclose(x, [0.5, 0.5, 0.5])

    def test_single_score_is_half(self) -> None:
        x = normalize_scores(np.array([7.0]), eps=0.1)
        np.testing.assert_allclose(x, [0.5])

    def test_output_bounded_by_eps(self) -> None:
        scores = np.array([1.0, 2.0, 3.0, 100.0])
        x = normalize_scores(scores, eps=0.05)
        assert x.min() >= 0.05 - 1e-9
        assert x.max() <= 0.95 + 1e-9


class TestBetaMixtureModelRecovery:
    def test_recovers_two_component_mixture_within_tolerance(self) -> None:
        rng = np.random.default_rng(0)
        n = 2000
        true_pi = 0.4
        n_low = int(n * true_pi)
        n_high = n - n_low
        low = rng.beta(2.0, 8.0, size=n_low)  # mean 0.2
        high = rng.beta(8.0, 2.0, size=n_high)  # mean 0.8
        x = np.concatenate([low, high])
        rng.shuffle(x)

        bmm = BetaMixtureModel(n_iter=100)
        params = bmm.fit(x)

        means = params.alpha / (params.alpha + params.beta)
        low_mean = means.min()
        high_mean = means.max()
        assert low_mean == pytest.approx(0.2, abs=0.05)
        assert high_mean == pytest.approx(0.8, abs=0.05)

        high_idx = int(np.argmax(means))
        assert params.high_compat_index == high_idx
        pi_high = params.pi[high_idx]
        assert pi_high == pytest.approx(1 - true_pi, abs=0.08)

    def test_log_space_e_step_does_not_underflow_near_boundaries(self) -> None:
        x = np.array([1e-8, 1e-6, 1e-4, 0.5, 1 - 1e-4, 1 - 1e-6, 1 - 1e-8])
        bmm = BetaMixtureModel(n_iter=20)
        params = bmm.fit(x)
        assert np.all(np.isfinite(params.pi))
        assert np.all(np.isfinite(params.alpha))
        assert np.all(np.isfinite(params.beta))
        posteriors = bmm.posterior(x)
        assert np.all(np.isfinite(posteriors))
        assert np.all(posteriors >= 0.0) and np.all(posteriors <= 1.0)

    def test_component_labels_order_invariant_for_decision(self) -> None:
        """A fit that swaps which index is 'high' must not change high_compat identification."""
        rng = np.random.default_rng(1)
        low = rng.beta(2.0, 8.0, size=500)
        high = rng.beta(8.0, 2.0, size=500)
        x = np.concatenate([low, high])

        bmm = BetaMixtureModel(n_iter=100)
        params = bmm.fit(x)
        means = params.alpha / (params.alpha + params.beta)
        # Regardless of which internal index ended up "component 0" vs "1",
        # high_compat_index must point at the larger-mean component.
        assert params.high_compat_index == int(np.argmax(means))

    def test_single_point_does_not_crash(self) -> None:
        bmm = BetaMixtureModel(n_iter=10)
        params = bmm.fit(np.array([0.5]))
        assert np.all(np.isfinite(params.alpha))
        assert np.all(np.isfinite(params.beta))

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError):
            BetaMixtureModel().fit(np.array([]))

    def test_posterior_before_fit_raises(self) -> None:
        bmm = BetaMixtureModel()
        with pytest.raises(RuntimeError):
            bmm.posterior(np.array([0.5]))

    def test_extreme_boundary_mass_yields_valid_beta_params(self) -> None:
        """A component's mu can be pushed toward exactly 0 or 1 when the data
        is heavily concentrated at one boundary; alpha/beta must stay > 0."""
        x = np.concatenate([np.full(50, 1e-9), np.array([0.5, 0.9, 0.95])])
        bmm = BetaMixtureModel(n_iter=100)
        params = bmm.fit(x)
        assert np.all(params.alpha > 0.0)
        assert np.all(params.beta > 0.0)
        assert np.all(np.isfinite(params.alpha))
        assert np.all(np.isfinite(params.beta))
        posteriors = bmm.posterior(x)
        assert np.all(np.isfinite(posteriors))


class TestSelectMergeTarget:
    def _scorer(self, score_map: dict[str, float]):
        def scorer(incoming, candidate):
            return score_map[candidate.episode_id]

        return scorer

    def test_empty_candidate_set_is_new(self) -> None:
        incoming = _episode("incoming")
        cfg = FusionConfig()
        decision = select_merge_target(incoming, [], self._scorer({}), cfg)
        assert decision.action == "NEW"
        assert decision.target_id is None

    def test_single_candidate_high_score_merges(self) -> None:
        incoming = _episode("incoming")
        candidates = [_episode("c0")]
        scorer = self._scorer({"c0": 0.9})
        cfg = FusionConfig(tau=0.1, m_min=1, em_iters=20)
        decision = select_merge_target(incoming, candidates, scorer, cfg)
        assert decision.action == "MERGE"
        assert decision.target_id == "c0"

    def test_all_identical_scores(self) -> None:
        incoming = _episode("incoming")
        candidates = [_episode(f"c{i}") for i in range(4)]
        scorer = self._scorer({f"c{i}": 0.5 for i in range(4)})
        cfg = FusionConfig(tau=0.5, m_min=1, em_iters=20)
        decision = select_merge_target(incoming, candidates, scorer, cfg)
        assert decision.action in {"MERGE", "NEW"}
        assert set(decision.normalized_scores) == {f"c{i}" for i in range(4)}
        # Identical raw scores -> identical normalized scores (0.5 each, Eq. 17 degenerate branch).
        assert all(v == pytest.approx(0.5) for v in decision.normalized_scores.values())

    def test_min_keep_fallback_fires_when_threshold_too_strict(self) -> None:
        incoming = _episode("incoming")
        candidates = [_episode("low"), _episode("mid"), _episode("high")]
        scorer = self._scorer({"low": 0.1, "mid": 0.5, "high": 0.9})
        # tau=1.01 is unreachable by any posterior (posteriors are in [0,1]),
        # forcing the min-keep fallback to be the only path to a MERGE.
        cfg = FusionConfig(tau=0.999999, m_min=1, em_iters=20)
        decision = select_merge_target(incoming, candidates, scorer, cfg)
        assert decision.action == "MERGE"
        # Fallback picks by raw normalized score x_i, so "high" (max raw score) wins.
        assert decision.target_id == "high"

    def test_m_min_zero_with_unreachable_threshold_is_new(self) -> None:
        incoming = _episode("incoming")
        candidates = [_episode(f"c{i}") for i in range(4)]
        # Identical raw scores -> Eq. 17's degenerate branch (x=0.5 for all)
        # -> the two initial BMM components are symmetric around 0.5, so no
        # point's posterior can plausibly clear a near-1.0 threshold.
        scorer = self._scorer({f"c{i}": 0.5 for i in range(4)})
        cfg = FusionConfig(tau=0.99, m_min=0, em_iters=20)
        decision = select_merge_target(incoming, candidates, scorer, cfg)
        assert decision.action == "NEW"

    def test_clear_high_compatibility_candidate_merges(self) -> None:
        incoming = _episode("incoming")
        # A cluster of low scores plus one clear high-compatibility outlier.
        low_scores = {f"low{i}": 0.05 + 0.01 * i for i in range(10)}
        candidates = [_episode(cid) for cid in low_scores] + [_episode("winner")]
        scores = dict(low_scores)
        scores["winner"] = 0.95
        cfg = FusionConfig(tau=0.7, m_min=1, em_iters=100)
        decision = select_merge_target(incoming, candidates, self._scorer(scores), cfg)
        assert decision.action == "MERGE"
        assert decision.target_id == "winner"

    def test_posterior_scores_present_for_all_candidates(self) -> None:
        incoming = _episode("incoming")
        candidates = [_episode(f"c{i}") for i in range(3)]
        scorer = self._scorer({"c0": 0.1, "c1": 0.5, "c2": 0.9})
        cfg = FusionConfig(tau=0.5, m_min=1, em_iters=20)
        decision = select_merge_target(incoming, candidates, scorer, cfg)
        assert set(decision.posterior_scores) == {"c0", "c1", "c2"}
        for v in decision.posterior_scores.values():
            assert 0.0 <= v <= 1.0
