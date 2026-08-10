"""Tests for fluxmem/ltsm.py -- promotion gate and FAISS store."""

from __future__ import annotations

import numpy as np
import pytest

from fluxmem.config import PromotionThresholds, UtilityWeights
from fluxmem.interfaces import EpisodicUnit, MemoryFormat, Turn
from fluxmem.ltsm import FaissVectorStore, is_eligible, promote, recency_gate
from fluxmem.mtem import MidTermEpisodicMemory, recency_decay, utility_score


def _episode(
    episode_id: str = "e0",
    access_count: int = 0,
    created_at: float = 0.0,
    last_access: float = 0.0,
    embedding: np.ndarray | None = None,
) -> EpisodicUnit:
    turn = Turn(
        turn_id=f"{episode_id}-t0", user="a b c", assistant="a b c", timestamp=0.0, last_access=0.0
    )
    return EpisodicUnit(
        episode_id=episode_id,
        turns=[turn],
        primary_format=MemoryFormat.HG,
        access_count=access_count,
        created_at=created_at,
        last_access=last_access,
        embedding=embedding,
    )


class TestRecencyGate:
    def test_zero_elapsed_is_one(self) -> None:
        episode = _episode(last_access=10.0)
        assert recency_gate(episode, now=10.0, max_age=5.0) == pytest.approx(1.0)

    def test_at_max_age_is_zero(self) -> None:
        episode = _episode(last_access=0.0)
        assert recency_gate(episode, now=5.0, max_age=5.0) == pytest.approx(0.0)

    def test_beyond_max_age_clamped_to_zero(self) -> None:
        episode = _episode(last_access=0.0)
        assert recency_gate(episode, now=100.0, max_age=5.0) == 0.0

    def test_non_positive_max_age_rejected(self) -> None:
        with pytest.raises(ValueError):
            recency_gate(_episode(), now=1.0, max_age=0.0)


class TestEligibilityTruthTable:
    """All four U-pass/r-pass combinations."""

    def _thresholds(self) -> PromotionThresholds:
        return PromotionThresholds(tau_u=0.5, tau_r=0.5, max_age=10.0)

    def test_u_pass_r_pass(self) -> None:
        weights = UtilityWeights(w1=1.0, w2=0.0, w3=0.0)
        episode = _episode(access_count=10, created_at=0.0, last_access=10.0)
        assert is_eligible(episode, weights, self._thresholds(), now=10.0) is True

    def test_u_pass_r_fail(self) -> None:
        weights = UtilityWeights(w1=1.0, w2=0.0, w3=0.0)
        episode = _episode(access_count=10, created_at=0.0, last_access=0.0)
        assert is_eligible(episode, weights, self._thresholds(), now=10.0) is False

    def test_u_fail_r_pass(self) -> None:
        weights = UtilityWeights(w1=1.0, w2=0.0, w3=0.0)
        episode = _episode(access_count=0, created_at=0.0, last_access=10.0)
        assert is_eligible(episode, weights, self._thresholds(), now=10.0) is False

    def test_u_fail_r_fail(self) -> None:
        weights = UtilityWeights(w1=1.0, w2=0.0, w3=0.0)
        episode = _episode(access_count=0, created_at=0.0, last_access=0.0)
        assert is_eligible(episode, weights, self._thresholds(), now=10.0) is False


class TestBoundaryIsGreaterEqual:
    def test_u_exactly_at_tau_u_passes(self) -> None:
        weights = UtilityWeights(w1=1.0, w2=0.0, w3=0.0)
        thresholds = PromotionThresholds(tau_u=0.5, tau_r=0.0, max_age=10.0)
        # age_in_turns = now - created_at = 9 -> access_frequency = 5/(9+1) = 0.5
        episode = _episode(access_count=5, created_at=1.0, last_access=10.0)
        u = utility_score(episode, weights, now=10.0)
        assert u == pytest.approx(0.5)
        assert is_eligible(episode, weights, thresholds, now=10.0) is True

    def test_r_exactly_at_tau_r_passes(self) -> None:
        # w3=1 so U (via recency_decay, default half_life) trivially clears a
        # tiny tau_u, isolating the r boundary as the only thing under test.
        weights = UtilityWeights(w1=0.0, w2=0.0, w3=1.0)
        thresholds = PromotionThresholds(tau_u=0.01, tau_r=0.5, max_age=10.0)
        episode = _episode(last_access=5.0)
        r = recency_gate(episode, now=10.0, max_age=10.0)
        assert r == pytest.approx(0.5)
        assert is_eligible(episode, weights, thresholds, now=10.0) is True


class TestRecencyRankOrderInvariant:
    def test_d_and_r_never_disagree_on_which_episode_is_more_recent(self) -> None:
        pairs = [(0.0, 5.0), (5.0, 0.0), (3.0, 3.0), (9.0, 1.0), (2.0, 8.0)]
        for last_access_a, last_access_b in pairs:
            a = _episode("a", last_access=last_access_a)
            b = _episode("b", last_access=last_access_b)
            now = 10.0
            d_a = recency_decay(a, now=now, half_life=5.0)
            d_b = recency_decay(b, now=now, half_life=5.0)
            r_a = recency_gate(a, now=now, max_age=10.0)
            r_b = recency_gate(b, now=now, max_age=10.0)
            # d and r must rank the pair the same way (both strictly
            # decreasing in dt from the same last_access).
            assert (d_a > d_b) == (r_a > r_b)
            assert (d_a < d_b) == (r_a < r_b)
            assert (d_a == d_b) == (r_a == r_b)


class TestWeightsChangePromotionSet:
    def test_changing_weights_changes_which_episodes_promote(self) -> None:
        # "frequent" is accessed often but low intensity/stale-ish; "intense"
        # is rarely accessed but long/intense and fresh.
        frequent = _episode("frequent", access_count=10, created_at=0.0, last_access=10.0)
        intense = _episode("intense", access_count=0, created_at=0.0, last_access=10.0)
        intense.turns = [
            Turn(
                turn_id="intense-t0",
                user=" ".join(["w"] * 100),
                assistant=" ".join(["w"] * 100),
                timestamp=0.0,
                last_access=0.0,
            )
        ]
        thresholds = PromotionThresholds(tau_u=0.3, tau_r=0.0, max_age=100.0)

        usage_only = UtilityWeights(w1=1.0, w2=0.0, w3=0.0)
        intensity_only = UtilityWeights(w1=0.0, w2=1.0, w3=0.0)

        assert is_eligible(frequent, usage_only, thresholds, now=10.0) is True
        assert is_eligible(intense, usage_only, thresholds, now=10.0) is False

        assert is_eligible(intense, intensity_only, thresholds, now=10.0) is True
        assert is_eligible(frequent, intensity_only, thresholds, now=10.0) is False


class TestFaissVectorStore:
    def test_round_trip_add_search(self) -> None:
        store = FaissVectorStore(dim=4)
        vec = np.array([1.0, 0.0, 0.0, 0.0])
        store.add(["e0"], vec.reshape(1, -1))
        results = store.search(vec, k=1)
        assert len(results) == 1
        assert results[0][0] == "e0"
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)

    def test_vectors_are_l2_normalized_before_add(self) -> None:
        store = FaissVectorStore(dim=3)
        vec = np.array([3.0, 4.0, 0.0])  # norm = 5
        store.add(["e0"], vec.reshape(1, -1))
        # Query with the unit vector in the same direction -> score ~1.0
        # only holds if the stored vector was normalized too.
        unit = vec / np.linalg.norm(vec)
        results = store.search(unit, k=1)
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)

    def test_empty_store_search_returns_empty(self) -> None:
        store = FaissVectorStore(dim=4)
        assert store.search(np.zeros(4), k=5) == []

    def test_len(self) -> None:
        store = FaissVectorStore(dim=2)
        assert len(store) == 0
        store.add(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]]))
        assert len(store) == 2

    def test_mismatched_ids_and_vectors_rejected(self) -> None:
        store = FaissVectorStore(dim=2)
        with pytest.raises(ValueError):
            store.add(["a", "b"], np.array([[1.0, 0.0]]))

    def test_wrong_dim_rejected(self) -> None:
        store = FaissVectorStore(dim=4)
        with pytest.raises(ValueError):
            store.add(["a"], np.array([[1.0, 0.0]]))

    def test_search_ranks_closest_first(self) -> None:
        store = FaissVectorStore(dim=2)
        store.add(["orthogonal", "same"], np.array([[0.0, 1.0], [1.0, 0.0]]))
        results = store.search(np.array([1.0, 0.0]), k=2)
        assert results[0][0] == "same"
        assert results[1][0] == "orthogonal"


class TestPromote:
    def test_promotion_removes_from_mtem_and_appears_in_store(self) -> None:
        mtem = MidTermEpisodicMemory(capacity=10, weights=UtilityWeights(w1=1.0, w2=0.0, w3=0.0))
        eligible = _episode(
            "eligible",
            access_count=10,
            created_at=0.0,
            last_access=10.0,
            embedding=np.array([1.0, 0.0]),
        )
        ineligible = _episode(
            "ineligible",
            access_count=0,
            created_at=0.0,
            last_access=0.0,
            embedding=np.array([0.0, 1.0]),
        )
        mtem.add(eligible, now=10.0)
        mtem.add(ineligible, now=10.0)

        store = FaissVectorStore(dim=2)
        thresholds = PromotionThresholds(tau_u=0.5, tau_r=0.5, max_age=10.0)
        weights = UtilityWeights(w1=1.0, w2=0.0, w3=0.0)

        promoted = promote(mtem, store, weights, thresholds, now=10.0)

        assert [e.episode_id for e in promoted] == ["eligible"]
        assert "eligible" not in mtem
        assert "ineligible" in mtem
        assert len(store) == 1
        results = store.search(np.array([1.0, 0.0]), k=1)
        assert results[0][0] == "eligible"

    def test_episodes_without_embedding_are_skipped(self) -> None:
        mtem = MidTermEpisodicMemory(capacity=10, weights=UtilityWeights(w1=1.0, w2=0.0, w3=0.0))
        no_embedding = _episode(
            "no-embed", access_count=10, created_at=0.0, last_access=10.0, embedding=None
        )
        mtem.add(no_embedding, now=10.0)

        store = FaissVectorStore(dim=2)
        thresholds = PromotionThresholds(tau_u=0.1, tau_r=0.1, max_age=10.0)
        weights = UtilityWeights(w1=1.0, w2=0.0, w3=0.0)

        promoted = promote(mtem, store, weights, thresholds, now=10.0)
        assert promoted == []
        assert "no-embed" in mtem
        assert len(store) == 0
