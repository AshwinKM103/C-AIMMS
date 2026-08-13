"""Tests for fluxmem/mtem.py -- utility score and MTEM tier."""

from __future__ import annotations

import math

import pytest

from fluxmem.config import UtilityWeights
from fluxmem.interfaces import EpisodicUnit, MemoryFormat, Turn
from fluxmem.mtem import (
    MidTermEpisodicMemory,
    access_frequency,
    interaction_intensity,
    recency_decay,
    utility_score,
)


def _episode(
    episode_id: str = "e0",
    access_count: int = 0,
    created_at: float = 0.0,
    last_access: float = 0.0,
    n_turns: int = 2,
    tokens_per_turn: int = 5,
) -> EpisodicUnit:
    words = " ".join(f"w{i}" for i in range(tokens_per_turn))
    turns = [
        Turn(
            turn_id=f"{episode_id}-t{i}",
            user=words,
            assistant=words,
            timestamp=float(i),
            last_access=float(i),
        )
        for i in range(n_turns)
    ]
    return EpisodicUnit(
        episode_id=episode_id,
        turns=turns,
        primary_format=MemoryFormat.HG,
        access_count=access_count,
        created_at=created_at,
        last_access=last_access,
    )


class TestAccessFrequency:
    def test_zero_access_count_is_zero(self) -> None:
        episode = _episode(access_count=0)
        assert access_frequency(episode, now=10.0) == 0.0

    def test_more_accesses_yield_higher_frequency(self) -> None:
        low = _episode(access_count=1, created_at=0.0)
        high = _episode(access_count=5, created_at=0.0)
        assert access_frequency(high, now=10.0) > access_frequency(low, now=10.0)

    def test_older_age_lowers_frequency_for_same_access_count(self) -> None:
        younger = _episode(access_count=3, created_at=8.0)
        older = _episode(access_count=3, created_at=0.0)
        assert access_frequency(younger, now=10.0) > access_frequency(older, now=10.0)

    def test_clipped_to_one(self) -> None:
        episode = _episode(access_count=1000, created_at=10.0)
        assert access_frequency(episode, now=10.0) == 1.0


class TestInteractionIntensity:
    def test_zero_turns_is_zero(self) -> None:
        episode = _episode(n_turns=0)
        assert interaction_intensity(episode, reference_length=50.0) == 0.0

    def test_more_tokens_yields_higher_intensity(self) -> None:
        short = _episode(tokens_per_turn=2)
        long = _episode(tokens_per_turn=20)
        assert interaction_intensity(long, reference_length=50.0) > interaction_intensity(
            short, reference_length=50.0
        )

    def test_clipped_to_one(self) -> None:
        episode = _episode(tokens_per_turn=1000)
        assert interaction_intensity(episode, reference_length=50.0) == 1.0

    def test_bounded_zero_to_one(self) -> None:
        for tokens in (0, 1, 10, 100, 10_000):
            episode = _episode(tokens_per_turn=tokens)
            value = interaction_intensity(episode, reference_length=50.0)
            assert 0.0 <= value <= 1.0


class TestRecencyDecay:
    def test_zero_elapsed_time_is_one(self) -> None:
        episode = _episode(last_access=10.0)
        assert recency_decay(episode, now=10.0, half_life=5.0) == pytest.approx(1.0)

    def test_more_elapsed_time_yields_lower_decay(self) -> None:
        recent = _episode(last_access=9.0)
        stale = _episode(last_access=0.0)
        assert recency_decay(recent, now=10.0, half_life=5.0) > recency_decay(
            stale, now=10.0, half_life=5.0
        )

    def test_half_life_semantics(self) -> None:
        episode = _episode(last_access=0.0)
        value = recency_decay(episode, now=5.0, half_life=5.0)
        assert value == pytest.approx(math.exp(-1.0))

    def test_non_positive_half_life_rejected(self) -> None:
        episode = _episode()
        with pytest.raises(ValueError):
            recency_decay(episode, now=10.0, half_life=0.0)


class TestUtilityScore:
    def test_zero_weights_yield_zero(self) -> None:
        episode = _episode(access_count=5, last_access=10.0)
        weights = UtilityWeights(w1=0.0, w2=0.0, w3=0.0)
        assert utility_score(episode, weights, now=10.0) == 0.0

    def test_isolates_c_term_when_other_weights_zeroed(self) -> None:
        episode = _episode(access_count=4, created_at=0.0, last_access=0.0, tokens_per_turn=0)
        weights = UtilityWeights(w1=1.0, w2=0.0, w3=0.0)
        expected = access_frequency(episode, now=10.0)
        assert utility_score(episode, weights, now=10.0) == pytest.approx(expected)

    def test_isolates_l_term_when_other_weights_zeroed(self) -> None:
        episode = _episode(tokens_per_turn=20)
        weights = UtilityWeights(w1=0.0, w2=1.0, w3=0.0)
        expected = interaction_intensity(episode, reference_length=DEFAULT_REFERENCE_LENGTH)
        assert utility_score(episode, weights, now=10.0) == pytest.approx(expected)

    def test_isolates_d_term_when_other_weights_zeroed(self) -> None:
        episode = _episode(last_access=7.0)
        weights = UtilityWeights(w1=0.0, w2=0.0, w3=1.0)
        expected = recency_decay(episode, now=10.0, half_life=DEFAULT_HALF_LIFE)
        assert utility_score(episode, weights, now=10.0) == pytest.approx(expected)

    def test_more_accesses_increase_utility_all_else_equal(self) -> None:
        weights = UtilityWeights(w1=1.0, w2=1.0, w3=1.0)
        low = _episode(access_count=1, created_at=0.0, last_access=10.0)
        high = _episode(access_count=10, created_at=0.0, last_access=10.0)
        assert utility_score(high, weights, now=10.0) > utility_score(low, weights, now=10.0)

    def test_more_elapsed_time_decreases_utility_all_else_equal(self) -> None:
        weights = UtilityWeights(w1=0.0, w2=0.0, w3=1.0)
        recent = _episode(last_access=9.0)
        stale = _episode(last_access=0.0)
        assert utility_score(recent, weights, now=10.0) > utility_score(stale, weights, now=10.0)


from fluxmem.mtem import DEFAULT_HALF_LIFE, DEFAULT_REFERENCE_LENGTH  # noqa: E402


class TestMidTermEpisodicMemory:
    def test_add_below_capacity_no_eviction(self) -> None:
        mtem = MidTermEpisodicMemory(capacity=10)
        evicted = mtem.add(_episode("e0"), now=0.0)
        assert evicted is None
        assert len(mtem) == 1

    def test_get_bumps_access_count_and_last_access(self) -> None:
        mtem = MidTermEpisodicMemory(capacity=10)
        mtem.add(_episode("e0", access_count=0), now=0.0)
        episode = mtem.get("e0", now=5.0)
        assert episode.access_count == 1
        assert episode.last_access == 5.0
        episode2 = mtem.get("e0", now=6.0)
        assert episode2.access_count == 2
        assert episode2.last_access == 6.0

    def test_capacity_cap_evicts_lowest_utility(self) -> None:
        weights = UtilityWeights(w1=1.0, w2=0.0, w3=0.0)
        mtem = MidTermEpisodicMemory(capacity=2, weights=weights)
        mtem.add(_episode("low", access_count=0, created_at=0.0), now=10.0)
        mtem.add(_episode("high", access_count=10, created_at=0.0), now=10.0)
        evicted = mtem.add(_episode("mid", access_count=3, created_at=0.0), now=10.0)
        assert evicted is not None
        assert evicted.episode_id == "low"
        assert len(mtem) == 2
        assert "high" in mtem
        assert "mid" in mtem

    def test_merge_combines_turns_and_access_count(self) -> None:
        mtem = MidTermEpisodicMemory(capacity=10)
        target = _episode("target", access_count=2, n_turns=1)
        incoming = _episode("incoming", access_count=3, n_turns=1, last_access=99.0)
        mtem.add(target, now=0.0)
        merged = mtem.merge("target", incoming)
        assert merged.access_count == 5
        assert len(merged.turns) == 2
        assert merged.last_access == 99.0

    def test_promotion_candidates_filters_by_tau_u(self) -> None:
        weights = UtilityWeights(w1=1.0, w2=0.0, w3=0.0)
        mtem = MidTermEpisodicMemory(capacity=10, weights=weights)
        mtem.add(_episode("low", access_count=0, created_at=0.0), now=10.0)
        mtem.add(_episode("high", access_count=10, created_at=0.0), now=10.0)
        candidates = mtem.promotion_candidates(tau_u=0.5, now=10.0)
        ids = {e.episode_id for e in candidates}
        assert ids == {"high"}

    def test_zero_capacity_rejected(self) -> None:
        with pytest.raises(ValueError):
            MidTermEpisodicMemory(capacity=0)

    def test_remove(self) -> None:
        mtem = MidTermEpisodicMemory(capacity=10)
        mtem.add(_episode("e0"), now=0.0)
        removed = mtem.remove("e0")
        assert removed.episode_id == "e0"
        assert "e0" not in mtem
