"""Tests for fluxmem/features.py -- the 7-scalar feature vector."""

from __future__ import annotations

import math

import pytest

from fluxmem.features import FEATURE_DIM, FEATURE_NAMES, extract_features
from fluxmem.interfaces import (
    Entity,
    EpisodicUnit,
    FakeEntityExtractor,
    MemoryFormat,
    SpacyEntityExtractor,
    Turn,
)


def _turn(turn_id: str, text: str, ts: float) -> Turn:
    return Turn(turn_id=turn_id, user=text, assistant=text, timestamp=ts, last_access=ts)


def _episode(turns: list[Turn], **kwargs) -> EpisodicUnit:
    return EpisodicUnit(episode_id="e0", turns=turns, primary_format=MemoryFormat.HG, **kwargs)


def _dup(text: str) -> str:
    """The extractor sees `f"{turn.user} {turn.assistant}"`; `_turn` sets both to `text`."""
    return f"{text} {text}"


class TestFeatureDim:
    def test_feature_dim_is_seven(self) -> None:
        assert FEATURE_DIM == 7
        assert len(FEATURE_NAMES) == FEATURE_DIM

    def test_vector_length_matches_feature_dim(self) -> None:
        episode = _episode([_turn("t0", "hello world", 0.0)])
        vector = extract_features(episode, FakeEntityExtractor())
        assert len(vector) == FEATURE_DIM
        assert vector.to_array().shape == (FEATURE_DIM,)


class TestEntityCountScaling:
    def test_one_entity_vs_ten_entities(self) -> None:
        one_text = "Alice called."
        ten_text = " ".join(f"Person{i}" for i in range(10)) + " met."
        one_script = {_dup(one_text): [Entity("Alice", "PERSON")]}
        ten_script = {_dup(ten_text): [Entity(f"Person{i}", "PERSON") for i in range(10)]}
        one_episode = _episode([_turn("t0", one_text, 0.0)])
        ten_episode = _episode([_turn("t0", ten_text, 0.0)])

        one_vector = extract_features(one_episode, FakeEntityExtractor(one_script))
        ten_vector = extract_features(ten_episode, FakeEntityExtractor(ten_script))

        assert one_vector["entity_count"] == pytest.approx(math.log1p(1))
        assert ten_vector["entity_count"] == pytest.approx(math.log1p(10))
        assert ten_vector["entity_count"] > one_vector["entity_count"]

    def test_zero_entities(self) -> None:
        episode = _episode([_turn("t0", "no entities here", 0.0)])
        vector = extract_features(episode, FakeEntityExtractor())
        assert vector["entity_count"] == 0.0


class TestCooccurrenceDensity:
    def test_single_entity_is_zero(self) -> None:
        text = "Alice spoke."
        episode = _episode([_turn("t0", text, 0.0)])
        extractor = FakeEntityExtractor({_dup(text): [Entity("Alice", "PERSON")]})
        vector = extract_features(episode, extractor)
        assert vector["cooccurrence_density"] == 0.0

    def test_no_entities_is_zero(self) -> None:
        episode = _episode([_turn("t0", "nothing", 0.0)])
        vector = extract_features(episode, FakeEntityExtractor())
        assert vector["cooccurrence_density"] == 0.0

    def test_pair_cooccurring_in_same_turn_is_one(self) -> None:
        text = "Alice met Bob."
        episode = _episode([_turn("t0", text, 0.0)])
        extractor = FakeEntityExtractor(
            {_dup(text): [Entity("Alice", "PERSON"), Entity("Bob", "PERSON")]}
        )
        vector = extract_features(episode, extractor)
        assert vector["cooccurrence_density"] == pytest.approx(1.0)

    def test_pair_never_cooccurring_is_zero(self) -> None:
        t0_text, t1_text = "Alice spoke.", "Bob spoke."
        episode = _episode([_turn("t0", t0_text, 0.0), _turn("t1", t1_text, 1.0)])
        extractor = FakeEntityExtractor(
            {_dup(t0_text): [Entity("Alice", "PERSON")], _dup(t1_text): [Entity("Bob", "PERSON")]}
        )
        vector = extract_features(episode, extractor)
        assert vector["cooccurrence_density"] == 0.0


class TestTemporalOrdering:
    def test_single_turn_is_vacuously_ordered(self) -> None:
        episode = _episode([_turn("t0", "hello", 0.0)])
        vector = extract_features(episode, FakeEntityExtractor())
        assert vector["temporal_ordering"] == pytest.approx(0.5)  # 1.0 monotonic, 0.0 markers

    def test_empty_episode_temporal_ordering(self) -> None:
        episode = _episode([])
        vector = extract_features(episode, FakeEntityExtractor())
        assert vector["temporal_ordering"] == pytest.approx(0.5)

    def test_increasing_timestamps_score_higher_than_non_monotonic(self) -> None:
        increasing = _episode([_turn("t0", "a", 0.0), _turn("t1", "b", 1.0), _turn("t2", "c", 2.0)])
        non_monotonic = _episode(
            [_turn("t0", "a", 2.0), _turn("t1", "b", 1.0), _turn("t2", "c", 0.0)]
        )
        inc_vector = extract_features(increasing, FakeEntityExtractor())
        nm_vector = extract_features(non_monotonic, FakeEntityExtractor())
        assert inc_vector["temporal_ordering"] > nm_vector["temporal_ordering"]

    def test_temporal_markers_raise_score(self) -> None:
        with_markers = _episode([_turn("t0", "then later finally", 0.0)])
        without_markers = _episode([_turn("t0", "apple banana cherry", 0.0)])
        wm_vector = extract_features(with_markers, FakeEntityExtractor())
        wom_vector = extract_features(without_markers, FakeEntityExtractor())
        assert wm_vector["temporal_ordering"] > wom_vector["temporal_ordering"]


class TestTopicDiversity:
    def test_repeated_words_lower_diversity(self) -> None:
        repetitive = _episode([_turn("t0", "cat cat cat cat", 0.0)])
        diverse = _episode([_turn("t0", "cat dog bird fish", 0.0)])
        rep_vector = extract_features(repetitive, FakeEntityExtractor())
        div_vector = extract_features(diverse, FakeEntityExtractor())
        assert div_vector["topic_diversity"] > rep_vector["topic_diversity"]
        # user==assistant==text, so both are duplicated in the token stream:
        # "cat cat cat cat cat cat cat cat" -> 1/8; "cat dog bird fish" x2 -> 4/8.
        assert rep_vector["topic_diversity"] == pytest.approx(0.125)
        assert div_vector["topic_diversity"] == pytest.approx(0.5)

    def test_empty_episode_topic_diversity_is_zero(self) -> None:
        episode = _episode([])
        vector = extract_features(episode, FakeEntityExtractor())
        assert vector["topic_diversity"] == 0.0


class TestTokenLength:
    def test_scales_with_token_count(self) -> None:
        short = _episode([_turn("t0", "a b", 0.0)])
        long = _episode([_turn("t0", " ".join(["word"] * 50), 0.0)])
        short_vector = extract_features(short, FakeEntityExtractor())
        long_vector = extract_features(long, FakeEntityExtractor())
        assert long_vector["token_length"] > short_vector["token_length"]

    def test_empty_episode_token_length_is_zero(self) -> None:
        episode = _episode([])
        vector = extract_features(episode, FakeEntityExtractor())
        assert vector["token_length"] == 0.0


class TestPlaceholderPassthrough:
    def test_hyperedge_density_and_visual_salience_pass_through_exactly(self) -> None:
        episode = _episode(
            [_turn("t0", "hello", 0.0)], hyperedge_density=0.42, visual_salience=0.77
        )
        vector = extract_features(episode, FakeEntityExtractor())
        assert vector["hyperedge_density"] == 0.42
        assert vector["visual_salience"] == 0.77

    def test_default_placeholders_are_zero(self) -> None:
        episode = _episode([_turn("t0", "hello", 0.0)])
        vector = extract_features(episode, FakeEntityExtractor())
        assert vector["hyperedge_density"] == 0.0
        assert vector["visual_salience"] == 0.0


class TestBoundedAndFinite:
    def test_all_features_finite(self) -> None:
        episode = _episode(
            [_turn("t0", "Alice met Bob then later finally", 0.0), _turn("t1", "regular text", 1.0)]
        )
        extractor = FakeEntityExtractor(
            {
                "Alice met Bob then later finally": [
                    Entity("Alice", "PERSON"),
                    Entity("Bob", "PERSON"),
                ]
            }
        )
        vector = extract_features(episode, extractor)
        for name in FEATURE_NAMES:
            value = vector[name]
            assert math.isfinite(value), f"{name} is not finite: {value}"

    def test_empty_episode_all_finite(self) -> None:
        episode = _episode([])
        vector = extract_features(episode, FakeEntityExtractor())
        for name in FEATURE_NAMES:
            assert math.isfinite(vector[name])


class TestSpacyIntegration:
    @pytest.mark.slow
    def test_real_extractor_end_to_end(self) -> None:
        episode = _episode(
            [
                _turn("t0", "Barack Obama visited Paris in July.", 0.0),
                _turn("t1", "Then he flew to Tokyo.", 1.0),
            ]
        )
        vector = extract_features(episode, SpacyEntityExtractor())
        assert len(vector) == FEATURE_DIM
        assert vector["entity_count"] > 0.0
        for name in FEATURE_NAMES:
            assert math.isfinite(vector[name])
