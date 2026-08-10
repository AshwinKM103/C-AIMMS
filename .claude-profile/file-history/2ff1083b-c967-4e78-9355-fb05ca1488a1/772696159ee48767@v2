"""Tests for fluxmem/interfaces.py: the boundary fakes and Protocols."""

from __future__ import annotations

import json
import pathlib

import pytest

from fluxmem.interfaces import (
    Entity,
    FakeEntityExtractor,
    MemoryFormat,
    SpacyEntityExtractor,
    StubEpisodeProducer,
)

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def _episode_to_dict(episode) -> dict:
    return {
        "episode_id": episode.episode_id,
        "primary_format": str(episode.primary_format),
        "created_at": episode.created_at,
        "last_access": episode.last_access,
        "turns": [
            {
                "turn_id": t.turn_id,
                "user": t.user,
                "assistant": t.assistant,
                "timestamp": t.timestamp,
            }
            for t in episode.turns
        ],
    }


class TestStubEpisodeProducer:
    def test_deterministic_for_fixed_seed(self) -> None:
        a = StubEpisodeProducer(seed=42).produce(5)
        b = StubEpisodeProducer(seed=42).produce(5)
        assert [_episode_to_dict(e) for e in a] == [_episode_to_dict(e) for e in b]

    def test_different_seeds_diverge(self) -> None:
        a = StubEpisodeProducer(seed=1).produce(5)
        b = StubEpisodeProducer(seed=2).produce(5)
        assert [_episode_to_dict(e) for e in a] != [_episode_to_dict(e) for e in b]

    def test_produce_count_respected(self) -> None:
        episodes = StubEpisodeProducer(seed=0).produce(7)
        assert len(episodes) == 7
        assert len({e.episode_id for e in episodes}) == 7

    def test_turns_per_episode_knob(self) -> None:
        episodes = StubEpisodeProducer(seed=0, turns_per_episode=6).produce(3)
        assert all(len(e.turns) == 6 for e in episodes)

    def test_entities_per_turn_knob_bounds_pool(self) -> None:
        producer = StubEpisodeProducer(seed=0, entities_per_turn=100)
        episode = producer.produce(1)[0]
        assert len(episode.turns) > 0

    def test_topic_spread_knob(self) -> None:
        narrow = StubEpisodeProducer(seed=0, topic_spread=1, turns_per_episode=10).produce(1)[0]
        wide = StubEpisodeProducer(seed=0, topic_spread=6, turns_per_episode=10).produce(1)[0]
        assert narrow.turns != wide.turns

    def test_zero_count_produces_empty_list(self) -> None:
        assert StubEpisodeProducer(seed=0).produce(0) == []

    def test_primary_format_is_valid_enum_member(self) -> None:
        episodes = StubEpisodeProducer(seed=0).produce(10)
        assert all(e.primary_format in MemoryFormat for e in episodes)

    def test_golden_snapshot(self) -> None:
        """Pins the fake so refactors can't silently change every downstream fixture."""
        episodes = StubEpisodeProducer(seed=7).produce(3)
        actual = [_episode_to_dict(e) for e in episodes]
        golden_path = FIXTURES_DIR / "stub_producer_seed7.json"
        expected = json.loads(golden_path.read_text())
        assert actual == expected


class TestFakeEntityExtractor:
    def test_returns_scripted_entities(self) -> None:
        script = {"Alice met Bob": [Entity("Alice", "PERSON"), Entity("Bob", "PERSON")]}
        extractor = FakeEntityExtractor(script)
        assert extractor.extract("Alice met Bob") == script["Alice met Bob"]

    def test_unscripted_input_returns_empty(self) -> None:
        extractor = FakeEntityExtractor({"foo": [Entity("foo", "X")]})
        assert extractor.extract("bar") == []

    def test_no_script_returns_empty_for_everything(self) -> None:
        assert FakeEntityExtractor().extract("anything") == []

    def test_returned_list_is_a_copy(self) -> None:
        entities = [Entity("Alice", "PERSON")]
        extractor = FakeEntityExtractor({"x": entities})
        result = extractor.extract("x")
        result.append(Entity("Bob", "PERSON"))
        assert extractor.extract("x") == entities


class TestSpacyEntityExtractor:
    @pytest.mark.slow
    def test_extracts_real_entities(self) -> None:
        extractor = SpacyEntityExtractor()
        entities = extractor.extract("Barack Obama visited Paris in July.")
        labels = {e.label for e in entities}
        texts = {e.text for e in entities}
        assert "Paris" in texts
        assert "GPE" in labels

    @pytest.mark.slow
    def test_empty_text_returns_empty(self) -> None:
        assert SpacyEntityExtractor().extract("") == []
