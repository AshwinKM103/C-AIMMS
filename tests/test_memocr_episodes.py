"""Tests for fluxmem/memocr_episodes.py -- MemOCR-backed EpisodeProducer."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from fluxmem.interfaces import MemoryFormat
from fluxmem.memocr_episodes import DEFAULT_PATCH_PIXELS, MemOCREpisodeProducer


@dataclass
class _FakeImage:
    size: tuple[int, int]


class _ScriptedMarkdownSource:
    """Fake `MarkdownSourceFn`: returns a fixed list, ignoring `count` beyond truncation."""

    def __init__(self, snapshots: list[str]) -> None:
        self._snapshots = snapshots

    def __call__(self, count: int) -> list[str]:
        return self._snapshots[:count]


class _ScriptedRenderer:
    """Fake `RenderFn`: maps each Markdown string to a pre-scripted image or None."""

    def __init__(self, images: list[_FakeImage | None]) -> None:
        self._images = images

    def __call__(self, snapshots: list[str]) -> list[_FakeImage | None]:
        assert len(snapshots) <= len(self._images)
        return self._images[: len(snapshots)]


class TestProduce:
    def test_produces_one_episode_per_snapshot(self) -> None:
        producer = MemOCREpisodeProducer(
            markdown_source=_ScriptedMarkdownSource(["# a", "# b"]),
            render_fn=_ScriptedRenderer([_FakeImage((100, 100)), _FakeImage((200, 200))]),
        )
        episodes = producer.produce(count=2)
        assert len(episodes) == 2
        assert [e.episode_id for e in episodes] == ["memocr-episode-0", "memocr-episode-1"]

    def test_all_episodes_are_visual_canvas_format(self) -> None:
        producer = MemOCREpisodeProducer(
            markdown_source=_ScriptedMarkdownSource(["# a"]),
            render_fn=_ScriptedRenderer([_FakeImage((50, 50))]),
        )
        (episode,) = producer.produce(count=1)
        assert episode.primary_format == MemoryFormat.VC

    def test_episodes_carry_no_raw_turns(self) -> None:
        """MemOCR's memory is already a compacted summary, not raw dialogue turns."""
        producer = MemOCREpisodeProducer(
            markdown_source=_ScriptedMarkdownSource(["# a"]),
            render_fn=_ScriptedRenderer([_FakeImage((50, 50))]),
        )
        (episode,) = producer.produce(count=1)
        assert episode.turns == []

    def test_respects_count_smaller_than_available_snapshots(self) -> None:
        producer = MemOCREpisodeProducer(
            markdown_source=_ScriptedMarkdownSource(["# a", "# b", "# c"]),
            render_fn=_ScriptedRenderer(
                [_FakeImage((10, 10)), _FakeImage((10, 10)), _FakeImage((10, 10))]
            ),
        )
        episodes = producer.produce(count=1)
        assert len(episodes) == 1


class TestVisualSalience:
    def test_full_budget_image_scores_near_one(self) -> None:
        budget_patches = 4
        width = height = int((budget_patches * DEFAULT_PATCH_PIXELS) ** 0.5)
        producer = MemOCREpisodeProducer(
            markdown_source=_ScriptedMarkdownSource(["# a"]),
            render_fn=_ScriptedRenderer([_FakeImage((width, height))]),
            budget_patches=budget_patches,
        )
        (episode,) = producer.produce(count=1)
        assert episode.visual_salience == 1.0

    def test_salience_is_clipped_at_one_for_over_budget_images(self) -> None:
        producer = MemOCREpisodeProducer(
            markdown_source=_ScriptedMarkdownSource(["# a"]),
            render_fn=_ScriptedRenderer([_FakeImage((10_000, 10_000))]),
            budget_patches=1,
        )
        (episode,) = producer.produce(count=1)
        assert episode.visual_salience == 1.0

    def test_failed_render_scores_zero_salience(self) -> None:
        producer = MemOCREpisodeProducer(
            markdown_source=_ScriptedMarkdownSource(["# a"]),
            render_fn=_ScriptedRenderer([None]),
        )
        (episode,) = producer.produce(count=1)
        assert episode.visual_salience == 0.0

    def test_small_image_scores_between_zero_and_one(self) -> None:
        producer = MemOCREpisodeProducer(
            markdown_source=_ScriptedMarkdownSource(["# a"]),
            render_fn=_ScriptedRenderer([_FakeImage((28, 28))]),
            budget_patches=256,
        )
        (episode,) = producer.produce(count=1)
        assert 0.0 < episode.visual_salience < 1.0


class TestConstruction:
    def test_rejects_non_positive_budget_patches(self) -> None:
        with pytest.raises(ValueError, match="budget_patches"):
            MemOCREpisodeProducer(
                markdown_source=_ScriptedMarkdownSource([]),
                render_fn=_ScriptedRenderer([]),
                budget_patches=0,
            )
