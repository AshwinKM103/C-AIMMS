"""Tests for fluxmem/supervision.py -- offline reward/labeling pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from fluxmem.config import RewardConfig
from fluxmem.interfaces import EpisodicUnit, MemoryFormat, Turn
from fluxmem.supervision import (
    FORMAT_ORDER,
    ExactMatchF1Judge,
    TokenBudgetMemUtil,
    label_episodes,
    per_format_rewards,
)


def _episode(episode_id: str = "e0") -> EpisodicUnit:
    turn = Turn(
        turn_id=f"{episode_id}-t0", user="hi", assistant="hi", timestamp=0.0, last_access=0.0
    )
    return EpisodicUnit(episode_id=episode_id, turns=[turn], primary_format=MemoryFormat.HG)


class FakeFormatRunner:
    """Deterministic FormatRunner fake -- no live generation/retrieval pipeline needed."""

    def __init__(
        self,
        responses: dict[tuple[str, MemoryFormat], str],
        references: dict[str, str],
        retrieved: dict[tuple[str, MemoryFormat], list[str]],
        evidence: dict[str, list[str]],
    ) -> None:
        self._responses = responses
        self._references = references
        self._retrieved = retrieved
        self._evidence = evidence

    def respond(self, episode: EpisodicUnit, fmt: MemoryFormat) -> str:
        return self._responses[(episode.episode_id, fmt)]

    def reference(self, episode: EpisodicUnit) -> str:
        return self._references[episode.episode_id]

    def retrieve(self, episode: EpisodicUnit, fmt: MemoryFormat) -> list[str]:
        return self._retrieved[(episode.episode_id, fmt)]

    def gold_evidence(self, episode: EpisodicUnit) -> list[str]:
        return self._evidence[episode.episode_id]


class TestExactMatchF1Judge:
    def test_identical_strings_score_one(self) -> None:
        judge = ExactMatchF1Judge()
        assert judge.score("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_disjoint_strings_score_zero(self) -> None:
        judge = ExactMatchF1Judge()
        assert judge.score("apple banana", "cherry durian") == 0.0

    def test_partial_overlap_is_between_zero_and_one(self) -> None:
        judge = ExactMatchF1Judge()
        score = judge.score("the cat sat on the mat", "the cat stood on a rug")
        assert 0.0 < score < 1.0

    def test_both_empty_scores_one(self) -> None:
        judge = ExactMatchF1Judge()
        assert judge.score("", "") == 1.0

    def test_one_empty_scores_zero(self) -> None:
        judge = ExactMatchF1Judge()
        assert judge.score("something", "") == 0.0
        assert judge.score("", "something") == 0.0

    def test_known_f1_value(self) -> None:
        # response tokens {a,a,b}, reference tokens {a,c}; common={a}: precision=1/3, recall=1/2
        # F1 = 2*(1/3*1/2)/(1/3+1/2) = 2*(1/6)/(5/6) = 2/5 = 0.4
        judge = ExactMatchF1Judge()
        assert judge.score("a a b", "a c") == pytest.approx(0.4)


class TestTokenBudgetMemUtil:
    def test_no_gold_evidence_is_zero(self) -> None:
        mem_util = TokenBudgetMemUtil()
        assert mem_util.score(["some retrieved text"], []) == 0.0

    def test_no_hits_is_zero(self) -> None:
        mem_util = TokenBudgetMemUtil()
        assert mem_util.score(["irrelevant text"], ["gold fact one"]) == 0.0

    def test_all_hits_full_hit_rate(self) -> None:
        mem_util = TokenBudgetMemUtil(reference_tokens=1000.0)
        score = mem_util.score(["gold fact one is here"], ["gold fact one"])
        assert score == pytest.approx(1.0, abs=0.01)

    def test_more_tokens_for_same_hits_lowers_score(self) -> None:
        mem_util = TokenBudgetMemUtil(reference_tokens=10.0)
        concise = mem_util.score(["gold fact one"], ["gold fact one"])
        verbose_retrieved = ["gold fact one"] + ["filler"] * 50
        verbose = mem_util.score(verbose_retrieved, ["gold fact one"])
        assert concise > verbose

    def test_partial_hits(self) -> None:
        mem_util = TokenBudgetMemUtil(reference_tokens=1000.0)
        score = mem_util.score(["fact one only"], ["fact one", "fact two"])
        assert score == pytest.approx(0.5, abs=0.01)

    def test_non_positive_reference_tokens_rejected(self) -> None:
        with pytest.raises(ValueError):
            TokenBudgetMemUtil(reference_tokens=0.0)


class TestPerFormatRewards:
    def test_weighted_combination(self) -> None:
        episode = _episode("e0")
        runner = FakeFormatRunner(
            responses={("e0", fmt): "the answer" for fmt in FORMAT_ORDER},
            references={"e0": "the answer"},
            retrieved={("e0", fmt): ["the answer"] for fmt in FORMAT_ORDER},
            evidence={"e0": ["the answer"]},
        )
        judge = ExactMatchF1Judge()
        mem_util = TokenBudgetMemUtil(reference_tokens=1000.0)
        config = RewardConfig(lambda_judge=0.7, lambda_mem=0.3)
        rewards = per_format_rewards(episode, runner, judge, mem_util, config)
        assert set(rewards) == set(FORMAT_ORDER)
        for fmt in FORMAT_ORDER:
            assert rewards[fmt] == pytest.approx(1.0, abs=0.01)

    def test_different_formats_get_different_rewards(self) -> None:
        episode = _episode("e0")
        runner = FakeFormatRunner(
            responses={
                ("e0", MemoryFormat.HG): "the exact answer",
                ("e0", MemoryFormat.VC): "something unrelated",
                ("e0", MemoryFormat.VS): "the exact answer",
            },
            references={"e0": "the exact answer"},
            retrieved={("e0", fmt): ["the exact answer"] for fmt in FORMAT_ORDER},
            evidence={"e0": ["the exact answer"]},
        )
        judge = ExactMatchF1Judge()
        mem_util = TokenBudgetMemUtil(reference_tokens=1000.0)
        config = RewardConfig()
        rewards = per_format_rewards(episode, runner, judge, mem_util, config)
        assert rewards[MemoryFormat.HG] > rewards[MemoryFormat.VC]
        assert rewards[MemoryFormat.VS] > rewards[MemoryFormat.VC]


class TestLabelEpisodes:
    def test_shapes(self) -> None:
        episodes = [_episode("e0"), _episode("e1")]
        runner = FakeFormatRunner(
            responses={(eid, fmt): "x" for eid in ("e0", "e1") for fmt in FORMAT_ORDER},
            references={"e0": "x", "e1": "x"},
            retrieved={(eid, fmt): ["x"] for eid in ("e0", "e1") for fmt in FORMAT_ORDER},
            evidence={"e0": ["x"], "e1": ["x"]},
        )
        judge = ExactMatchF1Judge()
        mem_util = TokenBudgetMemUtil()
        from fluxmem.interfaces import FakeEntityExtractor

        X, y = label_episodes(
            episodes, runner, judge, mem_util, RewardConfig(), FakeEntityExtractor()
        )
        assert X.shape[0] == 2
        assert y.shape == (2,)
        assert y.dtype == np.int64

    def test_ties_break_by_format_order(self) -> None:
        """All formats score identically; the argmax must pick FORMAT_ORDER[0] (HG)."""
        episode = _episode("e0")
        runner = FakeFormatRunner(
            responses={("e0", fmt): "same" for fmt in FORMAT_ORDER},
            references={"e0": "same"},
            retrieved={("e0", fmt): ["same"] for fmt in FORMAT_ORDER},
            evidence={"e0": ["same"]},
        )
        judge = ExactMatchF1Judge()
        mem_util = TokenBudgetMemUtil()
        from fluxmem.interfaces import FakeEntityExtractor

        X, y = label_episodes(
            [episode], runner, judge, mem_util, RewardConfig(), FakeEntityExtractor()
        )
        assert y[0] == 0  # index of MemoryFormat.HG in FORMAT_ORDER

    def test_empty_episode_list(self) -> None:
        runner = FakeFormatRunner({}, {}, {}, {})
        judge = ExactMatchF1Judge()
        mem_util = TokenBudgetMemUtil()
        from fluxmem.interfaces import FakeEntityExtractor

        X, y = label_episodes([], runner, judge, mem_util, RewardConfig(), FakeEntityExtractor())
        assert X.shape[0] == 0
        assert y.shape[0] == 0
