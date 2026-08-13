"""fluxmem/supervision.py -- offline reward/labeling pipeline (FluxMem Alg. 2, COLM Sec 1.3.3).

Split from fluxmem/selector.py so the ADR'd modeling choice (ADR 0002) can
be revised without touching the MLP, and to keep both files under the
300-line cap (`.claude/rules/coding-style.md`). See ADR 0002 for the
Judge/MemUtil concretization this module implements: `r_judge` is
token-overlap F1 against a reference answer; `r_mem` is retrieval hit-rate
normalized by retrieved token budget.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol, runtime_checkable

import numpy as np

from fluxmem.config import RewardConfig
from fluxmem.features import extract_features
from fluxmem.interfaces import EntityExtractor, EpisodicUnit, MemoryFormat

# Canonical tie-break order: on an exact reward tie, the earliest format here wins.
FORMAT_ORDER: tuple[MemoryFormat, ...] = (MemoryFormat.HG, MemoryFormat.VC, MemoryFormat.VS)


@runtime_checkable
class JudgeReward(Protocol):
    """`r_judge(s)`: scores a format's simulated response against a reference answer."""

    def score(self, response: str, reference: str) -> float: ...


@runtime_checkable
class MemUtilReward(Protocol):
    """`r_mem(s)`: scores a format's retrieved content against gold evidence."""

    def score(self, retrieved: list[str], gold_evidence: list[str]) -> float: ...


@runtime_checkable
class FormatRunner(Protocol):
    """Boundary for the (non-goal) generation/retrieval pipeline ADASTORE would sit behind.

    Real implementations would call ITERRET/WORKMEM; both are out of scope
    here, so tests supply a deterministic fake.
    """

    def respond(self, episode: EpisodicUnit, fmt: MemoryFormat) -> str: ...

    def reference(self, episode: EpisodicUnit) -> str: ...

    def retrieve(self, episode: EpisodicUnit, fmt: MemoryFormat) -> list[str]: ...

    def gold_evidence(self, episode: EpisodicUnit) -> list[str]: ...


def _token_f1(response: str, reference: str) -> float:
    """Token-overlap F1 between two strings (ADR 0002's default `JudgeReward`)."""
    response_tokens = response.lower().split()
    reference_tokens = reference.lower().split()
    if not response_tokens or not reference_tokens:
        return 1.0 if response_tokens == reference_tokens else 0.0
    common = Counter(response_tokens) & Counter(reference_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(response_tokens)
    recall = num_common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


class ExactMatchF1Judge:
    """Default `JudgeReward` (ADR 0002): token-overlap F1 against a reference answer."""

    def score(self, response: str, reference: str) -> float:
        return _token_f1(response, reference)


class TokenBudgetMemUtil:
    """Default `MemUtilReward` (ADR 0002): hit-rate normalized by retrieved token budget.

    `reference_tokens` sets the scale at which token budget starts eroding
    the reward -- a modeling choice, matching `fluxmem/features.py`'s
    `DEFAULT_REFERENCE_LENGTH` convention for consistency, not a paper value.
    """

    def __init__(self, reference_tokens: float = 50.0) -> None:
        if reference_tokens <= 0:
            raise ValueError(f"reference_tokens must be > 0, got {reference_tokens}")
        self._reference_tokens = reference_tokens

    def score(self, retrieved: list[str], gold_evidence: list[str]) -> float:
        if not gold_evidence:
            return 0.0
        joined = " ".join(retrieved).lower()
        hits = sum(1 for span in gold_evidence if span.lower() in joined)
        hit_rate = hits / len(gold_evidence)
        token_budget = sum(len(item.split()) for item in retrieved)
        return hit_rate / (1.0 + token_budget / self._reference_tokens)


def per_format_rewards(
    episode: EpisodicUnit,
    runner: FormatRunner,
    judge: JudgeReward,
    mem_util: MemUtilReward,
    reward_config: RewardConfig,
) -> dict[MemoryFormat, float]:
    """FluxMem Eq. 9: `r(s) = lambda_judge*r_judge(s) + lambda_mem*r_mem(s)` for each format."""
    reference = runner.reference(episode)
    gold_evidence = runner.gold_evidence(episode)
    rewards: dict[MemoryFormat, float] = {}
    for fmt in FORMAT_ORDER:
        r_judge = judge.score(runner.respond(episode, fmt), reference)
        r_mem = mem_util.score(runner.retrieve(episode, fmt), gold_evidence)
        rewards[fmt] = reward_config.lambda_judge * r_judge + reward_config.lambda_mem * r_mem
    return rewards


def _argmax_format_index(rewards: dict[MemoryFormat, float]) -> int:
    """`y = argmax_s r(s)`. Ties broken by `FORMAT_ORDER` (strict `>` keeps the first max)."""
    best_idx = 0
    best_value = float("-inf")
    for idx, fmt in enumerate(FORMAT_ORDER):
        value = rewards[fmt]
        if value > best_value:
            best_value = value
            best_idx = idx
    return best_idx


def label_episodes(
    episodes: list[EpisodicUnit],
    runner: FormatRunner,
    judge: JudgeReward,
    mem_util: MemUtilReward,
    reward_config: RewardConfig,
    entity_extractor: EntityExtractor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds `(X, y)` for `fluxmem.selector.FormatSelector.fit`.

    `X`: `(n, FEATURE_DIM)` feature matrix. `y`: `(n,)` int array, each an
    index into `FORMAT_ORDER` (COLM Eq. 6's `f*_j = argmax_f r_f_j`).
    """
    features = [extract_features(e, entity_extractor).to_array() for e in episodes]
    X = np.stack(features) if features else np.empty((0, 0))
    y = np.array(
        [
            _argmax_format_index(per_format_rewards(e, runner, judge, mem_util, reward_config))
            for e in episodes
        ],
        dtype=np.int64,
    )
    return X, y
