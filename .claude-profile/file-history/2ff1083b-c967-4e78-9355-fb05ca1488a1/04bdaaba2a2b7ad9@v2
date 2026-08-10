"""fluxmem/mtem.py -- MTEM tier and utility score U(e_j) (COLM Sec 1.3.2, Eq. 5).

`U(e_j) = w1*c(e_j) + w2*l(e_j) + w3*d(e_j)` over access frequency,
interaction intensity, and recency. Neither COLM nor FluxMem defines `c`,
`l`, `d` concretely -- each concretization below is a modeling choice,
individually swappable, and normalized to `[0,1]` so `w1,w2,w3` are
comparable (also a modeling choice, stated here rather than assumed).

Naming note: COLM's `c(e_j)` (access frequency, this module) and FluxMem
Eq. 6's `c(m)` (confidence, not implemented -- see fluxmem/ltsm.py) are the
same glyph for unrelated quantities. The field here is named
`access_frequency`, never `c`, so the two can never collide in code even
though they collide in the papers' notation.
"""

from __future__ import annotations

import math

from fluxmem.config import UtilityWeights
from fluxmem.interfaces import EpisodicUnit

DEFAULT_CAPACITY = 2000
DEFAULT_REFERENCE_LENGTH = 50.0
DEFAULT_HALF_LIFE = 100.0


def access_frequency(episode: EpisodicUnit, now: float) -> float:
    """`c(e_j)`: access_count normalized by elapsed turns since creation.

    `age_in_turns = now - created_at` (both are the same turn-index-like
    scalar the rest of fluxmem uses for "time" -- see fluxmem/interfaces.py's
    StubEpisodeProducer). Clipped to at most 1.0 to honor the table's stated
    `(0,1]` range; the papers do not state this bound, so the clip is a
    modeling choice, not a cited fact.
    """
    age_in_turns = max(0.0, now - episode.created_at)
    return min(1.0, episode.access_count / (age_in_turns + 1.0))


def interaction_intensity(episode: EpisodicUnit, reference_length: float) -> float:
    """`l(e_j)`: mean whitespace-token count per turn, divided by a reference length.

    Clipped to `[0,1]`. `reference_length` is a config'd scale, not a paper
    value -- neither source states one.
    """
    if not episode.turns or reference_length <= 0:
        return 0.0
    tokens_per_turn = [
        len(turn.user.split()) + len(turn.assistant.split()) for turn in episode.turns
    ]
    mean_tokens = sum(tokens_per_turn) / len(tokens_per_turn)
    return min(1.0, mean_tokens / reference_length)


def recency_decay(episode: EpisodicUnit, now: float, half_life: float) -> float:
    """`d(e_j)`: exponential decay of time since last access, `exp(-dt/half_life)`.

    Always in `(0,1]` for `dt >= 0`. This is the *soft* recency term inside
    `U` -- see fluxmem/ltsm.py for the separate *hard* recency gate `r(e_j)`,
    which uses a different (linear) transform of the same `dt` by design.
    """
    if half_life <= 0:
        raise ValueError(f"half_life must be > 0, got {half_life}")
    delta_t = max(0.0, now - episode.last_access)
    return math.exp(-delta_t / half_life)


def utility_score(
    episode: EpisodicUnit,
    weights: UtilityWeights,
    now: float,
    *,
    reference_length: float = DEFAULT_REFERENCE_LENGTH,
    half_life: float = DEFAULT_HALF_LIFE,
) -> float:
    """Pure function: COLM Eq. 5, `U = w1*c + w2*l + w3*d`."""
    c = access_frequency(episode, now)
    l = interaction_intensity(episode, reference_length)  # noqa: E741 -- matches paper notation
    d = recency_decay(episode, now, half_life)
    return weights.w1 * c + weights.w2 * l + weights.w3 * d


class MidTermEpisodicMemory:
    """MTEM tier: stores episodic units, ranked by `U(e_j)` for LTSM promotion.

    Capacity cap of 2000 units (FluxMem). Exceeding it on `add` evicts the
    lowest-utility episode, computed with this instance's weights/config at
    the time of eviction.
    """

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        weights: UtilityWeights | None = None,
        reference_length: float = DEFAULT_REFERENCE_LENGTH,
        half_life: float = DEFAULT_HALF_LIFE,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._weights = weights or UtilityWeights(w1=1.0, w2=1.0, w3=1.0)
        self._reference_length = reference_length
        self._half_life = half_life
        self._episodes: dict[str, EpisodicUnit] = {}

    def __len__(self) -> int:
        return len(self._episodes)

    def __contains__(self, episode_id: str) -> bool:
        return episode_id in self._episodes

    @property
    def weights(self) -> UtilityWeights:
        return self._weights

    def utility(self, episode: EpisodicUnit, now: float) -> float:
        return utility_score(
            episode,
            self._weights,
            now,
            reference_length=self._reference_length,
            half_life=self._half_life,
        )

    def add(self, episode: EpisodicUnit, now: float) -> EpisodicUnit | None:
        """Stores `episode`; returns the evicted lowest-utility episode, if any."""
        self._episodes[episode.episode_id] = episode
        if len(self._episodes) <= self._capacity:
            return None
        lowest_id = min(self._episodes, key=lambda eid: self.utility(self._episodes[eid], now))
        return self._episodes.pop(lowest_id)

    def get(self, episode_id: str, now: float) -> EpisodicUnit:
        """Fetches `episode_id`, bumping `access_count`/`last_access` -- the
        write that makes `access_frequency` and `recency_decay` move."""
        episode = self._episodes[episode_id]
        episode.access_count += 1
        episode.last_access = now
        return episode

    def merge(self, target_id: str, incoming: EpisodicUnit) -> EpisodicUnit:
        """Merges `incoming` into the stored episode `target_id`, in place."""
        target = self._episodes[target_id]
        target.turns.extend(incoming.turns)
        target.access_count += incoming.access_count
        target.last_access = max(target.last_access, incoming.last_access)
        return target

    def remove(self, episode_id: str) -> EpisodicUnit:
        return self._episodes.pop(episode_id)

    def promotion_candidates(self, tau_u: float, now: float) -> list[EpisodicUnit]:
        """Episodes clearing the utility gate `U(e) >= tau_u`.

        This is only the utility gate -- LTSM applies the remaining, separate
        recency gate `r(e) >= tau_r` (see fluxmem/ltsm.py's `is_eligible`).
        """
        return [e for e in self._episodes.values() if self.utility(e, now) >= tau_u]

    def episodes(self) -> list[EpisodicUnit]:
        return list(self._episodes.values())
