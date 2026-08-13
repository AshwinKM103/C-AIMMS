"""fluxmem/features.py -- 7-scalar feature vector x_j for the format selector (COLM Sec 1.3.3).

FluxMem App. C argues features should be "intentionally simple and
interpretable ... lightweight ... avoids introducing additional reasoning
overhead into the control layer" -- which rules out an LLM-based extractor
but not real statistical NER. `entity_count` and `cooccurrence_density` feed
the selector's training labels (fluxmem/supervision.py), so noisy entity
detection there propagates into supervision; a capitalization heuristic
fails on lowercase text, fires on sentence-initial words, and cannot type
entities. The default extractor is therefore spaCy's `en_core_web_sm` (real
statistical NER, ~10k words/s on CPU, deterministic, ~12MB), injected via
the `EntityExtractor` Protocol so tests can substitute `FakeEntityExtractor`
and stay hermetic. Rejected: `nltk.ne_chunk` (weaker, and its corpora
default to `~/nltk_data` on a near-full disk); `transformers` BERT-NER
(~430MB, slow, makes feature extraction model-dependent -- kept as a
drop-in upgrade behind the same Protocol if entity quality ever becomes the
bottleneck).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fluxmem.interfaces import Entity, EntityExtractor, EpisodicUnit, SpacyEntityExtractor, Turn

FEATURE_NAMES = (
    "entity_count",
    "cooccurrence_density",
    "temporal_ordering",
    "topic_diversity",
    "token_length",
    "hyperedge_density",
    "visual_salience",
)
FEATURE_DIM = len(FEATURE_NAMES)  # 7 -- never hardcode the 7

_TEMPORAL_MARKERS = frozenset(
    {
        "then",
        "later",
        "after",
        "before",
        "next",
        "finally",
        "meanwhile",
        "afterward",
        "subsequently",
    }
)


@dataclass(slots=True)
class FeatureVector:
    """A named 7-scalar feature vector, orderable into `fluxmem.selector`'s input space."""

    values: dict[str, float] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        return np.array([self.values[name] for name in FEATURE_NAMES], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, name: str) -> float:
        return self.values[name]


def _turn_text(turn: Turn) -> str:
    return f"{turn.user} {turn.assistant}"


def _tokens(text: str) -> list[str]:
    return text.split()


def _entity_key(entity: Entity) -> tuple[str, str]:
    return (entity.text, entity.label)


def _cooccurrence_density(entities_per_turn: list[list[Entity]]) -> float:
    """Entity pairs co-occurring within a turn, over all possible pairs in the episode.

    Bounded in [0,1]: every within-turn co-occurring pair is drawn from the
    same distinct-entity set used for the denominator, so the numerator can
    never exceed it.
    """
    distinct = sorted({_entity_key(e) for ents in entities_per_turn for e in ents})
    n = len(distinct)
    if n < 2:
        return 0.0
    possible_pairs = n * (n - 1) / 2
    cooccurring: set[tuple] = set()
    for ents in entities_per_turn:
        keys = sorted({_entity_key(e) for e in ents})
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                cooccurring.add((keys[i], keys[j]))
    return len(cooccurring) / possible_pairs


def _temporal_ordering(turns: list[Turn]) -> float:
    """Blend of adjacent-pair monotonic-timestamp fraction and temporal-marker rate.

    50/50 blend is a modeling choice (neither paper specifies one); both
    components are in [0,1], so the blend is too. A single turn (or none) is
    vacuously ordered.
    """
    if len(turns) < 2:
        monotonic_fraction = 1.0
    else:
        increasing = sum(1 for a, b in zip(turns, turns[1:]) if b.timestamp > a.timestamp)
        monotonic_fraction = increasing / (len(turns) - 1)

    words = [w.lower().strip(".,!?") for t in turns for w in _tokens(_turn_text(t))]
    marker_rate = (
        min(1.0, sum(1 for w in words if w in _TEMPORAL_MARKERS) / len(words)) if words else 0.0
    )

    return 0.5 * monotonic_fraction + 0.5 * marker_rate


def _topic_diversity(turns: list[Turn]) -> float:
    """Type-token ratio (distinct tokens / total tokens) across all turns. Bounded (0,1]."""
    tokens = [w.lower() for t in turns for w in _tokens(_turn_text(t))]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _total_token_count(turns: list[Turn]) -> int:
    return sum(len(_tokens(_turn_text(t))) for t in turns)


def extract_features(
    episode: EpisodicUnit, entity_extractor: EntityExtractor | None = None
) -> FeatureVector:
    """Computes the 7-scalar feature vector for `episode` (COLM Sec 1.3.3).

    `entity_extractor` defaults to `SpacyEntityExtractor`; inject
    `FakeEntityExtractor` for hermetic tests. `hyperedge_density` and
    `visual_salience` are read straight off the episode as opaque
    placeholders -- the HG/VC encoders that would produce them are an
    explicit non-goal (see fluxmem/interfaces.py).
    """
    entity_extractor = entity_extractor or SpacyEntityExtractor()
    turns = episode.turns

    entities_per_turn = [entity_extractor.extract(_turn_text(t)) for t in turns]
    distinct_entities = {_entity_key(e) for ents in entities_per_turn for e in ents}

    values = {
        "entity_count": math.log1p(len(distinct_entities)),
        "cooccurrence_density": _cooccurrence_density(entities_per_turn),
        "temporal_ordering": _temporal_ordering(turns),
        "topic_diversity": _topic_diversity(turns),
        "token_length": math.log1p(_total_token_count(turns)),
        "hyperedge_density": episode.hyperedge_density,
        "visual_salience": episode.visual_salience,
    }
    return FeatureVector(values)
