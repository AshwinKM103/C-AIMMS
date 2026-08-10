"""Boundary types and Protocols that the rest of fluxmem/ is tested against.

No dependencies on the other fluxmem modules -- this is the fake/interface
layer everything else builds on (STIM, MTEM, features, selector, fusion,
LTSM all import from here, never the reverse).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np


class MemoryFormat(StrEnum):
    """Opaque HETREP representation labels.

    HG (hypergraph), VC (visual canvas), VS (vector store) are treated as
    labels an upstream HETREP would attach -- the encoders themselves are an
    explicit non-goal of this package.
    """

    HG = "HG"
    VC = "VC"
    VS = "VS"


@dataclass(frozen=True, slots=True)
class Entity:
    """A single named-entity mention: surface form + type label."""

    text: str
    label: str


@dataclass(slots=True)
class Turn:
    """One raw dialogue turn, as STIM buffers it."""

    turn_id: str
    user: str
    assistant: str
    timestamp: float
    last_access: float


@dataclass(slots=True)
class EpisodicUnit:
    """A structured episodic unit e_j, as MTEM stores it (COLM Sec 1.3.2).

    `hyperedge_density` and `visual_salience` are opaque placeholders a real
    HETREP HG/VC encoder would populate (see fluxmem/features.py); that
    encoder is out of scope here, so they default to 0.0 and pass through
    feature extraction unchanged.
    """

    episode_id: str
    turns: list[Turn]
    primary_format: MemoryFormat
    features: np.ndarray | None = None
    embedding: np.ndarray | None = None
    access_count: int = 0
    last_access: float = 0.0
    created_at: float = 0.0
    hyperedge_density: float = 0.0
    visual_salience: float = 0.0


@runtime_checkable
class VectorStore(Protocol):
    """Minimal vector-store boundary; fluxmem/ltsm.py's FaissVectorStore implements it."""

    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    def __len__(self) -> int: ...


@runtime_checkable
class EpisodeProducer(Protocol):
    """Boundary fake for HETREP (segmentation + encoding), a non-goal here."""

    def produce(self, count: int) -> list[EpisodicUnit]: ...


@runtime_checkable
class EntityExtractor(Protocol):
    """NER boundary, injected into fluxmem/features.py."""

    def extract(self, text: str) -> list[Entity]: ...


class FakeEntityExtractor:
    """Scripted entity extractor for hermetic feature tests.

    `script` maps exact input text to the entity list `extract` returns for
    it. Missing keys return an empty list rather than raising, so producer
    filler text that isn't scripted doesn't need an entry.
    """

    def __init__(self, script: dict[str, list[Entity]] | None = None) -> None:
        self._script = script or {}

    def extract(self, text: str) -> list[Entity]:
        return list(self._script.get(text, []))


_SPACY_NLP = None


def _load_spacy_model():
    """Lazily loads en_core_web_sm once, at first use, cached module-wide.

    Per-episode loading was measured at a 10x slowdown vs. loading once.
    """
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy

        _SPACY_NLP = spacy.load("en_core_web_sm")
    return _SPACY_NLP


class SpacyEntityExtractor:
    """Real statistical NER via spaCy's en_core_web_sm (default extractor)."""

    def extract(self, text: str) -> list[Entity]:
        nlp = _load_spacy_model()
        doc = nlp(text)
        return [Entity(text=ent.text, label=ent.label_) for ent in doc.ents]


class StubEpisodeProducer:
    """Deterministic, seeded episode generator for testing tiers above HETREP.

    Not a real segmenter/encoder. Exposes knobs for entity density, topic
    spread, and turn/episode length so downstream feature tests have known
    ground truth to assert against.
    """

    _TOPIC_POOL = ("billing", "travel", "weather", "recipes", "sports", "music")
    _ENTITY_POOL = (
        ("Alice", "PERSON"),
        ("Bob", "PERSON"),
        ("Paris", "GPE"),
        ("Tokyo", "GPE"),
        ("Acme Corp", "ORG"),
        ("Monday", "DATE"),
    )
    _FILLER_WORDS = ("the", "a", "then", "later", "about", "regarding")

    def __init__(
        self,
        seed: int = 0,
        turns_per_episode: int = 3,
        entities_per_turn: int = 1,
        topic_spread: int = 1,
        tokens_per_turn: int = 8,
    ) -> None:
        self._rng = random.Random(seed)
        self._turns_per_episode = turns_per_episode
        self._entities_per_turn = entities_per_turn
        self._topic_spread = max(1, min(topic_spread, len(self._TOPIC_POOL)))
        self._tokens_per_turn = tokens_per_turn

    def produce(self, count: int) -> list[EpisodicUnit]:
        return [self._one_episode(i) for i in range(count)]

    def _one_episode(self, index: int) -> EpisodicUnit:
        topics = self._rng.sample(self._TOPIC_POOL, k=self._topic_spread)
        turns: list[Turn] = []
        base_ts = float(index * 1000)
        entities_k = min(self._entities_per_turn, len(self._ENTITY_POOL))
        for t in range(self._turns_per_episode):
            entities = self._rng.sample(self._ENTITY_POOL, k=entities_k)
            topic = self._rng.choice(topics)
            n_filler = max(0, self._tokens_per_turn - len(entities))
            filler = " ".join(self._rng.choice(self._FILLER_WORDS) for _ in range(n_filler))
            entity_text = " ".join(name for name, _label in entities)
            text = f"{entity_text} {topic} {filler}".strip()
            ts = base_ts + t
            turns.append(
                Turn(
                    turn_id=f"episode-{index}-turn-{t}",
                    user=text,
                    assistant=text,
                    timestamp=ts,
                    last_access=ts,
                )
            )
        primary_format = self._rng.choice(list(MemoryFormat))
        return EpisodicUnit(
            episode_id=f"episode-{index}",
            turns=turns,
            primary_format=primary_format,
            created_at=base_ts,
            last_access=base_ts + max(0, self._turns_per_episode - 1),
        )
