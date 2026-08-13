#!/usr/bin/env python
"""End-to-end smoke test for the ADASTORE write phase (COLM Alg. 1 lines 1-7 shape).

stub producer -> 6 turns into STIM -> 2 evicted to MTEM -> features extracted
-> selector assigns a format -> BMM fusion decides merge-vs-new -> one
episode crosses both promotion thresholds and lands in the FAISS store.
Asserts the tier counts at each stage. All logic lives in fluxmem/; this
script only wires the tiers together.

Segmentation (turning a raw evicted turn into an EpisodicUnit) is COLM
Sec 1.3.1, surprise-based, and out of scope for this package -- `_turn_to_episode`
below is an explicit placeholder shim, not a real segmenter.
"""

from __future__ import annotations

import numpy as np

from fluxmem.config import FusionConfig, PromotionThresholds, UtilityWeights
from fluxmem.features import extract_features
from fluxmem.fusion import select_merge_target
from fluxmem.interfaces import EpisodicUnit, FakeEntityExtractor, MemoryFormat, StubEpisodeProducer
from fluxmem.ltsm import FaissVectorStore, promote
from fluxmem.mtem import MidTermEpisodicMemory
from fluxmem.selector import FormatSelector
from fluxmem.stim import ShortTermInteractionMemory

_EMBEDDING_DIM = 8


def _turn_to_episode(turn, index: int) -> EpisodicUnit:
    """Wraps a single evicted turn into a trivial one-turn episode (segmentation shim)."""
    rng = np.random.default_rng(index)
    return EpisodicUnit(
        episode_id=f"episode-from-{turn.turn_id}",
        turns=[turn],
        primary_format=MemoryFormat.HG,
        created_at=turn.timestamp,
        last_access=turn.timestamp,
        embedding=rng.normal(size=_EMBEDDING_DIM),
    )


def main() -> None:
    stim = ShortTermInteractionMemory(capacity=4)
    producer = StubEpisodeProducer(seed=0, turns_per_episode=1)
    turns = [episode.turns[0] for episode in producer.produce(6)]

    evicted = [ev for turn in turns if (ev := stim.push(turn)) is not None]
    assert len(stim) == 4, f"expected STIM to hold 4 turns, got {len(stim)}"
    assert len(evicted) == 2, (
        f"expected 2 evictions from 6 pushes at capacity 4, got {len(evicted)}"
    )
    print(f"STIM: {len(stim)} turns buffered, {len(evicted)} evicted")

    mtem = MidTermEpisodicMemory(capacity=100)
    for i, turn in enumerate(evicted):
        mtem.add(_turn_to_episode(turn, i), now=10.0)
    assert len(mtem) == 2, f"expected 2 episodes in MTEM, got {len(mtem)}"
    print(f"MTEM: {len(mtem)} episodes")

    episodes = mtem.episodes()
    entity_extractor = FakeEntityExtractor()
    feature_vectors = [extract_features(e, entity_extractor) for e in episodes]
    assert all(len(fv) == 7 for fv in feature_vectors), "expected FEATURE_DIM=7 vectors"
    print(f"features: extracted {len(feature_vectors)} 7-dim vectors")

    # Selector: fit on a trivial synthetic set so predict() is callable --
    # this smoke test demonstrates pipeline shape, not selector quality.
    selector = FormatSelector()
    rng = np.random.default_rng(0)
    X_synth = rng.normal(size=(6, 7))
    y_synth = np.array([0, 1, 2, 0, 1, 2])
    selector.fit(X_synth, y_synth, X_synth, y_synth)
    assigned_format = selector.predict(feature_vectors[0].to_array())
    print(f"selector: assigned format {assigned_format}")

    incoming, *candidates = episodes

    def scorer(a: EpisodicUnit, b: EpisodicUnit) -> float:
        return float(np.dot(a.embedding, b.embedding))

    decision = select_merge_target(incoming, candidates, scorer, FusionConfig(tau=0.5, m_min=1))
    print(f"fusion: decision={decision.action} target={decision.target_id}")

    for episode in mtem.episodes():
        episode.access_count = 100
        episode.last_access = 10.0
    weights = UtilityWeights(w1=1.0, w2=0.0, w3=0.0)
    thresholds = PromotionThresholds(tau_u=0.5, tau_r=0.0, max_age=1000.0)
    store = FaissVectorStore(dim=_EMBEDDING_DIM)
    promoted = promote(mtem, store, weights, thresholds, now=10.0)
    assert len(promoted) >= 1, "expected at least one episode to promote"
    assert len(store) == len(promoted)
    print(f"LTSM: promoted {len(promoted)} episode(s), store size={len(store)}")

    print("smoke test OK")


if __name__ == "__main__":
    main()
