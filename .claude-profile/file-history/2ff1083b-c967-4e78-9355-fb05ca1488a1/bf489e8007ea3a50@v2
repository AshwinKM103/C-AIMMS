"""fluxmem/ltsm.py -- LTSM tier: promotion gate + FAISS store (COLM Sec 1.3.2).

## Gate 1 is the composite `U`, not usage (the one real conflict between the papers)

COLM Sec 1.3.2 writes `U(e_j) >= tau_u`, where capital `U` is Eq. 5's composite
(`fluxmem/mtem.py`). FluxMem Eq. 6 writes `u(m) >= tau_u` and defines `u(m)` as
*usage alone*. Per this project's Reading rule (COLM wins on conflicts, FluxMem
fills COLM's silences), `is_eligible` below uses the composite. The difference
is compensatory vs. non-compensatory: under COLM's reading, a rarely-accessed
episode can still promote if it is long/intense and recent (`w2*l + w3*d`
compensate for low `c`); under FluxMem's reading, access count alone decides
and nothing compensates. Two consequences that must survive into experiment
records: (1) promotion is coupled to `w1,w2,w3` -- changing the weights
changes which episodes promote, a coupling that does not exist under FluxMem's
reading; record the weights alongside `tau_u` or the promotion rate is not
reproducible; (2) `tau_u` is on composite `U`'s scale (`[0, w1+w2+w3]`), not
FluxMem's raw usage-count scale (`[0, inf)`) -- FluxMem's tuned threshold
values are not portable here.

## The confidence gate is not implemented

FluxMem Eq. 6's third conjunct `(AND c(m) >= tau_c)` is parenthesised and
called "optional" in its own source, and COLM omits it entirely. It is also
unproducible in this scope: confidence would have to come from a fact
extractor rating its own output, and this package's write path has no step
that emits one (HETREP/ITERRET are non-goals). Omitting it is coherent, not
an oversight, and safe: dropping a pure AND-conjunct makes promotion strictly
more permissive, so it can never reject anything the two remaining gates
accept. Known, non-silent gap: if LTSM precision measures poorly against
COLM's "compact and high-precision" claim, this is the first knob to revisit
-- but it does not get a disabled `tau_c` config flag (absent beats dead; see
`fluxmem/config.py`'s module docstring).

## `r(e_j)` and `d(e_j)` are both recency, deliberately

Not a conflict -- both papers do this. `d(e_j)` (inside `U`, `fluxmem/mtem.py`)
is a soft, tradeable addend: `exp(-dt/half_life)`, smooth, no cliff, right for
a term being summed against others. `r(e_j)` (this module) is a hard,
non-negotiable threshold: `1 - min(dt/max_age, 1)`, linear, so `tau_r=0.8`
reads as "touched within the most recent 20% of the horizon." Both derive
from the same `dt = now - last_access` and are strictly decreasing in `dt`,
so they can never disagree in rank order (tested below) -- but they can
disagree in verdict: a slightly-stale, heavily-used episode can clear `tau_u`
via compensation while still failing the hard `tau_r` gate, which is the
point (belt-and-braces: the composite admits useful-but-aged episodes, the
hard gate still refuses anything genuinely ancient).

## Note on storage-invariants.md

`.claude/rules/storage-invariants.md` is path-scoped and `configs/` matches
it, but its `dense_weight`/`bm25_weight` invariant describes FluxMem's
hybrid retriever in `LightMem/`. This package builds no BM25 fusion --
retrieval (ITERRET) is an explicit non-goal -- so there are no fusion
weights to record. Stated here explicitly rather than left for a reader to
assume the invariant was overlooked.
"""

from __future__ import annotations

import faiss
import numpy as np

from fluxmem.config import PromotionThresholds, UtilityWeights
from fluxmem.interfaces import EpisodicUnit, VectorStore
from fluxmem.mtem import MidTermEpisodicMemory, utility_score


def recency_gate(episode: EpisodicUnit, now: float, max_age: float) -> float:
    """`r(e_j)`: hard recency gate, linear in `dt = now - last_access`. See module docstring."""
    if max_age <= 0:
        raise ValueError(f"max_age must be > 0, got {max_age}")
    delta_t = max(0.0, now - episode.last_access)
    return 1.0 - min(delta_t / max_age, 1.0)


def is_eligible(
    episode: EpisodicUnit,
    weights: UtilityWeights,
    thresholds: PromotionThresholds,
    now: float,
) -> bool:
    """Exactly `U(e) >= tau_u AND r(e) >= tau_r`. Two conjuncts, no third (see module docstring)."""
    u = utility_score(episode, weights, now)
    r = recency_gate(episode, now, thresholds.max_age)
    return u >= thresholds.tau_u and r >= thresholds.tau_r


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


class FaissVectorStore:
    """`VectorStore` (see fluxmem/interfaces.py) backed by `faiss.IndexFlatIP`.

    Vectors are L2-normalized before indexing, so inner product equals
    cosine similarity. `IndexFlatIP` has no native string ids, so this
    keeps an explicit id<->row map alongside it.
    """

    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._row_to_id: list[str] = []

    def __len__(self) -> int:
        return len(self._row_to_id)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(f"vectors must be shape (n, {self._dim}), got {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors must have the same length")
        self._index.add(_l2_normalize(vectors))
        self._row_to_id.extend(ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if len(self) == 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        scores, rows = self._index.search(_l2_normalize(query), min(k, len(self)))
        return [
            (self._row_to_id[row], float(score))
            for row, score in zip(rows[0], scores[0], strict=True)
            if row != -1
        ]


def promote(
    mtem: MidTermEpisodicMemory,
    store: VectorStore,
    weights: UtilityWeights,
    thresholds: PromotionThresholds,
    now: float,
) -> list[EpisodicUnit]:
    """Moves every `is_eligible` episode from `mtem` into `store`, embeds via
    the episode's own `embedding` field.

    Episodes without an embedding are skipped: producing one is HETREP's
    job (a non-goal here, see fluxmem/interfaces.py) -- `promote` never
    synthesizes one.
    """
    promoted: list[EpisodicUnit] = []
    for episode in list(mtem.episodes()):
        if episode.embedding is None:
            continue
        if not is_eligible(episode, weights, thresholds, now):
            continue
        store.add([episode.episode_id], episode.embedding.reshape(1, -1))
        mtem.remove(episode.episode_id)
        promoted.append(episode)
    return promoted
