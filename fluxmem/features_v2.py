"""Feature extraction for selector MLP (v2: 15 features with critical fixes).

This module extracts 15 features from encoder outputs (VS/HG/VC) for the
selector MLP that routes episodes to appropriate memory format. It replaces
the two opaque placeholder scalars `hyperedge_density`/`visual_salience`
that `fluxmem/features.py`'s 7-feature vector reads off `EpisodicUnit` with
a superset of 15 features computed directly from VS/HG/VC encoder output --
see `docs/workflows/feature-engineering-selector/phase3_output.md` (Phase 3,
the feature spec this module implements) and `phase3_features.py` (the
validation script this module's formulas are ported from) in the same
directory.

Key improvements over the Phase 3 validation script:
1. Input validation: rejects NaN/inf in `vs_embedding`/`hg_adjacency`/
   `vc_feature_map` at the boundary (Phase 4b critical fix #1).
2. Output clamping: the 12 features formula-bounded to [0, 1] are clamped
   defensively against floating-point drift (Phase 4b critical fix #2).
3. Zero-vector handling: `vs_embedding_stability` returns the documented
   neutral value 0.5 for a near-zero-norm embedding instead of dividing by
   ~0 (Phase 4b critical fix #3); `vs_energy_concentration` and
   `vs_semantic_specificity` are independently zero-guarded (see their
   docstrings for why they use a *different* fallback value than 0.5 --
   the "neutral, can't-tell" semantics of a stability proxy don't transfer
   to a concentration or specificity statistic).
4. Adjacency validation: negative edge weights raise `ValueError` rather
   than silently corrupting the HG-derived features (Phase 4b fix #4).
5. Epsilon guards: every division in this module (footprint ratios, degree
   distributions, cosine similarity) carries a `+EPS` denominator guard
   (Phase 4b fix #5), and a real bug in the Phase 3 script's `hg_spread` --
   it built the degree distribution *before* checking `n < 2`, which raises
   `ZeroDivisionError` on a 0-node adjacency matrix -- is fixed by
   reordering the guard first (see `_hg_spread`).
6. 62 edge cases in `tests/test_features_v2.py` (zero vectors, empty/1-node
   graphs, all-zero VC maps, huge-magnitude inputs, mismatched shapes,
   negative weights, NaN/inf injection at every input site) plus a
   regression test asserting no NaN/inf ever reaches the returned array.

Deviations from the task brief worth flagging explicitly (evidence
discipline: a prompt's illustrative code is not itself the spec):
- The task brief's example feature names (`vs_dimensionality`,
  `vc_canvas_size`, `vs_hg_efficiency_gap`, `efficiency_variance`, ...)
  do not match the actual, already-decided Phase 3 feature set in
  `phase3_output.md`. This module implements the real, settled 15 --
  `vs_energy_concentration`, `hg_hierarchy_depth`, `vc_layout_concentration`,
  `hg_cost_position`, `hg_information_per_kb`, `vs_hg_structural_agreement`,
  `vs_vc_structural_agreement`, `hg_vc_structural_agreement`,
  `complementarity_score`, `vs_embedding_stability`, `hg_graph_stability`,
  `vc_layout_stability`, `vs_semantic_specificity`, `hg_relational_richness`,
  `vc_layout_prominence` -- rather than re-deriving a different set from the
  brief's placeholder names. See phase3_output.md Part 0 for why even *these*
  are validated as "formula runs correctly", not yet as "discriminates real
  episodes" (every Phase 3 sample array is synthetic).
- `PHASE_4A_VALIDATION_REPORT.md`, `phase4b_report.md`, and
  `edge_case_test.py`, named in the task brief as reference material, do not
  exist anywhere in this repository (verified: `find . -iname` for each
  returns nothing). The 6 critical/important fixes are instead sourced from
  `docs/workflows/feature-engineering-selector/phase4_output_critical_fixes.txt`,
  the one Phase 4 artifact that does exist, whose checklist matches the task
  brief's "Critical Context" section. The 62 edge cases are newly written in
  `tests/test_features_v2.py`, not ported from a missing file.
- `EpisodicUnit` (fluxmem/interfaces.py) had no fields for the *structured*
  raw HG/VC encoder outputs (`hg_adjacency`, `vc_feature_map`) this feature
  set needs -- only the two scalar placeholders `hyperedge_density`/
  `visual_salience`. Both real encoders (hetrep/hg, hetrep/vc) are still
  stubs (ADR 0003), so there is no live producer of these arrays yet. Two
  fields were added to `EpisodicUnit`, defaulting to `None`, matching the
  existing 0.0-default placeholder convention; `extract_features_v2`
  degrades to a documented degenerate structure (see `_resolve_hg_adjacency`
  / `_resolve_vc_feature_map`) rather than raising, so the selector's
  feature extraction path doesn't hard-fail before Phase 3 (HG/VC) ships.
  `unit.embedding` (VS), by contrast, *is* raising if missing -- VS is
  already Phase 1, implemented, and every downstream feature that needs a
  vector needs it to be a real vector, not a documented stand-in.
- Per-group extraction functions take raw encoder-output arrays, not the
  `EpisodicUnit` itself -- this matches phase3_output.md's own "Next Steps
  for Implementation" #2 ("extraction function shape ... mirrors
  phase3_features.py's functions directly"), and keeps each group
  independently unit-testable without constructing a full `EpisodicUnit`.
  Only the top-level `extract_features_v2` touches `EpisodicUnit`.

Usage:
    unit = EpisodicUnit(episode_id="e0", turns=[...], primary_format=..., embedding=...)
    features = extract_features_v2(unit)  # np.ndarray shape (15,)

    # vs_semantic_specificity needs a batch/corpus context (the one feature
    # in this set that isn't a pure function of a single episode -- see
    # phase3_output.md "Next Steps" #4); pass the running VS-embedding mean
    # once a batch exists, otherwise it degrades to the documented default.
    features = extract_features_v2(unit, corpus_vs_mean=running_vs_mean)

Test:
    python -m pytest tests/test_features_v2.py -v
    python fluxmem/features_v2.py --test-edge-cases  # forwards to the suite above
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from fluxmem.interfaces import EpisodicUnit

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
FEATURE_NAMES_V2 = (
    "vs_energy_concentration",
    "hg_hierarchy_depth",
    "vc_layout_concentration",
    "hg_cost_position",
    "hg_information_per_kb",
    "vs_hg_structural_agreement",
    "vs_vc_structural_agreement",
    "hg_vc_structural_agreement",
    "complementarity_score",
    "vs_embedding_stability",
    "hg_graph_stability",
    "vc_layout_stability",
    "vs_semantic_specificity",
    "hg_relational_richness",
    "vc_layout_prominence",
)
FEATURE_DIM_V2 = len(FEATURE_NAMES_V2)  # 15 -- never hardcode the 15

# The 12 features phase3_output.md's Range column formula-bounds to [0, 1]
# (hg_cost_position, hg_information_per_kb are unbounded above in principle;
# vc_layout_prominence is bounded to [-1, 1], not [0, 1] -- none of the three
# are clamped here). Phase 4b critical fix #2.
BOUNDED_FEATURES_V2 = frozenset(
    {
        "vs_energy_concentration",
        "hg_hierarchy_depth",
        "vc_layout_concentration",
        "vs_hg_structural_agreement",
        "vs_vc_structural_agreement",
        "hg_vc_structural_agreement",
        "complementarity_score",
        "vs_embedding_stability",
        "hg_graph_stability",
        "vc_layout_stability",
        "vs_semantic_specificity",
        "hg_relational_richness",
    }
)
assert len(BOUNDED_FEATURES_V2) == 12  # matches Phase 4b's "clamp 12 features" finding

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
EPS = 1e-12
# Below this L2 norm, a vector is treated as numerically zero -- Phase 4b
# critical fix #3's threshold, applied consistently everywhere a norm
# appears in a denominator.
_ZERO_NORM_EPS = 1e-10

VS_ITEMSIZE_BYTES = 4  # float32
VC_ITEMSIZE_BYTES = 4  # float32
DEFAULT_VC_SHAPE = (7, 7, 512)  # architecture-fixed VC canvas grid x channels

# Documented placeholder JSON-serialization-size estimates (phase3_output.md
# Group 2), pending measurement against a real HyperMem payload.
HG_BYTES_PER_NODE = 150.0
HG_BYTES_PER_EDGE = 80.0

_DEFAULT_RNG_SEED = 42  # phase3_features.py's seed, kept for reproducibility
_STABILITY_TRIALS_VS = 20
_STABILITY_TRIALS_HG = 20
_STABILITY_TRIALS_VC = 10
_STABILITY_SIGMA_VS = 0.01
_STABILITY_SIGMA_HG = 0.02
_STABILITY_SIGMA_VC = 0.02


# ---------------------------------------------------------------------------
# Input validation (Phase 4b critical fix #1, important fix #4)
# ---------------------------------------------------------------------------
def _validate_vs_embedding(v: np.ndarray) -> None:
    """Rejects a malformed or NaN/inf VS embedding before any formula runs."""
    if v.ndim != 1:
        raise ValueError(f"vs_embedding must be 1-D, got shape {v.shape}")
    if v.size == 0:
        raise ValueError("vs_embedding must be non-empty")
    if np.any(np.isnan(v)):
        raise ValueError("vs_embedding contains NaN")
    if np.any(np.isinf(v)):
        raise ValueError("vs_embedding contains inf")


def _validate_hg_adjacency(adj: np.ndarray) -> None:
    """Rejects a malformed, NaN/inf, or negative-weight adjacency matrix.

    Negative edge weights (Phase 4b important fix #4) would silently corrupt
    every HG-derived feature downstream (a "richness" that can go negative,
    a degree distribution that isn't a real probability distribution) --
    rejecting here is cheaper than debugging a negative-richness episode
    three features later.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"hg_adjacency must be a square 2-D matrix, got shape {adj.shape}")
    if np.any(np.isnan(adj)):
        raise ValueError("hg_adjacency contains NaN")
    if np.any(np.isinf(adj)):
        raise ValueError("hg_adjacency contains inf")
    if np.any(adj < 0):
        raise ValueError("hg_adjacency contains negative edge weights")


def _validate_vc_feature_map(feat: np.ndarray) -> None:
    """Rejects a malformed or NaN/inf VC feature map.

    Shape is validated as 3-D (H, W, C) but not pinned to the architecture
    default (7, 7, 512) -- footprint/spread formulas derive H/W/C from the
    array itself (matching phase3_features.py), so a differently-shaped real
    VC encoder output is still handled correctly rather than rejected.
    """
    if feat.ndim != 3:
        raise ValueError(f"vc_feature_map must be 3-D (H, W, C), got shape {feat.shape}")
    if feat.size == 0:
        raise ValueError("vc_feature_map must be non-empty")
    if np.any(np.isnan(feat)):
        raise ValueError("vc_feature_map contains NaN")
    if np.any(np.isinf(feat)):
        raise ValueError("vc_feature_map contains inf")


def _validate_corpus_vs_mean(corpus_mean: np.ndarray, vs_dim: int) -> None:
    if corpus_mean.ndim != 1 or corpus_mean.shape[0] != vs_dim:
        raise ValueError(
            f"corpus_vs_mean must be 1-D with shape ({vs_dim},), got shape {corpus_mean.shape}"
        )
    if np.any(np.isnan(corpus_mean)) or np.any(np.isinf(corpus_mean)):
        raise ValueError("corpus_vs_mean contains NaN/inf")


# ---------------------------------------------------------------------------
# Shared low-level statistics (private -- group functions compose these)
# ---------------------------------------------------------------------------
def _vs_spread(v: np.ndarray) -> float:
    """Participation ratio of squared components, rescaled to [0, 1].

    High = embedding energy spread across many dims (isotropic); low = a few
    dims dominate (peaky). Zero-vector guarded: a vector with ~0 norm has no
    energy to be concentrated or spread, so this returns 0.0 (the formula's
    natural floor) rather than the ZeroDivisionError-adjacent NaN a raw
    `(pr - 1/d)/(1 - 1/d)` would produce when `sum(v**4)` underflows to 0.
    Also guards `d < 2`, where `(1 - 1/d)` is 0 in the denominator -- never
    hit by a real 384-d VS embedding, defensive only.

    Rescales by `max(|v|)` before squaring: the participation ratio is scale
    invariant (`pr(v) == pr(k*v)` for any `k != 0`), but the naive formula
    squares an already-squared sum (`norm_sq**2`), which overflows float64
    for embeddings with extreme-magnitude components (e.g. ~1e150) well
    before the true, scale-invariant answer would. Rescaling first keeps
    every intermediate quantity near O(1) regardless of the input's absolute
    scale -- caught by `TestExtremeMagnitudeInputs` in
    `tests/test_features_v2.py`.
    """
    d = v.shape[0]
    if d < 2:
        return 0.0
    scale = float(np.max(np.abs(v)))
    if scale < _ZERO_NORM_EPS:
        return 0.0
    vn = v / scale
    norm_sq = float(np.sum(vn**2))
    if norm_sq < EPS:
        return 0.0
    num = norm_sq**2
    den = d * float(np.sum(vn**4)) + EPS
    pr = num / den
    return float(np.clip((pr - 1.0 / d) / (1.0 - 1.0 / d), 0.0, 1.0))


def _degree_dist(adj: np.ndarray) -> np.ndarray:
    """Weighted total (in+out) degree per node, normalized to a distribution.

    Only ever called with `n >= 2` (see `_hg_spread`) -- `len(deg) == 0`
    would otherwise raise `ZeroDivisionError` on `1.0 / len(deg)`, a real bug
    in the Phase 3 validation script's call order (it built this
    distribution *before* checking `n < 2`).
    """
    deg = adj.sum(axis=0) + adj.sum(axis=1)
    total = deg.sum()
    if total <= EPS:
        return np.full_like(deg, 1.0 / len(deg))
    return deg / total


def _hg_spread(adj: np.ndarray) -> float:
    """Normalized Shannon entropy of the weighted degree distribution, [0, 1].

    High = relational importance spread evenly across nodes; low = a few hub
    nodes dominate. Guards `n < 2` *before* building the degree distribution
    (fixes the Phase 3 script's order-of-operations bug -- see `_degree_dist`).
    """
    n = adj.shape[0]
    if n < 2:
        return 0.0
    p = _degree_dist(adj)
    p_safe = p[p > 0]
    if p_safe.size == 0:
        return 0.0
    h = -float(np.sum(p_safe * np.log(p_safe)))
    return float(np.clip(h / np.log(n), 0.0, 1.0))


def _vc_spatial_map(feat: np.ndarray) -> np.ndarray:
    """Channel-averaged spatial activation map, shape (H, W) -> flattened (H*W,)."""
    return feat.mean(axis=-1).reshape(-1)


def _vc_spread(feat: np.ndarray) -> float:
    """Normalized Shannon entropy of the spatial activation map, [0, 1].

    An all-zero/near-zero map (the degenerate-placeholder case, and the
    Phase 3 sample data's near-ceiling case -- phase3_output.md 4.1) has no
    concentration to speak of; returns 1.0 (maximally "spread", the formula's
    documented degenerate convention, unchanged from phase3_features.py) so
    downstream `vc_layout_concentration = 1 - spread` reads 0.0.
    """
    s = np.clip(_vc_spatial_map(feat), 0, None)
    total = s.sum()
    if total <= EPS:
        return 1.0
    p = s / total
    p_safe = p[p > 0]
    if len(s) < 2 or p_safe.size == 0:
        return 1.0
    h = -float(np.sum(p_safe * np.log(p_safe)))
    return float(np.clip(h / np.log(len(s)), 0.0, 1.0))


def _vs_footprint_kb(vs_dim: int) -> float:
    return vs_dim * VS_ITEMSIZE_BYTES / 1024.0


def _vc_footprint_kb(vc_shape: tuple[int, int, int]) -> float:
    h, w, c = vc_shape
    return h * w * c * VC_ITEMSIZE_BYTES / 1024.0


def _hg_footprint_kb(adj: np.ndarray) -> float:
    n = adj.shape[0]
    e = int(np.count_nonzero(adj))
    return (n * HG_BYTES_PER_NODE + e * HG_BYTES_PER_EDGE) / 1024.0


def _hg_relational_richness(adj: np.ndarray) -> float:
    """Confidence-weighted relational density: `sum(edge weights) / (N*(N-1))`, [0, 1]."""
    n = adj.shape[0]
    if n < 2:
        return 0.0
    return float(np.clip(adj.sum() / (n * (n - 1)), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Group 1: Geometry & Scale
# ---------------------------------------------------------------------------
def extract_geometry_features(
    vs_embedding: np.ndarray, hg_adjacency: np.ndarray, vc_feature_map: np.ndarray
) -> dict[str, float]:
    """VS/HG/VC dimensionality and structure properties (phase3_output.md Group 1)."""
    return {
        "vs_energy_concentration": _vs_spread(vs_embedding),
        # Future-only: the 3-layer Facts/Episodes/Topics typing has no
        # analog in today's flat, untyped adjacency -- fixed 1/3 (a single
        # flat layer) until typed 3-layer HG output exists.
        "hg_hierarchy_depth": 1.0 / 3.0,
        "vc_layout_concentration": 1.0 - _vc_spread(vc_feature_map),
    }


# ---------------------------------------------------------------------------
# Group 2: Efficiency & Memory Trade-off
# ---------------------------------------------------------------------------
def extract_efficiency_features(
    hg_adjacency: np.ndarray, vs_dim: int, vc_shape: tuple[int, int, int]
) -> dict[str, float]:
    """Memory footprint and efficiency metrics (phase3_output.md Group 2).

    Deliberately excludes `task_accuracy`-based ratios -- those are the
    selector's training *label* (fluxmem/supervision.py), not an input
    feature; including one would be target leakage (phase3_output.md Part
    0.2). Epsilon-guarded denominator (Phase 4b important fix #5), even
    though `vc_kb - vs_kb` is a fixed architecture constant (~96.5) that
    never reaches zero in practice.
    """
    vs_kb = _vs_footprint_kb(vs_dim)
    vc_kb = _vc_footprint_kb(vc_shape)
    hg_kb = _hg_footprint_kb(hg_adjacency)
    richness = _hg_relational_richness(hg_adjacency)
    return {
        "hg_cost_position": float((hg_kb - vs_kb) / (vc_kb - vs_kb + EPS)),
        "hg_information_per_kb": float(richness / (hg_kb + EPS)),
    }


# ---------------------------------------------------------------------------
# Group 3: Complementarity & Overlap
# ---------------------------------------------------------------------------
def extract_overlap_features(
    vs_spread: float, hg_spread: float, vc_spread: float
) -> dict[str, float]:
    """Redundancy and complementarity between encoder pairs (phase3_output.md Group 3).

    Structural agreement, not semantic agreement -- compares each encoder's
    self spread/concentration statistic, not whether they agree on *which*
    content matters (phase3_output.md 4.4). Takes the raw, un-flipped spread
    stats (same `vc_spread` Group 1 flips into `vc_layout_concentration`).
    """
    vs_hg = 1.0 - abs(vs_spread - hg_spread)
    vs_vc = 1.0 - abs(vs_spread - vc_spread)
    hg_vc = 1.0 - abs(hg_spread - vc_spread)
    return {
        "vs_hg_structural_agreement": vs_hg,
        "vs_vc_structural_agreement": vs_vc,
        "hg_vc_structural_agreement": hg_vc,
        "complementarity_score": 1.0 - float(np.mean([vs_hg, vs_vc, hg_vc])),
    }


# ---------------------------------------------------------------------------
# Group 4: Robustness & Stability
# ---------------------------------------------------------------------------
def extract_robustness_features(
    vs_embedding: np.ndarray,
    hg_adjacency: np.ndarray,
    vc_feature_map: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Noise sensitivity and consistency (phase3_output.md Group 4).

    Output-space perturb-and-remeasure proxy, not true input-text
    paraphrase-robustness -- the live encoders and original episode text
    aren't available to this module (phase3_output.md's Group 4 preamble,
    stated as a limitation, not silently absorbed into "tested").
    """
    return {
        "vs_embedding_stability": _vs_embedding_stability(vs_embedding, rng),
        "hg_graph_stability": _hg_graph_stability(hg_adjacency, rng),
        "vc_layout_stability": _vc_layout_stability(vc_feature_map, rng),
    }


def _vs_embedding_stability(
    v: np.ndarray,
    rng: np.random.Generator,
    sigma: float = _STABILITY_SIGMA_VS,
    trials: int = _STABILITY_TRIALS_VS,
) -> float:
    """Phase 4b critical fix #3: 0.5 (neutral -- "can't say") for a near-zero-norm
    embedding, rather than dividing by a ~0 norm when renormalizing the
    perturbed vector. 0.5 (not 0.0, unlike `_vs_spread`'s zero-vector floor)
    because "stability" asks a different question than "concentration": a
    zero vector isn't necessarily *unstable*, there is simply no basis to
    measure Lipschitz sensitivity around a point that doesn't exist, so this
    reports maximal uncertainty rather than asserting either extreme.
    """
    if float(np.linalg.norm(v)) < _ZERO_NORM_EPS:
        return 0.5
    v_norm = v / (float(np.linalg.norm(v)) + EPS)
    dists = np.empty(trials)
    for i in range(trials):
        noise = rng.normal(0, sigma, size=v.shape)
        v_pert = v + noise
        v_pert = v_pert / (float(np.linalg.norm(v_pert)) + EPS)
        dists[i] = np.linalg.norm(v_norm - v_pert)
    mean_dist = float(np.mean(dists))
    return float(np.clip(1.0 - min(mean_dist / 2.0, 1.0), 0.0, 1.0))


def _hg_graph_stability(
    adj: np.ndarray,
    rng: np.random.Generator,
    sigma: float = _STABILITY_SIGMA_HG,
    trials: int = _STABILITY_TRIALS_HG,
) -> float:
    """1 - mean absolute change in `hg_relational_richness` under edge-weight jitter.

    Uses richness (continuous, weight-sensitive) rather than a count-based
    edge-density statistic, which the Phase 3 validation found saturates at
    a constant 1.0 -- jitter this small essentially never flips a weight
    across the zero threshold (phase3_output.md 4.1).
    """
    base = _hg_relational_richness(adj)
    if adj.shape[0] < 2:
        return 1.0  # no edges possible to perturb; vacuously stable
    deltas = np.empty(trials)
    for i in range(trials):
        noise = rng.normal(0, sigma, size=adj.shape)
        adj_pert = np.clip(adj + noise * (adj > 0), 0, 1)
        deltas[i] = abs(_hg_relational_richness(adj_pert) - base)
    mean_delta = float(np.mean(deltas))
    return float(np.clip(1.0 - min(mean_delta, 1.0), 0.0, 1.0))


def _vc_layout_stability(
    feat: np.ndarray,
    rng: np.random.Generator,
    sigma: float = _STABILITY_SIGMA_VC,
    trials: int = _STABILITY_TRIALS_VC,
) -> float:
    base = 1.0 - _vc_spread(feat)
    deltas = np.empty(trials)
    for i in range(trials):
        noise = rng.normal(0, sigma, size=feat.shape)
        feat_pert = np.clip(feat + noise, 0, 1)
        deltas[i] = abs((1.0 - _vc_spread(feat_pert)) - base)
    mean_delta = float(np.mean(deltas))
    return float(np.clip(1.0 - min(mean_delta, 1.0), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Group 5: Task-Relevant / Domain Fit
# ---------------------------------------------------------------------------
def extract_task_features(
    vs_embedding: np.ndarray,
    corpus_vs_mean: np.ndarray,
    hg_adjacency: np.ndarray,
    vc_feature_map: np.ndarray,
) -> dict[str, float]:
    """Encoder-specific properties (phase3_output.md Group 5)."""
    return {
        "vs_semantic_specificity": _vs_semantic_specificity(vs_embedding, corpus_vs_mean),
        "hg_relational_richness": _hg_relational_richness(hg_adjacency),
        "vc_layout_prominence": _vc_layout_prominence(vc_feature_map),
    }


def _vs_semantic_specificity(v: np.ndarray, corpus_mean: np.ndarray) -> float:
    """1 - max(cos(v, corpus_mean), 0), [0, 1].

    A zero `v` already degrades safely without a special case: `dot` is 0,
    the denominator's `+EPS` prevents literal 0/0, so `cos` is 0 and the
    feature reads 1.0 ("maximally distinctive" -- a defensible reading of
    "resembles nothing", not a crash).
    """
    denom = float(np.linalg.norm(v)) * float(np.linalg.norm(corpus_mean)) + EPS
    cos = float(np.dot(v, corpus_mean)) / denom
    return float(np.clip(1.0 - max(cos, 0.0), 0.0, 1.0))


def _vc_layout_prominence(feat: np.ndarray) -> float:
    """Peak-to-mean ratio of the spatial map: `(max - mean) / (max + mean)`, [-1, 1]."""
    s = _vc_spatial_map(feat)
    peak, mean = float(s.max()), float(s.mean())
    return float((peak - mean) / (peak + mean + EPS))


# ---------------------------------------------------------------------------
# Degenerate-input resolution (HG/VC encoders are still stubs -- see module
# docstring's "Deviations" section)
# ---------------------------------------------------------------------------
def _resolve_hg_adjacency(unit: EpisodicUnit) -> np.ndarray:
    """A single isolated node, no edges -- the smallest well-defined graph.

    Every formula above is `n < 2`-guarded, so this produces well-defined
    (near-zero, non-signaling) feature values rather than crashing, matching
    `EpisodicUnit.hyperedge_density`'s existing 0.0-default convention.
    """
    if unit.hg_adjacency is not None:
        return np.asarray(unit.hg_adjacency, dtype=np.float64)
    return np.zeros((1, 1), dtype=np.float64)


def _resolve_vc_feature_map(unit: EpisodicUnit) -> np.ndarray:
    """An all-zero map at the architecture-fixed (7, 7, 512) default shape."""
    if unit.vc_feature_map is not None:
        return np.asarray(unit.vc_feature_map, dtype=np.float64)
    return np.zeros(DEFAULT_VC_SHAPE, dtype=np.float64)


def _resolve_corpus_vs_mean(v: np.ndarray, corpus_vs_mean: np.ndarray | None) -> np.ndarray:
    """Degrades to `v` itself (a single-episode "batch") when no corpus context
    is available yet -- yields `cos == 1` -> `vs_semantic_specificity == 0.0`
    ("as generic as a batch of one can be"), a defensible boundary value
    rather than an arbitrary one.
    """
    if corpus_vs_mean is not None:
        return np.asarray(corpus_vs_mean, dtype=np.float64)
    return v


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------
def extract_features_v2(
    unit: EpisodicUnit,
    *,
    corpus_vs_mean: np.ndarray | None = None,
    rng_seed: int = _DEFAULT_RNG_SEED,
) -> np.ndarray:
    """Extracts all 15 features for `unit`, with the Phase 4b critical fixes applied.

    `unit.embedding` (VS) is required -- raises if `None` or invalid, since
    VS is already-implemented Phase 1 and every VS-dependent feature needs a
    real vector. `unit.hg_adjacency`/`unit.vc_feature_map` (HG/VC) are
    optional -- both real encoders are still stubs (ADR 0003), so `None`
    degrades to a documented placeholder structure (see
    `_resolve_hg_adjacency`/`_resolve_vc_feature_map`) instead of raising.

    `rng_seed` seeds a local `np.random.Generator` for Group 4's
    perturb-and-remeasure trials -- a local generator, not global
    `np.random.seed`, so this function's output is deterministic given its
    inputs regardless of call order or what else in the process has touched
    numpy's global RNG state (`.claude/rules/testing.md`: deterministic
    tests).

    Raises:
        ValueError: `unit.embedding` is missing/invalid, or any provided
            raw encoder output is malformed/NaN/inf/negative-weighted, or
            (defensively) the assembled feature vector still contains
            NaN/inf after every per-feature guard above.
    """
    if unit.embedding is None:
        raise ValueError("extract_features_v2 requires unit.embedding (VS output) to be populated")
    vs_embedding = np.asarray(unit.embedding, dtype=np.float64)
    _validate_vs_embedding(vs_embedding)

    hg_adjacency = _resolve_hg_adjacency(unit)
    _validate_hg_adjacency(hg_adjacency)

    vc_feature_map = _resolve_vc_feature_map(unit)
    _validate_vc_feature_map(vc_feature_map)

    resolved_corpus_mean = _resolve_corpus_vs_mean(vs_embedding, corpus_vs_mean)
    _validate_corpus_vs_mean(resolved_corpus_mean, vs_embedding.shape[0])

    vs_spread = _vs_spread(vs_embedding)
    hg_spread = _hg_spread(hg_adjacency)
    vc_spread = _vc_spread(vc_feature_map)

    vc_shape = (vc_feature_map.shape[0], vc_feature_map.shape[1], vc_feature_map.shape[2])
    rng = np.random.default_rng(rng_seed)

    values: dict[str, float] = {}
    values.update(extract_geometry_features(vs_embedding, hg_adjacency, vc_feature_map))
    values.update(extract_efficiency_features(hg_adjacency, vs_embedding.shape[0], vc_shape))
    values.update(extract_overlap_features(vs_spread, hg_spread, vc_spread))
    values.update(extract_robustness_features(vs_embedding, hg_adjacency, vc_feature_map, rng))
    values.update(
        extract_task_features(vs_embedding, resolved_corpus_mean, hg_adjacency, vc_feature_map)
    )

    feature_array = np.array([values[name] for name in FEATURE_NAMES_V2], dtype=np.float64)

    # Phase 4b critical fix #2: clamp the 12 formula-bounded-to-[0,1]
    # features against floating-point drift (e.g. 1.0000000000000002).
    for i, name in enumerate(FEATURE_NAMES_V2):
        if name in BOUNDED_FEATURES_V2:
            feature_array[i] = np.clip(feature_array[i], 0.0, 1.0)

    # Phase 4b critical fix #3 (regression backstop): every per-feature
    # guard above should already prevent this, but a silent NaN/inf reaching
    # the selector's MLP is exactly the failure mode Phase 4b exists to
    # catch, so this is checked explicitly rather than trusted implicitly.
    if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
        bad = {
            name: float(feature_array[i])
            for i, name in enumerate(FEATURE_NAMES_V2)
            if not np.isfinite(feature_array[i])
        }
        raise ValueError(f"Feature extraction produced NaN/inf: {bad}")

    return feature_array


if __name__ == "__main__":
    if "--test-edge-cases" in sys.argv:
        import pytest

        suite = Path(__file__).resolve().parent.parent / "tests" / "test_features_v2.py"
        raise SystemExit(pytest.main(["-v", str(suite)]))
    print(f"FEATURE_NAMES_V2 ({FEATURE_DIM_V2}):")
    for name in FEATURE_NAMES_V2:
        bounded = "clamped [0,1]" if name in BOUNDED_FEATURES_V2 else "unclamped"
        print(f"  {name:<30} {bounded}")
