# Deployment Guide: Feature Implementation & Integration

**For**: ML engineers implementing Phase 5  
**Time Estimate**: 2–3 weeks (implementation + testing + integration)  
**Success Criteria**: All 15 features implemented, 62 edge cases passing, MLP training converges cleanly

---

## Quick Start (1 Day Overview)

**Time**: ~6–8 hours reading + orientation

1. **Read this section** (20 min)
2. **Read FEATURE_ENGINEERING_REPORT.md, sections 1–3** (1 hour) — executive summary, background, specs
3. **Read FEATURE_REFERENCE.md, Group 1** (30 min) — understand one feature deeply
4. **Skim phase3_features.py** (30 min) — see how all 15 are implemented
5. **Understand the 6 critical fixes** (Phase 4 handoff checklist) (30 min)
6. **Check the integration point** (fluxmem/selector.py lines ~80–120) (30 min)

After this, you have enough context to start implementation. Deep knowledge of each feature comes during coding.

---

## Step 1: Environment Setup (30 minutes)

### Activate Canonical Environment

```bash
conda activate caimms  # Python 3.11.15, torch >=2.7, sentence-transformers
```

Verify:

```bash
python --version  # Python 3.11.15
pip show torch sentence-transformers  # Check versions
```

### Verify Sample Data Exists

```bash
ls -la /tmp/selector_sample_data/
# Expected: vs_embeddings.npy, vc_features.npy, hg_data/ (with adjacency_*.npy, node_features_*.npy)
```

If missing, generate:

```bash
cd /home/durgesh/aditya/C-AIMMS/docs/workflows/feature-engineering-selector
python sample_data_generator.py
```

### Verify phase3_features.py Runs

```bash
python phase3_features.py
# Expected output: Summary statistics table + correlation matrix
```

This is your reference implementation. Keep it open while coding Phase 5.

---

## Step 2: Implementation (8–12 hours)

### Create features_v2.py Structure

**Location**: `/home/durgesh/aditya/C-AIMMS/fluxmem/features_v2.py`

**Template**:

```python
"""fluxmem/features_v2.py -- 20-feature vector for the format selector (Phase 3 design).

Replaces fluxmem/features.py (7 features) with 15-feature set designed for
VS/HG/VC encoder outputs. Includes all 6 critical fixes from Phase 4b.

This file is the reference implementation; use phase3_features.py as template
for all formulas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fluxmem.interfaces import EpisodicUnit, EntityExtractor, SpacyEntityExtractor

# ---- Feature naming and dimensions ----
FEATURE_NAMES_OLD = (  # Kept for backward compatibility
    "entity_count",
    "cooccurrence_density",
    "temporal_ordering",
    "topic_diversity",
    "token_length",
)

FEATURE_NAMES_NEW = (  # The 15 new features
    "vs_energy_concentration",
    "hg_hierarchy_depth",
    "vc_layout_concentration",
    # ... (all 15 names)
)

FEATURE_NAMES = FEATURE_NAMES_OLD + FEATURE_NAMES_NEW
FEATURE_DIM = len(FEATURE_NAMES)  # 20

# ... rest of implementation
```

### Group-Based Implementation

Implement one group at a time (each ~1 hour):

**Group 1: Geometry & Scale (1 hour)**

```python
def extract_geometry_features(
    vs_embedding: np.ndarray,
    hg_adjacency: np.ndarray,
    vc_features: np.ndarray,
) -> dict[str, float]:
    """Extract Group 1 features.

    Args:
        vs_embedding: shape (384,), float32
        hg_adjacency: shape (N, N), float32, weighted [0,1]
        vc_features: shape (7, 7, 512), float32

    Returns:
        dict with 3 keys: vs_energy_concentration, hg_hierarchy_depth, vc_layout_concentration
    """
    # Copy implementations from phase3_features.py (lines 46–125)
    # Each function is self-contained; just copy the logic

    return {
        "vs_energy_concentration": vs_spread(vs_embedding),
        "hg_hierarchy_depth": hg_hierarchy_depth(n_layers_present=1),  # Fixed at 1/3 for stub
        "vc_layout_concentration": vc_layout_concentration(vc_features),
    }
```

**Group 2: Efficiency & Memory (1 hour)**

```python
def extract_efficiency_features(hg_adjacency: np.ndarray) -> dict[str, float]:
    """Extract Group 2 features."""
    return {
        "hg_cost_position": hg_cost_position(hg_adjacency),
        "hg_information_per_token": hg_information_per_token(hg_adjacency),  # memory budget = tokens, not disk space; see DESIGN_RATIONALE.md#memory-budget--tokens-not-disk-space
    }
```

**Group 3: Complementarity (1.5 hours)**

```python
def extract_overlap_features(
    vs_embedding: np.ndarray,
    hg_adjacency: np.ndarray,
    vc_features: np.ndarray,
) -> dict[str, float]:
    """Extract Group 3 features."""
    # Reuse spread functions from Group 1, compare them
    vs_spread_val = vs_spread(vs_embedding)
    hg_spread_val = hg_spread(hg_adjacency)
    vc_spread_val = vc_spread(vc_features)

    vs_hg_agree = 1.0 - abs(vs_spread_val - hg_spread_val)
    vs_vc_agree = 1.0 - abs(vs_spread_val - vc_spread_val)
    hg_vc_agree = 1.0 - abs(hg_spread_val - vc_spread_val)

    return {
        "vs_hg_structural_agreement": vs_hg_agree,
        "vs_vc_structural_agreement": vs_vc_agree,
        "hg_vc_structural_agreement": hg_vc_agree,
        "complementarity_score": 1.0 - np.mean([vs_hg_agree, vs_vc_agree, hg_vc_agree]),
    }
```

**Group 4: Robustness (2 hours)**

```python
def extract_robustness_features(
    vs_embedding: np.ndarray,
    hg_adjacency: np.ndarray,
    vc_features: np.ndarray,
    seed: int = 42,
) -> dict[str, float]:
    """Extract Group 4 features.

    This group requires stochastic perturbation (20–10 trials).
    Seed for reproducibility.
    """
    np.random.seed(seed)  # Fix seed for deterministic results

    return {
        "vs_embedding_stability": vs_embedding_stability(vs_embedding, n_trials=20, sigma=0.01),
        "hg_graph_stability": hg_graph_stability(hg_adjacency, n_trials=20, sigma=0.02),
        "vc_layout_stability": vc_layout_stability(vc_features, n_trials=10, sigma=0.02),
    }
```

**Group 5: Task-Relevant (1.5 hours)**

```python
def extract_task_features(
    vs_embedding: np.ndarray,
    hg_adjacency: np.ndarray,
    vc_features: np.ndarray,
    corpus_mean: np.ndarray | None = None,
) -> dict[str, float]:
    """Extract Group 5 features.

    Note: vs_semantic_specificity requires corpus_mean (not per-episode computable).
    Pass None if corpus not available (feature will be NaN, flag for warning).
    """
    features = {
        "hg_relational_richness": hg_relational_richness(hg_adjacency),
        "vc_layout_prominence": vc_layout_prominence(vc_features),
    }

    if corpus_mean is not None:
        features["vs_semantic_specificity"] = vs_semantic_specificity(vs_embedding, corpus_mean)
    else:
        features["vs_semantic_specificity"] = np.nan  # Flag for warning

    return features
```

### Keep Helper Functions Organized

Place all helper functions in a "Helper Functions" section at the top (after imports, before group functions):

```python
# ---- Helper Functions (Group 1 & others use these) ----
EPS = 1e-12

def vs_spread(v: np.ndarray) -> float:
    """Participation ratio, rescaled to [0,1]."""
    # Copy from phase3_features.py, lines 46–56
    ...

def hg_spread(adj: np.ndarray) -> float:
    """Normalized Shannon entropy of weighted degree distribution."""
    # Copy from phase3_features.py, lines 89–101
    ...

# ... other helpers
```

### Total Implementation Time

- Group 1: 1 hour
- Group 2: 1 hour
- Group 3: 1.5 hours
- Group 4: 2 hours (stochastic; needs care with seed)
- Group 5: 1.5 hours
- Integration (main entry point + old features backward compat): 1.5 hours
- **Total: 8–10 hours**

---

## Step 3: Critical Fixes (2–3 hours)

**MUST implement all 6 before unit testing.** These prevent silent training failures.

### Fix 1: Input Validation (30 min)

**Add at start of extract_features_v2():**

```python
def extract_features_v2(
    unit: EpisodicUnit,
    corpus_mean: np.ndarray | None = None,
    entity_extractor: EntityExtractor | None = None,
) -> FeatureVector:
    """Extract 20-feature vector.

    CRITICAL: All inputs validated for NaN/inf before extraction.
    """
    # ---- INPUT VALIDATION ----
    if not hasattr(unit, 'embedding') or unit.embedding is None:
        raise ValueError("Episode unit missing VS embedding")

    if np.any(np.isnan(unit.embedding)) or np.any(np.isinf(unit.embedding)):
        raise ValueError(f"VS embedding contains NaN/inf: {unit.embedding}")

    # Similar checks for HG and VC if present
    if hasattr(unit, 'hyperedge_adjacency') and unit.hyperedge_adjacency is not None:
        if np.any(np.isnan(unit.hyperedge_adjacency)):
            raise ValueError("HG adjacency contains NaN")

    # ... proceed with extraction
```

### Fix 2: Zero-Vector Guard (30 min)

**Add to vs_embedding_stability():**

```python
def vs_embedding_stability(v: np.ndarray, n_trials: int = 20, sigma: float = 0.01) -> float:
    """1 - expected L2 distance from perturbed embedding.

    CRITICAL: Guard against zero-norm (cannot re-normalize).
    """
    norm = np.linalg.norm(v)
    if norm < EPS:
        # Cannot normalize zero-vector; return 0.0 (minimum stability)
        return 0.0

    v_normalized = v / norm
    distances = []
    for _ in range(n_trials):
        eps = np.random.randn(*v.shape) * sigma
        v_pert = v + eps
        v_pert_norm = v_pert / (np.linalg.norm(v_pert) + EPS)  # Extra EPS guard
        dist = np.linalg.norm(v_pert_norm - v_normalized)
        distances.append(dist)

    return float(1.0 - np.mean(distances) / 2.0)
```

### Fix 3: Output Clamping (30 min)

**Add before returning FeatureVector:**

```python
def extract_features_v2(...) -> FeatureVector:
    """Extract 20 features with full validation."""

    # [Extraction code for all 5 groups]

    # ---- OUTPUT CLAMPING ----
    # Clamp 12 features to [0, 1]
    bounded_features = [
        "vs_energy_concentration", "hg_hierarchy_depth", "vc_layout_concentration",
        "vs_hg_structural_agreement", "vs_vc_structural_agreement", "hg_vc_structural_agreement",
        "complementarity_score", "vs_embedding_stability", "hg_graph_stability",
        "vc_layout_stability", "vs_semantic_specificity", "hg_relational_richness",
        "vc_layout_prominence",
    ]

    for name in bounded_features:
        features[name] = np.clip(features[name], 0.0, 1.0)

    # Leave hg_cost_position and hg_information_per_token unbounded (observed range for
    # hg_cost_position now legitimately exceeds 1.0 under token accounting; do not clamp
    # to the old disk-KB-era range)

    # ---- FINAL VALIDATION ----
    feature_array = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
    if np.any(np.isnan(feature_array)):
        raise ValueError(f"Extracted features contain NaN: {feature_array}")

    return FeatureVector(values=features)
```

### Fix 4: Adjacency Validation (15 min)

**Add to any HG feature function:**

```python
def hg_cost_position(adj: np.ndarray) -> float:
    """Position of HG footprint between VS and VC."""
    # CRITICAL: Validate adjacency
    if np.any(adj < 0):
        raise ValueError(f"Adjacency contains negative weights: min={np.min(adj)}")
    if np.any(adj > 1.0):
        raise ValueError(f"Adjacency weights exceed 1.0: max={np.max(adj)}")

    # ... rest of function
```

### Fix 5: Epsilon Guards (15 min)

**Add to division-heavy functions:**

```python
def hg_cost_position(adj: np.ndarray) -> float:
    """..."""
    hg_tokens = n_nodes * HG_TOKENS_PER_NODE + n_edges * HG_TOKENS_PER_EDGE
    vs_tokens = 384.0
    vc_tokens = 1176.0

    denominator = vc_tokens - vs_tokens  # Could theoretically be zero or negative
    if denominator <= EPS:
        # Guard: if bookends are too close, return neutral value
        return 0.0

    return float((hg_tokens - vs_tokens) / denominator)
```

### Fix 6: Regression Test (15 min, automated in Unit Testing)

See Step 4 (Unit Testing) for MLP training convergence check.

---

## Step 4: Unit Testing (8–12 hours)

### Test File: edge_case_test.py

**Location**: `tests/test_edge_cases_features_v2.py` (or alongside features_v2.py)

**Structure** (62 test cases total):

```python
"""Unit tests for features_v2.py -- 62 edge case suite."""

import pytest
import numpy as np
from fluxmem.features_v2 import (
    extract_features_v2,
    extract_geometry_features,
    extract_efficiency_features,
    # ... etc
)

# ---- Group 1 Tests (6 cases) ----
class TestGeometryFeatures:
    def test_vs_energy_concentration_on_random_vector(self):
        """Sanity: feature runs and is in [0, 1]."""
        v = np.random.randn(384)
        v = v / np.linalg.norm(v)
        feat = extract_geometry_features(v, np.eye(5), np.random.randn(7, 7, 512))
        assert 0 <= feat["vs_energy_concentration"] <= 1

    def test_vs_energy_concentration_zero_vector(self):
        """Edge case: zero-norm embedding -> should guard."""
        v = np.zeros(384)
        feat = extract_geometry_features(v, np.eye(5), np.random.randn(7, 7, 512))
        # Should not crash; might return 0.0 or NaN (test for expected behavior)
        assert not np.isinf(feat["vs_energy_concentration"])

    def test_hg_hierarchy_depth_constant(self):
        """Expected: always 1/3 until real HG lands."""
        feat = extract_geometry_features(np.random.randn(384), np.eye(5), np.random.randn(7, 7, 512))
        assert feat["hg_hierarchy_depth"] == pytest.approx(1/3)

    def test_vc_layout_concentration_on_noise(self):
        """Sanity: runs on random spatial map."""
        vc = np.random.randn(7, 7, 512)
        feat = extract_geometry_features(np.random.randn(384), np.eye(5), vc)
        assert 0 <= feat["vc_layout_concentration"] <= 1

    def test_vc_layout_concentration_uniform_map(self):
        """Edge case: uniform spatial map -> low concentration (high entropy)."""
        vc = np.ones((7, 7, 512))
        feat = extract_geometry_features(np.random.randn(384), np.eye(5), vc)
        assert feat["vc_layout_concentration"] == pytest.approx(0.0, abs=0.01)

    def test_vc_layout_concentration_peaked_map(self):
        """Edge case: single cell dominates -> high concentration (low entropy)."""
        vc = np.zeros((7, 7, 512))
        vc[0, 0, :] = 1.0
        feat = extract_geometry_features(np.random.randn(384), np.eye(5), vc)
        assert feat["vc_layout_concentration"] > 0.9

# ---- Group 2 Tests (4 cases) ----
class TestEfficiencyFeatures:
    def test_hg_cost_position_single_node(self):
        """Edge case: single-node graph -> minimal footprint."""
        adj = np.array([[0.0]])
        feat = extract_efficiency_features(adj)
        # Cost should be very negative (cheaper than VS)
        assert feat["hg_cost_position"] < 0

    def test_hg_cost_position_large_dense_graph(self):
        """Edge case: many nodes, many edges -> expensive."""
        adj = np.ones((10, 10)) * 0.5
        np.fill_diagonal(adj, 0)
        feat = extract_efficiency_features(adj)
        # Cost should be high (approaching or exceeding VC cost)
        # Exact value depends on byte constants
        assert feat["hg_cost_position"] > 0

    def test_hg_information_per_token_zero_footprint(self):
        """Edge case: single node, no edges."""
        adj = np.array([[0.0]])
        feat = extract_efficiency_features(adj)
        # Zero richness (n_nodes < 2 guard) -> return 0.0
        assert feat["hg_information_per_token"] == 0.0

    def test_hg_information_per_token_dense_small_graph(self):
        """Sanity: small dense graph has high information per token."""
        adj = np.ones((3, 3)) * 0.8
        np.fill_diagonal(adj, 0)
        feat = extract_efficiency_features(adj)
        # Richness high, footprint small -> ratio should be moderate-to-high
        assert feat["hg_information_per_token"] > 0

# ---- Group 3 Tests (8 cases) ----
class TestComplementarityFeatures:
    def test_structural_agreement_perfect(self):
        """All encoders have same spread -> high agreement."""
        # Construct inputs with controlled spreads
        # (This requires creating special arrays with known spread values)
        # Placeholder test:
        feat = extract_overlap_features(
            np.random.randn(384), np.eye(5), np.random.randn(7, 7, 512)
        )
        assert 0 <= feat["vs_hg_structural_agreement"] <= 1
        assert 0 <= feat["complementarity_score"] <= 1

    # ... 7 more cases covering corner cases, saturations, NaN propagation

# ---- Group 4 Tests (12 cases) ----
class TestRobustnessFeatures:
    def test_vs_stability_deterministic_seed(self):
        """Stability is deterministic given seed."""
        v = np.random.randn(384)
        v = v / np.linalg.norm(v)
        feat1 = extract_robustness_features(v, np.eye(5), np.random.randn(7, 7, 512), seed=42)
        feat2 = extract_robustness_features(v, np.eye(5), np.random.randn(7, 7, 512), seed=42)
        assert feat1["vs_embedding_stability"] == feat2["vs_embedding_stability"]

    def test_hg_stability_metric_choice(self):
        """Graph stability uses weight-sensitive metric (not count-based)."""
        # Count-based saturates at 1.0; weight-sensitive should show variance
        adj_sparse = np.eye(10) * 0.1
        adj_dense = np.ones((10, 10)) * 0.1
        np.fill_diagonal(adj_dense, 0)

        feat_sparse = extract_robustness_features(np.random.randn(384), adj_sparse, np.random.randn(7, 7, 512))
        feat_dense = extract_robustness_features(np.random.randn(384), adj_dense, np.random.randn(7, 7, 512))

        # Stability should differ (if using weight-sensitive metric)
        # Exact values depend on specific graphs
        assert feat_sparse["hg_graph_stability"] != feat_dense["hg_graph_stability"]

    # ... 10 more cases covering edge cases, zero-vector guards, etc.

# ---- Group 5 Tests (8 cases) ----
class TestTaskRelevantFeatures:
    def test_vs_semantic_specificity_requires_corpus_mean(self):
        """Specificity needs corpus mean; None raises or returns NaN."""
        v = np.random.randn(384)
        feat = extract_task_features(v, np.eye(5), np.random.randn(7, 7, 512), corpus_mean=None)
        # Should handle gracefully (NaN or warning, not crash)
        assert np.isnan(feat["vs_semantic_specificity"]) or feat["vs_semantic_specificity"] == 0.0

    def test_vs_semantic_specificity_with_corpus_mean(self):
        """Specificity computes correctly with corpus mean."""
        v = np.random.randn(384)
        v = v / np.linalg.norm(v)
        corpus_mean = np.ones(384) / np.sqrt(384)  # Arbitrary mean
        feat = extract_task_features(v, np.eye(5), np.random.randn(7, 7, 512), corpus_mean=corpus_mean)
        assert 0 <= feat["vs_semantic_specificity"] <= 1

    # ... 6 more cases covering HG richness, VC prominence edge cases

# ---- Regression Tests (12 cases) ----
class TestNaNPropagation:
    def test_nan_in_vs_propagates_and_is_caught(self):
        """NaN in input -> caught by validation."""
        v_nan = np.random.randn(384)
        v_nan[0] = np.nan
        with pytest.raises(ValueError):
            extract_features_v2(MockUnit(embedding=v_nan, adj=np.eye(5), vc=np.random.randn(7,7,512)))

    def test_output_features_no_nan(self):
        """All extracted features are finite."""
        feat = extract_features_v2(MockUnit(
            embedding=np.random.randn(384),
            adj=np.eye(5),
            vc=np.random.randn(7,7,512)
        ))
        for name, val in feat.values.items():
            assert not np.isnan(val), f"Feature {name} is NaN"
            assert not np.isinf(val), f"Feature {name} is inf"

    # ... 10 more regression tests
```

### Run Tests

```bash
pytest tests/test_edge_cases_features_v2.py -v

# Expected: All 62 tests pass
# ============== 62 passed in 45s ==============
```

### Coverage Check

```bash
pytest tests/test_edge_cases_features_v2.py --cov=fluxmem.features_v2 --cov-report=term-missing

# Target: >95% coverage
```

---

## Step 5: Integration with fluxmem/selector.py (1 hour)

### Update Import

**File**: `fluxmem/selector.py` (around line 10–20)

**Change**:

```python
# OLD:
from fluxmem.features import extract_features, FEATURE_DIM

# NEW:
from fluxmem.features_v2 import extract_features_v2, FEATURE_DIM
```

### Update Supervision Loop

**File**: `fluxmem/supervision.py` or wherever features are extracted during training

**Change**:

```python
# OLD:
features = extract_features(unit)

# NEW:
features = extract_features_v2(unit, corpus_mean=train_corpus_mean)
```

Note: If `vs_semantic_specificity` is not needed, pass `corpus_mean=None` (feature will be NaN; handle with warning or default value).

### Retrain MLP from Scratch

**Critical**: Old pickled models are invalid (feature dimension changed: 7 → 20).

```python
# Delete old pickles
rm -f ~/.cache/caimms/selector_*.pkl

# Retrain
python fluxmem/train_selector.py \
    --output-model ./selector_v2.pkl \
    --data-path ./episodes_labeled.jsonl \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001
```

### Backward Compatibility

If you want to keep old features available (e.g., for ablation studies):

```python
# In features_v2.py:
# Keep old 7-feature extraction
from fluxmem.features import extract_features  # Old function

# New version calls both:
def extract_features_v2_extended(unit, corpus_mean=None):
    old_features = extract_features(unit)  # 7 features
    new_features = extract_features_v2(unit, corpus_mean)  # 15 features
    combined = {**old_features, **new_features}
    return combined
```

But this is optional; if replacing entirely, just use features_v2.

---

## Step 6: Validation (4–6 hours)

### Sanity Checks

**Run on sample data:**

```bash
cd /home/durgesh/aditya/C-AIMMS/docs/workflows/feature-engineering-selector

# 1. Verify outputs match phase3_features.py
python validate_features_v2.py  # Custom script comparing new vs. reference

# 2. Verify ranges are reasonable
python -c "from fluxmem.features_v2 import extract_features_v2; ..."  # Spot checks
```

### Integration Test

**Test MLP training with new features:**

```bash
python -c "
import numpy as np
from fluxmem.selector import FormatSelector
from fluxmem.features_v2 import extract_features_v2

# Create 100 synthetic episodes
episodes = [...]
labels = [...]  # Ground truth format (0=VS, 1=HG, 2=VC)

# Train new selector
selector = FormatSelector(feature_dim=20)
selector.fit(episodes, labels, epochs=50)

# Check: loss converges, no NaN
print('Training completed successfully. Final loss:', selector.loss_history[-1])
assert not np.any(np.isnan(selector.loss_history)), 'NaN in loss history'
"
```

### Regression Test

**Ensure old features still work (if keeping them):**

```bash
# Test old 7-feature selector still trains
python -c "
from fluxmem.selector import FormatSelector
from fluxmem.features import extract_features
# ... (same test as above, but with old features)
"
```

---

## Step 7: Documentation & Handoff (2 hours)

### Update CLAUDE.md (if needed)

If C-AIMMS's main CLAUDE.md mentions feature engineering, update the status:

```markdown
| `fluxmem/features_v2.py` | 15-feature selector (Phase 5, 2026-08-15) | Complete, tested |
```

### Create Implementation Notes

**File**: `docs/workflows/feature-engineering-selector/IMPLEMENTATION_NOTES.md`

```markdown
# Implementation Notes: features_v2.py

## What Changed

- 7 features → 20 features (5 old + 15 new)
- All 6 critical edge-case fixes applied
- 62 unit tests passing
- Fully backward compatible with selector.py training pipeline

## What to Monitor

- MLP training loss (should converge smoothly, no NaN spikes)
- Feature range distributions (z-score normalization should be meaningful)
- Selector's held-out test accuracy (should match or exceed baseline)

## Known Limitations

- `hg_hierarchy_depth` is constant (1/3) until HyperMem lands
- `vc_layout_*` features have zero variance on synthetic data (expected)
- Group 4 (Robustness) measures output-space, not input-perturbation sensitivity
- Group 3 (Complementarity) measures structural, not semantic agreement

## Future Work

See FEATURE_ENGINEERING_REPORT.md, Section "Remaining Work" for validation gates.
```

### Prepare for Phase 3+ Validation

Ensure you have clear notes for when HG/VC encoders arrive:

```markdown
# Re-Validation Checkpoints (Phase 3+)

1. **When HyperMem lands**:
   - Re-compute correlations: `hg_information_per_token` vs. `hg_relational_richness`
   - If r > 0.95, drop redundant one (reduces to 14 features)
   - Measure real rendered-into-context token cost; update token constants (not byte constants -- memory budget is tokens, see DESIGN_RATIONALE.md#memory-budget--tokens-not-disk-space)

2. **When MemOCR lands**:
   - Re-validate `vc_layout_*` features on real renders
   - Check if `vs_vc_structural_agreement` decorrelates from `complementarity_score`
   - Measure real rendering overhead

3. **Full Paraphrase Robustness Testing** (once live encoders exist):
   - Replace Group 4 output-space metrics with true input-perturbation tests
```

---

## Deployment Checklist

Use this checklist to verify Phase 5 is complete:

### Implementation ✅

- [ ] features_v2.py created with all 15 features implemented
- [ ] All 5 group functions written (extract_geometry, extract_efficiency, extract_overlap, extract_robustness, extract_task)
- [ ] Helper functions (vs_spread, hg_spread, vc_spread, etc.) copied from phase3_features.py
- [ ] FEATURE_NAMES and FEATURE_DIM updated (20 features)
- [ ] Code reviewed for clarity and correctness

### Critical Fixes ✅

- [ ] Fix 1: Input validation (reject NaN/inf at boundary)
- [ ] Fix 2: Zero-vector guard (vs_embedding_stability)
- [ ] Fix 3: Output clamping (clamp 12 features to [0, 1])
- [ ] Fix 4: Adjacency validation (reject negative weights)
- [ ] Fix 5: Epsilon guards (safe division)
- [ ] Fix 6: Regression test (no NaN loss cascade)

### Testing ✅

- [ ] phase3_features.py runs without error (reference implementation works)
- [ ] edge_case_test.py created with 62 test cases
- [ ] All 62 tests pass
- [ ] >95% code coverage achieved
- [ ] No NaN/inf in extracted features (regression test passes)
- [ ] MLP training converges without NaN loss

### Integration ✅

- [ ] Import updated in selector.py
- [ ] Supervision loop updated to use extract_features_v2
- [ ] FEATURE_DIM updated (7 → 20)
- [ ] Old selector models deleted (force retrain)
- [ ] MLP retraining completes successfully on real/synthetic data
- [ ] Backward compatibility maintained (or documented as breaking change)

### Documentation ✅

- [ ] FEATURE_ENGINEERING_REPORT.md complete
- [ ] FEATURE_REFERENCE.md complete (15 features documented)
- [ ] DESIGN_RATIONALE.md complete (all decisions explained)
- [ ] DEPLOYMENT_GUIDE.md complete (this file, you are here)
- [ ] IMPLEMENTATION_NOTES.md created (monitoring, known issues)
- [ ] Code comments added (why each fix is needed)
- [ ] CLAUDE.md updated (if applicable)

### Handoff ✅

- [ ] All files committed to git
- [ ] PR created and reviewed
- [ ] Phase 5 sign-off checklist complete
- [ ] Ready for Phase 3+ encoder integration

---

## Troubleshooting

### Issue: "AttributeError: 'EpisodicUnit' has no attribute 'hyperedge_adjacency'"

**Cause**: EpisodicUnit interface doesn't define the attribute name expected by features_v2.

**Fix**: Check the interface in `fluxmem/interfaces.py`. Update feature extraction to use the correct attribute names (e.g., `unit.hg_adjacency` instead of `unit.hyperedge_adjacency`).

### Issue: "All extracted feature values are NaN"

**Cause**: Usually input validation failed silently, or an edge case guard is missing.

**Fix**:

1. Check input validation passes (add debug print statements).
2. Verify zero-vector guard in vs_embedding_stability.
3. Check epsilon guards in division-heavy functions.

### Issue: "MLP training loss becomes NaN at epoch 5"

**Cause**: Feature values contain NaN/inf that propagate through backprop.

**Fix**:

1. Run edge_case_test.py to isolate which feature is broken.
2. Add print statements in extract_features_v2 to see which feature fails.
3. Most likely: missing epsilon guard or output clamping.

### Issue: "New features seem to have no discriminative power (selector accuracy unchanged)"

**Cause**: This is expected on synthetic data! Sample data is Gaussian noise, not real encoder outputs.

**Expected behavior**: On synthetic data, features exercise formulas correctly but don't encode real discriminative signal. Once real HG/VC encoders land (Phase 3+), features will activate and show real variance.

**What to do**:

1. Verify formulas are correct (run edge_case_test.py).
2. Proceed to Phase 3 (HyperMem/MemOCR integration).
3. Re-validate features on real data once encoders exist.

---

## Timeline & Resource Estimate

| Phase               | Task                             | Effort        | Effort Type    | Owner                |
| ------------------- | -------------------------------- | ------------- | -------------- | -------------------- |
| **1 (Setup)**       | Environment, sample data         | 30 min        | Independent    | Implementer          |
| **2 (Coding)**      | Implement 15 features + 5 groups | 8–10 hrs      | Deep focus     | Implementer          |
| **3 (Fixes)**       | Apply 6 critical fixes           | 2–3 hrs       | Careful/detail | Implementer          |
| **4 (Testing)**     | Write & run 62 edge case tests   | 8–12 hrs      | QA + debug     | QA + Implementer     |
| **5 (Integration)** | Update selector.py, retrain      | 1–2 hrs       | Integration    | ML Eng + Implementer |
| **6 (Validation)**  | Sanity checks, regression tests  | 4–6 hrs       | Verification   | QA + ML Eng          |
| **7 (Handoff)**     | Documentation, sign-off          | 2 hrs         | Documentation  | Implementer          |
| **TOTAL**           |                                  | **26–35 hrs** | ~3.5 days      | Team                 |

**Calendar estimate**: 2–3 weeks (accounting for code review rounds, debugging, and integration with other Phase 5 tasks).

---

## Success Criteria (Final Check)

**Phase 5 is complete when**:

✅ All 15 features implemented and tested  
✅ All 6 critical edge-case fixes verified  
✅ 62 edge case tests passing (100%)  
✅ MLP training converges cleanly (no NaN loss)  
✅ Backward compatible with fluxmem/selector.py  
✅ Documentation complete and published  
✅ Ready for Phase 3+ HG/VC encoder integration

---

**Prepared by**: AI technical writer  
**Date**: 2026-08-15  
**Status**: Ready for Phase 5 implementation  
**Questions?** See FEATURE_ENGINEERING_REPORT.md or FEATURE_REFERENCE.md
