# Feature Reference Guide

**Quick lookup for each of the 15 selector features.** Each feature is self-contained for easy implementation and review.

---

## Group 1: Geometry & Scale

### Feature 1: vs_energy_concentration

**Definition**: Participation ratio of squared embedding components, rescaled to [0, 1].

**Formula**:

```
PR = (Σ v_i²)² / (D · Σ v_i⁴)
feature = (PR - 1/D) / (1 - 1/D)
where D = 384 (embedding dimensionality)
```

**Expected Range**: (0, 1] on real sentence embeddings; observed [0.24, 0.35] on sample data.

**Testability**: ✅ Testable now. Formula is mechanically sound on any 384-d vector. Sample data is synthetic Gaussian, not real sentence-transformers output, so variance is from random geometry, not semantic patterns.

**Why It Matters**:

- High = embedding energy is distributed across many dimensions (isotropic). Good for nearest-neighbor search generalization.
- Low = a few dimensions dominate (peaky/concentrated). Possibly overfit to one lexical pattern; nearest-neighbor search may fail to generalize.

**Edge Cases**:

- Zero-vector input: \|\|v\|\| < 1e-8 → return 0.0 (guard implemented in Phase 5)
- Very small vectors: add ε=1e-12 to denominator to prevent division by zero

**Implementation** (reference: phase3_features.py, lines 46-56):

```python
def vs_energy_concentration(v: np.ndarray) -> float:
    d = v.shape[0]  # 384
    num = (np.sum(v**2)) ** 2
    den = d * np.sum(v**4) + EPS
    pr = num / den
    return float((pr - 1 / d) / (1 - 1 / d))
```

**Usage in MLP**: Group 1 geometry signal; input to standard scaler normalization.

**References**: Phase 3 Output, Group 1.

---

### Feature 2: hg_hierarchy_depth

**Definition**: Number of distinct abstraction levels present in the hypergraph, divided by 3 (the intended maximum).

**Formula**:

```
feature = (distinct_layers_present) / 3
```

**Expected Range**: Fixed at 1/3 = 0.3333 (constant) until real HyperMem output exists.

**Testability**: ❌ Untestable now. Requires typed-layer field on each node (Facts/Episodes/Topics). Current stub has flat, untyped node set.

**Why It Matters**:

- Measures whether HG output includes the full three-layer hierarchy (Facts, Episodes, Topics).
- High = hierarchical richness; low = flat structure.
- Essential for assessing HG's intended reasoning capability (coarse-to-fine navigation).

**Edge Cases**:

- No layers present (flat graph): value = 0.0
- Partial hierarchy (e.g., only Facts and Episodes): value = 2/3
- Full 3-layer hierarchy: value = 1.0

**Implementation** (reference: phase3_features.py, lines 70-77):

```python
def hg_hierarchy_depth(n_layers_present: int = 1, max_layers: int = 3) -> float:
    """n_layers_present defaults to 1 (flat, single layer) for current stub."""
    return float(n_layers_present / max_layers)
```

**Update Path**: When HyperMem lands, implement layer-counting from node type field: `{node.layer for node in graph.nodes}`.

**Usage in MLP**: Group 1 geometry signal; constant now, signal when HG implemented.

**References**: Phase 2 Output, Section B (HG intended design).

---

### Feature 3: vc_layout_concentration

**Definition**: 1 minus the normalized Shannon entropy of the spatial activation map. High concentration (low entropy) = information concentrated in few regions.

**Formula**:

```
spatial_map = vc_features.mean(axis=-1).reshape(-1)  # shape (7,7,512) -> (49,)
entropy = -Σ (p_i * log(p_i)) where p = spatial_map / sum(spatial_map)
feature = 1 - (entropy / log(49))
```

**Expected Range**: [0, 1]; observed [0.0001, 0.0003] on sample data (high entropy/low concentration).

**Testability**: ⚠️ Partial. Formula runs on any (7, 7, 512) tensor. Sample data is noise, not real MemOCR renders, so concentration is near-maximum-entropy (opposite of intended behavior).

**Why It Matters**:

- MemOCR's design principle: crucial evidence occupies high-visibility regions (concentrated information).
- High concentration = layout is doing its job (evidence clearly marked as important).
- Low concentration = information spread uniformly (no visual priority guidance).

**Edge Cases**:

- Uniform spatial map (all cells equal): entropy = log(49), feature = 0.0 (maximum spread).
- Single cell dominates: entropy ≈ 0, feature ≈ 1.0 (maximum concentration).
- All-zero spatial map: handle via clip(spatial_map, 0, None) and total = sum(); if total ≤ ε, return 1.0 (treat as no information).

**Implementation** (reference: phase3_features.py, lines 104-125):

```python
def vc_layout_concentration(feat: np.ndarray) -> float:
    """feat: shape (7, 7, 512), float32"""
    spatial_map = feat.mean(axis=-1).reshape(-1)
    spatial_map = np.clip(spatial_map, 0, None)
    total = spatial_map.sum()
    if total <= EPS:
        return 1.0  # No information = maximally "concentrated" (vacuously)
    p = spatial_map / total
    p_safe = p[p > 0]
    h = -np.sum(p_safe * np.log(p_safe))
    return float(1.0 - h / np.log(len(spatial_map)))
```

**Usage in MLP**: Group 1 geometry signal; constant on current noise, signal when MemOCR lands.

**References**: Phase 2 Output, Section C (VC correct design); MemOCR paper, Eq. 6, Sec. 3.2.

---

## Group 2: Efficiency & Memory Trade-off

### Feature 4: hg_cost_position

**Definition**: Position of HG's per-episode footprint on a scale between VS (cheap) and VC (expensive).

**Formula**:

```
hg_footprint_kb = (n_nodes * 150 + n_edges * 80) / 1024
vs_footprint_kb = 384 * 4 / 1024 = 1.5
vc_footprint_kb = 7 * 7 * 512 * 4 / 1024 = 98.0
feature = (hg_footprint_kb - vs_footprint_kb) / (vc_footprint_kb - vs_footprint_kb)
```

**Expected Range**: Observed [−0.0039, 0.0465]; unbounded above in principle (if HG grows very large).

**Testability**: ⚠️ Partial. Formula runs; VS and VC footprints are architecture-fixed (real). HG footprint uses placeholder constants (150 bytes/node, 80 bytes/edge from estimated JSON serialization). Measure real values once HyperMem lands.

**Why It Matters**:

- Directly measures the denominator of the reward function (`task_accuracy / memory_budget`).
- 0 = HG is as cheap as VS (best efficiency).
- 1 = HG is as expensive as VC (worst efficiency).
- Selector uses this to balance accuracy gains against memory cost.

**Edge Cases**:

- HG footprint equals VS footprint: feature = 0.0.
- HG footprint very large (large graph): feature → ∞ (unbounded above).
- Zero-node graph (n_nodes = 0): footprint ≈ 0, feature → negative (cheap).

**Implementation** (reference: phase3_features.py, lines 139-149):

```python
def hg_cost_position(adj: np.ndarray) -> float:
    """adj: shape (N, N), weighted adjacency matrix"""
    n_nodes = adj.shape[0]
    n_edges = np.count_nonzero(adj)
    hg_kb = (n_nodes * HG_BYTES_PER_NODE + n_edges * HG_BYTES_PER_EDGE) / 1024.0
    vs_kb = 1.5
    vc_kb = 98.0
    if vc_kb - vs_kb <= EPS:
        return 0.0  # Guard: denominator too small
    return float((hg_kb - vs_kb) / (vc_kb - vs_kb))
```

**Update Path**: Replace `HG_BYTES_PER_NODE = 150` and `HG_BYTES_PER_EDGE = 80` with measured constants once real HyperMem output exists.

**Usage in MLP**: Group 2 efficiency signal; critical for reward-function alignment.

**References**: Phase 3 Output, Group 2; ADR 0002 (reward function).

---

### Feature 5: hg_information_per_kb

**Definition**: Confidence-weighted relational density (richness) divided by HG's memory footprint. Proxy for information-per-byte efficiency.

**Formula**:

```
relational_richness = Σ(edge_weights) / (N * (N-1))
hg_footprint_kb = (n_nodes * 150 + n_edges * 80) / 1024
feature = relational_richness / hg_footprint_kb
```

**Expected Range**: [0.01, 0.21] observed; unbounded above in principle.

**Testability**: ⚠️ Partial. Formula is mechanically sound. Numerator (richness) is real on mock generator (which uses realistic ~10% edge sparsity). Denominator (footprint) is a placeholder estimate.

**Why It Matters**:

- Information-per-byte: high = HG packs a lot of relational information into small footprint (efficient).
- Low = HG is footprint-heavy relative to its relational content (wasteful).
- Does NOT require ground-truth accuracy (unlike naive `accuracy/footprint` which is target leakage).
- Complements `hg_cost_position` (absolute cost) with a density signal (value per unit cost).

**Edge Cases**:

- Zero-footprint graph (single node, no edges): divide by near-zero → large value.
- No edges (all zeros): richness = 0, feature = 0 (no relational information).
- Very dense graph: richness → 1, small footprint → large feature.

**Implementation** (reference: phase3_features.py, lines 139-162, adapted):

```python
def hg_information_per_kb(adj: np.ndarray) -> float:
    """adj: shape (N, N), weighted adjacency matrix"""
    n_nodes = adj.shape[0]
    if n_nodes < 2:
        return 0.0

    # Relational richness
    richness = np.sum(np.abs(adj)) / (n_nodes * (n_nodes - 1))

    # Footprint
    n_edges = np.count_nonzero(adj)
    hg_kb = (n_nodes * HG_BYTES_PER_NODE + n_edges * HG_BYTES_PER_EDGE) / 1024.0

    if hg_kb <= EPS:
        return 0.0  # No footprint = no information
    return float(richness / hg_kb)
```

**Correlation Note**: r = 0.967 with `hg_relational_richness` on current sample (mechanically explained: edge count ∝ node count on mock generator). Likely to persist on real data, but re-validate once HyperMem lands.

**Usage in MLP**: Group 2 efficiency signal; value-per-byte proxy.

**References**: Phase 3 Output, Group 2; Phase 4 Validation, Section 4.2.

---

## Group 3: Complementarity & Overlap

### Feature 6: vs_hg_structural_agreement

**Definition**: 1 minus the absolute difference in information spread (entropy-based concentration) between VS and HG encoders.

**Formula**:

```
vs_spread = participation_ratio(embedding)  # [0, 1]
hg_spread = normalized_entropy(degree_distribution)  # [0, 1]
feature = 1 - |vs_spread - hg_spread|
```

**Expected Range**: [0, 1]; observed [0.31, 0.40].

**Testability**: ❌ Untestable for semantic agreement. Formula runs; result measures structural agreement (do both encoders show similar concentration?), not semantic agreement (do they agree on _what_ is important?).

**Why It Matters**:

- High = both encoders concentrate information similarly.
  - Redundant signal: if one encoder compresses, so does the other.
  - Selector may prefer one for other reasons (cost, robustness).
- Low = encoders distribute information differently.
  - Complementary signal: one may capture what the other misses.
  - Selector benefits from having both available.

**Limitation**: Measures structural overlap (shape), not semantic overlap (content). True semantic agreement requires:

1. Extracting episode summaries from HG nodes.
2. Embedding those summaries with the same VS encoder.
3. Computing cosine similarity against the original VS embedding.
   This is only feasible once HyperMem is implemented and producing real node summaries.

**Edge Cases**:

- Both spread values identical: feature = 1.0 (perfect agreement).
- One spread = 0, other = 1: feature = 0.0 (no agreement).

**Implementation** (reference: phase3_features.py, adapted):

```python
def vs_hg_structural_agreement(vs_spread: float, hg_spread: float) -> float:
    """vs_spread: participation ratio; hg_spread: normalized entropy"""
    return float(1.0 - abs(vs_spread - hg_spread))
```

**Usage in MLP**: Group 3 complementarity signal; input to `complementarity_score` aggregate.

**References**: Phase 3 Output, Group 3; Phase 4 Validation, Section 4.4 (limitations).

---

### Feature 7: vs_vc_structural_agreement

**Definition**: 1 minus the absolute difference in information spread between VS and VC encoders.

**Formula**: Same as Feature 6, but comparing VS spread to VC spatial entropy.

**Expected Range**: [0, 1]; observed [0.24, 0.35].

**Testability**: ❌ Same limitation as Feature 6. Additionally, current VC sample data sits at near-maximum entropy (noise), causing r = −1.0 with `complementarity_score` (artifact of stub data, not structural).

**Why It Matters**: Same as Feature 6, for VS/VC pair.

**Edge Cases**: Same as Feature 6.

**Implementation**: Same pattern as Feature 6, with VC spread instead of HG spread.

**Correlation Note**: Currently perfectly correlated with `complementarity_score` due to stub data ceiling (expected to decorrelate once real rendered layouts with realistic concentration exist).

**Usage in MLP**: Group 3 complementarity signal; input to `complementarity_score` aggregate.

**References**: Phase 3 Output, Group 3; Phase 4 Validation, Section 4.5.

---

### Feature 8: hg_vc_structural_agreement

**Definition**: 1 minus the absolute difference in information spread between HG and VC encoders.

**Formula**: Same pattern as Features 6–7, comparing HG and VC spreads.

**Expected Range**: [0, 1]; observed [0.89, 0.97] (high agreement on current sample).

**Testability**: ❌ Same structural-only limitation as Features 6–7.

**Why It Matters**: Same as Feature 6, for HG/VC pair.

**Implementation**: Same pattern.

**High Observed Agreement Note**: r = 0.943 mean on sample. Both HG (degree-distribution entropy) and VC (spatial entropy) tend to sit at moderate-to-high entropy on this generator → high agreement value. Expected to show more variance on real data.

**Usage in MLP**: Group 3 complementarity signal; input to `complementarity_score` aggregate.

**References**: Phase 3 Output, Group 3.

---

### Feature 9: complementarity_score

**Definition**: Aggregate of pairwise structural agreement: measures overall encoder diversity.

**Formula**:

```
feature = 1 - mean(vs_hg_structural_agreement, vs_vc_structural_agreement, hg_vc_structural_agreement)
```

**Expected Range**: [0, 1]; observed [0.43, 0.51].

**Testability**: ❌ Untestable for true complementarity. Measures structural diversity (do encoders have different concentration patterns?), not semantic diversity (do they encode different information?).

**Why It Matters**:

- High complementarity = three encoders encode information very differently.
  - Selector's choice of which encoder to use has a big impact.
  - High ROI for having three options.
- Low complementarity = encoders show similar structure.
  - Selector's choice matters less; all three may be roughly equivalent.
  - Consider consolidating to fewer encoders.

**Edge Cases**:

- All three agreement values = 1 (perfect agreement): feature = 0 (no complementarity).
- All three agreement values = 0 (perfect disagreement): feature = 1 (maximum complementarity).

**Implementation**:

```python
def complementarity_score(
    vs_hg_agree: float,
    vs_vc_agree: float,
    hg_vc_agree: float
) -> float:
    return 1.0 - np.mean([vs_hg_agree, vs_vc_agree, hg_vc_agree])
```

**Usage in MLP**: Group 3 signal; high-level encoder diversity metric.

**References**: Phase 3 Output, Group 3; Task brief (explicitly requested).

---

## Group 4: Robustness & Stability

### Feature 10: vs_embedding_stability

**Definition**: 1 minus expected L2 distance from perturbed embedding (after re-normalization), averaged over trials.

**Formula**:

```
For 20 trials:
  1. Perturb v: v' = v + ε, ε ~ N(0, 0.01²I)
  2. Normalize: v'_norm = v' / ||v'||
  3. Distance: d = ||v'_norm - v|| (both unit norm)
feature = 1 - mean(d over 20 trials)
```

**Expected Range**: [0, 1]; observed [0.9026, 0.9074] (high stability).

**Testability**: ⚠️ Partial. Measures output-space sensitivity (local Lipschitz stability), not input robustness (paraphrase invariance). True robustness = perturb the episode text, re-run SentenceTransformer, measure shift. Not feasible on static `.npy` arrays.

**Why It Matters**:

- High stability = embedding is robust to small perturbations.
  - Good for nearest-neighbor search (small changes in input → small changes in output).
- Low stability = embedding is sensitive to perturbations.
  - Risky for retrieval (similar episodes may have dissimilar embeddings).

**Edge Cases**:

- Zero-norm embedding: ||v|| < 1e-8 → return 0.0 (cannot re-normalize).
- Very small embedding: after perturbation, direction may flip significantly (captured as low stability).

**Implementation** (reference: phase3_features.py, adapted):

```python
def vs_embedding_stability(v: np.ndarray, n_trials: int = 20, sigma: float = 0.01) -> float:
    """v: shape (384,), float32"""
    if np.linalg.norm(v) < EPS:
        return 0.0  # Guard: zero-vector

    distances = []
    for trial in range(n_trials):
        eps = np.random.randn(*v.shape) * sigma
        v_perturbed = v + eps
        v_norm_target = v / np.linalg.norm(v)
        v_perturbed_norm = v_perturbed / (np.linalg.norm(v_perturbed) + EPS)
        dist = np.linalg.norm(v_perturbed_norm - v_norm_target)
        distances.append(dist)

    return float(1.0 - np.mean(distances) / 2.0)  # Normalize by max possible L2 distance (2)
```

**Limitation**: σ=0.01 is small relative to typical embedding norms (~1.0 for unit vectors). Captures very-local sensitivity only. For robustness assessment, also pair with corpus-wide evaluations (does same query text produce same top-k neighbors across different paraphrases?).

**Usage in MLP**: Group 4 robustness signal; confidence in VS output stability.

**References**: Phase 3 Output, Group 4 preamble (limitations); Phase 4b Edge Cases.

---

### Feature 11: hg_graph_stability

**Definition**: 1 minus expected change in relational richness under edge-weight perturbation, averaged over trials.

**Formula**:

```
For 20 trials:
  1. Perturb adjacency: A' = clip(A + ε, 0, 1), ε ~ N(0, 0.02²)
  2. Re-compute relational richness: r' = sum(A') / (N(N-1))
  3. Change: |r' - r|
feature = 1 - mean(|Δr| over 20 trials)
```

**Expected Range**: [0, 1]; observed [0.9975, 0.9998] (very high stability).

**Testability**: ⚠️ Partial. Same output-space sensitivity limitation as Feature 10. Uses weight-sensitive `hg_relational_richness`, not count-based density (which would saturate at 1.0 due to rarely flipping edges).

**Why It Matters**:

- High stability = HG's relational signal is robust to small edge-weight changes.
  - Reliable reasoning (small noise in extraction doesn't flip conclusions).
- Low stability = HG's relational signal is brittle.
  - Risky to rely on HG for multi-hop reasoning if small extraction errors flip edge weights significantly.

**Edge Cases**:

- Negative edge weights (error condition): reject at boundary (validate adjacency >= 0).
- Very small graphs (few nodes): perturbation noise dominates → lower stability (expected).
- Very dense graphs (many edges): perturbations average out → high stability (expected).

**Implementation** (reference: phase3_features.py, adapted):

```python
def hg_graph_stability(adj: np.ndarray, n_trials: int = 20, sigma: float = 0.02) -> float:
    """adj: shape (N, N), weighted adjacency [0, 1]"""
    n_nodes = adj.shape[0]
    if n_nodes < 2:
        return 0.0

    richness_original = np.sum(adj) / (n_nodes * (n_nodes - 1))

    deltas = []
    for trial in range(n_trials):
        eps = np.random.randn(*adj.shape) * sigma
        adj_perturbed = np.clip(adj + eps, 0, 1)
        richness_perturbed = np.sum(adj_perturbed) / (n_nodes * (n_nodes - 1))
        delta = abs(richness_perturbed - richness_original)
        deltas.append(delta)

    return float(1.0 - np.mean(deltas))
```

**Metric Choice Note**: Evaluated count-based density first; it saturated at 1.0 (perturbations never flip edges across zero threshold at σ=0.02). Switched to weight-sensitive richness, which shows real variance (see Phase 4, Section 4.1).

**Usage in MLP**: Group 4 robustness signal; confidence in HG stability.

**References**: Phase 3 Output, Group 4; Phase 4 Validation, Section 4.1 (metric choice bug).

---

### Feature 12: vc_layout_stability

**Definition**: 1 minus expected change in layout concentration under spatial-map perturbation, averaged over trials.

**Formula**:

```
For 10 trials:
  1. Perturb spatial map: S' = S + ε, ε ~ N(0, 0.02²)
  2. Clamp: S' = clip(S', 0, ∞)
  3. Re-compute layout concentration
  4. Change: |conc' - conc|
feature = 1 - mean(|Δconc| over 10 trials)
```

**Expected Range**: [0, 1]; observed [1.0, 1.0] (saturated at ceiling).

**Testability**: ❌ Untestable now. Current VC sample data is noise sitting at near-maximum entropy (ceiling). Perturbations can't push it higher → change is always zero → stability = 1.0 (constant).

**Why It Matters**:

- High stability = layout's concentration property is robust to noise in rendering.
  - Reliable spatial reasoning (small rendering errors don't destroy layout structure).
- Low stability = concentration is fragile.
  - Risky to rely on VC if rendering noise significantly flattens or spikes the importance distribution.

**Edge Cases**:

- All-zero spatial map: entropy = 0 (undefined with log); clip to treat as max-entropy → feature = 1.0.
- Maximum-entropy (uniform) map: concentration = 0; perturbations stay at ceiling → stability = 1.0 (expected on current noise, bug on real concentrated layouts).

**Implementation**:

```python
def vc_layout_stability(feat: np.ndarray, n_trials: int = 10, sigma: float = 0.02) -> float:
    """feat: shape (7, 7, 512)"""
    conc_original = vc_layout_concentration(feat)

    deltas = []
    for trial in range(n_trials):
        eps = np.random.randn(*feat.shape) * sigma
        feat_perturbed = np.clip(feat + eps, 0, None)
        conc_perturbed = vc_layout_concentration(feat_perturbed)
        delta = abs(conc_perturbed - conc_original)
        deltas.append(delta)

    return float(1.0 - np.mean(deltas))
```

**Data Artifact Note**: Saturation at 1.0 is a symptom of stub data (high-entropy noise), not a bug in the formula. Expect real variance once MemOCR lands with concentrated renders (where perturbations can push concentration down toward entropy ceiling).

**Usage in MLP**: Group 4 robustness signal; currently constant, signal when VC lands.

**References**: Phase 3 Output, Group 4; Phase 4 Validation, Section 4.1 (saturation).

---

## Group 5: Task-Relevant / Domain Fit

### Feature 13: vs_semantic_specificity

**Definition**: 1 minus cosine similarity to the batch-mean embedding (clipped at 0). Measures how distinctive an embedding is relative to the corpus.

**Formula**:

```
corpus_mean = mean(all_episode_embeddings_in_batch)  # shape (384,)
sim = cos(embedding, corpus_mean)  # [-1, 1]
feature = 1 - max(sim, 0)
```

**Expected Range**: [0, 1]; observed [0.66, 0.79].

**Testability**: ⚠️ Partial. Formula runs. Sample data is synthetic Gaussian, so "distinctiveness" is from random geometry, not semantic domain properties.

**Why It Matters**:

- High specificity = embedding is distinctive within the batch.
  - VS's nearest-neighbor search will discriminate this episode well.
  - Selector may prefer VS if this signal is high.
- Low specificity = embedding is generic/near-corpus-mean.
  - VS's retrieval will be noisy (many similar episodes in the batch).
  - Selector should prefer HG/VC's explicit structure, which may do better on generic content.

**Unique Aspect**: Only feature in the 15-set that requires corpus-level context (batch mean). All others are computable per-episode.

**Edge Cases**:

- Embedding identical to corpus mean: similarity = 1, feature = 0.0 (maximally generic).
- Embedding orthogonal to corpus mean: similarity = 0, feature = 1.0 (maximally specific).
- Negative similarity (opposite to corpus mean): clipped to 0, feature = 1.0.

**Implementation** (reference: phase3_features.py, lines 165-174):

```python
def vs_semantic_specificity(embedding: np.ndarray, corpus_mean: np.ndarray) -> float:
    """
    embedding: shape (384,), unit-norm
    corpus_mean: shape (384,), computed as mean of all training embeddings
    """
    # Cosine similarity
    sim = np.dot(embedding, corpus_mean) / (np.linalg.norm(corpus_mean) + EPS)
    return float(1.0 - np.clip(sim, 0, None))
```

**Integration Note**: This feature requires passing a `corpus_mean` into the feature extraction function. The mean should be computed once on training data and cached (or updated incrementally during inference). Do NOT compute a fresh mean per-episode.

**Usage in MLP**: Group 5 task-relevant signal; batch-context signal for VS.

**References**: Phase 3 Output, Group 5; Implementation note in Phase 3, Section 5.

---

### Feature 14: hg_relational_richness

**Definition**: Confidence-weighted relational density. Sum of all edge weights (treating weights as confidence scores) divided by possible edges.

**Formula**:

```
richness = sum(adjacency_weights) / (N * (N-1))
```

**Expected Range**: [0, 1]; observed [0.07, 0.25].

**Testability**: ⚠️ Partial. Formula is sound. Sample graph generator produces ~10% edge sparsity with variable weights, so richness variance is real on this mock data. True test requires real HyperMem extraction.

**Why It Matters**:

- High richness = HG's output is full of relations (high-confidence edges).
  - Strong signal for multi-hop reasoning (many paths to explore).
  - Selector may prefer HG if this signal is high.
- Low richness = HG is sparse (few relations extracted).
  - Weak relational signal; HG may underperform on entity-centric queries.
  - Selector should prefer VS/VC's other strengths.

**Edge Cases**:

- No edges (all zeros): richness = 0.0.
- Complete graph with maximum weights: richness = 1.0.
- Single node (N=1): richness = 0.0 (denominator = 0).

**Implementation** (reference: phase3_features.py, lines 167-173):

```python
def hg_relational_richness(adj: np.ndarray) -> float:
    """adj: shape (N, N), weighted adjacency [0, 1]"""
    n = adj.shape[0]
    if n < 2:
        return 0.0
    return float(np.sum(adj) / (n * (n - 1)))
```

**Correlation Note**: r = 0.967 with `hg_information_per_kb` on sample (edge count scales linearly with node count on generator). Likely structural, but re-validate on real HG data. If confirmed, drop one of the pair in production.

**Usage in MLP**: Group 5 task-relevant signal; HG reasoning-capability proxy.

**References**: Phase 3 Output, Group 5; Phase 4 Validation, Section 4.5.

---

### Feature 15: vc_layout_prominence

**Definition**: Peak-to-mean ratio of spatial activation map. Measures whether one region stands out as more important than others.

**Formula**:

```
spatial_map = vc_features.mean(axis=-1).reshape(-1)  # (49,)
feature = (max(spatial_map) - mean(spatial_map)) / (max(spatial_map) + mean(spatial_map))
```

**Expected Range**: [−1, 1]; observed [0.031, 0.057] (very low prominence).

**Testability**: ❌ Untestable now. Current VC sample is noise with uniform activation → low prominence (correct). Real rendered layouts should show high prominence (one region stands out).

**Why It Matters**:

- High prominence = one region stands out as most important (matches MemOCR design: crucial evidence gets prominent placement).
  - Layout is functioning as intended.
  - Spatial reasoning should be reliable (visual hierarchy is clear).
- Low prominence = all regions are equally important.
  - Layout is failing to guide attention (or the episode has truly uniform importance).
  - Selector may doubt VC's ability to guide reasoning.

**Edge Cases**:

- All equal activation: max = mean, feature = 0.0 (no prominence).
- One cell dominates: max >> mean, feature → 1.0 (maximum prominence).
- All-zero map: max = mean = 0, feature = 0/0 → NaN (edge case: guard with if total ≤ ε: return 0.0).

**Implementation** (reference: phase3_features.py, lines 167-169):

```python
def vc_layout_prominence(feat: np.ndarray) -> float:
    """feat: shape (7, 7, 512)"""
    spatial_map = feat.mean(axis=-1).reshape(-1)
    maximum = np.max(spatial_map)
    mean_val = np.mean(spatial_map)

    denominator = maximum + mean_val
    if denominator <= EPS:
        return 0.0  # All-zero map

    return float((maximum - mean_val) / denominator)
```

**Expected Behavior on Real Data**: Current observed [0.031, 0.057] indicates very low prominence (noise-like). Real MemOCR renders should show much higher values (e.g., [0.3, 0.8]), with high variance across episodes.

**Usage in MLP**: Group 5 task-relevant signal; VC layout-quality proxy.

**References**: Phase 3 Output, Group 5; MemOCR paper, Sec. 3.2 (layout-guided importance).

---

## Quick Reference: Testing & Implementation

### Edge Case Checklist (Phase 5)

- [ ] **NaN/inf input**: Reject in extract_features_v2() boundary
- [ ] **Zero-vector embedding**: Guard vs_embedding_stability with ||v|| < 1e-8
- [ ] **Single-node graph**: Handle N=1 case (denominator = 0)
- [ ] **Negative adjacency weights**: Validate (reject if min < 0)
- [ ] **All-zero spatial map**: Clip, return 0.0
- [ ] **Output clamping**: Clamp 12 [0,1] features after extraction
- [ ] **No NaN propagation**: Assert all feature values are finite before MLP

### Normalization

All 15 features will pass through `StandardScaler` (fit on training data only) in fluxmem/selector.py. The scaler handles:

- Bounded features [0, 1]: Z-score normalization removes bounds, but OK (MLP handles).
- Unbounded features (hg_cost_position, hg_information_per_kb): Z-score essential.

Do NOT double-normalize; StandardScaler is sufficient.

### Reference Implementation

Use `phase3_features.py` (in this directory) as the template for all 15 functions. It is the canonical reference; copy formulas directly for Phase 5 implementation.

---

**Last Updated**: 2026-08-15  
**Status**: Ready for Phase 5 implementation  
**Questions?** See FEATURE_ENGINEERING_REPORT.md (comprehensive) or DESIGN_RATIONALE.md (decision-focused).
