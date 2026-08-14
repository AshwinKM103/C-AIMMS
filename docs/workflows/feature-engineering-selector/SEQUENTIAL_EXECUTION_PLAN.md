# Sequential Extraction Plan: Feature Engineering Prompt

**Date**: 2026-08-14  
**Strategy**: Sequential (Phase 1 → 2 → 3 → 4 → 5)  
**Total Duration**: ~5 hours  
**Model Policy**: Haiku (default) → Sonnet (decision-making) → Opus (reserved for critical ties)

---

## Overview: 5-Phase Sequential Extraction

```
Phase 1: FluxMem Baseline       [HAIKU] ← foundation
   ↓ (handoff: 12 features + selector MLP spec)
Phase 2: Encoder Characterization [HAIKU/SONNET] ← specifications
   ↓ (handoff: VS/HG/VC output specs)
Phase 3: Feature Engineering     [SONNET] ⭐ CORE (decision-heavy)
   ↓ (handoff: M features + pseudocode)
Phase 4: Validation             [HAIKU] ← testing on real data
   ↓ (handoff: validated feature set)
Phase 5: Implementation         [SONNET] ← production-ready code
   ↓
Done: Feature set + code + docs
```

---

## PHASE 1: FluxMem Baseline Understanding

### Effort Level: MEDIUM (not complex, but thorough reading required)

### Input to Agent

```markdown
Task: Extract and document FluxMem's 12-feature selector baseline.

Read and document:

1. fluxmem/features.py (all 12 features)
   - Name, mathematical definition
   - What modality it measures (linear/graph/hierarchical)
   - Input type, output range, semantic meaning

2. fluxmem/selector.py (MLP implementation)
   - Architecture (layers, activations, shapes)
   - Feature preprocessing/normalization
   - Training signal and inference routing

3. FluxMem paper (if available in docs/ or references)
   - Why these 12 features were chosen
   - Assumptions about input/output space
   - Limitations mentioned

Output: Structured markdown with 12 features listed, MLP architecture documented,
rationale explained. Format as: Feature table with columns:
[Name | Definition | Modality | Input | Output | Why It Matters]
```

### Agents to Spawn (Sequential)

#### Agent 1.1: `academic-researcher` [HAIKU]

```
Model: Haiku
Task: Extract FluxMem paper + feature rationale
Input: selector_feature_engineering_prompt.md (Phase 1.1, 1.3)
Output: 12 features documented + paper rationale
Time: 30 min
```

#### Agent 1.2: `code-reviewer` [HAIKU]

```
Model: Haiku
Task: Review selector.py and features.py for implementation
Input: Output from Agent 1.1 + selector_feature_engineering_prompt.md (Phase 1.2)
Output: MLP architecture + preprocessing details documented
Time: 15 min
```

### Handoff to Phase 2

**Output artifact**: `phase1_output.md`

```markdown
## FluxMem 12 Features (Documented)

| Feature   | Definition | Modality | Input   | Output  | Purpose           |
| --------- | ---------- | -------- | ------- | ------- | ----------------- |
| feature_1 | [math]     | graph    | [shape] | [range] | Differentiates... |
| ...       | ...        | ...      | ...     | ...     | ...               |

## MLP Architecture

- Input: 12 features → normalized to [0, 1]
- Layers: [spec from selector.py]
- Output: 3-way softmax (VS, HG, VC)
- Training signal: [how it's supervised]
```

---

## PHASE 2: Encoder Output Characterization

### Effort Level: MEDIUM-HIGH (code exploration + interpretation)

### Input to Agents

```markdown
Context: You have FluxMem's 12 features (from Phase 1).
Now characterize the three encoders that will be INPUTS to the new selector MLP.

Task A (Explore agent): Locate + extract encoder output specs
Task B (ml-engineer agent): Interpret what each output MEANS semantically
```

### Agents to Spawn (Parallel after Phase 1)

#### Agent 2.1: `Explore` [HAIKU]

```
Model: Haiku
Task: Locate and extract encoder output specifications
Input: Codebase queries for:
  - hetrep/ (VS encoder output)
  - HyperMem/hypermem/structure.py + extractors/ (HG encoder output)
  - MemOCR/scripts/eval.sh + memory_img*.py (VC encoder output)
  - fluxmem/fusion.py, memocr_episodes.py, tests/test_selector.py

Deliverables:
  1. File locations for each encoder
  2. Example code showing output shapes/types
  3. Metadata available from each encoder (confidence, density, etc.)

Output format: File reference table with path → output_shape → output_type
```

#### Agent 2.2: `ml-engineer` [SONNET] ⭐ Decision-making

```
Model: Sonnet (requires semantic interpretation)
Task: Interpret encoder outputs from ML pipeline perspective
Input: Output from Agent 2.1 + selector_feature_engineering_prompt.md (Phase 2)

Questions to answer:
  - What does each encoder's output represent semantically?
  - What are the key properties of each representation?
  - What metadata is preserved that could drive selection?
  - Memory footprint of each encoding?

Output: Semantic characterization document with:
  VS: [semantic meaning, key properties, memory profile]
  HG: [semantic meaning, key properties, memory profile]
  VC: [semantic meaning, key properties, memory profile]
```

### Handoff to Phase 3

**Output artifact**: `phase2_output.md`

```markdown
## Encoder Specifications

### VS (Vector Semantic) - VectorDB

- **Output Shape**: (batch, 384) [sentence-transformers embedding]
- **Output Type**: float32, cosine-normalized
- **Memory**: 384 × 4 bytes = 1.5 KB per embedding
- **Key Properties**: Isotropy, alignment with language models
- **Metadata**: None (deterministic output)

### HG (Hierarchical Graph) - HyperMem

- **Output Shape**: (N, N) adjacency matrix + node_features (N, D)
- **Output Type**: float32 sparse graph
- **Memory**: O(edges) + O(N × D)
- **Key Properties**: Connectivity patterns, hierarchy depth, node count
- **Metadata**: Graph density, centrality measures available

### VC (Visual/Conceptual) - MemOCR

- **Output Shape**: (7, 7, 512) CNN features [from final-step rendering]
- **Output Type**: float32, spatial feature maps
- **Memory**: 7 × 7 × 512 × 4 ≈ 100 KB
- **Key Properties**: Spatial coherence, feature map activation
- **Metadata**: Confidence scores per region, rendering quality metrics

## Key Insight

Your selector must choose between these THREE fundamentally different representations.
The 12 FluxMem features (designed for linear/graph/hierarchical) may not capture
the right properties for VS/HG/VC selection.
```

---

## PHASE 3: Feature Engineering (CORE)

### Effort Level: HIGH ⭐ (Decision-making, creative synthesis, complex reasoning)

### Input to Agent

```markdown
Context:

- Phase 1 output: FluxMem 12 features (baseline to understand, not copy)
- Phase 2 output: VS/HG/VC output specifications (concrete shapes/types)
- Your reward function: selector optimizes task_accuracy / memory_budget
- CRITICAL: Bottom-up design. Don't inherit from ADRs. Trust the data.

Task: Design M ≤ 20 features for the selector MLP.

Features should measure:

1. GEOMETRY: Dimensionality, sparsity, scale, structure of each encoding
2. OVERLAP: Redundancy/complementarity between encoder pairs
3. EFFICIENCY: Compression ratio (task_accuracy / memory_footprint) for each
4. ROBUSTNESS: Noise sensitivity, uncertainty quantification
5. DOMAIN FIT: Task-relevant properties (graph quality, visual coverage, etc.)

All features must be:

- Modality-agnostic (computable for all three encoders)
- Normalized to [0, 1] or [-1, 1]
- Interpretable (clear meaning, unit, why it matters)

Deliverable:

- List of M features with name, definition, computation formula, expected range
- Pseudocode for feature extraction
- Justification for each feature (why include it?)
```

### Agents to Spawn (Sequential after Phase 2)

#### Agent 3.1: `data-scientist` [SONNET] ⭐ PRIMARY WORK

```
Model: Sonnet (decision-making, trade-off analysis, validation reasoning)
Task: Design feature set from encoder outputs (BOTTOM-UP, no ADR bias)
Input: phase1_output.md + phase2_output.md + selector_feature_engineering_prompt.md (Phase 3)
       + sample data (real or synthetic: HyperMem graph, MemOCR features, VectorDB embedding)

Critical context to include in prompt:
  "Selector reward = task_accuracy / memory_budget
   Design features that capture this trade-off, not just accuracy or efficiency alone.
   Include derived feature: estimated_performance / memory_cost"

Output:
  - M ≤ 20 features (target 12-16)
  - Each with: name | definition | formula | expected range | why it matters
  - Grouped by category: Geometry, Overlap, Efficiency, Robustness, Domain
  - Pseudocode for extraction
  - Validation on sample data (ranges, correlations, edge cases)

Time: 90 min (longest phase - this is the creative core)
```

#### Agent 3.2: `research-analyst` [HAIKU] (PARALLEL with 3.1)

```
Model: Haiku
Task: Literature review on feature selection for modality routing
Input: Feature engineering + attention mechanisms keywords

Questions:
  - How do other systems select between encodings (mixture-of-experts, dynamic routing)?
  - What features do they use to make routing decisions?
  - Best practices in feature engineering for multi-modal selection?

Output:
  - Related work summary (3-5 key papers)
  - Best practice patterns
  - Inform data-scientist's choices

Time: 30 min (runs in parallel, provides context)
```

### Handoff to Phase 4

**Output artifact**: `phase3_output.md`

```markdown
## Designed Feature Set (M ≤ 20 features)

### Geometry Features (5 features)

1. **embedding_dimensionality**: Size of representation space
   - Formula: len(encoder_output.shape)
   - Expected range: [10, 1000]
   - Why: Bigger ≠ better; trade-off between expressiveness and memory

2. **sparsity**: Fraction of near-zero values
   - Formula: sum(abs(output) < threshold) / output.size
   - Expected range: [0, 1]
   - Why: Sparse representations may be more efficient

3. [etc. 3 more geometry features]

### Overlap Features (3 features)

4. **vs_hg_mutual_information**: Redundancy between VS and HG
   - Formula: MI(embed_vs, embed_hg)
   - Expected range: [0, 1]
   - Why: Highly correlated encoders may be redundant

5. [etc. 2 more overlap features]

### Efficiency Features (4 features) ⭐ KEY FOR YOUR REWARD

6. **vs_efficiency_ratio**: VS task_accuracy / memory_cost
   - Formula: (task_accuracy_with_vs) / (memory_footprint_vs)
   - Expected range: [0, 10]
   - Why: Directly optimizes your reward function

7. **hg_efficiency_ratio**: HG task_accuracy / memory_cost
8. **vc_efficiency_ratio**: VC task_accuracy / memory_cost
9. **efficiency_variance**: Spread in efficiency across encoders
   - Why: If all encoders are equally efficient, selector has less to do

### Robustness Features (3 features)

10. **vs_noise_sensitivity**: Change in output when input perturbed
11. **hg_noise_sensitivity**: Change in graph when input perturbed
12. **vc_noise_sensitivity**: Change in visual features when input perturbed

### Domain-Specific Features (5 features)

13. **graph_hierarchy_quality**: Is the HG hierarchy meaningful or degenerate?
14. **visual_coverage**: Fraction of episode visually rendered in VC
15. **embedding_isotropy**: Are all directions equally used in VS?
16. [etc. 2 more domain features]

## Validation Results

✅ All features compute without error on sample data
✅ Ranges are as expected (no NaNs, infinities)
✅ Pairwise correlations < 0.9 (low redundancy)
✅ Features show variation across encoders (discriminative)
```

---

## PHASE 4: Validation & Refinement

### Effort Level: MEDIUM (testing, statistics, edge cases)

### Input to Agents

```markdown
Context: You have a proposed feature set from Phase 3 (M features + pseudocode).
Task: Validate these features work on REAL DATA from each encoder.

1. Test feature extraction (no crashes, expected ranges)
2. Check correlations (drop redundant features)
3. Identify edge cases (what breaks?)
4. Verify features discriminate encoder types
```

### Agents to Spawn (Parallel after Phase 3)

#### Agent 4.1: `benchmarking-specialist` [HAIKU]

```
Model: Haiku
Task: Rigorous feature validation on real data
Input: phase3_output.md (feature set + pseudocode) + real encoder outputs

Validation steps:
  1. Run feature extraction on 10+ real HyperMem graphs
  2. Run feature extraction on 10+ real MemOCR outputs
  3. Run feature extraction on 10+ real VectorDB embeddings
  4. Compute feature statistics (mean, std, min, max)
  5. Compute pairwise correlations (drop if > 0.9)
  6. Try to predict encoder type from features (classification accuracy)

Output:
  - Feature validation table (mean, std, range for each feature)
  - Correlation matrix (identify redundancy)
  - Encoder type prediction accuracy (bonus: not required)
  - Recommendations (drop/keep/refine each feature)

Time: 40 min
```

#### Agent 4.2: `error-detective` [HAIKU] (PARALLEL with 4.1)

```
Model: Haiku
Task: Find edge cases and failure modes
Input: Feature pseudocode + edge case scenarios

Edge cases to test:
  - Empty graph (HG with 0 nodes)
  - Zero embedding (VS with all zeros)
  - Blank image (VC with no visual content)
  - Degenerate graphs (linear chain, star topology)
  - Very high-dimensional embeddings (>1000 dims)
  - Very sparse graphs (1% connectivity)

Output:
  - List of features that fail on edge cases
  - Proposed fixes (clipping, safeguards, NaN handling)
  - Robustness report

Time: 15 min
```

### Handoff to Phase 5

**Output artifact**: `phase4_output.md`

```markdown
## Validation Results

### Feature Statistics (from 30+ real encoder outputs)

| Feature                  | Mean | Std  | Min  | Max  | Status |
| ------------------------ | ---- | ---- | ---- | ---- | ------ |
| embedding_dimensionality | 384  | 0    | 384  | 384  | ✅ OK  |
| sparsity                 | 0.15 | 0.08 | 0.02 | 0.42 | ✅ OK  |
| vs_hg_mutual_information | 0.23 | 0.11 | 0.05 | 0.67 | ✅ OK  |
| ...                      | ...  | ...  | ...  | ...  | ...    |

### Correlation Analysis

✅ Max pairwise correlation: 0.87 (acceptable, no redundancy > 0.9)
✅ All features have variance (discriminative)

### Edge Case Testing

✅ All features handle empty graphs (safe defaults)
✅ All features handle zero embeddings (clipped to min value)
⚠️ Feature X fails on degenerate graphs → Proposed fix: [safeguard]

### Feature Importance Ranking

Top 5 features for encoder discrimination:

1. efficiency_ratio (separates VS from HG/VC best)
2. memory_footprint (separates VC from VS/HG)
3. noise_sensitivity (differentiates HG best)
4. graph_hierarchy_quality (HG-specific, informative)
5. embedding_isotropy (VS-specific, informative)

### Final Count

✅ Starting features: 20
✅ Removed (too correlated): 1 (overlapping_redundancy)
✅ Added (from research-analyst suggestions): 0
✅ Final count: 19 features
```

---

## PHASE 5: Implementation & Documentation

### Effort Level: MEDIUM-HIGH (code quality, testability, documentation)

### Input to Agents

```markdown
Context: You have a validated, refined feature set (M features with ranges + edge case handling).
Task A: Implement as production-ready Python module
Task B: Document the feature engineering process
```

### Agents to Spawn (Parallel after Phase 4)

#### Agent 5.1: `python-engineer` [SONNET] ⭐ Implementation Quality

```
Model: Sonnet (code quality, edge cases, testing)
Task: Implement feature extraction module
Input: phase4_output.md (validated features) + selector_feature_engineering_prompt.md (Phase 5)

Deliverables:
  1. features_caimms.py module with:
     - Function for each feature group (geometry_features(), overlap_features(), etc.)
     - Unified extract_features(vs_output, hg_output, vc_output) function
     - Normalization logic (scale to [0, 1])
     - Error handling (NaN, infinity, edge cases)
     - Type hints, docstrings, examples

  2. Unit tests (tests/test_features_caimms.py):
     - Test each feature on toy data
     - Test edge cases (zeros, empty, degenerate)
     - Test normalization (ranges correct)
     - Test on real encoder outputs (if available)

  3. Integration example:
     - How to use with selector MLP
     - Expected input/output shapes
     - Performance considerations

Output: Production-ready code with 80%+ test coverage, clear documentation

Time: 40 min
```

#### Agent 5.2: `technical-writer` [HAIKU] (PARALLEL with 5.1)

```
Model: Haiku
Task: Document the feature engineering process
Input: phase1_output.md + phase2_output.md + phase3_output.md + phase4_output.md

Deliverables:
  1. Feature Engineering Report (comprehensive summary)
  2. Feature Reference Guide (table with all M features)
  3. Implementation notes (how to use features_caimms.py)
  4. Design decisions document (why these features, not others)
  5. Future work (gaps, extensions, open questions)

Output: Complete documentation package

Time: 30 min
```

### Final Polish

#### Skill: `/code-review` [HAIKU-level]

```
Task: Review generated code from python-engineer
Focus: Correctness, edge cases, maintainability, performance
Output: Feedback + improvement suggestions
Time: 15 min
```

#### Skill: `/checkpoint`

```
Task: Archive all work, decisions, findings
Output: Persistent memory for future sessions
Time: 5 min
```

---

## Summary: Execution Checklist

```
PHASE 1: FluxMem Baseline (45 min total)
  ☐ Agent 1.1: academic-researcher [Haiku] → 30 min
  ☐ Agent 1.2: code-reviewer [Haiku] → 15 min
  ↓ Handoff: phase1_output.md

PHASE 2: Encoder Characterization (45 min total)
  ☐ Agent 2.1: Explore [Haiku] → 20 min
  ☐ Agent 2.2: ml-engineer [Sonnet] → 25 min (parallel with 2.1, but depends on its output)
  ↓ Handoff: phase2_output.md

PHASE 3: Feature Engineering (120 min total)
  ☐ Agent 3.1: data-scientist [Sonnet] → 90 min ⭐ CORE WORK
  ☐ Agent 3.2: research-analyst [Haiku] → 30 min (parallel with 3.1)
  ↓ Handoff: phase3_output.md

PHASE 4: Validation (55 min total)
  ☐ Agent 4.1: benchmarking-specialist [Haiku] → 40 min
  ☐ Agent 4.2: error-detective [Haiku] → 15 min (parallel with 4.1)
  ↓ Handoff: phase4_output.md

PHASE 5: Implementation (85 min total)
  ☐ Agent 5.1: python-engineer [Sonnet] → 40 min
  ☐ Agent 5.2: technical-writer [Haiku] → 30 min (parallel with 5.1)
  ☐ Skill: /code-review → 15 min
  ☐ Skill: /checkpoint → 5 min

TOTAL: ~350 minutes (~5.8 hours)
```

---

## Model Allocation Summary

| Phase | Agent(s)                           | Model(s)       | Why                                                          |
| ----- | ---------------------------------- | -------------- | ------------------------------------------------------------ |
| 1     | academic-researcher, code-reviewer | Haiku × 2      | Straightforward reading + code review                        |
| 2     | Explore, ml-engineer               | Haiku + Sonnet | ml-engineer needs semantic interpretation (decision)         |
| 3     | data-scientist, research-analyst   | Sonnet + Haiku | data-scientist is CORE (high decision-making)                |
| 4     | benchmarking, error-detective      | Haiku × 2      | Testing & validation (pattern matching)                      |
| 5     | python-engineer, technical-writer  | Sonnet + Haiku | python-engineer: code quality matters; writer: summarization |

**Total Model Usage**:

- Haiku: 60% (8 agent runs)
- Sonnet: 40% (4 agent runs with decision-making)
- Opus: Reserved (0% unless critical tie in Phase 3)

---

## Ready to Execute?

When you're ready, I can:

1. **Spawn Phase 1 agents** (academic-researcher + code-reviewer)
2. **Wait for Phase 1 output** → then spawn Phase 2
3. **Continue sequentially** through all phases
4. **Track progress** with checkpoint updates

**Before starting Phase 1, confirm:**

- ✅ Do you have the FluxMem paper (or is it OK if academic-researcher searches for it)?
- ✅ For Phase 3, should I help generate sample data, or do you have real encoder outputs?
- ✅ Ready to spawn agents now?
