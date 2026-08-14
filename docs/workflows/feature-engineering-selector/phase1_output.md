# Phase 1 Output: FluxMem Baseline Understanding

**Status**: ✅ Complete  
**Date**: 2026-08-14  
**Agents**: academic-researcher [Haiku], code-reviewer [Haiku]

---

## Executive Summary

**Two critical findings for C-AIMMS feature engineering:**

1. **FluxMem Paper** (Lu et al. 2026) defines **12 features** that strategically differentiate LINEAR, GRAPH, and HIERARCHICAL memory structures by measuring temporal, relational, and hierarchical conversational properties.

2. **C-AIMMS Implementation** (fluxmem/selector.py) currently uses a **simplified 7-feature subset** due to Phase 2/3 dependencies (features 6-7 are placeholders: `hyperedge_density`, `visual_salience`).

**Critical Insight**: The gap between **FluxMem's 12 paper features** and **C-AIMMS's 7 current features** means the new selector MLP (for VS/HG/VC selection) must be **designed from first principles**, not just copied. The 7 current features measure conversation-level properties (entities, temporal flow, topics), but C-AIMMS needs features that measure **encoder output properties** (VS embedding geometry, HG graph structure, VC visual features).

---

## Part A: FluxMem Paper Analysis (Academic-Researcher)

### The 12 FluxMem Features

FluxMem's selector chooses among LINEAR, GRAPH, HIERARCHICAL structures by extracting 12 lightweight features at each turn:

#### **Group 1: Temporal Properties (Linear-favoring)**

1. **page_count** — Number of interaction pages in context window (scale of conversation scope)
2. **avg_page_length** — Average tokens/turns per page (information density)
3. **is_qna_pattern** — Binary indicator of Q&A-dominated dialogue (turn-by-turn reasoning)
4. **temporal_density** — Interactions per unit time (recency-driven reasoning indicator)
5. **time_span** — Temporal coverage of context window (scope depth)

#### **Group 2: Relational Properties (Graph-favoring)**

6. **entity_density** — Named entities per page (entity-centric signal)
7. **relation_indicators** — Explicit relational expressions ("compared to", "depends on") (graph structure signal)
8. **is_entity_centric** — Binary: repeated entity-focused queries (relation-following demand)
9. **semantic_complexity** — Semantic variability across pages (structured content signal)

#### **Group 3: Hierarchical Properties (Hierarchical-favoring)**

10. **topic_diversity** — Distinct topics in recent interactions (branching discourse indicator)
11. **topic_transitions** — Frequency of topic shifts (rapid context switching)
12. **is_decision_tree** — Binary: decision-tree-like nested conditionals (coarse-to-fine reasoning)

### Design Rationale

**Why these 12?**

- **Conversational dimension coverage**: Temporal (page-level turn ordering), Relational (entity & semantic links), Hierarchical (topic evolution & nesting)
- **Lightweight computation**: Extractable via simple NLP (entity recognition, topic modeling) without embedding overhead
- **Discriminative**: Each group strongly signals when one structure is optimal
- **Empirical validation**: +9.18% vs. best single-structure baseline (LoCoMo), +2.1% (PERSONAMEM)

### FluxMem Selector Implementation

```
Input: 12-feature vector from conversation turn
  ↓
MLP: Linear(12, 4) → ReLU → Linear(4, 3)
  ↓
Output: 3-way softmax (LINEAR, GRAPH, HIERARCHICAL)
```

**Training**: Offline supervised learning (Algorithm 3)

- Run agent under each structure, compute reward
- Assign ground-truth label based on best reward
- Train MLP via cross-entropy loss

### Key Assumptions (from Paper)

✅ Lightweight computation prioritized over semantic depth  
✅ Conversation structure is locally stable (features over recent pages predict next turn)  
✅ Single structure does NOT dominate all conversations  
✅ Offline supervised learning is feasible (interaction-derived labels)

### Limitations (Acknowledged by Authors)

⚠️ Feature simplicity may miss deep semantic patterns  
⚠️ Local stability assumption breaks in rapidly shifting conversations  
⚠️ Label quality depends on reward signal reliability  
⚠️ No cross-task generalization tested

---

## Part B: C-AIMMS Implementation Analysis (Code-Reviewer)

### Current Selector Architecture (fluxmem/selector.py)

```
Input: 7-feature vector from episode
  ↓
StandardScaler (z-score normalization, fit on train only)
  ↓
MLP: Linear(7, 16) → ReLU → Linear(16, 3)
  ↓
Softmax → argmax → MemoryFormat (HG | VC | VS)
```

### The 7 Current Features (fluxmem/features.py)

| #   | Feature                | Computation                                   | Status     | Purpose                                            |
| --- | ---------------------- | --------------------------------------------- | ---------- | -------------------------------------------------- |
| 1   | `entity_count`         | log(distinct_entities)                        | ✅ Active  | Entity presence signal (Graph-leaning)             |
| 2   | `cooccurrence_density` | entity pair co-occurrence rate                | ✅ Active  | Entity relation density (Graph-leaning)            |
| 3   | `temporal_ordering`    | timestamp monotonicity + temporal marker rate | ✅ Active  | Temporal coherence (Linear-leaning)                |
| 4   | `topic_diversity`      | type-token ratio (distinct/total tokens)      | ✅ Active  | Semantic variability (Hierarchical-leaning)        |
| 5   | `token_length`         | log(total tokens)                             | ✅ Active  | Episode size (general signal)                      |
| 6   | `hyperedge_density`    | **placeholder (0.0)**                         | ⏳ Phase 2 | HG-specific signal (awaiting HyperMem integration) |
| 7   | `visual_salience`      | **placeholder (0.0)**                         | ⏳ Phase 3 | VC-specific signal (awaiting MemOCR integration)   |

### MLP Training Pipeline

**Architecture Decision**: PyTorch (not scikit-learn) for transparency:

- COLM Eq. 6 cross-entropy loss is explicit, not hidden in library calls
- Training loop reviewable line-by-line

**Preprocessing**:

- StandardScaler fit on train only (prevents data leakage)
- z-score normalization: (x - mean_train) / std_train

**Training**:

- Loss: PyTorch F.cross_entropy (COLM Eq. 6 literal)
- Optimizer: Adam (default lr=0.01)
- Early stopping: Validation loss plateau (patience=10 epochs)
- Reproducibility: Explicit torch seeding at **init** and fit()

**Inference**:

- Decision: argmax(softmax(output))
- Tie-breaking: Deterministic (HG > VC > VS priority)
- Confidence available via predict_proba()

### Code Quality Assessment

✅ **Strengths**:

- Transparent training loop (reviewable loss computation)
- Proper train/val separation in StandardScaler (no data leakage)
- Comprehensive tests (seeding, early stopping, serialization)
- Clear error messages (unfitted model detection)

⚠️ **Known Issues** (from ADR 0012):

- Features 2, 3, 4 are near-constant or collinear (amplify noise vs. signal)
- No feature importance analysis
- Pickle artifacts are version-sensitive

✅ **Production Readiness**: YES — awaits Phase 2/3 placeholder replacement

---

## Part C: Critical Gap Analysis

### The Challenge: FluxMem vs. C-AIMMS Encoders

| Dimension          | FluxMem (12 features)                         | C-AIMMS Current (7 features)        | C-AIMMS Need (NEW DESIGN)                          |
| ------------------ | --------------------------------------------- | ----------------------------------- | -------------------------------------------------- |
| **Measures**       | Conversation patterns (Q&A, entities, topics) | Episode-level summaries             | **Encoder OUTPUT properties**                      |
| **Input Source**   | Dialogue turns, NER, topic models             | Episode turns, spaCy NER            | **VS embeddings, HG graphs, VC CNN features**      |
| **Dimensionality** | Scalar features extracted from text           | Scalar features extracted from text | **Must handle 384-dim VS, (N,N) HG, (7,7,512) VC** |
| **Distinguishes**  | LINEAR vs. GRAPH vs. HIERARCHICAL             | Not yet optimized for VS/HG/VC      | **VS vs. HG vs. VC encoder suitability**           |

### Key Insight

**FluxMem's 12 features don't directly apply to C-AIMMS's selector because:**

1. FluxMem measures **conversation-level** properties (temporal patterns, entity density, topic flow)
2. C-AIMMS needs to measure **encoder output** properties (embedding geometry, graph structure, visual features)
3. The three encoders (VS, HG, VC) produce **fundamentally different data types** (vectors, graphs, spatial maps)

**Therefore**: The new feature set must be **designed from scratch** to characterize VS/HG/VC encoder outputs in a modality-agnostic way.

---

## Part D: What This Means for Phase 3 (Feature Engineering)

### Starting Point (What We Know)

✅ FluxMem's 12-feature design patterns are proven effective (+9% gain)  
✅ C-AIMMS's 7-feature infrastructure is production-ready  
✅ The three C-AIMMS encoders are characterized in Phase 2 (coming next)  
✅ Sample data is generated (10 episodes of VS/HG/VC outputs)

### Constraints for New Design

❌ Can't just copy FluxMem's 12 features (different problem domain)  
✅ Must design **modality-agnostic** features (work for all three encoders)  
✅ Must measure **encoder output properties** (not conversation patterns)  
✅ Must be **computable from real data** (VectorDB, HyperMem, MemOCR)  
✅ Should align with **reward function**: task_accuracy / memory_budget

### Phase 3 Task (Clarified)

**Design M ≤ 20 features that:**

1. **Geometry**: Measure dimensionality, sparsity, scale, structure of each encoder's output
2. **Overlap**: Measure redundancy/complementarity between encoder pairs
3. **Efficiency**: Measure compression ratio (task_accuracy / memory_footprint) ← KEY (your reward function)
4. **Robustness**: Measure noise sensitivity, uncertainty quantification
5. **Domain Fit**: Measure task-relevant properties (graph quality, visual coverage, etc.)

**All features must be**:

- Modality-agnostic (computable for VS, HG, VC)
- Normalized [0, 1]
- Interpretable (clear why it matters)

---

## Deliverables: Phase 1 Complete ✅

| Artifact                     | File                                                             | Content                                                          |
| ---------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| **FluxMem Paper Extraction** | `/tmp/claude-1005/.../FluxMem_12_Feature_Selector_Extraction.md` | 12 features + design rationale + limitations + empirical results |
| **C-AIMMS Code Review**      | `/tmp/claude-1005/.../SELECTOR_IMPLEMENTATION_REVIEW.md`         | Architecture + training + feature pipeline + known issues        |
| **Phase 1 Summary**          | (this file)                                                      | Gap analysis + next steps                                        |

---

## Next: Phase 2 (Encoder Characterization)

**Agents to Spawn**:

- ✅ `Explore` [Haiku] — Locate VS/HG/VC encoder files, extract output specs
- ✅ `ml-engineer` [Sonnet] — Interpret encoder outputs semantically

**Depends on**: Phase 1 complete (current)  
**Duration**: ~45 min  
**Produces**: phase2_output.md (encoder specs + semantics)  
**Then**: Phase 3 feature engineering begins

---

**Phase 1 Status**: ✅ COMPLETE  
**Ready for**: Phase 2 (Encoder Characterization)
