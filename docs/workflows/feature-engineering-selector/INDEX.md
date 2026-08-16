# Selector MLP Feature Engineering: Complete Index

## Files in This Deliverable

```
docs/workflows/feature-engineering-selector/
├── INDEX.md                                    ← You are here
├── FEATURE_ENGINEERING_REPORT.md               ← Start here: complete overview
├── FEATURE_REFERENCE.md                        ← Feature catalog (all 15 features)
├── DESIGN_RATIONALE.md                         ← Architecture & token-budget accounting
├── DEPLOYMENT_GUIDE.md                         ← Integration & code examples
├── phase3_output.md                            ← Full specifications & validation
├── phase1_output.md                            ← Baseline analysis
├── phase2_output.md                            ← Encoder characterization
└── SEQUENTIAL_EXECUTION_PLAN.md                ← Execution record
```

---

## Document Map & Read Path

### **For Quick Reference (10 min)**

1. Read: `FEATURE_REFERENCE.md` — All 15 features with formulas

---

### **For Understanding (40 min)**

1. Read: `FEATURE_ENGINEERING_REPORT.md` — Complete methodology & results
2. Read: `DESIGN_RATIONALE.md` → Architecture section (why these features)
3. Skim: `FEATURE_REFERENCE.md` → Your features of interest

---

### **For Implementation (30 min)**

1. Read: `DEPLOYMENT_GUIDE.md` — Code patterns & integration
2. Skim: `FEATURE_REFERENCE.md` → Feature formulas
3. Reference: `phase3_output.md` → Validation gates & bounded features

---

## Overview: What Is This Task?

### The Problem

You have a **selector MLP** that chooses between three encodings:

- **VS** (Vector Semantic): dense embeddings from text/episodes
- **HG** (Hierarchical Graph): structured graph representations from HyperMem
- **VC** (Visual/Conceptual): visual or conceptual features from MemOCR

The selector is based on **FluxMem's selector**, which has **12 hand-crafted features** designed to differentiate between linear, graph, and hierarchical modalities.

### The Challenge

**Can't directly reuse FluxMem's features because:**

1. Different representations (embeddings vs. graphs vs. images)
2. Different extraction sources (VectorDB vs. HyperMem vs. MemOCR)
3. Different downstream uses (C-AIMMS pipeline vs. FluxMem pipeline)

### The Goal

**Design a new feature set** (M ≤ 20 features) that:

- ✅ Works for all three encodings
- ✅ Derives from their actual outputs (not theoretical)
- ✅ Helps an MLP choose between them
- ✅ Is interpretable and justified (not just tuned)
- ✅ Addresses the non-bias requirement: bottom-up, not inherited from ADRs

---

## Task Structure: 5 Phases

| Phase                      | Goal                             | Key Questions                            | Duration  |
| -------------------------- | -------------------------------- | ---------------------------------------- | --------- |
| **1: Baseline**            | Understand FluxMem's 12 features | Why these 12? How do they work?          | 30 min    |
| **2: Encoder Outputs**     | Characterize VS, HG, VC outputs  | What shape/type/range does each produce? | 45 min    |
| **3: Feature Engineering** | Design new feature set           | What properties matter for selection?    | 90 min ⭐ |
| **4: Validation**          | Test on real data                | Do features work? Are they redundant?    | 55 min    |
| **5: Implementation**      | Code + docs                      | Can this be used in practice?            | 70 min    |

**Total**: 5–6 hours (sequential) or 4–4.5 hours (parallel)

---

## Key Artifacts to Produce

### Deliverable 1: Feature Engineering Report

- FluxMem baseline analysis (12 features, why they work)
- Encoder output characterization (VS, HG, VC specifications)
- Derived feature set (M ≤ 20 features, definitions, pseudocode)
- Feature importance ranking
- Validation results and gaps

### Deliverable 2: Feature Extraction Code

- Python module with feature computation functions
- Type hints and docstrings
- Unit tests and edge case handling
- Example usage and integration instructions

### Deliverable 3: Feature Reference Guide

- Table mapping FluxMem → C-AIMMS features (if applicable)
- Interpretation guide (what does each feature mean?)
- Configuration and usage examples

---

## Execution Paths

### Path A: Sequential (Recommended)

Best for: Learning, debugging, iterative feedback

```
1. academic-researcher (Phase 1)
   ↓ (output: 12 FluxMem features)
2. code-reviewer (Phase 1 validation)
   ↓
3. Explore (Phase 2)
   ↓ (output: encoder specs)
4. ml-engineer (Phase 2 interpretation)
   ↓
5. data-scientist (Phase 3) ⭐ CORE
   ↓ (output: M features, pseudocode)
6. research-analyst (Phase 3, parallel with data-scientist)
   ↓
7. benchmarking-specialist (Phase 4)
   ↓ (output: validation report)
8. error-detective (Phase 4, parallel)
   ↓
9. python-engineer (Phase 5)
   ↓ (output: implementation)
10. technical-writer (Phase 5, parallel)
    ↓ (output: documentation)
11. /code-review skill (Polish)
12. /checkpoint skill (Archive)
```

**Time**: ~5 hours
**Best for**: Getting the most accurate, well-validated result

---

### Path B: Parallel (Fastest)

Best for: Time-critical, established team, can handle coordination

```
Batch 1 (30 min):
  academic-researcher (Phase 1) ║ Explore (Phase 2)
  ↓
  → deliverables: 12 FluxMem features + encoder specs

Batch 2 (25 min):
  code-reviewer (Phase 1 validation) ║ ml-engineer (Phase 2)
  ↓
  → deliverables: validated FluxMem features + encoder semantics

Batch 3 (90 min):
  data-scientist (Phase 3) ⭐ CORE ║ research-analyst (Phase 3)
  ↓
  → deliverables: M features + literature context

Batch 4 (55 min):
  benchmarking-specialist (Phase 4) ║ error-detective (Phase 4)
  ↓
  → deliverables: validated features + robustness report

Batch 5 (70 min):
  python-engineer (Phase 5) ║ technical-writer (Phase 5)
  ↓
  → deliverables: implementation + docs

Polish (20 min):
  /code-review + /checkpoint
```

**Time**: ~4–4.5 hours
**Best for**: Teams with capacity to manage multiple agents in parallel

---

### Path C: Hybrid (Flexible)

Best for: Incremental progress, multi-session work

**Session 1 (90 min)**: Phases 1–2

- academic-researcher (FluxMem baseline)
- Explore (encoder structures)
- Deliverable: Clear understanding of baseline + outputs

**Session 2 (120 min)**: Phases 3–4

- data-scientist (feature engineering)
- benchmarking-specialist (validation)
- Deliverable: Validated feature set v1

**Session 3 (90 min)**: Phase 5 + polish

- python-engineer (implementation)
- technical-writer (docs)
- /code-review + /checkpoint
- Deliverable: Production-ready code + docs

**Time**: Spread over 3 days, ~300 minutes of agent work
**Best for**: Iterating, gathering feedback between sessions

---

## Critical Constraints & Requirements

### Non-Bias Requirement

❌ **Don't**: Inherit feature choices from past ADRs or architectural decisions
✅ **Do**: Derive features bottom-up from encoder outputs
✅ **Do**: Cite only papers and code, not design docs
✅ **Do**: Validate with data, not intuition

### Modality-Agnostic Design

❌ **Don't**: Design separate feature sets for VS, HG, VC
✅ **Do**: Design one feature set that works for all three
✅ **Do**: Use properties that can be computed for any encoder output

### Feature Count

❌ **Don't**: Create >20 features
✅ **Do**: Target 12–16 features (match FluxMem baseline)
✅ **Do**: Rank by importance if you exceed the limit

### Real Data Validation

❌ **Don't**: Assume features work without testing
✅ **Do**: Validate on actual HyperMem graphs, MemOCR images, VectorDB embeddings
✅ **Do**: Check for NaNs, infinities, out-of-range values

---

## Key Code Paths (Relative)

| What                        | Where                             |
| --------------------------- | --------------------------------- |
| FluxMem selector            | `fluxmem/selector.py`             |
| FluxMem features (baseline) | `fluxmem/features.py`             |
| FluxMem MLP + training      | `scripts/train_selector.py`       |
| HyperMem graph extraction   | `HyperMem/hypermem/extractors/`   |
| MemOCR visual encoding      | `MemOCR/` (esp. `memory_img*.py`) |
| HetRep VS encoder           | `hetrep/` (all Python files)      |
| Selector tests              | `tests/test_selector.py`          |
| Papers & ADRs               | `docs/adr/`, `docs/papers/`       |

---

## Checkpoints During Execution

### ✅ After Phase 1

**Questions**:

- Can you articulate why FluxMem chose these 12 features?
- Do you understand the math behind each feature?
- Can you see how they differentiate linear/graph/hierarchical?

**If no**: Ask academic-researcher or code-reviewer for clarification.

---

### ✅ After Phase 2

**Questions**:

- Do you know the exact shape, type, and range of each encoder output?
- Can you explain what each encoder "means" in terms of information content?
- Do you have or can you create toy examples of VS, HG, VC outputs?

**If no**: Ask ml-engineer to interpret or Explore to find examples in tests.

---

### ✅ After Phase 3

**Questions**:

- Can you list M features and explain each in one sentence?
- Do these features make intuitive sense (not arbitrary)?
- Can you compute each feature by hand on a simple example?
- Do features span different dimensions (geometry, efficiency, robustness)?

**If no**: Iterate with data-scientist; consider starting with simpler features.

---

### ✅ After Phase 4

**Questions**:

- Do all features compute without errors?
- Are feature ranges as expected (no NaNs, outliers)?
- Is there redundancy (feature correlation > 0.9)?
- Can features predict encoder type? (Nice to have, not required)

**If no**: Refine in Phase 3 or fix edge cases with error-detective.

---

### ✅ After Phase 5

**Questions**:

- Does the code run without errors?
- Are tests passing?
- Is documentation complete and clear?
- Can someone else use this code to extract features?

**If no**: Iterate with python-engineer or technical-writer.

---

## Recommended Next Steps (After This Task)

1. **Collect ground truth**: Label episodes with their "best" encoder (VS, HG, or VC)
2. **Train selector MLP**: Use M features as input, labels as supervision
3. **Benchmark**: Compare new selector accuracy vs. FluxMem baseline
4. **Deploy**: Integrate into `fluxmem/selector.py`
5. **Iterate**: Ablate features, add/remove based on importance

These are _not_ part of this feature engineering task, but keep them in mind as you design.

---

## Quick Q&A

**Q: Can I skip Phase 1?**
A: No. You need to understand the baseline to avoid just copying FluxMem features.

**Q: Can I skip Phase 2?**
A: No. Without knowing what encoders output, your features will miss the mark.

**Q: Is Phase 3 really the hardest?**
A: Yes. That's where creativity and domain knowledge matter most. Budget extra time.

**Q: What if I don't have labeled selector targets?**
A: Validation will be qualitative (interpretability). Quantitative validation (discriminative power) requires labels.

**Q: Should I use `/prompt-engineer`?**
A: Yes. Takes 10 minutes, saves hours in agent efficiency and token usage.

**Q: How many agents should I spawn in parallel?**
A: Path B (parallel) spawns up to 2 agents at a time. More than that gets chaotic.

---

## Success Criteria

- [x] Feature set derived bottom-up from encoder outputs, not inherited
- [x] M ≤ 20 features, each with clear definition and pseudocode
- [x] All features compute without errors on real data
- [x] Features span multiple dimensions (not all measuring the same thing)
- [x] Design decisions documented with rationale
- [x] Production-ready Python code with tests and docs
- [x] Feature engineering report explains "why these features"

---

## Status: Complete ✅

**Workflow completed** (2026-08-15). 15-feature selector MLP fully designed, validated, and implemented.

- ✅ Phase 1–2: Baseline analysis + encoder characterization
- ✅ Phase 3: Feature engineering (15 features designed)
- ✅ Phase 4: Validation on real data (96 tests passing)
- ✅ Phase 5: Implementation in `fluxmem/features_v2.py`

**See FEATURE_ENGINEERING_REPORT.md for full results.**
