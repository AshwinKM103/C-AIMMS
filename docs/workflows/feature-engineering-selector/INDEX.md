# Selector MLP Feature Engineering: Complete Index

## Files in This Scratchpad

```
scratchpad/
├── INDEX.md                                    ← You are here
├── QUICK_START_GUIDE.md                       ← Start here: 5 min overview
├── selector_feature_engineering_prompt.md     ← Main prompt: detailed task spec
└── agents_skills_recommendation.md            ← Agent lineup and execution strategies
```

---

## Document Map & Read Path

### **For the Impatient (5 min)**
1. Read: `QUICK_START_GUIDE.md` — What you need to know to start
2. Action: Pick an execution sequence (Sequential, Parallel, or Hybrid)
3. Action: Run `/prompt-engineer` on the main prompt
4. Action: Spawn your first agent

---

### **For the Thorough (20 min)**
1. Read: `QUICK_START_GUIDE.md` — Full guide with checkpoints and Q&A
2. Read: `INDEX.md` → "Overview" section (this document)
3. Skim: `selector_feature_engineering_prompt.md` → Phases 1–3 only (get a feel for the task)
4. Read: `agents_skills_recommendation.md` → Your chosen execution sequence
5. Action: Run `/prompt-engineer`, spawn agents

---

### **For the Completist (45 min)**
1. Read: `QUICK_START_GUIDE.md` — Full guide
2. Read: `INDEX.md` → All sections (this document)
3. Read: `selector_feature_engineering_prompt.md` — Entire prompt, all phases
4. Read: `agents_skills_recommendation.md` → Entire agent lineup
5. Answer: The "Questions to Ask Before Starting" in agents_skills_recommendation.md
6. Action: Run `/prompt-engineer`, spawn agents, track with checkpoints

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

| Phase | Goal | Key Questions | Duration |
|-------|------|---------------|----------|
| **1: Baseline** | Understand FluxMem's 12 features | Why these 12? How do they work? | 30 min |
| **2: Encoder Outputs** | Characterize VS, HG, VC outputs | What shape/type/range does each produce? | 45 min |
| **3: Feature Engineering** | Design new feature set | What properties matter for selection? | 90 min ⭐ |
| **4: Validation** | Test on real data | Do features work? Are they redundant? | 55 min |
| **5: Implementation** | Code + docs | Can this be used in practice? | 70 min |

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

| What | Where |
|------|-------|
| FluxMem selector | `fluxmem/selector.py` |
| FluxMem features (baseline) | `fluxmem/features.py` |
| FluxMem MLP + training | `scripts/train_selector.py` |
| HyperMem graph extraction | `HyperMem/hypermem/extractors/` |
| MemOCR visual encoding | `MemOCR/` (esp. `memory_img*.py`) |
| HetRep VS encoder | `hetrep/` (all Python files) |
| Selector tests | `tests/test_selector.py` |
| Papers & ADRs | `docs/adr/`, `docs/papers/` |

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

These are *not* part of this feature engineering task, but keep them in mind as you design.

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

## File Sizes & Token Estimates

| File | Size | Read Time | Content |
|------|------|-----------|---------|
| `QUICK_START_GUIDE.md` | ~4 KB | 5 min | Overview, checkpoints, Q&A |
| `INDEX.md` (this file) | ~8 KB | 10 min | Map, paths, execution strategies |
| `selector_feature_engineering_prompt.md` | ~15 KB | 20 min | Detailed 5-phase task spec |
| `agents_skills_recommendation.md` | ~20 KB | 25 min | Agent lineup, effort estimates, pitfalls |

**Total**: ~47 KB, ~60 minutes to read completely
(But you can start with QUICK_START_GUIDE and fold in others as needed)

---

## How to Use These Files

### **Scenario 1: "I want to start today"**
1. Read `QUICK_START_GUIDE.md` (5 min)
2. Run `/prompt-engineer` on the main prompt (10 min)
3. Pick Path C (Hybrid): start with academic-researcher + Explore today (90 min)
4. Schedule data-scientist for tomorrow

### **Scenario 2: "I want to do this carefully"**
1. Read `QUICK_START_GUIDE.md` (5 min)
2. Read `agents_skills_recommendation.md` → your execution path (15 min)
3. Review the main prompt (`selector_feature_engineering_prompt.md`) completely (20 min)
4. Answer the "Questions to Ask Before Starting" in agents_skills_recommendation.md
5. Run `/prompt-engineer` (10 min)
6. Start with Path A (Sequential): follow the phase-by-phase sequence

### **Scenario 3: "I need this done fast"**
1. Skim `QUICK_START_GUIDE.md` (3 min)
2. Jump to `agents_skills_recommendation.md` → Path B (Parallel) (5 min)
3. Run `/prompt-engineer` with aggressive token limits (10 min)
4. Spawn batches in parallel, track with checkpoints

---

## Support & Debugging

### If You Get Stuck

**Problem**: Encoder outputs are not clear
- **Solution**: Ask `Explore` agent to find examples in tests or `ml-engineer` to explain

**Problem**: Features seem arbitrary or disconnected
- **Solution**: Go back to Phase 2 output (encoder specs) and ensure each feature addresses a real property

**Problem**: Phase 3 is taking too long
- **Solution**: Start with 5–7 simple features, validate them, then add more

**Problem**: Validation fails (features don't compute or are out of range)
- **Solution**: Use `error-detective` agent to identify edge cases

**Problem**: Too many features (>20)
- **Solution**: Rank by variance or mutual information, cut to top 12–16

---

## Ready?

1. ✅ Review `QUICK_START_GUIDE.md`
2. ✅ Pick an execution path (Sequential, Parallel, or Hybrid)
3. ✅ Run `/prompt-engineer` on the main prompt
4. ✅ Spawn your first agent
5. ✅ Track progress with checkpoints
6. ✅ Use `/checkpoint` at the end to archive findings

**You're ready to go!**

---

## Last Updated
Created: 2026-08-14
Prompt version: 1.0
Agent lineup: 10 agents + 5 skills

