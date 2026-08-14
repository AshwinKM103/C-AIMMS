# Recommended Agents & Skills for Selector Feature Engineering

## Executive Summary

This task requires **multi-disciplinary expertise**: literature review (papers), code archaeology (understanding three disparate systems), mathematical modeling (feature design), and empirical validation (testing on data). Below is a **layered agent lineup** organized by phase and expertise.

---

## Phase-by-Phase Agent Recommendations

### **Phase 1: Baseline Understanding (FluxMem Selector)**

#### Primary Agent: `academic-researcher`
**Why**: This phase requires deep reading of the FluxMem paper and extracting the rationale for 12 features.

**Scope**:
- Read FluxMem paper (if in `docs/` or linked in ADRs)
- Extract feature definitions with mathematical rigor
- Synthesize why these features differentiate linear/graph/hierarchical modalities
- Produce structured feature documentation

**Tools needed**: Read, Write, Bash (to find paper references)
**Estimated effort**: 20–30 minutes

#### Secondary Agent: `code-reviewer`
**Why**: Understand the selector implementation quality and design patterns.

**Scope**:
- Review `fluxmem/selector.py` for architecture, preprocessing, normalization
- Review `fluxmem/features.py` for how features are computed
- Identify assumptions and constraints
- Suggest refactoring if needed for the new feature set

**Tools needed**: Read, Bash (grep for usages)
**Estimated effort**: 15 minutes

---

### **Phase 2: Encoder Output Analysis**

#### Primary Agent: `Explore` (fast mode)
**Why**: Quickly locate and characterize the three encoder implementations without reading the full files redundantly.

**Scope**:
- Find key functions in each encoder (input/output)
- Identify output data structures (shapes, types, ranges)
- Locate test files or example usage
- Build a quick reference of what each encoder produces

**Query examples**:
- `"HyperMem graph extractor output structure"`
- `"MemOCR visual feature extraction pipeline"`
- `"hetrep vector semantic embedding output"`

**Tools needed**: All tools (grep, read targeted sections, Bash)
**Estimated effort**: 20 minutes

#### Secondary Agent: `ml-engineer`
**Why**: Interpret the outputs in the context of ML pipeline design.

**Scope**:
- Understand what these outputs are meant to represent (task-level semantics)
- Identify what properties matter for downstream use
- Check if outputs are normalized, sparse, or have special structure
- Identify potential preprocessing steps

**Tools needed**: Read, Bash, Write
**Estimated effort**: 25 minutes

---

### **Phase 3: Feature Engineering (Core)**

#### Primary Agent: `data-scientist`
**Why**: This is the core feature engineering task—deriving meaningful features from data/representations.

**Scope**:
- Design intrinsic properties (dimensionality, sparsity, structure, scale)
- Design cross-modality features (redundancy, complementarity, efficiency)
- Design domain-specific features (task-relevant properties)
- Compute and validate features on sample data
- Analyze feature correlations and importance

**Deliverable**: A concrete feature set (M ≤ 20 features) with pseudocode and validation.

**Tools needed**: Read, Write, Bash (to run test data), Python (compute features)
**Estimated effort**: 60–90 minutes (core work)

#### Alternative/Complementary: `research-analyst`
**Why**: Systematic literature review on feature selection methodologies.

**Scope**:
- Research papers on feature engineering for modality selection (e.g., attention mechanisms, gating networks)
- Compare existing approaches (FluxMem, mixture-of-experts, dynamic routing)
- Extract best practices and design principles
- Inform the feature design in Phase 3

**Tools needed**: alphaXiv (paper search), Read, Write
**Estimated effort**: 30 minutes

---

### **Phase 4: Validation & Refinement**

#### Primary Agent: `benchmarking-specialist`
**Why**: Rigorously test and validate the feature set on real data.

**Scope**:
- Design a validation plan (test on sample inputs)
- Implement feature extraction on real data
- Check for correctness (no NaNs, expected ranges)
- Compute correlations between features
- Test separation power (can you predict encoder type from features?)
- Compare to FluxMem features (feature mapping)

**Deliverable**: Validation report with test results, correlations, gaps.

**Tools needed**: Read, Write, Bash, Python (numpy, scipy for stats)
**Estimated effort**: 40 minutes

#### Secondary: `error-detective`
**Why**: Identify edge cases and failure modes.

**Scope**:
- Test on boundary cases (empty graphs, blank images, zero embeddings)
- Identify features that might fail or be undefined
- Suggest robustness improvements

**Tools needed**: Read, Bash, Python
**Estimated effort**: 15 minutes

---

### **Phase 5: Documentation & Code Generation**

#### Primary Agent: `python-engineer`
**Why**: Generate clean, reusable feature extraction code.

**Scope**:
- Implement feature extraction functions from pseudocode
- Add type hints, docstrings, error handling
- Organize into a new module (`fluxmem/features_caimms.py`)
- Write unit tests for each feature

**Deliverable**: Production-ready Python module with tests.

**Tools needed**: Write, Edit, Bash (run tests)
**Estimated effort**: 40 minutes

#### Secondary: `technical-writer`
**Why**: Document the feature engineering process for future reference.

**Scope**:
- Write the full Feature Engineering Report (phases 1–4)
- Create a feature reference guide (table with definitions, ranges, usage)
- Document design decisions and trade-offs
- Create a README for the new feature set

**Deliverable**: Complete documentation package.

**Tools needed**: Read, Write, Edit
**Estimated effort**: 30 minutes

---

## Skills to Invoke Before/After

### Before Starting (Prompt Optimization)
**Skill**: `/prompt-engineer`
- Optimize the main prompt for token efficiency
- Tailor to the primary agent (e.g., data-scientist)
- Compress overlapping sections
- Prioritize phases by importance

**Estimated effort**: 10 minutes

### After Feature Design (Experiment Setup)
**Skill**: `/experiment-discipline`
- Design a rigorous experiment to validate features
- Define metrics (correlation, separation power, etc.)
- Create a reproducible test pipeline
- Set success thresholds

**Estimated effort**: 15 minutes

### After Implementation (Code Review)
**Skill**: `/code-review` (if not using code-reviewer agent)
- Review the feature extraction code
- Check for edge cases, performance, maintainability
- Suggest improvements

**Estimated effort**: 15 minutes

### End of Session
**Skill**: `/checkpoint`
- Save progress, decisions, and recommendations
- Prepare for next session if multi-day work

---

## Recommended Execution Sequence

### **Option 1: Sequential (Recommended for Clarity)**

1. **Optimize prompt** (`/prompt-engineer` on main prompt) — 10 min
2. **Phase 1**: `academic-researcher` + `code-reviewer` in parallel — 30 min
3. **Phase 2**: `Explore` (fast) → `ml-engineer` — 45 min
4. **Phase 3**: `data-scientist` with `research-analyst` in parallel — 90 min (core)
5. **Phase 4**: `benchmarking-specialist` + `error-detective` in parallel — 55 min
6. **Phase 5**: `python-engineer` + `technical-writer` in parallel — 70 min
7. **Polish**: `/code-review` skill — 15 min
8. **Checkpoint**: `/checkpoint` — 5 min

**Total**: ~5 hours (mostly Phase 3)

---

### **Option 2: Parallel (Faster, Requires Coordination)**

**Batch 1 (in parallel)**:
- `academic-researcher` on FluxMem paper + feature extraction
- `Explore` on encoder structures
- Estimated: 30 min

**Batch 2 (in parallel)**:
- `code-reviewer` on selector.py
- `ml-engineer` on encoder outputs (fed by Batch 1 results)
- Estimated: 25 min

**Batch 3 (core, sequential)**:
- `data-scientist` on feature engineering (fed by Batch 2)
- `research-analyst` on related work (in parallel)
- Estimated: 90 min

**Batch 4 (in parallel)**:
- `benchmarking-specialist` on validation
- `error-detective` on edge cases
- Estimated: 55 min

**Batch 5 (in parallel)**:
- `python-engineer` on code generation
- `technical-writer` on documentation
- Estimated: 70 min

**Polish**: `/code-review` + `/checkpoint`
- Estimated: 20 min

**Total**: ~4–4.5 hours (parallelized, Phase 3 still dominates)

---

### **Option 3: Hybrid (Start Now, Iterate)**

If time is limited:

1. **Today** (90 min):
   - `academic-researcher` on FluxMem paper
   - `Explore` on encoder structures
   - `data-scientist` starts feature brainstorming
   - Deliverable: Initial feature list (v0)

2. **Tomorrow** (120 min):
   - `data-scientist` refines features with Phase 2 inputs
   - `benchmarking-specialist` validates v0 on real data
   - Feedback loop: iterate if needed
   - Deliverable: Feature list v1 (validated)

3. **Later** (90 min):
   - `python-engineer` implements
   - `technical-writer` documents
   - `/code-review` polish
   - Deliverable: Production-ready code + docs

---

## Critical Success Factors

### 1. **Non-Bias Commitment**
- Ensure agents understand the "Explicit Non-Bias Clause" in the prompt
- Tell them: ignore past ADRs and design decisions; trust only the data
- Validate: feature set should be derived bottom-up, not top-down from prior work

### 2. **Clear Input/Output Contracts**
- Between Phase 1 → 2: encode the "12 FluxMem features" as a structured table for Phase 2
- Between Phase 2 → 3: encode encoder outputs as a data structure spec (shapes, types, ranges)
- Between Phase 3 → 4: encode proposed features as pseudocode or test functions
- Between Phase 4 → 5: encode validated features + gaps as a reference spec

### 3. **Real Data**
- Don't just theorize; test on actual HyperMem graphs, MemOCR images, VectorDB embeddings
- If no labeled data exists, the agent should flag this and suggest synthetic or toy data for validation

### 4. **Iterative Feedback**
- After Phase 3 output, review: are the features interpretable? Can you understand why one modality is preferred?
- If not, iterate: refine definitions, add clarifying features, remove redundant ones

---

## Agent Profile Quick Reference

| Agent | Expertise | Key Phases | Effort |
|-------|-----------|-----------|--------|
| `academic-researcher` | Paper reading, feature rationale | 1 | 30 min |
| `code-reviewer` | Code quality, implementation patterns | 1 | 15 min |
| `Explore` | Quick code navigation | 2 | 20 min |
| `ml-engineer` | ML pipeline semantics | 2 | 25 min |
| `data-scientist` | Feature engineering, validation | **3, 4** | **120+ min** |
| `research-analyst` | Feature selection literature | 3 | 30 min |
| `benchmarking-specialist` | Rigorous testing, statistics | 4 | 40 min |
| `error-detective` | Edge cases, robustness | 4 | 15 min |
| `python-engineer` | Code implementation | 5 | 40 min |
| `technical-writer` | Documentation | 5 | 30 min |

---

## Skills Before/After

| Skill | When | Purpose | Effort |
|-------|------|---------|--------|
| `/prompt-engineer` | Before all | Optimize main prompt | 10 min |
| `/experiment-discipline` | After Phase 3 | Formalize validation | 15 min |
| `/code-review` | After Phase 5 | Polish code quality | 15 min |
| `/checkpoint` | End | Save progress | 5 min |

---

## Common Pitfalls & Mitigations

### Pitfall 1: Inheriting Bias from ADRs
**Mitigation**: Make the "Explicit Non-Bias Clause" the first thing agents read. Have them cite only papers and code.

### Pitfall 2: Features That Don't Separate Modalities
**Mitigation**: After Phase 4 validation, test the discriminative power of the feature set. If features can't predict encoder type at >70% accuracy, iterate on Phase 3.

### Pitfall 3: Missing Edge Cases
**Mitigation**: After Phase 4, run `error-detective` to test boundary conditions (empty graphs, blank images, zero embeddings).

### Pitfall 4: Scope Creep in Phase 3
**Mitigation**: Cap the feature set at 20 features. If you have more candidates, rank by importance (variance, mutual information with selector targets) and cut.

### Pitfall 5: No Real Data for Validation
**Mitigation**: If labeled selector targets don't exist, use synthetic data or ask the team for example episodes where one modality is clearly better.

---

## Handoff to Selector Training

Once this feature engineering is complete, the next phases are:

1. **Collect ground truth**: For a set of episodes, annotate which encoding (VS, HG, VC) should the selector prefer (highest accuracy, lowest latency, best robustness?)
2. **Train selector MLP**: Use the M-dimensional feature vector as input, selector target as supervision
3. **Benchmark**: Compare new selector accuracy/efficiency vs. FluxMem baseline
4. **Deploy**: Integrate the new selector and feature extraction into `fluxmem/selector.py`

These are out of scope for this feature engineering task but should be kept in mind as motivation.

---

## Questions to Ask Before Starting

1. **Do you have labeled data** for selector targets? (i.e., "for this episode, use VS" / "use HG" / "use VC")
   - If yes: validation will be quantitative (discriminative power)
   - If no: validation will be qualitative (interpretability, plausibility)

2. **Are there examples of episodes where one modality clearly outperforms others?**
   - If yes: use them to validate that features capture the right properties
   - If no: suggest creating toy examples to drive feature design

3. **Do you want features that are modality-agnostic or modality-specific?**
   - Modality-agnostic: same feature can be computed for any encoder
   - Modality-specific: different features for VS vs HG vs VC
   - Recommendation: start with modality-agnostic, then add modality-specific if needed

4. **What's the deployment context?** (inference latency, training time, hardware)
   - This affects feature complexity (e.g., avoid expensive statistics if latency-sensitive)

---

## Ready to Go?

Once you've answered the above questions and selected an execution sequence (Option 1, 2, or 3), start with `/prompt-engineer` to optimize the main prompt, then spawn the first agent.

