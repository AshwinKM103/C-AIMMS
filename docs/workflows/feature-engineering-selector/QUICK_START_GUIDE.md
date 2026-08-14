# Selector Feature Engineering: Quick Start Guide

## What You Have

Three files in this scratchpad:

1. **`selector_feature_engineering_prompt.md`** — The main prompt for the feature engineering task
   - 5 phases with concrete tasks and deliverables
   - Explicit non-bias clause to prevent inheritance from past ADRs
   - Relative paths to all key codebases
   - Success criteria at the end

2. **`agents_skills_recommendation.md`** — Recommended agents and skills
   - Phase-by-phase breakdown of which agents to use
   - Three execution sequences (sequential, parallel, hybrid)
   - Effort estimates for each agent
   - Pitfall mitigations

3. **`QUICK_START_GUIDE.md`** — This file. How to use the above.

---

## How to Get Started (5 Minutes)

### Step 1: Review the Task
Read the **Objective** section in `selector_feature_engineering_prompt.md`.

**Key question**: Do you understand what "feature engineering for a selector MLP" means in this context?
- Selector MLP: neural network that chooses between VS, HG, VC encodings
- Feature engineering: designing input features that help the MLP make good choices
- The features come from analyzing the *outputs* of the three encoders

If no, ask for clarification before proceeding.

### Step 2: Check Your Resources
Do you have:
- ✅ Access to `fluxmem/`, `HyperMem/`, `MemOCR/`, `hetrep/` codebases?
- ✅ The FluxMem paper (or reference to it)?
- ✅ Examples of how each encoder is used?
- ❓ Labeled data for selector targets (optional, but helpful)?

If you're missing critical resources (e.g., FluxMem paper), note that before starting.

### Step 3: Pick an Execution Sequence

From `agents_skills_recommendation.md`, choose:

**Option 1: Sequential** (Recommended)
- **Pros**: Clear dependencies, easy to debug, iterative feedback
- **Cons**: Slower (~5 hours)
- **Pick this if**: You have time and want to be thorough

**Option 2: Parallel** (Fastest)
- **Pros**: Fast (~4–4.5 hours)
- **Cons**: Requires coordination, more context switches
- **Pick this if**: Time is critical and you can manage multiple agents

**Option 3: Hybrid** (Flexible)
- **Pros**: Start now, refine later
- **Cons**: Spread over multiple sessions
- **Pick this if**: You want to make progress today but can iterate

### Step 4: Optimize the Prompt
Run `/prompt-engineer` on `selector_feature_engineering_prompt.md` with a brief context:

```
Optimize this prompt for a feature engineering task in an ML codebase.
Agent: data-scientist (primary), research-analyst (secondary).
Context: Designing features for a selector MLP that chooses between 3 encoders.
Goal: Derive bottom-up features from encoder outputs, not inherit from past work.
Constraint: Token budget ~10k for the main agent, 5k for secondary.
```

This will produce an optimized version tailored to your agents and constraints.

### Step 5: Spawn Your First Agent
Based on your execution sequence, start with:

**Option 1 (Sequential)**: 
```
/prompt-engineer → academic-researcher (Phase 1) → code-reviewer (Phase 1)
```

**Option 2 (Parallel)**:
```
Spawn 2 agents in parallel: academic-researcher, Explore
```

**Option 3 (Hybrid)**:
```
Start with academic-researcher + Explore today
Schedule data-scientist for tomorrow
```

---

## During Execution: Key Checkpoints

### After Phase 1 (Baseline)
**Deliverable**: Structured list of FluxMem's 12 features.

**Checklist**:
- [ ] Each feature has a clear definition (not vague)
- [ ] I understand why these 12 differentiate linear/graph/hierarchical
- [ ] I can see how the selector MLP uses them (preprocessing, normalization)

**Red flag**: If you can't articulate why a feature matters, ask for clarification.

### After Phase 2 (Encoder Outputs)
**Deliverable**: Structured specification of VS, HG, VC outputs.

**Checklist**:
- [ ] I know the exact shape/type of each encoder output (e.g., "HyperMem outputs an adjacency matrix of shape (N, N) with float32 weights in [0, 1]")
- [ ] I know what metadata is available (e.g., "MemOCR provides confidence scores for each OCR token")
- [ ] I can visualize or inspect a real example

**Red flag**: If encoder outputs are unclear, ask for sample data or code examples.

### After Phase 3 (Feature Design)
**Deliverable**: Concrete list of M ≤ 20 features with pseudocode.

**Checklist**:
- [ ] Each feature has a one-line definition (e.g., "cosine distance between VS and HG embeddings")
- [ ] I can compute each feature by hand on a simple example
- [ ] Features cover different dimensions (geometry, overlap, efficiency, robustness, domain fit)
- [ ] No feature is obviously redundant with others

**Red flag**: If features are too many (>20) or not interpretable, iterate on the design.

### After Phase 4 (Validation)
**Deliverable**: Validation report with test results.

**Checklist**:
- [ ] All features compute without errors on real data
- [ ] Feature ranges are as expected (no NaNs, infinities, or outliers)
- [ ] Features show some variation across episodes (not all constant)
- [ ] Correlations between features are < 0.9 (low redundancy)
- [ ] Features somewhat separate different encoder types (bonus, not required)

**Red flag**: If many features fail or correlate highly, go back to Phase 3 and refine.

### After Phase 5 (Implementation)
**Deliverable**: Production-ready code + documentation.

**Checklist**:
- [ ] Code runs without errors
- [ ] All functions have docstrings and type hints
- [ ] Tests pass for happy paths and edge cases
- [ ] Feature engineering report is readable and complete
- [ ] Feature reference table is clear and usable

**Red flag**: If code quality is low or documentation is missing, iterate with python-engineer or technical-writer.

---

## Common Q&A

### Q: Should I read the full FluxMem paper or just the feature section?
**A**: Start with the feature section (referenced in Phase 1.3). If the rationale isn't clear, read the broader paper or ask an expert for context.

### Q: What if HyperMem or MemOCR outputs are not documented?
**A**: Spawn the `Explore` agent to find example usage in tests or scripts. If still unclear, run the code with sample input and inspect the output.

### Q: Do I need labeled data for selector targets?
**A**: No, but it helps. If you don't have it, validation will be qualitative (interpretability). If you do, validation can be quantitative (discriminative power).

### Q: How many features should I end up with?
**A**: Target 12–16 features to match FluxMem's baseline. If you have more, rank by importance and cut. If fewer, consider adding domain-specific features.

### Q: What if I get stuck in Phase 3?
**A**: That's the hardest part. Take a step back:
1. Are the encoder outputs clearly understood? (Phase 2)
2. Do you have example data to work with? (ask for toy examples)
3. Are you trying to be too clever? (simpler features often work better)
4. Ask the team: "What should the selector care about when choosing between VS, HG, VC?"

### Q: Can I skip Phase 2 and just look at the code?
**A**: No. Phase 2 is critical for understanding what you're designing features *for*. Skip it and your features will likely miss the mark.

### Q: Should I use the `/prompt-engineer` skill?
**A**: Yes. It will compress the main prompt and tailor it to your agents, saving tokens and time. Takes 10 minutes, saves hours.

### Q: Is this a one-session or multi-session task?
**A**: Depends on your time budget:
- **One long session** (5–6 hours): Option 1 (sequential)
- **Two sessions** (4 hours + 4 hours): Option 3 (hybrid)
- **Today**: Optimize + Phase 1 + Phase 2 (90 minutes)
- **Tomorrow**: Phases 3 + 4 (120 minutes)
- **Next day**: Phase 5 + polish (90 minutes)

### Q: What's the deliverable I should hand off after this?
**A**: A new feature set (pseudocode or code) that the next phase (selector training) can use. Include:
- M features with definitions and computation formulas
- Validation report showing they work on real data
- Python module with feature extraction functions
- Design document explaining why these features and not others

---

## Token Budget Hints

If you're using Claude Code with a token limit:

- **Main prompt**: ~5k tokens (after `/prompt-engineer` optimization)
- **Phase 1 (academic-researcher)**: ~3k tokens
- **Phase 2 (Explore + ml-engineer)**: ~4k tokens
- **Phase 3 (data-scientist)**: ~8–10k tokens (core work)
- **Phase 4 (benchmarking-specialist)**: ~5k tokens
- **Phase 5 (python-engineer + technical-writer)**: ~6k tokens

**Total**: ~30–35k tokens (one high-capability Claude model, or 2–3 Haiku agents in sequence)

If you're under a strict budget:
- Prioritize Phase 3 (feature engineering)
- Phase 1 + 2 can be condensed by the academic-researcher and Explore agents
- Phase 4 + 5 are less critical if Phase 3 is solid

---

## Final Checklist Before You Start

- [ ] I've read the "Objective" section in the main prompt
- [ ] I understand the "Explicit Non-Bias Clause" (don't inherit from ADRs)
- [ ] I have access to the necessary codebases (fluxmem, HyperMem, MemOCR, hetrep)
- [ ] I've chosen an execution sequence (Option 1, 2, or 3)
- [ ] I'm ready to ask clarifying questions if encoder outputs are unclear
- [ ] I've set aside time (5–6 hours for sequential, 4–4.5 hours for parallel)
- [ ] I have a clear definition of what the selector should optimize for (accuracy? latency? robustness?)

If all checkboxes are ticked, you're ready to start!

---

## Next Actions

1. **Read the main prompt** (`selector_feature_engineering_prompt.md`) completely. Time: 15 minutes.

2. **Run `/prompt-engineer`** on the main prompt to optimize it for your agents. Time: 10 minutes.

3. **Spawn your first agent** based on your chosen execution sequence. (See "Step 5" above.)

4. **Track progress** using this guide's "During Execution" checkpoints.

5. **At the end**, run `/checkpoint` to save findings, decisions, and recommendations for future sessions.

---

## Support

If you get stuck:
- Review the "Common Q&A" section above
- Check the "Red flags" in the checkpoint sections
- Ask for clarification on encoder outputs or feature definitions
- Consider starting with a simpler feature set (fewer features, clearer definitions) and iterating

Good luck! The feature engineering phase is the most creative and impactful part of this task. Take your time to get it right.

