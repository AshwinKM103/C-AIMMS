# Agent Lineup Quick Reference Card

Use this card to quickly identify which agents to spawn and in what order.

---

## The Stack: Agents & Their Roles

### **Tier 1: Foundation (Understanding)**
These agents build the baseline knowledge you need.

| Agent | Task | Time | Output | Run |
|-------|------|------|--------|-----|
| `academic-researcher` | Read FluxMem paper, extract 12 features | 30 min | Structured feature list | First |
| `Explore` (fast) | Find + characterize VS/HG/VC encoder outputs | 20 min | Encoder specs (shapes, types, ranges) | Parallel |

**Pair with**: Read the main prompt Phases 1–2 first so they know what to look for.

---

### **Tier 2: Validation (Sanity Check)**
These verify Tier 1 outputs are correct.

| Agent | Task | Time | Output | Run |
|-------|------|------|--------|-----|
| `code-reviewer` | Review selector.py for quality + patterns | 15 min | Implementation notes | After academic-researcher |
| `ml-engineer` | Interpret encoder outputs in ML context | 25 min | Semantic characterization | After Explore |

**Pair with**: Let these agents verify Tier 1 before moving to Tier 3.

---

### **Tier 3: Core Feature Engineering (The Heavy Lifting)**
This is where the real work happens.

| Agent | Task | Time | Output | Run |
|-------|------|------|--------|-----|
| `data-scientist` ⭐ | Design M ≤ 20 features from encoder outputs | 90 min | Feature list + pseudocode | Main work |
| `research-analyst` | Literature review on feature selection | 30 min | Best practices + context | Parallel |

**Pair with**: Feed Tier 1 + 2 outputs to data-scientist. research-analyst runs in parallel.

**Key input**: Tier 1 (FluxMem 12 features) + Tier 2 (encoder specs)
**Key output**: Concrete list of M features, each with definition + pseudocode

---

### **Tier 4: Validation (Testing)**
These verify Tier 3 outputs work on real data.

| Agent | Task | Time | Output | Run |
|-------|------|------|--------|-----|
| `benchmarking-specialist` | Test features on real data (compute, ranges, correlation) | 40 min | Validation report | After data-scientist |
| `error-detective` | Find edge cases + failure modes | 15 min | Robustness report | Parallel |

**Pair with**: Let these agents validate Tier 3 before moving to Tier 5.

---

### **Tier 5: Implementation (Making It Real)**
These produce production-ready deliverables.

| Agent | Task | Time | Output | Run |
|-------|------|------|--------|-----|
| `python-engineer` | Implement feature extraction code | 40 min | Module with tests + docs | After Tier 4 |
| `technical-writer` | Document the feature engineering process | 30 min | Report + reference guide | Parallel |

**Pair with**: Both can run in parallel. Merge outputs for final deliverable.

---

### **Post-Phase Skills (Polish & Archive)**
These wrap up.

| Skill | Task | Time | Output | Run |
|-------|------|------|--------|-----|
| `/code-review` | Review the generated code for quality | 15 min | Feedback + suggestions | After python-engineer |
| `/checkpoint` | Archive work, decisions, findings | 5 min | Persistent notes for next session | End |

---

## Three Execution Strategies

### **STRATEGY A: SEQUENTIAL (Recommended for Accuracy)**

```
Optimize prompt → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Polish → Archive
academic-researcher → (Explore + ml-engineer) → data-scientist → benchmarking → python-engineer
```

**Command sequence**:
```bash
/prompt-engineer  # Optimize main prompt

# Then spawn agents in order:
academic-researcher  # Phase 1 (30 min)
Explore  # Phase 2 (20 min)
ml-engineer  # Phase 2 validation (25 min)
code-reviewer  # Phase 1 validation (15 min)
data-scientist  # Phase 3 (90 min) ← Most important
research-analyst  # Phase 3 parallel (30 min)
benchmarking-specialist  # Phase 4 (40 min)
error-detective  # Phase 4 parallel (15 min)
python-engineer  # Phase 5 (40 min)
technical-writer  # Phase 5 parallel (30 min)
/code-review  # Polish (15 min)
/checkpoint  # Archive (5 min)
```

**Total time**: ~5 hours
**Best for**: Thorough, iterative, getting the best result

---

### **STRATEGY B: PARALLEL (Fastest)**

**Batch 1** (Run in parallel):
```
academic-researcher + Explore
(30 min → 12 FluxMem features + encoder specs)
```

**Batch 2** (Run in parallel):
```
code-reviewer + ml-engineer
(25 min → validated FluxMem features + encoder semantics)
```

**Batch 3** (Core work, run in parallel):
```
data-scientist ⭐ + research-analyst
(90 min → M features + context)
```

**Batch 4** (Run in parallel):
```
benchmarking-specialist + error-detective
(40 min → validated features + robustness)
```

**Batch 5** (Run in parallel):
```
python-engineer + technical-writer
(40 min → implementation + docs)
```

**Polish**:
```
/code-review + /checkpoint
(20 min → code review + archive)
```

**Total time**: ~4–4.5 hours (parallelized)
**Best for**: Time-critical, teams that can manage multiple agents

---

### **STRATEGY C: HYBRID (Incremental)**

**Session 1 (Today)** — 90 min:
```
/prompt-engineer
academic-researcher  # Phase 1
Explore              # Phase 2
→ Deliverable: Baseline understanding
```

**Session 2 (Tomorrow)** — 120 min:
```
data-scientist ⭐  # Phase 3 (main work)
research-analyst   # Phase 3 parallel
benchmarking-specialist  # Phase 4
→ Deliverable: Validated feature set v1
```

**Session 3 (Next day)** — 90 min:
```
python-engineer     # Phase 5
technical-writer    # Phase 5 parallel
/code-review + /checkpoint
→ Deliverable: Production-ready code + docs
```

**Total time**: Spread over 3 days, ~300 min of agent work
**Best for**: Iterating, feedback between sessions, parallel work with human review

---

## Quick Selection Matrix

| Scenario | Strategy | First Agent(s) |
|----------|----------|-----------------|
| "I have 5 hours" | A (Sequential) | `/prompt-engineer` → `academic-researcher` |
| "I need this done today" | B (Parallel) | `/prompt-engineer` → spawn Batch 1 |
| "I want to iterate" | C (Hybrid) | `/prompt-engineer` → `academic-researcher` + `Explore` |
| "Time is unlimited" | A (Sequential) | `/prompt-engineer` → follow the sequence |
| "Limited tokens" | B (Parallel batches) | Use batches, skip research-analyst if needed |

---

## Skills Invocation Points

### Before Starting
```bash
/prompt-engineer \
  "Optimize for data-scientist (primary) + research-analyst (secondary). \
   Task: Feature engineering for selector MLP. \
   Token budget: 10k main, 5k secondary."
```

### After Phase 3 (Optional)
```bash
/experiment-discipline \
  "Design validation plan for feature set. \
   Metrics: compute error, range correctness, correlation, separation power. \
   Success: features work on real data, no correlation >0.9."
```

### After Phase 5
```bash
/code-review \
  "Review the feature extraction code for quality, edge cases, maintainability."
```

### End of Session
```bash
/checkpoint \
  "Archive: feature engineering findings, agent outputs, design decisions, next steps."
```

---

## Critical Handoff Points (Between Agents)

### Handoff 1: academic-researcher → ml-engineer
**academic-researcher outputs**: FluxMem's 12 features (definitions + rationale)
**ml-engineer inputs**: These 12 features + task: "Interpret these in the context of the C-AIMMS pipeline"
**ml-engineer outputs**: Semantic understanding of what each feature represents

---

### Handoff 2: Explore → data-scientist
**Explore outputs**: Encoder specs (shapes, types, ranges, examples)
**data-scientist inputs**: These specs + task: "Design M ≤ 20 features from these outputs"
**data-scientist outputs**: Concrete feature list + pseudocode

---

### Handoff 3: data-scientist → benchmarking-specialist
**data-scientist outputs**: M features + pseudocode
**benchmarking-specialist inputs**: Pseudocode + task: "Test on real data (HyperMem, MemOCR, VectorDB)"
**benchmarking-specialist outputs**: Validation report (success/failures, ranges, correlations)

---

### Handoff 4: Validation outputs → python-engineer
**benchmarking-specialist outputs**: Validated feature list (with confidence levels)
**python-engineer inputs**: Feature list + pseudocode + task: "Implement as production-ready module"
**python-engineer outputs**: `features_caimms.py` (or similar) with tests

---

## Gotchas & Mitigations

### Gotcha 1: "Features are too similar to FluxMem"
**Mitigation**: Explicitly tell agents: "Start from zero. Don't inherit. Data-driven, not theory-driven."

### Gotcha 2: "Phases take longer than estimated"
**Mitigation**: Time estimates are ~±20%. Budget 1.5x if you're new to this.

### Gotcha 3: "Real data is messy (NaNs, outliers)"
**Mitigation**: Reserve 15 min of error-detective time to handle edge cases.

### Gotcha 4: "Features don't separate encoder types"
**Mitigation**: This is OK. Separation is nice-to-have, not required. Features just need to be meaningful.

### Gotcha 5: "Too many features (>20)"
**Mitigation**: Rank by variance or mutual information. Cut to top 12–16.

---

## Agent Cheat Sheet

### academic-researcher
- **Strengths**: Literature, paper reading, feature rationale
- **Limitations**: Not a coder; not great at testing
- **Give them**: FluxMem paper, selector.py, Phase 1 task
- **Expect**: Structured analysis + written explanation

### Explore
- **Strengths**: Fast code location, navigation
- **Limitations**: Reads excerpts, not whole files
- **Give them**: Codebase queries + encoder component names
- **Expect**: File locations, example usage, quick reference

### code-reviewer
- **Strengths**: Code quality, design patterns, implementation
- **Limitations**: Not domain-expert in feature engineering
- **Give them**: selector.py + features.py + task: "Validate Phase 1 output"
- **Expect**: Code quality assessment, improvement suggestions

### ml-engineer
- **Strengths**: ML pipeline semantics, representation theory
- **Limitations**: Not a coder; not great at literature
- **Give them**: Encoder specs + task: "What do these outputs mean?"
- **Expect**: Semantic interpretation, properties to measure

### data-scientist ⭐ (CORE)
- **Strengths**: Feature engineering, statistical thinking, data validation
- **Limitations**: Can get lost in details without clear input spec
- **Give them**: Encoder specs + FluxMem baseline + clear feature design task
- **Expect**: Concrete feature set, pseudocode, validation results

### research-analyst
- **Strengths**: Literature review, synthesis, best practices
- **Limitations**: Not hands-on implementation
- **Give them**: Feature engineering methods + task: "What's known about modality selection?"
- **Expect**: References, design principles, context

### benchmarking-specialist
- **Strengths**: Rigorous testing, statistics, validation plans
- **Limitations**: Not a domain expert
- **Give them**: Feature list + real data + task: "Test rigorously"
- **Expect**: Validation report, correlations, failure analysis

### error-detective
- **Strengths**: Edge cases, failure modes, robustness
- **Limitations**: Not a domain expert
- **Give them**: Feature functions + edge case list (empty graphs, zero embeddings, etc.)
- **Expect**: Failure modes + fixes

### python-engineer
- **Strengths**: Clean code, tests, documentation
- **Limitations**: Not a domain expert
- **Give them**: Validated features + pseudocode + task: "Implement as module"
- **Expect**: Production-ready code

### technical-writer
- **Strengths**: Documentation, explanation, structure
- **Limitations**: Can't validate technical correctness
- **Give them**: All Phase outputs + task: "Document the feature engineering story"
- **Expect**: Complete documentation package

---

## Estimated Token Usage (Claude 3.5 Sonnet)

| Agent | Tokens In | Tokens Out | Total |
|-------|-----------|------------|-------|
| `/prompt-engineer` | 3k | 2k | 5k |
| academic-researcher | 4k | 2k | 6k |
| Explore | 2k | 2k | 4k |
| code-reviewer | 3k | 1k | 4k |
| ml-engineer | 3k | 2k | 5k |
| data-scientist ⭐ | 5k | 5k | 10k |
| research-analyst | 3k | 3k | 6k |
| benchmarking-specialist | 4k | 3k | 7k |
| error-detective | 2k | 1k | 3k |
| python-engineer | 4k | 4k | 8k |
| technical-writer | 5k | 3k | 8k |
| `/code-review` | 3k | 2k | 5k |
| `/checkpoint` | 2k | 1k | 3k |
| **TOTAL** | ~45k | ~32k | **~77k** |

**Notes**:
- Parallel agents don't double token cost (they don't read each other's outputs, just handoff)
- Haiku would be ~60% of above; Opus would be similar or slightly higher
- Optimize main prompt before sharing → saves ~2k tokens per agent

---

## The Golden Rule

✅ **DO**:
- Start with `/prompt-engineer` to optimize for your agents
- Follow the 5-phase structure (don't skip phases)
- Keep agent outputs as structured handoffs (don't make agents re-discover)
- Use checkpoints to track progress
- Validate on real data (Phase 4)

❌ **DON'T**:
- Skip Phase 1 or 2 (foundation matters)
- Have more than 2 agents in parallel (gets chaotic)
- Inherit constraints from ADRs (explicit non-bias clause)
- Accept "looks good" without testing on data
- Create >20 features without justification

---

## Ready? Pick Your Path

### **Quick Start (Today)**
1. Read: `QUICK_START_GUIDE.md` (5 min)
2. Run: `/prompt-engineer` on main prompt (10 min)
3. Start: `academic-researcher` + `Explore` (90 min)
4. Total: ~2 hours today

### **Full Immersion (This Week)**
1. Read: This card + `agents_skills_recommendation.md` (20 min)
2. Run: `/prompt-engineer` (10 min)
3. Follow: **Strategy A (Sequential)** or **Strategy B (Parallel)**
4. Total: ~5 hours (or 4.5 hours for parallel)

### **Incremental (Multi-Session)**
1. Run: `/prompt-engineer` (10 min)
2. Follow: **Strategy C (Hybrid)** — 3 sessions, ~100 min each
3. Iterate based on feedback between sessions

---

**Good luck!** 🚀

