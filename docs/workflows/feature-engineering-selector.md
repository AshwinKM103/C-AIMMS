# Selector MLP Feature Engineering Workflow

**Status**: Ready to use
**Location**: `docs/workflows/feature-engineering-selector/`
**Created**: 2026-08-14

## Quick Links

| File                                                                                                            | Purpose                    | Read Time |
| --------------------------------------------------------------------------------------------------------------- | -------------------------- | --------- |
| [`README.md`](feature-engineering-selector/README.md)                                                           | Overview & navigation      | 5 min     |
| [`QUICK_START_GUIDE.md`](feature-engineering-selector/QUICK_START_GUIDE.md)                                     | How to start + checkpoints | 20 min    |
| [`selector_feature_engineering_prompt.md`](feature-engineering-selector/selector_feature_engineering_prompt.md) | Main prompt (5 phases)     | 20 min    |
| [`agents_skills_recommendation.md`](feature-engineering-selector/agents_skills_recommendation.md)               | Agent lineup & strategies  | 25 min    |
| [`AGENT_LINEUP_CARD.md`](feature-engineering-selector/AGENT_LINEUP_CARD.md)                                     | Quick agent reference      | 15 min    |
| [`INDEX.md`](feature-engineering-selector/INDEX.md)                                                             | Complete map & paths       | 20 min    |

## What This Is

A complete feature engineering system for designing a new selector MLP that chooses between three encodings (VS, HG, VC) in C-AIMMS. The system includes:

- **Detailed prompt** with explicit non-bias requirement (bottom-up feature design, not inherited from ADRs)
- **10 recommended agents** with clear task assignments and effort estimates
- **3 execution strategies**: Sequential (5h), Parallel (4.5h), Hybrid (flexible)
- **Built-in checkpoints** to verify each phase
- **Production-ready structure** for implementation

## Start Here

### Option 1: Quick Start (5 minutes)

```bash
# Read the README
cat docs/workflows/feature-engineering-selector/README.md

# Then read the quick start guide
cat docs/workflows/feature-engineering-selector/QUICK_START_GUIDE.md
```

### Option 2: Full Deep Dive (60 minutes)

Read all files in order:

1. `README.md`
2. `QUICK_START_GUIDE.md`
3. `selector_feature_engineering_prompt.md` (main prompt for agents)
4. `agents_skills_recommendation.md` (agent assignments)
5. `AGENT_LINEUP_CARD.md` (quick reference)
6. `INDEX.md` (complete map)

### Option 3: Start Immediately

1. Read `QUICK_START_GUIDE.md` section "How to Get Started"
2. Run `/prompt-engineer` on `selector_feature_engineering_prompt.md`
3. Pick an agent from `AGENT_LINEUP_CARD.md` → "Quick Selection Matrix"
4. Spawn agent with the main prompt

## The Task

Design a new feature set (M ≤ 20 features) for a selector MLP that chooses between:

- **VS** (Vector Semantic): dense embeddings from VectorDB
- **HG** (Hierarchical Graph): structured graphs from HyperMem
- **VC** (Visual/Conceptual): visual/conceptual features from MemOCR

The feature set must:

- Derive from encoder **outputs**, not prior architectural decisions
- Be **modality-agnostic** (works for all three encodings)
- Be **validated** on real data
- Be **interpretable** (not just tuned)

## Key Constraints

✅ **Explicit non-bias requirement**: Derive bottom-up from data, don't inherit from ADRs
✅ **Relative paths**: All code references use repo-relative paths
✅ **Real data validation**: Test on actual HyperMem, MemOCR, VectorDB outputs
✅ **Feature count**: ≤ 20 features (target 12–16 to match FluxMem baseline)

## Recommended Execution

### Path A: Sequential (Best for Accuracy)

```
academic-researcher (FluxMem baseline)
  ↓
Explore (encoder outputs)
  ↓
data-scientist ⭐ (feature design)
  ↓
benchmarking-specialist (validation)
  ↓
python-engineer (implementation)
  ↓
/code-review + /checkpoint (polish)
```

**Time**: ~5 hours

### Path B: Parallel (Fastest)

```
Batch 1: academic-researcher || Explore
Batch 2: code-reviewer || ml-engineer
Batch 3: data-scientist ⭐ || research-analyst
Batch 4: benchmarking-specialist || error-detective
Batch 5: python-engineer || technical-writer
```

**Time**: ~4.5 hours

### Path C: Hybrid (Flexible)

```
Session 1: Foundation (academic-researcher + Explore)
Session 2: Core + Validation (data-scientist + benchmarking-specialist)
Session 3: Implementation (python-engineer + technical-writer)
```

**Time**: Spread over 3 sessions, ~300 min agent work

See `AGENT_LINEUP_CARD.md` for detailed command sequences.

## Next Steps

1. **Read** `README.md` (5 min orientation)
2. **Choose** an execution strategy
3. **Run** `/prompt-engineer` on the main prompt
4. **Spawn** your first agent
5. **Track** progress with built-in checkpoints
6. **Archive** findings with `/checkpoint` at the end

## Questions?

Refer to:

- **"What is this task?"** → `INDEX.md` → "Overview"
- **"Which agent to use?"** → `AGENT_LINEUP_CARD.md` → "Quick Selection Matrix"
- **"How do I start?"** → `QUICK_START_GUIDE.md` → "How to Get Started"
- **"What if I get stuck?"** → `QUICK_START_GUIDE.md` → "Common Q&A"
- **"What are the phases?"** → `selector_feature_engineering_prompt.md` → "Phase 1-5"

## All Files

```
docs/workflows/feature-engineering-selector/
├── README.md                                    [Start here: overview]
├── QUICK_START_GUIDE.md                        [How to get going]
├── selector_feature_engineering_prompt.md      [Main prompt for agents]
├── agents_skills_recommendation.md             [Agent lineup & strategies]
├── AGENT_LINEUP_CARD.md                        [Quick agent reference]
└── INDEX.md                                    [Complete map]
```

**Total**: 88 KB | **Read time**: 5 min (quick) to 60 min (complete)

---

## Related Documentation

- **ADRs**: See `docs/adr/0004, 0007, 0010, 0011, 0014` for architectural context (reference only, not bias)
- **Code**: `fluxmem/selector.py`, `HyperMem/`, `MemOCR/`, `hetrep/`
- **Workflows**: `docs/workflows/new-experiment.md`, `debug-a-metric.md`, `review-and-merge.md`

---

**Ready?** Start with `README.md` in this directory!
