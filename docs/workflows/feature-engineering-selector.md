# Selector MLP Feature Engineering Workflow

**Status**: Ready to use
**Location**: `docs/workflows/feature-engineering-selector/`
**Created**: 2026-08-14

## Output Documents

| File                                                                                          | Purpose                                         | Read Time |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------- |
| [`FEATURE_ENGINEERING_REPORT.md`](feature-engineering-selector/FEATURE_ENGINEERING_REPORT.md) | Feature analysis & design decisions             | 30 min    |
| [`FEATURE_REFERENCE.md`](feature-engineering-selector/FEATURE_REFERENCE.md)                   | 15-feature specification & validation           | 20 min    |
| [`DESIGN_RATIONALE.md`](feature-engineering-selector/DESIGN_RATIONALE.md)                     | Architecture & token-budget accounting          | 20 min    |
| [`DEPLOYMENT_GUIDE.md`](feature-engineering-selector/DEPLOYMENT_GUIDE.md)                     | Integration & usage examples                    | 15 min    |
| [`phase3_output.md`](feature-engineering-selector/phase3_output.md)                           | Complete feature definitions & validation gates | 40 min    |
| [`INDEX.md`](feature-engineering-selector/INDEX.md)                                           | Document map & read paths                       | 10 min    |

## What This Is

A complete feature engineering system for designing a new selector MLP that chooses between three encodings (VS, HG, VC) in C-AIMMS. The system includes:

- **Detailed prompt** with explicit non-bias requirement (bottom-up feature design, not inherited from ADRs)
- **10 recommended agents** with clear task assignments and effort estimates
- **3 execution strategies**: Sequential (5h), Parallel (4.5h), Hybrid (flexible)
- **Built-in checkpoints** to verify each phase
- **Production-ready structure** for implementation

## How to Use These Documents

Start with **FEATURE_ENGINEERING_REPORT.md** for a complete overview, then drill into specific sections:

- **Understanding the feature set** → `FEATURE_REFERENCE.md` (all 15 features with formulas & ranges)
- **Deployment & integration** → `DEPLOYMENT_GUIDE.md` (code examples, validation gates)
- **Architecture decisions** → `DESIGN_RATIONALE.md` (token-budget accounting, why these features)
- **Detailed specifications** → `phase3_output.md` (complete validation, bounded/unbounded features, clamping rules)

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

## Next Steps

1. **Read** `FEATURE_ENGINEERING_REPORT.md` — complete overview & methodology
2. **Reference** `FEATURE_REFERENCE.md` — feature definitions & validation gates
3. **Implement** using `DEPLOYMENT_GUIDE.md` — code patterns & testing

## Questions?

Refer to:

- **"What features are implemented?"** → `FEATURE_REFERENCE.md` → "Feature Catalog"
- **"How do I use this in code?"** → `DEPLOYMENT_GUIDE.md` → "Integration"
- **"Why these features?"** → `DESIGN_RATIONALE.md` → full rationale
- **"What are the bounded/unbounded features?"** → `phase3_output.md` → "Feature Properties"

## All Files

```
docs/workflows/feature-engineering-selector/
├── INDEX.md                                    [Document map & read paths]
├── FEATURE_ENGINEERING_REPORT.md               [Complete analysis & methodology]
├── FEATURE_REFERENCE.md                        [Feature catalog & specifications]
├── DESIGN_RATIONALE.md                         [Architecture & design decisions]
├── DEPLOYMENT_GUIDE.md                         [Integration & code examples]
├── phase3_output.md                            [Detailed specs & validation gates]
├── phase1_output.md                            [Baseline analysis]
├── phase2_output.md                            [Encoder characterization]
└── SEQUENTIAL_EXECUTION_PLAN.md                [Phase-by-phase execution record]
```

**Total**: 130 KB | **Read time**: 10 min (reference) to 90 min (complete)

---

## Related Documentation

- **ADRs**: See `docs/adr/0004, 0007, 0010, 0011, 0014` for architectural context (reference only, not bias)
- **Code**: `fluxmem/selector.py`, `HyperMem/`, `MemOCR/`, `hetrep/`
- **Workflows**: `docs/workflows/new-experiment.md`, `debug-a-metric.md`, `review-and-merge.md`

---

**Ready?** Start with `README.md` in this directory!
