---
name: experiment-discipline
description: How C-AIMMS records, compares, and reproduces experiment runs. Use when starting a run, logging metrics, comparing configurations, investigating a metric that moved, or writing up results for the paper. Wraps the /track and /compare conventions with this project's storage and benchmark specifics.
---

# Experiment Discipline

`/track` and `/compare` (experiment-tracker) provide the shape. This skill fills in what those
generic commands cannot know: our benchmarks, our storage, our failure modes.

**These are conventions, not a tracking server.** Nothing here talks to MLflow or W&B. If you are
already using one, keep it — this layer exists so that a run is reconstructable from the repo alone.

## Where runs live

FluxMem has no experiment harness in-repo (`docs/adr/0001-isolate-fluxmem.md` — the
LoCoMo/LongMemEval/EgoLife harnesses were LightMem-core/EM²Mem-only and were deleted along
with those subsystems; building a new `experiments/fluxmem/` was explicitly deferred, not
built). Evaluation stays ad hoc via `FluxMem.md`'s inline usage examples until that changes.
Still write a run record for anything you'd want to defend later: `experiments/runs/fluxmem/
<YYYY-MM-DD>-<slug>.md` — committed, diffable, reviewable in a PR. Result payloads (JSON/CSV)
sit beside them. FAISS index files never go in git; they live under
`/mnt/ssd/users/durgesh/` and are referenced by path.

## A run record must contain

Anything missing here means the number cannot be defended in review.

```markdown
# <benchmark> — <slug>
Date: YYYY-MM-DD   Commit: <sha>   Author:

## Hypothesis
What this run tests, and what result would falsify it.

## Configuration
- Subsystem: fluxmem | baseline:<name>
- LLM / embedder backend: openai (+ model id) — FluxMem's interfaces are API-based, not local
- Vector store: FAISS index path=<...>
- Retrieval fusion: bm25_weight / dense_weight (see `.claude/rules/storage-invariants.md`)
- Seed(s): <...>
- Dataset version / split: <...>

## Environment
conda env, commit sha of any modified baseline

## Metrics
| metric | value | n | notes |

## Result
Confirmed / falsified / inconclusive — and why.
```

## Non-negotiables

1. **Log the seed.** Every run. A number without a seed is an anecdote.
2. **Log the dataset version.** `/compare` is only valid across runs on the same data.
3. **Never overwrite a previous run record.** New file, new slug. Supersede, don't edit —
   the same discipline as ADRs.
4. **One variable at a time**, or the comparison attributes the delta to the wrong thing.
5. **Baselines are load-bearing.** If you touched a `baseline:<name>` comparison point, say so
   explicitly — every prior number in that column is now suspect.
6. **Single runs do not support conclusions.** Report n, and report variance when you have it.

## When a metric moves unexpectedly

Do this before theorising (see `.claude/rules/evidence-discipline.md`):

1. Did `bm25_weight` / `dense_weight` change without being logged? Fusion-weight drift is
   the classic cause of "recall changed for no reason" in FluxMem's retriever.
2. Did the FAISS index get rebuilt from a different dataset version without noting it?
3. `/bisect` (debug-session) across commits — bisect on the *metric*, not on a crash.
4. `/logic-review` (logic-lens) the metric-computation path. This is the failure mode that
   matters most here: code that runs clean and reports the wrong number. Linters and mypy
   cannot see it.
5. Only then form two competing hypotheses and test them.

## Benchmarks and external baselines

No benchmark harness lives in-repo today (see "Where runs live" above). Two external systems
publish on `longmemeval`/`locomo` and are tracked as comparison points in
`docs/related-work-agent-memory.md`:
Cortex (LongMemEval 98.2% R@10 / 0.915 MRR; also LoCoMo, BEAM) and axme-code (LongMemEval 89.2%
E2E / 97.8% R@5). Cite their numbers as *reported by them* — we have not reproduced either.
