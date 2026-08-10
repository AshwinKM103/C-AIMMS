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

`experiments/{longmemeval,locomo,egolife}/` holds the harnesses. Run records go in
`experiments/runs/<benchmark>/<YYYY-MM-DD>-<slug>.md` — committed, diffable, reviewable in a PR.
Result payloads (JSON/CSV) sit beside them. Model weights, `*.pkl`, and Qdrant stores never go in
git; they live under `/mnt/ssd/users/durgesh/` and are referenced by path.

## A run record must contain

Anything missing here means the number cannot be defended in review.

```markdown
# <benchmark> — <slug>
Date: YYYY-MM-DD   Commit: <sha>   Author:

## Hypothesis
What this run tests, and what result would falsify it.

## Configuration
- Subsystem: lightmem | fluxmem | em2mem | baseline:<name>
- Memory manager: openai | deepseek | ollama | vllm | transformers  (+ model id)
- Embedding model: <path from EMBEDDING_MODEL_PATH>, dims: 1024
- Vector store: qdrant path=<...>, on_disk=<bool>, collection=<...>, distance=COSINE
- Pre-compressor: llmlingua_2 | entropy | none
- Topic segmenter: <...>
- FluxMem only: bm25_weight / dense_weight
- Seed(s): <...>
- Dataset version / split: <...>

## Environment
conda env, torch/transformers versions, GPU, commit sha of any modified baseline

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
5. **Baselines are load-bearing.** If you touched anything under
   `memory_toolkits/memories/layers/baselines/**`, say so explicitly — you changed a comparison
   point, and every prior number in that column is now suspect.
6. **Single runs do not support conclusions.** Report n, and report variance when you have it.

## When a metric moves unexpectedly

Do this before theorising (see `.claude/rules/evidence-discipline.md`):

1. Did the store get wiped? `on_disk=False` on an existing path silently `rmtree`s it —
   the classic cause of "recall collapsed for no reason". See the `qdrant-ops` skill.
2. Did the embedding model or its dims change without recreating the collection?
3. `/bisect` (debug-session) across commits — bisect on the *metric*, not on a crash.
4. `/logic-review` (logic-lens) the metric-computation path. This is the failure mode that
   matters most here: code that runs clean and reports the wrong number. Linters and mypy
   cannot see it.
5. Only then form two competing hypotheses and test them.

## Benchmarks and external baselines

`longmemeval`, `locomo`, `egolife` are the harnesses in-repo. Two external systems publish on the
same benchmarks and are tracked as comparison points in `docs/related-work-agent-memory.md`:
Cortex (LongMemEval 98.2% R@10 / 0.915 MRR; also LoCoMo, BEAM) and axme-code (LongMemEval 89.2%
E2E / 97.8% R@5). Cite their numbers as *reported by them* — we have not reproduced either.
