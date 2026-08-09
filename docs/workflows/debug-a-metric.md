# Workflow: A metric moved unexpectedly

Work down this list in order. Steps 1–2 explain most surprises and cost a minute each; theorising
first wastes hours.

## 1. Did the store get wiped?
The single most common cause of "recall collapsed for no reason". Constructing the Qdrant client
with `on_disk=False` (the default) against an existing `path` runs `shutil.rmtree` on it first.

```bash
ls -la /mnt/ssd/users/durgesh/qdrant/<run>/     # empty or newer than the run?
```

## 2. Did the embedding config drift?
- `EMBEDDING_MODEL_PATH` in `.env` changed?
- Model output width still equals `embedding_model_dims` (1024)?
- Collection recreated after a model change? Reusing a collection built with different embeddings
  gives quietly terrible recall rather than an error.

## 3. Bisect on the metric, not on a crash
```bash
/bisect      # debug-session plugin
```
Mark "good" and "bad" by **metric value against a threshold**, not by pass/fail. This is the tool
that answers "which commit moved it".

## 4. Review the metric-computation path
```bash
/logic-review <file>     # logic-lens
```
This is the failure mode that matters most in a research repo: code that runs clean, passes ruff
and mypy, and reports the wrong number. Logic-lens traces Premises → Trace → Divergence → Trigger
→ Remedy and targets exactly the classes linters miss (state mutation, control-flow escape,
callee-contract mismatch, boundary blindspots).

## 5. Check whether a baseline changed
Anything touched under `memory_toolkits/memories/layers/baselines/**` changes a *comparison point*.
Your number may be fine and the baseline moved.

## 6. Only now, hypothesise
Two competing hypotheses minimum, each with a cheap discriminating test.
See `.claude/rules/evidence-discipline.md`.

## Related
`/bisect` · `/logic-review` · `error-detective` agent · `experiment-discipline` skill
