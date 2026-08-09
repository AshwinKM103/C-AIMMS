# Workflow: Start a new experiment

**Loop:** Question → Collect → Synthesize → Close (research, not development).

## 1. Write the hypothesis before the code
State what the run tests and **what result would falsify it**. If no result could falsify it, you
have a demo, not an experiment. Put it at the top of the run record.

## 2. Create the run record
`experiments/runs/<benchmark>/<YYYY-MM-DD>-<slug>.md` from the template in the
`experiment-discipline` skill. Fill Configuration and Environment *before* running — reconstructing
them afterwards is how details get lost.

## 3. Pick the storage path
```
/mnt/ssd/users/durgesh/qdrant/<benchmark>-<slug>/
```
Unique per run. `on_disk=True`. Never `/tmp/qdrant`, never under `/home`.
If you must reuse a path, back it up first — see the `qdrant-ops` skill for why.

## 4. Change one variable
One. If you change the embedding model *and* the compressor, the delta is unattributable and the
run is wasted compute.

## 5. Run, then record
Metrics with `n`. Seeds. Dataset version. Paste real output — do not summarise numbers you did
not read.

## 6. Close it
- Confirmed / falsified / inconclusive, and why.
- If the finding changes the architecture, that is an ADR: `/adr`.
- `/checkpoint` before switching tasks.
- Commit the record: `exp: <benchmark> <slug> — <one-line result>`

## Related
`experiment-discipline` skill · `qdrant-ops` skill · `/track`, `/compare` · `.claude/rules/evidence-discipline.md`
