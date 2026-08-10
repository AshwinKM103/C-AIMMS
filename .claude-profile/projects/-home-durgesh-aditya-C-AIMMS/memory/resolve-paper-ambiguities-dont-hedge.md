---
name: resolve-paper-ambiguities-dont-hedge
description: "When implementing from papers, follow the paper being implemented on conflicts; never park a resolved ambiguity behind a config flag"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 06c5580f-af8e-49c2-89bf-e907a31a27da
  modified: 2026-08-10T04:42:50.872Z
---

When implementing a spec from a paper: **where two papers conflict, follow the paper being
implemented** (for C-AIMMS work that is the COLM PDF; cited sources like the FluxMem paper fill
its *silences* only). And once an ambiguity is resolved, **implement the resolution** — do not
ship a config flag offering both readings, and do not ship an omitted feature as "disabled by
default." Absent beats dead.

**Why:** on the ADASTORE plan (2026-08-10) I had proposed a `promotion_gate: "utility" | "usage"`
flag and a `tau_c` confidence gate "disabled by default since COLM omits it." The user pushed
back: that is hedging, not deciding. A flag that re-opens a settled question is dead config that
misleads the next reader about what the spec actually says.

**How to apply:** state the conflict explicitly, name which paper wins and why, implement that one
path, and document the rejected reading in a docstring or ADR — not in `config.py`. Config is for
tuning, not for hedging. Related: [[explain-modeling-choices-in-depth]].
