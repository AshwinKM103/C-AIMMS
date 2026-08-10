---
name: explain-modeling-choices-in-depth
description: "Wants substantive explanation of the reasoning behind each modeling/approximation choice, not just a flag that one was made"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 06c5580f-af8e-49c2-89bf-e907a31a27da
  modified: 2026-08-10T04:43:04.247Z
---

When a plan flags something as "a modeling choice" or "an approximation," expect to be asked to
**explain it in depth** — what the alternatives were, what behavioral difference each produces,
and what it costs downstream. Flagging alone is not enough.

**Why:** reviewing the ADASTORE plan (2026-08-10), the user picked out nearly every hedged item
(heuristic NER vs. real NER; `U(e)` vs. `u(m)` as the promotion gate; recency appearing both
inside the utility score and as a separate threshold; LRU vs. FIFO on STIM) and asked "explain
what this means / why." Several of those pushbacks found genuine weak choices — the
capitalization-regex NER was replaced with spaCy behind a Protocol as a direct result.

**How to apply:** when writing a plan, pre-empt this — give the alternatives table and the
consequence of each choice inline rather than deferring to review. A cheap approximation is fine
when justified (e.g. FluxMem App. C explicitly argues features should be lightweight), but say
*which* argument justifies it and how far it extends. Related:
[[resolve-paper-ambiguities-dont-hedge]].
