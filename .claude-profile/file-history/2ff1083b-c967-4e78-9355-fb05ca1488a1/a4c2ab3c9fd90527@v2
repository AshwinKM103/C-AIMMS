# 0002. Format-selector reward design: concretizing Judge(·) and MemUtil(·)

## Status

Accepted — 2026-08-10. Gates `fluxmem/selector.py` and `fluxmem/supervision.py`
(plan Step 6); no code in either file lands before this ADR.

## Context

COLM Sec 1.3.3 specifies offline supervision for the format-adaptive selector: for a
held-out set, compute per-format rewards `r_HG_j, r_VC_j, r_VS_j` by "evaluat[ing] the
downstream response quality and memory utilization effectiveness of all three formats,"
then set the training target `f*_j = argmax_f r_f_j`. COLM does not say how to compute
either quantity.

FluxMem (the cited source for this section) is more concrete but only partially closes
the gap. Eq. 9 / App. D Alg. 2 give the reward's _shape_:

```
r(s) = lambda_judge * r_judge(s) + lambda_mem * r_mem(s)      lambda = 0.7 / 0.3
y = argmax_s r(s)
```

This is a cited fact, not a decision — `lambda_judge=0.7`, `lambda_mem=0.3` land in
`fluxmem/config.py`'s `RewardConfig` as defaults, config-exposed but not re-derived here.
What FluxMem does _not_ specify is what `Judge(·)` and `MemUtil(·)` compute internally —
that concretization is this ADR's actual scope, per `.claude/rules/evidence-discipline.md`:
training data downstream of the selector depends on this choice, so it must be recorded
as a modeling choice, not presented as a paper-specified fact.

A second, scope-defining constraint: this package does not implement HETREP, ITERRET, or
WORKMEM (see `fluxmem/interfaces.py`'s `EpisodeProducer` boundary and the plan's Out of
Scope section). There is no real generation pipeline and no real retrieval controller in
this repo to produce an actual "downstream response" or "retrieved evidence" from a live
agent loop. Both reward components must therefore be defined over inputs a `runner`
object supplies (real or fake), never assume a live LLM call is available, and be fully
computable against `fluxmem/interfaces.py`'s `StubEpisodeProducer` output in tests.

## Decision

### `r_judge`: exact-match / F1 against a reference answer

Two candidates were compared:

| Option                       | Description                                                                     | Tradeoff                                                                                                                                                                                                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (a) Exact-match / F1         | Token-overlap F1 between a per-format simulated response and a reference answer | Cheap, deterministic, fully offline-testable; brittle on free-form or paraphrased text                                                                                                                                                                          |
| (b) LLM-as-judge, 0–1 rubric | An LLM scores response quality against a rubric                                 | Matches FluxMem's own GPT-4.1/Qwen3 evaluation setup and captures paraphrase/semantic quality; non-deterministic, requires one API call per (episode x format) — 3x cost per labeled example, and makes every reward computation an external-network dependency |

**Decision: (a), exact-match/F1.** Two reasons beyond the general cost/determinism
tradeoff: first, this package has no real generation pipeline to produce a response for
an LLM judge to score in the first place (ITERRET/WORKMEM are non-goals) — so "LLM judge"
would need a second LLM to _also_ fabricate the response being judged, doubling the
external dependency for no offline-test benefit. Second,
`.claude/rules/testing.md` requires the default test loop to stay hermetic; an
LLM-as-judge default would make `fluxmem/supervision.py`'s core training-label path
untestable without live API credentials. F1 against a reference keeps the entire
offline-supervision path deterministic and testable with fakes.

This is the same shape of tradeoff Step 4 resolved for NER (real spaCy over an LLM
extractor) and for the same underlying reason: the paper's own emphasis on avoiding
"additional reasoning overhead" applies to the control layer generally, not just
`fluxmem/features.py`. **Known gap, not a silent one**: LLM-as-judge is the paper's own
evaluation setup and is a strictly better match to "response quality" as a concept; if
offline supervision quality is ever found wanting, this is the first knob to revisit, and
the `JudgeReward` Protocol below is designed so that revisit doesn't touch `selector.py`.

### `r_mem`: retrieval hit-rate normalized by retrieved token budget

| Option                      | Description                                                        | Tradeoff                                                                                                                                              |
| --------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| (a) Hit-rate                | Fraction of gold evidence spans surfaced by retrieval              | Simple, no extra dependency; ignores _how much_ was retrieved to get there                                                                            |
| (b) Hit-rate / token budget | Hit-rate normalized by whitespace-token count of retrieved content | Closer to "memory utilization _effectiveness_" (COLM's literal phrase — effectiveness implies a cost term, not just recall); adds a tokenization step |

**Decision: (b), hit-rate normalized by retrieved token budget.** COLM's phrase is
"memory utilization effectiveness," not "retrieval accuracy" — effectiveness reads as
recall-per-cost, which (a) does not capture: a format that retrieves everything gets a
perfect hit-rate under (a) regardless of how much irrelevant content it drags along,
which is the opposite of what a memory _format_ comparison should reward (HG/VC/VS
differ precisely in how compactly they can represent the same information). The "extra
dependency" in (b) is nominal: token counting uses the same whitespace-split approach
`fluxmem/features.py`'s `token_length` feature already uses, not a model-specific
subword tokenizer — so this does not add a new dependency to the project.

### Design for both: Protocols with deterministic fakes

```python
class JudgeReward(Protocol):
    def score(self, response: str, reference: str) -> float: ...

class MemUtilReward(Protocol):
    def score(self, retrieved: list[str], gold_evidence: list[str]) -> float: ...
```

`fluxmem/supervision.py` (Step 6) implements the exact-match/F1 and hit-rate/token-budget
concretizations above as the default `JudgeReward`/`MemUtilReward`, but every call site
takes the Protocol, not the concrete class — so no test in `fluxmem/` ever needs a live
LLM call or a real retrieval index, and the Step 6 gap noted above (LLM-as-judge) is a
drop-in replacement rather than a rewrite.

`RewardConfig(lambda_judge=0.7, lambda_mem=0.3)` lives in `fluxmem/config.py`, added in
Step 6. `lambda` stays config-exposed (per FluxMem's own framing as a tunable), but the
_shape_ of `r(s)` (linear combination, `argmax` selection) is cited, not reopened.

## Consequences

- `fluxmem/selector.py`'s training labels (Step 6) are only as good as F1-against-reference
  and hit-rate/token-budget as quality proxies; both are strictly cheaper and more brittle
  than the paper's own LLM-judge-based evaluation. This is an explicit, documented
  approximation, not an oversight.
- No `fluxmem` test ever makes a network call or requires API credentials — the entire
  offline-supervision path (Step 6) is testable with `FakeEntityExtractor`-style fakes
  for `JudgeReward`/`MemUtilReward`, consistent with `.claude/rules/testing.md`.
- If LLM-as-judge or a real retrieval-based `r_mem` are adopted later (e.g. once ITERRET
  exists), only the `JudgeReward`/`MemUtilReward` implementations swap in — `selector.py`
  and the `RewardConfig`/`per_format_rewards` call sites are unaffected, because they
  depend on the Protocols, not the concrete classes.
- Token-budget normalization for `r_mem` reuses the whitespace-token convention from
  `fluxmem/features.py`, so no new dependency (e.g. a subword tokenizer) enters the
  project for this decision.
