# 0004. HyperMem integration and gap resolution strategy

## Status

Proposed — 2026-08-12 (revised same day, second pass). Supersedes the first pass of this ADR
in place (not a new numbered ADR — the philosophy underneath every decision changed, so the prior
text is replaced rather than appended to). Complements 0003 (MemOCR/FluxMem: adapter-in pattern)
by applying the same component-adapter discipline, but under a different retrieval stance than
0003 assumed.

## Context

**C-AIMMS composes memory components from multiple papers** — segmentation, encoding, storage,
retrieval — each independently swappable and eventually ablatable. HyperMem's retrieval stage
(`stage4_hypergraph_retrieval.py`) is the active retrieval implementation for this project. The
system runs it as a baseline, not as deferred or reference material. Gaps in the active retrieval
component are bugs to fix, not theoretical issues in code being read but not run.

**FluxMem's architecture unchanged:** `fluxmem/ltsm.py` contains no retriever (STIM/MTEM/LTSM
handles encoding/storage/summarization only). HyperMem runs as a standalone harness alongside
FluxMem, maintaining the adapter boundary from ADR 0003 — retrieval code is not inside FluxMem's
core module, but it is a first-class component of the overall memory system. `CLAUDE.md` has
been updated to reflect this stance (see separate CLAUDE.md PR).

**Pickle discipline** (`.claude/rules/storage-invariants.md`) is unaffected by the philosophy
change: any HyperMem artifact reused for C-AIMMS experiments (embeddings, BM25 indices) must not
be pickled and committed. `stage3_hypergraph_index.py` still builds indices in memory/disk-cache
per-run against `data/locomo10.json`; this stays disposable.

HyperMem (`HyperMem/hypermem/`, Yue et al., 2025, arXiv:2604.08256) is a vendored three-level
hypergraph memory system (Topic → Episode → Fact nodes, hyperedges linking each level), six
stages (`stage1_memory_extraction.py` through `stage6_eval.py`, `eval.py`, `config.py`). Line
counts and stage roles are unchanged from the prior pass:

| Stage file                        | Lines | Role                                         |
| --------------------------------- | ----- | -------------------------------------------- |
| `stage1_memory_extraction.py`     | 428   | raw dialogue → episode/fact extraction       |
| `stage2_hypergraph_extraction.py` | 1100  | hypergraph construction (topics, hyperedges) |
| `stage3_hypergraph_index.py`      | 1022  | BM25 + embedding index build                 |
| `stage4_hypergraph_retrieval.py`  | 1341  | retrieval + rerank                           |
| `stage5_response.py`              | 377   | answer generation                            |
| `stage6_eval.py`                  | 560   | LoCoMo scoring (BLEU/ROUGE/BERTScore/METEOR) |
| `eval.py`                         | 251   | orchestration/CLI                            |
| `config.py`                       | 185   | `ExperimentConfig`                           |

`EM-LLM` is vendored but not yet wired in. The dual-segmentation reconciliation from the first
pass (EM-LLM primary, HyperMem's LLM boundary detector arbitrates low-confidence regions) is
unchanged by this revision and is not repeated in full below — see the prior ADR text in git
history if the segmentation section is needed; nothing in this revision touches it.

## Decision

### Gap resolution (G1–G8), overriding the prior pass where noted

Each row below states what was re-investigated, what was found, and the new resolution. Where
the resolution changed from the first pass, the override reason is explicit.

#### G1 — topic aggregation skips retrieval-seeded candidate discovery (§3.3.1 / §3.2.2)

**Prior resolution:** accept as limitation, defer (retrieval component was being treated as
inert).

**New resolution: keep as a documented limitation, but re-scope the reason.** The paper's §3.2.2
describes comparing a current episode against existing topics via lexical+semantic similarity
(`C^E`) before falling to LLM batch matching; the code does O(n_topics) LLM matching directly.
This is not the same class of gap as G2/G4/G5 — it's an efficiency/cost optimization the paper
specifies and the code skips, not a correctness bug or a discarded signal. Fixing it means adding
a BM25/dense pre-filter stage ahead of the LLM matching loop in `stage2_hypergraph_extraction.py`,
which is real new code, not a parameter flip. Given the project is actively running HyperMem's
retrieval as baseline, this now sits in the "improve the baseline" bucket rather than "irrelevant
to a component we don't run" — but it's the lowest-ROI item in that bucket: it doesn't affect
answer quality, only extraction-time LLM cost, and there is no ablation question it unblocks.

**Confidence: Medium.** Paper is explicit about the intended design (§3.2.2, "compare `v^E_cur`
with `C^E`"); the code's substitution of pure LLM batch matching is a confirmed divergence, not
an ambiguity. Impact tier: **Low** (cost/efficiency only, not correctness or the ablation
roadmap).

#### G2 — new topics seeded from single episodes, conflating Case 1 (Initialization) vs. Case 2 (Topic Creation)

**Prior resolution:** fix in code, conditionally on stage2 being retained as baseline.

**New resolution: fix in code now, unconditionally.** Paper §3.2.2 defines three cases (Topic
Initialization when `C^E = ∅`, Topic Creation when `C^E ≠ ∅` but no match, Topic Update when a
match exists) — a documented decision tree the code under-implements by not branching on whether
an existing topic passes a similarity threshold. Since HyperMem is the active retrieval baseline,
not read-only reference, this is a correctness bug in running code, not a hypothetical fix gated
on a reuse decision that no longer applies.

**Confidence: Medium** (the fix itself — branch on similarity threshold before topic creation —
is straightforward, but choosing the right threshold value requires empirical tuning against
LoCoMo, which is where the medium confidence sits, not in whether the bug exists). Impact tier:
**Medium** (affects topic-graph structure, which downstream affects G1's candidate matching and
retrieval quality).

#### G3 — `should_wait` signal

**Investigation performed** (per this ADR's instruction, before deciding): read
`docs/cited-papers/HyperMem_Yue2025_arXiv2604.08256.pdf` §3.2.1 directly (`pdftotext -layout`,
verified locally rather than assumed) and re-read `extractors/conv_episode_extractor.py` end to
end, not just the `:176` slice the first pass looked at.

1. **Paper confirms `should_wait` is part of the published algorithm, not editorial confusion.**
   §3.2.1: "The detector outputs two signals: `should_end`... and `should_wait`, i.e., the event
   is still unfolding and requires further input." Figure 6's prompt template output schema is
   `{should_end: bool, should_wait: bool, confidence: float, topic_summary: ...}` — the LLM is
   explicitly asked for it (contradicting the first pass's claim "prompt never asks for it").
2. **The signal is computed and even locally consumed**, just not by the caller the first pass
   inspected. `conv_episode_extractor.py:349-352` reads `should_wait` off the boundary-detection
   result into `StatusResult(should_wait=should_wait)`, and `:389` branches on it for a debug log
   (`elif should_wait: logger.debug(...)`). The parser default (`data.get("should_wait", True)` at
   `:261`) is a safe fallback for a field the LLM is asked to return, not evidence of a field the
   prompt never requests.
3. **The actual gap:** `main/stage1_memory_extraction.py:176` reads only `result[0]`
   (the `Episode` or `None`) and discards `result[1]` (`StatusResult`, carrying `should_wait`)
   entirely. Stage1's buffer-accumulation logic (`history_raw_data_list.append(...)`) already runs
   unconditionally whenever `episode_result is None`, which happens to cover both
   "`should_wait=True`, still unfolding" and "`should_end=False` for other reasons" — so the
   distinct signal the paper defines collapses into a single "keep buffering" path at the call
   site, even though the extractor itself computes and briefly acts on the finer-grained signal.

**New resolution: fix in code — wire `should_wait` from `StatusResult` into `stage1`'s control
flow**, distinguishing "genuinely still unfolding, keep buffering as-is" from other reasons
`episode_result` came back `None` (e.g., a message the paper's boundary detector considered
irrelevant, or a retry/failure path). At minimum, log or gate on the distinction so C-AIMMS's own
extraction stats (episodes/hour, buffer-wait duration) reflect what the paper's algorithm actually
signals, rather than an approximation stage1 currently makes.

**Confidence: High** the paper specifies this — the section quote above is unambiguous.
**Confidence: Medium** on the specific fix (the exact behavioral difference `should_wait=False`
vs. `should_wait=True` should produce at the stage1 call site isn't fully specified by the paper
beyond "requires further input," so the implementation needs a judgment call on what "further
input" changes about stage1's buffering cadence). Impact tier: **Low** — this is a segmentation
diagnostics/fidelity fix, not something that changes retrieval-time behavior or ranking; it
belongs with the other small correctness fixes, not the high-impact retrieval changes below.

#### G4 — hyperedge weights `w^E`/`w^F` computed via LLM call but discarded at retrieval

**Prior resolution:** accept as limitation, do not restore (retrieval was out of scope).

**New resolution: OVERRIDE — implement weight fusion at retrieval time.** Re-investigated the
exact discard points; confirmed at all four sites the prior pass's fidelity analysis flagged:

- `stage4_hypergraph_retrieval.py:683` — "Use BM25 score directly, without fusing with hyperedge
  weights" (episode filtering from topics, via `get_connected_episodes_and_weights`, which itself
  computes `episode_weights` at lines 407-435 and then the caller ignores them).
- `:706` — vector-similarity episode results, same discard, same comment pattern.
- `:798` — fact filtering from episodes via `get_connected_facts_weights_relations` (443-470),
  same discard for BM25 fact results.
- `:821` — vector-similarity fact results, same discard.

The weights (`we,v^E ∈ [0,1]`, paper §3.2.2, "the LLM assigns an importance weight... based on its
contribution to the topic") are paid for on every extraction call and then never read at query
time. Under the old framing (HyperMem retrieval is inert reference code) this was correctly
flagged as wasted spend with no compensating benefit. Under the new framing — HyperMem's retrieval
is the active baseline C-AIMMS runs and reports numbers from — the right fix is not to stop paying
for the weights, it's to make the LLM cost earn its keep by fusing the weights into the ranking
that already runs at those four sites, i.e., modify the RRF/score-combination step so
`final_score = f(bm25_or_vector_score, w^E_or_w^F)` instead of passing the raw retrieval score
through unmodified.

**Confidence: Medium.** The paper computes these weights as part of the published algorithm (so
something should consume them), but the paper's own ablations do not isolate weight-fusion's
effect on retrieval quality — meaning this ADR cannot cite a paper result that says fusion helps,
only that the paper's design implies it should be used. This is exactly the ROI question the task
brief calls out: resolve it by measuring, not by assuming either "the LLM cost is wasted" or "the
paper's implied design is automatically correct." Impact tier: **High** — this changes ranking
behavior on every retrieval call and is the kind of change that needs a full before/after
comparison on LoCoMo, not a one-line patch merged without evaluation.

#### G5 — reranker uses `episode_description` (long) not `summary` (short)

**Re-verified, not conditional this time.** `stage4_hypergraph_retrieval.py:624` passes
`text_field="summary"` for topic-level reranking (correct, matches paper) but `:742` passes
`text_field="episode_description"` for episode-level reranking — confirmed by direct grep, both
call sites present in the current file. This breaks §4.5's token-efficiency claims because
reranking against the long field defeats the purpose of having a short `summary` field at all.

**New resolution: fix in code now**, not gated on a "kept as baseline" decision — HyperMem's
retrieval is the active baseline, so this fix applies unconditionally: change
`text_field="episode_description"` to `text_field="summary"` at the episode-reranking call site
(`:742`, and confirm episode records actually populate a `summary` field distinct from
`episode_description` before flipping the flag — `stage1_memory_extraction.py`'s inline comment
"summary is already set by `_generate_episode_memory` as `content[:200]+"..."`" suggests `summary`
may currently be a truncation of the long field rather than an LLM-generated short one; that needs
a one-line check before the swap, not an assumption).

**Confidence: High** that the field-swap bug exists and that fixing it is correct per the paper.
Impact tier: **Medium** — it's cheap to fix, but changes what text the reranker scores against on
every episode-level query, which is a measurable ranking change, not a pure bug fix with no
retrieval-quality consequence.

#### G6 — single pooled BM25 corpus across Topic/Episode/Fact levels

**Prior resolution:** accept as limitation (paper silent, "no ground truth to fix toward").

**New resolution: re-examine as a solvable design problem, not dismiss as an accepted
limitation.** Confirmed the pooling: `stage3_hypergraph_index.py:301-309` builds
`all_corpus = fact_corpus + episode_corpus + topic_corpus` and calls a single
`BM25Okapi(all_corpus)` — one IDF/avgdl statistic shared across all three node types, which are
different in typical document length and vocabulary (short topic summaries vs. longer episode
narratives vs. terse fact assertions). The paper is genuinely silent on this (no equation or
description of per-level vs. pooled indexing in §3.3), so there's no fidelity target to match —
but "the paper doesn't say" is not the same as "there's nothing to investigate." The consequence
(soft downranking of terms that are common in one level but rare across the pooled corpus, or vice
versa) is measurable, not merely theoretical.

**New decision:** implement per-level BM25 indexing (one `BM25Okapi` index per node type) as a
variant, and run a retrieval-quality comparison (pooled vs. per-level) on LoCoMo before deciding
which becomes the default. This is explicitly an experiment, not a silent code change — it
produces a result C-AIMMS can cite either way ("pooling was fine" or "per-level indexing improves
ranking by X").

**Confidence: Low** going in (unspecified in paper, no prior evidence either direction) but
**high ROI if it moves the numbers**, which is exactly why it's worth the implementation cost
rather than being written off. Impact tier: **High** — it's an architecture-level retrieval
change requiring a full comparison run, same tier as G4.

#### G7 — config defaults diverge from paper

Unchanged from the first pass: `config.py`'s `retrieval_config` (`topic_top_k=15,
episode_top_k=20, fact_top_k=30` vs. paper's stated 10/10/30), `use_reranker` defaulting to
`False` (`config.py:79`, gated on `HYPERMEM_USE_RERANKER` env var), and the λ pooling weight
(`node_emb_update_weight=0.5`, `hyperedge_emb_aggregate_type="sum"` — re-verified at
`config.py:20-21`, matching the paper's stated 0.5/sum, so this specific sub-claim from the first
pass's fidelity analysis was already correct in code; only the top-k values and reranker default
are the actual divergences). **Resolution: fix in code, unconditionally** — this is now easier to
justify than before, since `config.py` is being kept in full (not trimmed) as the config for the
active retrieval baseline, not just for a salvaged `stage6_eval.py`.

**Confidence: High.** Impact tier: **Low** (numeric defaults only, no architectural change,
though `use_reranker=True` as default will change retrieval latency/cost — worth flagging when
this lands, not a reason to avoid the fix).

#### G8 — no tests anywhere in HyperMem

**Resolution unchanged, tightened.** HyperMem is vendored, but under the new philosophy it is an
_actively run_ baseline, not read-mostly reference — `.claude/rules/testing.md` applies to
whatever code C-AIMMS runs, and running `stage4_hypergraph_retrieval.py` in CI/experiments now
counts as "actively developing" for the modules being touched by G2/G4/G5/G6 above. **New
resolution: any file touched by this ADR's fixes (G2, G4, G5, G6, G7) gets tests in the same PR
that changes it** — this is not optional for a component the project reports numbers from.
Untouched HyperMem files stay untested (stage5, stage2's untouched portions, etc.) until they too
become actively-run code.

**Confidence: High.** Impact tier: **Low** (process requirement, not a code change itself, but
gates all the Medium/High tier work above).

### Code reuse assessment: what's active vs. what's still read-only

The prior pass framed most modules as "remove from active use, keep as reference." Under the new
philosophy, more of the pipeline is active:

| Module                            | Status under this ADR                                                                                                                                                                                                                                                                                                                                                                                             |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stage3_hypergraph_index.py`      | **Active baseline, gets G6 fix.**                                                                                                                                                                                                                                                                                                                                                                                 |
| `stage4_hypergraph_retrieval.py`  | **Active baseline, gets G4 + G5 fixes.**                                                                                                                                                                                                                                                                                                                                                                          |
| `stage2_hypergraph_extraction.py` | **Active baseline, gets G2 fix; G1 is a documented, lower-priority limitation within the same active component (not "read, don't run").**                                                                                                                                                                                                                                                                         |
| `stage1_memory_extraction.py`     | **Active baseline, gets G3 fix.**                                                                                                                                                                                                                                                                                                                                                                                 |
| `config.py`                       | **Active baseline, kept in full (not trimmed), gets G7 fix.**                                                                                                                                                                                                                                                                                                                                                     |
| `stage6_eval.py`                  | **Kept, repurposed as a generic answer-scoring harness** (unchanged from the first pass — this decision did not depend on the retrieval-scope framing; the scoring logic was always dataset-agnostic).                                                                                                                                                                                                            |
| `stage5_response.py`, `eval.py`   | **Removed from active import paths, unchanged from the first pass** — `stage5` is answer generation coupled to a response-shape C-AIMMS doesn't need yet (FluxMem has no generation-consuming component to feed it), and `eval.py` is glue code over stages already individually triaged. Nothing here reintroduces retrieval into FluxMem's own module boundary; it stays a standalone runner under `HyperMem/`. |

**Net effect:** HyperMem's stages 1-4 plus `config.py` are now an actively maintained retrieval
baseline (not dormant reference code), run as a standalone harness outside `fluxmem/` per the
adapter boundary in 0003. `stage6_eval.py` remains the reusable scoring component. `stage5` and
`eval.py` remain out of scope, unchanged from the prior pass — this ADR does not extend "active
baseline" status to the response-generation layer, since nothing downstream in C-AIMMS consumes
it yet.

## Implementation Roadmap

Each phase is a separate session with concise context (model assignments reflect reasoning
depth). Phases group by decision impact and are ordered for cumulative progress.

### Phase 1 — Low-impact fixes (Sonnet low reasoning, ~6-8 min)

**Files:** `stage1_memory_extraction.py`, `config.py`  
**Context:** G3 = wire should_wait into stage1 buffer logic (paper §3.2.1). G7 = fix numeric
defaults (top-k, reranker). G1 = add code comment documenting retrieval limitation.

- **G3:** Stage1 currently discards `result[1]` (StatusResult with should_wait); wire it so
  should_wait=True distinguishes "still unfolding" from other None-episode cases. Add test.
- **G7:** Update config.py: `topic_top_k=10`, `episode_top_k=10`, `use_reranker=True`. Add
  regression test on config values.
- **G1:** Add comment at stage2's LLM batch-matching call site noting §3.2.2's retrieval-seeded
  design is not implemented (efficiency gap only, not correctness).

### Phase 2 — Medium-impact fixes (Sonnet low reasoning, ~10-15 min)

**Files:** `stage2_hypergraph_extraction.py`, `stage1_memory_extraction.py`, `stage4_hypergraph_retrieval.py`  
**Context:** G2 = add paper's 3-case decision tree (Case 1/2/3 on similarity threshold). G5 =
verify summary field, then fix reranking text_field.

- **G2:** Implement similarity-threshold check before topic creation in stage2's topic-matching
  logic. Requires threshold tuning against LoCoMo; current value is TBD.
- **G5:** Check stage1's `_generate_episode_memory` — is `summary` a truncation of `content` or
  a distinct short field? If truncation, fix it. Then flip `text_field="summary"` at
  stage4:742 (episode reranking).

### Phase 3 — High-impact research (Sonnet medium reasoning, ~15-25 min + eval runs)

**Files:** `stage4_hypergraph_retrieval.py`, `stage3_hypergraph_index.py`  
**Context:** G4 and G6 are ranking experiments requiring before/after LoCoMo comparison. Decisions
on weight fusion and per-level indexing depend on measured impact.

- **G4 (weight fusion):** Implement score combination using w^E/w^F at stage4 lines 683, 706, 798, 821.
  Modify RRF to fuse hyperedge weights with BM25/vector scores. Run LoCoMo before/after;
  decide whether fusion justifies LLM cost.
- **G6 (per-level BM25):** Implement one BM25Okapi index per node type (Topic/Episode/Fact) in
  stage3 as variant to pooled corpus. Run LoCoMo comparison; decide default based on measured
  ranking quality.

### Phase 4 — Testing (per G8, can overlap with Phases 1-3)

**Scope:** Every file modified in Phases 1-3 gets tests for changed logic (same PR as fix).  
Per `.claude/rules/testing.md`, active code requires test coverage.

### Phase 5 — CLAUDE.md update (separate PR)

Already done; no additional work needed.

## Implementation Status (2026-08-12)

Phases 1–3 implemented in this session (`HyperMem/` is untracked in git at time of writing — no
commits made). Phase 4 (testing) was folded into Phases 1–3 per G8, not run separately.

- **Phase 1 (G3, G7, G1):** done. `stage1_memory_extraction.py` wires `should_wait`;
  `config.py` top-k/reranker defaults corrected; G1 limitation documented as a comment in
  `stage2_hypergraph_extraction.py`.
- **Phase 2 (G2, G5):** done. `topic_extractor.py`/`topic_prompts.py` add a `confidence` field
  and threshold gate (`topic_match_confidence_threshold=0.6`, TBD pending LoCoMo tuning);
  verified `summary` is LLM-generated (not a truncation of `episode_description`) before
  flipping `text_field="summary"` at both stage4 reranking call sites.
- **Phase 3 (G4, G6):** mechanism implemented, **not evaluated**. `fuse_hyperedge_weight()`
  (multiplicative `score * weight`, gated by `config.fuse_hyperedge_weights`, default `False`)
  and `BM25Corpus` (pooled vs. `bm25_indexing_mode="per_level"`, default `"pooled"`) are in place
  in `stage4_hypergraph_retrieval.py` / `stage3_hypergraph_index.py`. Per the roadmap, this phase
  is a toggle plus test coverage, not the LoCoMo before/after run itself — that run has not been
  executed, and the current defaults (fusion off, pooled indexing) are unchanged from the
  pre-ADR baseline until it is.
- **Tests:** 22 new tests across `HyperMem/tests/` (one file per gap), `pytest tests/ -q` →
  `22 passed`.
- **Unplanned fix:** `stage4_hypergraph_retrieval.py` had a pre-existing import-time `TypeError`
  from a `callable | None` annotation (unrelated to any gap, but blocked importing the module for
  Phase 4 tests). Fixed with `from __future__ import annotations`.

**Still open:** the Phase 3 LoCoMo before/after comparisons (item 1 under Timeline, below) — this
is the work that would justify flipping `fuse_hyperedge_weights` or `bm25_indexing_mode` away
from their current (pre-ADR-equivalent) defaults.

## Consequences

**Cost breakdown by phase:**

- **Phase 1 (Low impact):** Sonnet low reasoning; 2 PRs (one for G3/G7 fixes + tests, one for G1
  comment). Bounded changes, clear paper citations.
- **Phase 2 (Medium impact):** Sonnet low reasoning; 2 PRs (one for G2 + threshold tuning + tests,
  one for G5 + field check + tests). Requires empirical tuning (G2 threshold, G5 field audit).
- **Phase 3 (High impact):** Sonnet medium reasoning; 2 PRs (one per experiment: G4 weight fusion,
  G6 per-level BM25). Both require LoCoMo before/after runs and result interpretation.
- **Phase 4 (Testing):** Tests land in same PR as each fix (Phase 1-3), not separately.

**Timeline:** No external deadline. Priority order for COLM timeline:

1. Phase 3 experiments (G4, G6) produce before/after numbers for related-work section
2. Phase 1/2 fixes improve correctness but don't generate comparison results

**Risk:**

- **G4 & G6:** Both change active retrieval ranking. Require measured comparison to know impact.
  Landing without LoCoMo eval would violate ADR's "measure, don't assume" principle.
- **G2 & G5:** Lower risk — paper specifies correct behavior. Risk is empirical parameters
  (threshold, field structure), not correctness ambiguity.
- **G3 & G7:** Lowest risk — config values and control-flow fixes with paper backing.

## Scope boundaries

- **HyperMem stays external to `fluxmem/ltsm.py`.** The adapter pattern from ADR 0003 is
  preserved — HyperMem runs as a standalone harness alongside FluxMem, not inside STIM/MTEM/LTSM
  core. This is a code-organization boundary, not a functionality boundary.
- **G4 and G6 are experiments, not pre-approved defaults.** Weight fusion (G4) and per-level BM25
  (G6) require before/after LoCoMo comparisons; this ADR defines the experiment, not the outcome.
- **Dual-segmentation (EM-LLM + HyperMem extractor fallback)** remains unchanged from the first
  pass and is Low confidence, pending EM-LLM integration.
