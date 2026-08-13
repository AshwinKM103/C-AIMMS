# 0012. Encoding-adaptive feature selection

## Status

Proposed — 2026-08-13. Governs `fluxmem/features.py` (`FEATURE_NAMES`, `extract_features`) and the
Phase 2/3 specifications of `hetrep/hg/encoder.py`'s `hyperedge_density` and the VC arm's
`visual_salience`. Supersedes the feature list transcribed from COLM §1.3.3, not COLM itself.

Full investigation, with executed evidence and per-feature reasoning:
`docs/selector-feature-investigation.md`.

## Context

`fluxmem/features.py` extracts a 7-scalar vector — `entity_count`, `cooccurrence_density`,
`temporal_ordering`, `topic_diversity`, `token_length`, `hyperedge_density`, `visual_salience` — and
`fluxmem/selector.py` maps it to one of HG / VC / VS. The FluxMem fidelity audit
(`docs/FluxMem-Fidelity-Audit-2026-08-13.md`, D-4) flagged two of these as constant-zero
placeholders pending HetRep, and classified the 7-vs-12 divergence from FluxMem as a reading-rule
choice rather than drift: the code matches COLM §1.3.3, and ADR 0008 makes COLM authoritative.

That defends the code's _fidelity_. This ADR asks the different question of its _fitness_.

### Encoding is unconditional, and that determines the analysis

COLM §1's framework description states it directly:

> In the **write phase**, each new episodic memory unit is segmented from raw dialogue, **encoded in
> three formats by HETREP** and stored with adaptively chosen primary format by ADASTORE.

Algorithm 1 orders it identically (line 4 `Encode e_j ← HETREP`, line 5 `Select`), and
`hetrep/encoder.py:45-50` runs all three arms with no content-dependent branching — the `None`
guards exist for phasing and ablation, not for episode routing. Eq. 1's "**up to** three encodings"
means a unit may lack one, not that encoding is conditional on the selector.

Three consequences frame every decision below:

1. **The selector chooses what to _read_, not what to _build_.** `f_j` routes among three artifacts
   that already exist.
2. **Measurements of the built artifacts are legitimate, zero-marginal-cost features.**
   `hyperedge_density` and `visual_salience` are not chicken-and-egg problems; they are the most
   direct evidence available about how each encoding actually turned out.
3. **Write cost is constant across the decision**, so FluxMem Eq. 9's reward — whose `r_mem` term
   (`fluxmem/supervision.py`) measures _retrieval_ token efficiency — is correctly specified for
   this target. The absence of a write-cost term is right, not a gap.

### Findings

1. **The 7 are COLM's prose paraphrase of FluxMem, not FluxMem's features.** FluxMem App. C.2
   Table 5 defines 12; App. G confirms "12-dimensional input, hidden size 4." COLM §1.3.3 summarizes
   these in one sentence and asserts they are "the same classes of features"; `features.py`
   implements that sentence. Five are dropped, including `semantic complexity` — Table 5 defines it
   as "embedding dispersion," the single most relevant Table 5 entry to the new target.

2. **Three features are degenerate or redundant, not two.** Executed against the code
   (`docs/selector-feature-investigation.md` §2.2–2.4):

   | Feature                | Observed behaviour                                                                                                                                                                                                                                   |
   | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | `temporal_ordering`    | Range ≈ [0.50, 0.58]. Half the feature is `monotonic_fraction`, which is 1.0 for any turn list in arrival order (`interfaces.py:212` assigns `ts = base_ts + t`). Exactly 0.5 for every VC episode, since `MemOCREpisodeProducer` emits `turns=[]`.  |
   | `topic_diversity`      | Type-token ratio: 0.92 → 0.06 as an episode grows 50 → 3200 tokens under an identical generative process. Largely a decreasing transform of `token_length`, already in the vector.                                                                   |
   | `cooccurrence_density` | Saturates at exactly 1.0 whenever one turn names every entity in the episode — 1.0 for 3 entities and for 24 alike. Pairs are deduplicated into a `set` (`features.py:95`), so a _recurring_ association contributes nothing past its first mention. |

3. **The two placeholders are the best-positioned features in the vector, with wrong or missing
   specifications.** Neither is circular (see above). But `hyperedge_density`'s spec
   (`hetrep/hg/encoder.py:23,51`: `|incident_edges| / |possible_edges|`) never says whether
   "incident" is scoped within-episode or store-wide — and the two readings measure opposite things.
   And `visual_salience`'s only numeric definition (`memocr_episodes.py:97-113`) is rendered pixel
   area over budget: that is render _size_, with no contrast term, so a uniformly-important large
   canvas and a large canvas with a tiny crucial core score identically.

4. **Two decision axes have no coordinate at all.**
   - **VC's advantage is budget-conditional**: MemOCR scores 62.2% vs MemAgent's 31.6% at a
     16-token budget, but at B = 1024 with 100K context the gain is +0.9 and **not statistically
     significant (p = 0.3419)**. The decision is read-side and budget is a read-side parameter, so
     this is a direct mismatch. Nothing in the vector represents budget.
   - **HG's advantage is cross-episode temporal binding**: HyperMem attributes its multi-hop result
     (93.62%, +9.58 over LightRAG) to hyperedges that "bind topically related episodes **scattered
     across time**." Nothing represents that time spread; `temporal_ordering` measures _within_-episode
     ordering, which is a constant.

5. **VS is defined negatively.** COLM §1.2.3 assigns it to "episodes with weak relational or
   temporal structure," so features earn their place by evidencing HG or VC, with VS as the residual.

## Decision

**Replace the feature vector with seven features that measure the three built artifacts wherever
possible, rather than proxying those quantities from raw text. Retain both placeholder fields as
inputs with corrected specifications.**

| #   | Feature               | Source        | Captures                                       | Change                       |
| --- | --------------------- | ------------- | ---------------------------------------------- | ---------------------------- |
| 1   | `token_length`        | episode       | episode scale                                  | **unchanged**                |
| 2   | `budget_ratio`        | read context  | episode size against the read budget           | new                          |
| 3   | `semantic_dispersion` | turns + `v_j` | whether one centroid can represent the episode | replaces `topic_diversity`   |
| 4   | `fact_node_yield`     | `H_j`         | atomic, separately-addressable facts extracted | replaces `entity_count`      |
| 5   | `hyperedge_density`   | `H_j` + store | how strongly the episode binds into the store  | **kept**, spec resolved      |
| 6   | `hyperedge_time_span` | `H_j` + store | how far across time those bindings reach       | replaces `temporal_ordering` |
| 7   | `visual_salience`     | `I_j`         | concentration of importance on the canvas      | **kept**, redefined          |

`cooccurrence_density` is removed outright. `FEATURE_DIM` stays 7 and is already derived from
`len(FEATURE_NAMES)`, so no call site hardcodes it.

### Two specifications this ADR resolves

**`hyperedge_density` is store-wide**: the fraction of this episode's nodes participating in at
least one hyperedge that _also contains nodes from a different episode_, bounded [0,1]. The
within-episode reading measures internal interconnection, which is close to what the removed
`cooccurrence_density` groped at and is not what HyperMem's multi-hop result is attributed to. The
existing formula is a Phase-2 plan in a no-op function, not a paper formula, so this is a
definitional decision rather than a fidelity question — but it is exactly the kind of ambiguity the
`resolve-paper-ambiguities-dont-hedge` rule says to close now rather than rediscover in Phase 2.

**`visual_salience` is the share of rendered canvas area occupied by the high-visibility (crucial)
region**, bounded [0,1], replacing pixel-area-over-budget. Its predictive content is interactional
rather than monotone: a _small_ salient share with large `token_length` and tight `budget_ratio` is
MemOCR Fig. 1(b)'s picture exactly — a small crucial core inside a large auxiliary remainder, under
pressure. A two-layer MLP can represent that interaction, so the feature need not be independently
monotone in VC-goodness.

### Why measure rather than proxy

Under unconditional encoding, a text heuristic that estimates a quantity HETREP has already computed
exactly is strictly worse information at the same cost. Four otherwise-plausible candidates fall to
this and are rejected in the investigation: `multi_entity_rate` (≥3-entity co-occurrence, a proxy
for hyperedge formation), `store_affinity` (cosine similarity to store episodes, a proxy for
cross-episode binding), `entity_density` (NER mentions, a proxy for fact yield), and
`auxiliary_ratio` (entity/date/numeral-free sentence share, a proxy for the crucial/detail split).
Each estimates something features 4, 5, or 7 measure directly.

This does not violate FluxMem App. C.2's "avoids introducing additional reasoning overhead into the
control layer." That constraint targets overhead _in the feature extractor_; reading a field off an
artifact built elsewhere in the pipeline adds none.

### Why not keep the 7 and wait for HetRep to fill the placeholders

| Option                                                                            | Tradeoff                                                                                                                                                                                                |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (a) Keep all 7 as specified; fill the two in Phases 2–3                           | Minimal change, stays literally on COLM §1.3.3; but preserves three degenerate or collinear columns, leaves both missing decision axes unrepresented, and ships `visual_salience` measuring render size |
| (b) Keep the 7 and _add_ new ones                                                 | No deletions to defend; but carries `topic_diversity`'s collinearity with `token_length` and the saturating `cooccurrence_density` into a wider input, on a dataset the project does not have at scale  |
| (c) Seven artifact-first features, placeholders retained and respecified (chosen) | Departs from COLM §1.3.3's enumeration; but every column has demonstrated variance or a stated mechanism, and the two most-criticized fields become the best-grounded ones                              |

**Decision: (c).** ADR 0008 makes COLM authoritative over a _component's_ implementation choices.
This is not that situation: COLM §1.3.3 is a one-sentence paraphrase of another paper's appendix,
written while describing a selector COLM has not built, and COLM is the paper this project is
writing rather than a fixed external spec. Following it literally would ship three columns measured
to be near-constant or collinear. The paraphrase is superseded on evidence; COLM's Eq. 6 selector
formulation, its three-format taxonomy, §1.2.3's VS assignment, and Algorithm 1's unconditional
encoding are all untouched.

### `semantic_dispersion` must not change how `v_j` is computed

Feature 3 needs per-turn embeddings. The tempting optimization — mean-pool the turn vectors to
produce `v_j` and get dispersion for free — is forbidden: COLM specifies
`v_j = Enc(summary(e_j))`, and a mean of turn vectors is not that. Per ADR 0008 this is exactly the
silent substitution that ADR rules out. Compute per-turn embeddings in a separate pass and leave
`v_j` alone.

## Consequences

### Positive

- **The two placeholder fields become the best-grounded features in the vector** rather than
  liabilities, and their Phase 2/3 implementations now have unambiguous specifications to build to.
- **Every column has demonstrated variance or a stated mechanism.** The three degenerate columns are
  gone, so `StandardScaler` is no longer amplifying noise on near-constant inputs.
- **Both missing decision axes are represented**: budget (feature 2) and cross-episode temporal
  reach (feature 6).
- **Near-zero net compute cost.** Features 4, 5, and 7 are field reads off artifacts already built;
  removing the spaCy NER pass may make the net change negative.
- **Dimension is unchanged at 7**, so `selector.py`, `supervision.py`, and the persisted
  scaler/state_dict format need no structural change.

### Negative

- **Departs from COLM §1.3.3's literal enumeration.** Any write-up saying "the selector uses the
  features COLM describes" becomes false and must cite this ADR instead. Accepted knowingly.
- **Five of seven features are phase-gated** — three available in Phase 1, three more in Phase 2,
  the last in Phase 3. This is not a regression, since under unconditional encoding there is no
  configuration in which HG or VC is selectable but unbuilt, but it does mean the vector is
  incomplete until Phase 3.
- **`extract_features` gains dependencies**: a store-wide hyperedge index (features 5–6) and canvas
  region metadata (feature 7). It stops being a pure function of one episode.
- **`visual_salience`'s redefinition is an integration requirement on the Phase 3 renderer.**
  MemOCR's drafter produces structured markdown with headings and highlights, so the crucial region
  is identifiable in principle, but the renderer must surface its geometry as metadata.
- **One real new cost:** `semantic_dispersion` needs `n_turns` MiniLM passes instead of one.
  Negligible against the LLM extraction and VLM read the same episode already pays unconditionally,
  but not yet benchmarked.

### Risks

- **This does not unblock selector training.** `hetrep/vc/stub.py:6-9` records that with VC stubbed,
  Eq. 6's label `f*_j = argmax_f r_j^f` is unmeasurable for VC. No feature set fixes a missing arm.
  **The 3-class head and `FORMAT_ORDER = (HG, VC, VS)` are unchanged by this ADR** — retargeting the
  selector was raised and rejected previously and is not re-proposed here. What is gated is the
  training *data*: in Phase 2 no example can carry a VC label, so the VC logit stays untrained and
  any accuracy figure is interpretable only as HG-vs-VS discrimination and must be reported as such.
  Reporting three-way accuracy before Phase 3 would report a number whose VC labels do not exist.
- **`budget_ratio` is redundant at a fixed budget**, degenerating to a monotone transform of
  `token_length`. It earns its place only if label collection sweeps budgets, as MemOCR's
  budget-aware training does. If the project settles on one budget, drop it.
- **`hyperedge_density` depends on store state at encode time.** An episode encoded when the store
  is nearly empty scores low through no property of its own. Whether to recompute as the store grows
  or accept the time-of-encoding snapshot is unresolved and affects reproducibility of any selector
  trained on it. This is the most likely source of a silently wrong number in the proposed set.
- **Write-time selection predicts a read-time payoff.** HG's benefit appears on multi-hop questions,
  VC's under tight budgets; neither is known when `f_j` is assigned. Eq. 6's labels absorb the query
  distribution implicitly, so a selector trained under one query mix may not transfer. Mitigated
  considerably by unconditional encoding — a wrong `f_j` degrades the primary read path rather than
  losing information — but this assumes ITERRET can fall back to a non-primary encoding, which is
  outside this ADR's scope and worth confirming.

## Related

- **ADR 0002** — the reward design (token-F1 judge, token-budget MemUtil) producing the labels these
  features are trained against. Under unconditional encoding its retrieval-only cost term is
  correctly specified for this decision; no change needed.
- **ADR 0003** — established explicit-approximation-over-silent-deviation, the standard
  `visual_salience`'s pixel-occupancy proxy was documented under. That standard is what made this
  ADR's respecification straightforward to argue; the proxy is superseded, the standard is not.
- **ADR 0005** — the `EpisodeEncoder` seam. Features 4, 5, and 7 read across it by design, which
  Algorithm 1's ordering (encode, then select) makes sound.
- **ADR 0007** — HetRep's reuse of HyperMem's `structure.py` data model, which is where features 4–6
  will be computed from.
- **ADR 0008** — COLM's authority over component implementations. The "Why not keep the 7" section
  distinguishes that case from this one; the `v_j` constraint above is a direct application of it.
- **`docs/FluxMem-Fidelity-Audit-2026-08-13.md`** — finding D-4 (7 vs 12 features) is the audit
  observation this ADR resolves. Its recommendation to record that "2 of 7 selector features are
  constant zero until HetRep lands" stands, with the correction that those two are respecified
  rather than merely awaited.
- **`docs/selector-feature-investigation.md`** — the full investigation, including the executed
  probes behind finding 2 and eight considered-and-rejected candidates.
