# Selector feature investigation: what predicts encoding choice?

**Date:** 2026-08-13 · **Scope:** `fluxmem/features.py`, `fluxmem/selector.py`, `fluxmem/interfaces.py`
· **Decision recorded in:** ADR 0012

The C-AIMMS selector picks one of three _encodings_ — hypergraph (HG), visual canvas (VC), vector
store (VS) — from a 7-scalar feature vector. This document asks whether that vector can express the
distinction it is being asked to make, and proposes a replacement.

Verification status is marked throughout: **[verified]** means read in the paper text or executed
against the code in this session; **[inferred]** means my reading of a paper that does not state the
claim directly; **[assumed]** means a modeling judgement with no source.

---

## 0. The architectural fact that frames everything

**All three encodings are built for every episode, unconditionally.** COLM §1 states it in the
framework description **[verified verbatim]**:

> In the **write phase**, each new episodic memory unit is segmented from raw dialogue, **encoded in
> three formats by HETREP** and stored with adaptively chosen primary format by ADASTORE.

Algorithm 1 orders it the same way — line 4 `Encode e_j ← HETREP`, line 5 `Select` **[verified]** —
and `hetrep/encoder.py:45-50` runs all three arms with no content-dependent branching (the `None`
guards exist for phasing and ablation, not for episode routing) **[verified]**. Eq. 1's phrasing
that a unit is "associated with **up to** three encodings" means a unit may lack one, not that
encoding is conditional on the selector's output.

Three consequences follow, and they determine the whole analysis:

1. **The selector chooses what to _read_, not what to _build_.** `f_j` is a routing label over three
   artifacts that already exist. There is no write-side cost to avoid.
2. **Measurements of the built artifacts are legitimate, zero-marginal-cost features.**
   `hyperedge_density` and `visual_salience` are not chicken-and-egg problems; they are the most
   direct evidence available about how each encoding actually turned out for this episode.
3. **Misselection is recoverable.** A wrong `f_j` costs retrieval quality on the primary path, but
   the other two encodings are still stored and reachable. This lowers the stakes of the decision
   relative to a system that discards what it does not select. **[inferred — COLM does not discuss
   fallback, but Eq. 1 stores all three, so nothing is lost]**

The right question, then, is not "which encoding is worth paying for?" but **"given three artifacts
that all exist, which one will best serve a future query about this episode?"**

---

## 1. Summary of findings

1. **The 7 features are not FluxMem's features.** FluxMem App. C.2 Table 5 defines **12**. The
   code's 7 transcribe COLM §1.3.3's _prose paraphrase_, which drops 5 — including
   `semantic complexity`, defined by Table 5 as "embedding dispersion," the most relevant Table 5
   entry to the new target. **[verified]**
2. **Three features are near-constant or redundant, not two.** Executed against the code:
   `temporal_ordering` is pinned to ≈[0.50, 0.58]; `topic_diversity` is largely a decreasing
   transform of `token_length`; `cooccurrence_density` saturates at exactly 1.0 (§3). **[verified]**
3. **The two placeholders should be _defined correctly_, not removed.** Both are measurements of
   artifacts that always exist. `hyperedge_density`'s specification is ambiguous and should be
   resolved store-wide; `visual_salience`'s only numeric definition measures the wrong quantity
   (render size rather than importance concentration) — a defect independent of the encoding
   schedule (§3.6).
4. **Two decision axes have no coordinate at all.** VC's advantage is _budget-conditional_ and
   vanishes into statistical insignificance at ample budget (p = 0.3419); nothing in the vector
   represents budget. HG's advantage is _cross-episode temporal binding_; nothing represents the
   time spread of the episodes an episode binds to (§2).
5. **A replacement set of 7 features** keeps the dimension unchanged, retains `token_length`
   unchanged, retains both placeholder fields with corrected definitions, and replaces the four
   degenerate or mistargeted columns (§4).
6. **Selector training remains blocked on Phase 3**, unchanged by any of this: `r^VC` cannot be
   measured without a real canvas (§5.1).

---

## Part 1 — Investigation report

### 1.1 The three papers are not selecting between the same kinds of thing

| Paper        | Chooses between               | What varies across the options                              |
| ------------ | ----------------------------- | ----------------------------------------------------------- |
| **FluxMem**  | LINEAR / GRAPH / HIERARCHICAL | how the _same text_ is indexed and traversed                |
| **COLM**     | HG / VC / VS                  | which of three _co-existing media_ is the primary read path |
| **MemOCR**   | (not a choice — one encoding) | how a fixed token budget is _allocated_ within one canvas   |
| **HyperMem** | (not a choice — one encoding) | which associations are made explicit and traversable        |

FluxMem's options are organizational schemes over one substrate; everything stays text, and what
differs is which query shape is served. Its features are accordingly _discourse-shape_ indicators —
`is qna pattern`, `decision tree`, `entity centric` — well matched to that target, because a
decision-tree dialogue really does suggest hierarchical organization. **[verified — Table 5's
motivation column reads exactly this way]**

COLM's options are three different media that all exist simultaneously. Asking "is this a decision
tree?" tells you nothing about whether a future query is better served by traversing a hypergraph,
reading a canvas, or fetching a vector. The question has changed category.

### 1.2 FluxMem: the actual 12 features

Table 5, App. C.2 **[verified verbatim]**:

| Feature             | Description (paper's words, condensed)                       |
| ------------------- | ------------------------------------------------------------ |
| page count          | Number of interaction pages in the context window            |
| avg page length     | Average length of recent pages                               |
| entity density      | Average named entities **per page**                          |
| relation indicators | Frequency of explicit relational expressions                 |
| topic diversity     | **Number of distinct topics detected**                       |
| topic transitions   | Frequency of topic shifts across consecutive pages           |
| is qna pattern      | Binary: question–answer dominated interaction                |
| decision tree       | Binary: decision-tree-like dialogue flow                     |
| entity centric      | Binary: entity-focused, repeated mentions                    |
| time span           | Temporal span covered by recent interactions                 |
| temporal density    | Interaction frequency per unit time                          |
| semantic complexity | Semantic variability across pages (**embedding dispersion**) |

App. G confirms "a two-layer MLP with a 12-dimensional input and hidden size 4, incurring negligible
overhead... executed once per interaction." **[verified]**

App. C.2's design constraint — "intentionally simple and interpretable... avoids introducing
additional reasoning overhead into the control layer" — rules out an LLM _inside the feature
extractor_. It does not constrain features that read artifacts an LLM already produced elsewhere in
the pipeline: those cost nothing extra at selection time. **[inferred — the paper's concern is
overhead in the control layer, and reading a field off a built artifact adds none]** This is what
makes artifact measurements admissible under FluxMem's own stated bar.

### 1.3 COLM: what each encoding is for

§1.2.3 gives VS its niche in one clause **[verified]**:

> VS serves as the primary format for episodes with **weak relational or temporal structure**.

This is a _negative_ definition, and it matters: VS is the residual. A selector does not need
positive evidence for VS; it needs evidence for HG and VC, with VS as what remains. **[inferred —
COLM does not say "residual," but defines VS only by the absence of the other arms' preconditions]**

### 1.4 MemOCR: VC's advantage is budget-conditional

MemOCR's framing of the problem **[verified]**:

> text imposes **uniform information density**: to maintain 100 tokens of crucial information, the
> system is compelled to retain a substantial volume of auxiliary details (Figure 1(b))

The canvas breaks that coupling by rendering crucial evidence in a high-visibility region and
auxiliary detail in lower-priority regions, so the two scale independently. Three results pin down
when this pays:

- **Obs. 2.1** — "Visual layout significantly enhances low-budget robustness. As budget tightens,
  MemOCR degrades most gracefully... Removing visual layout causes a marked additional drop."
  **[verified]** Measured: 62.2% at a 16-token budget (−16.6% from its own 1024-token baseline)
  against MemAgent's 31.6% (−53.3%). **[verified from Table 1]**
- **Obs. 2.2** — "MemOCR achieves 8× token-efficiency gain at extreme budgets." **[verified]**
- **Saturation** — "At maximum budget (B = 1024), the gain becomes small at 100K context (only
  +0.9) and **not statistically significant (p = 0.3419)**, indicating saturation when both memory
  and context are ample." **[verified]**

The last is the most useful finding here and the easiest to overlook. **VC's advantage is not a
property of the episode alone.** The same episode is worth reading as a canvas at a 16-token budget
and worth nothing at a 1024-token budget with ample context. A feature vector with no budget
coordinate is being asked to predict a budget-conditional quantity from budget-free inputs. Since
the selector's decision is now understood to be purely read-side (§0), and budget is a read-side
parameter, this is a direct mismatch rather than a subtlety.

Figure 5's oracle analysis isolates the second condition: ground-truth evidence is injected into
either the Crucial or the Detailed region and both help, by different amounts. **[verified that the
experiment exists and compares the two regions; the per-dataset margins live in the figure, which
the text extract does not resolve numerically]** The design implication holds regardless of margin:
the canvas has a _privileged region of limited size_, so it pays off when a small crucial core sits
inside a large auxiliary remainder. If everything is uniformly important there is nothing to demote,
and the canvas degenerates to uniform density — the very thing it was built to escape.

### 1.5 HyperMem: HG's advantage is cross-episode

LoCoMo by category **[verified, Table 1]**: single-hop 96.08, multi-hop 93.62, temporal 89.72,
open-domain 70.83, overall 92.73. The paper attributes the multi-hop result specifically:

> On Multi-hop questions requiring evidence aggregation across multiple dialogue segments, HyperMem
> reaches 93.62%, outperforming LightRAG by 9.58%, demonstrating **hyperedges effectively bind
> topically related episodes scattered across time**.

**[verified]** Two things follow.

First, the operative property is _scattered across time_. A topic hyperedge earns its keep when it
groups episodes that recency-ordered retrieval would not co-retrieve. An episode whose hyperedges
stay within its own boundaries, or link only to its immediate temporal neighbours, gains little from
HG traversal that VS with recency would not already give. **[inferred — HyperMem does not phrase
this as a selection criterion, since HyperMem does not select; the inference is from what its
hyperedges are shown to do]**

Second, the weakest category is open-domain (70.83), consistent with COLM's assignment of
weakly-structured content to VS rather than HG.

### 1.6 Where the encodings diverge, and what is measurable

Because all three artifacts exist at selection time (§0), each decision point can be checked against
what is _directly measurable on the built artifact_ rather than what must be proxied from raw text.

| Question                                                    | Favours | Directly measurable on | Measured today?                 |
| ----------------------------------------------------------- | ------- | ---------------------- | ------------------------------- |
| Does this episode bind into the store's hypergraph?         | HG      | `H_j`                  | Intended, but spec is ambiguous |
| Are the episodes it binds to temporally distant?            | HG      | `H_j` + store          | **No**                          |
| How many atomic, separately-addressable facts did it yield? | HG      | `H_j`                  | Proxied by NER `entity_count`   |
| Is a small crucial core buried in bulky auxiliary detail?   | VC      | `I_j`                  | **No** — proxy measures size    |
| Is the read budget tight relative to this episode?          | VC      | context                | Partly (`token_length`)         |
| Is the content coherent enough for one centroid?            | VS      | `v_j` + turns          | **No** (TTR ≠ dispersion)       |

Four of six have no coordinate. Two more are measurable on artifacts the pipeline already builds but
are currently proxied from raw text — which, once the artifacts are known to exist, is strictly
worse information for the same cost.

### 1.7 Trade-offs

|             | HG                                | VC                                       | VS                     |
| ----------- | --------------------------------- | ---------------------------------------- | ---------------------- |
| Read cost   | coarse-to-fine traversal          | one image, `O(L·s²)` visual tokens       | ANN top-k              |
| Wins when…  | evidence is scattered across time | budget is tight and importance is skewed | content is homogeneous |
| Fails when… | episode is topically isolated     | importance is uniform                    | episode is multi-topic |
| Evidence    | multi-hop 93.62 vs LightRAG +9.58 | 62.2% vs 31.6% @ B=16                    | COLM §1.2.3 assignment |

Note what is _absent_ from this table: write cost. Under unconditional encoding it is identical
across all three options and therefore not attributable to the decision. FluxMem Eq. 9's reward
`r = λ_judge·r_judge + λ_mem·r_mem`, with `r_mem` implemented in `fluxmem/supervision.py` as
`hit_rate / (1 + token_budget/reference_tokens)`, measures _retrieval_ token efficiency. That is the
correct quantity for this decision, and the absence of a write-cost term is correct rather than a
gap. **[verified against Eq. 9 and `supervision.py`; the sufficiency argument follows from §0]**

---

## Part 2 — Feature analysis

### 2.0 Provenance

COLM §1.3.3 describes the vector in prose — "the number of distinct named entities and their
co-occurrence density... the degree of temporal ordering... the topic diversity... and the total
token length," then "extended with two additional dimensions for hyperedge density and visual
salience" — and asserts these are "the same classes of features used by FluxMem." **[verified]**

`fluxmem/features.py` implements that sentence literally. The paraphrase drops seven Table 5
features and reinterprets two: `entity density` (per page) becomes a raw count, and `topic
diversity` (distinct topics) becomes a type-token ratio.

This is not a considered adaptation of FluxMem's feature engineering to a new target; it is a
transcription of a one-sentence summary. The existing audit
(`docs/FluxMem-Fidelity-Audit-2026-08-13.md:61`) correctly classifies this as a reading-rule choice
rather than drift — the code matches COLM, and ADR 0008 makes COLM authoritative. That defends
_fidelity_; this document asks about _fitness_.

### 2.1 `entity_count` — replace with an artifact measurement

`math.log1p(len(distinct_entities))` via spaCy NER. Predicts how many fact nodes an HG encoding
would yield — a reasonable thing to want. But the HG arm _actually builds_ those nodes, so once
Phase 2 lands the extractor can report the real count instead of a proxy for it. An NER mention
count and an LLM-extracted assertion count are not the same quantity, and it is the latter that
determines whether HyperMem's fact layer has anything to index.

Defect regardless: unnormalized, so confounded with episode length, and `token_length` is already in
the vector. FluxMem's Table 5 uses `entity density` — per page — for exactly this reason.

### 2.2 `cooccurrence_density` — remove

Executed against `features.py:83-101`:

```
A. 3 turns, k distinct entities per turn, no sharing
   k=2  n=6   density=0.2000      k=4  n=12  density=0.2727
   k=3  n=9   density=0.2500      k=5  n=15  density=0.2857

B. same 3 entities in every turn; turn 0 padded with extra entities
   n=3   density=1.0000           n=12  density=1.0000
   n=6   density=1.0000           n=24  density=1.0000
```

**[verified — executed]** Case B is the defect. The numerator is a `set` of co-occurring pairs
(line 95), so any single turn naming all `n` entities enumerates all `C(n,2)` possible pairs and
pins the feature to exactly 1.0 — for 3 entities and for 24 alike. One verbose turn saturates it.
Because pairs are deduplicated, an association that _recurs_ across turns contributes nothing past
its first occurrence.

Beyond the saturation bug, this is the wrong primitive twice over. HyperMem's contribution is that
pairwise relations cannot express joint dependencies among ≥3 elements — so a pairwise density
cannot see the structure a hypergraph is for. And once the hypergraph is actually built, counting
_hypothetical_ co-occurrences in raw text is a guess at a quantity that can be read off `H_j`
directly.

### 2.3 `temporal_ordering` — replace

```
strictly increasing timestamps, varying temporal-marker density
   markers/turn=0   value=0.5000
   markers/turn=1   value=0.5250
   markers/turn=3   value=0.5750
```

**[verified — executed]** The feature is `0.5·monotonic_fraction + 0.5·marker_rate`
(`features.py:104-122`). `monotonic_fraction` counts adjacent pairs with `b.timestamp > a.timestamp`
— 1.0 for any turn list in arrival order. `StubEpisodeProducer` assigns `ts = base_ts + t`
(`interfaces.py:212`), and any segmenter over sequential dialogue does the same. So half the feature
is a constant; the other half is a rate over a 9-word marker frozenset, empirically a few percent.
Realistic range: **[0.50, 0.58]**.

`MemOCREpisodeProducer` emits `turns=[]`, taking the `len(turns) < 2` branch to
`monotonic_fraction = 1.0` with no words — **exactly 0.5** for every VC episode.

The audit reported two constant-zero features. This is a third near-constant column, and
`StandardScaler` on a near-degenerate column amplifies noise rather than signal.

What was wanted is _wall-clock spread_, which FluxMem's `time span` motivates as "Longer spans
reduce effectiveness of pure recency-based retrieval" **[verified]** — and which HyperMem's
"scattered across time" identifies as the precise condition HG addresses. Within-episode ordering is
not that; the relevant span is _between_ an episode and the ones it binds to.

### 2.4 `topic_diversity` — replace

```
type-token ratio vs. length, identical generative process
   tokens=50    TTR=0.9200        tokens=800   TTR=0.2450
   tokens=200   TTR=0.6150        tokens=3200  TTR=0.0625
```

**[verified — executed]** TTR falls monotonically with length by construction; those four episodes
are drawn from the same 200-word vocabulary with the same distribution and differ only in length. So
`topic_diversity` is substantially a decreasing transform of `token_length`, already feature #5 —
two of seven columns largely redundant.

It also does not measure its name. FluxMem defines topic diversity as "Number of distinct topics
detected"; TTR is lexical variety, which a single-topic technical passage scores high on and a
multi-topic plain-language exchange scores low on.

FluxMem's `semantic complexity` — "Estimated semantic variability across recent pages (e.g.,
**embedding dispersion**)" **[verified]** — is the feature that was wanted, and COLM's paraphrase
dropped it.

### 2.5 `token_length` — keep unchanged

`math.log1p(total_tokens)`. The most directly relevant of the seven: it is the numerator of the
compression ratio that decides VC, and it bounds what VS's single vector is being asked to stand
for. Keep as-is.

### 2.6 `hyperedge_density` and `visual_salience` — keep as inputs, fix the definitions

Both are constant 0.0 today because their arms are stubs: `hetrep/hg/encoder.py:37-54` is a literal
no-op, `hetrep/vc/stub.py:29` assigns 0.0. **[verified]** Given §0, neither is circular — each is a
measurement of an artifact that is always built, available at the selector for free. They are the
best-positioned features in the vector, not the worst. The problems are definitional.

**`hyperedge_density` is under-specified.** `hetrep/hg/encoder.py:23,51` gives
`|incident_edges| / |possible_edges|` without saying whether "incident" is scoped to the episode's
own node set or to the store-wide hypergraph. The two readings measure opposite things:

- _Within-episode:_ how internally interconnected this episode's own nodes are. Close to what
  `cooccurrence_density` gropes at, and largely irrelevant to HyperMem's demonstrated advantage.
- _Store-wide:_ how strongly this episode binds into hyperedges that also contain other episodes.
  This is the quantity HyperMem's multi-hop result is attributed to.

The store-wide reading is the one supported by evidence, and the spec should say so. **[inferred —
the code comment is a Phase-2 plan, not a paper formula, so this is a definitional decision rather
than a fidelity question]**

**`visual_salience` measures the wrong quantity.** The only numeric definition anywhere is
`memocr_episodes.py:97-113`: rendered pixel area over a budget, capped at 1.0. That is render
_size_. MemOCR's mechanism is the _contrast_ between the crucial and the detail region (§1.4), and
an area-over-budget scalar has no contrast term — a large canvas that is uniformly important and a
large canvas with a tiny crucial core score identically. Its docstring is admirably explicit that
the proxy is "a modeling choice, not a cited fact"; the finding is that even as a proxy it measures
the wrong thing for this purpose. Since the canvas is actually rendered, the crucial/detail split is
directly observable and does not need proxying.

### 2.7 Verdict table

| Feature                | Measures                    | Verdict  | Reason                                                      |
| ---------------------- | --------------------------- | -------- | ----------------------------------------------------------- |
| `entity_count`         | NER entity mentions         | Replace  | Proxy for a count `H_j` reports directly; also unnormalized |
| `cooccurrence_density` | distinct pair coverage      | Remove   | Saturates at 1.0; pairwise primitive; subsumed by `H_j`     |
| `temporal_ordering`    | ≈0.5 + marker rate          | Replace  | Near-constant; the relevant span is cross-episode           |
| `topic_diversity`      | type-token ratio            | Replace  | Collinear with length; not topic detection                  |
| `token_length`         | episode size                | **Keep** | Directly relevant to VC and VS                              |
| `hyperedge_density`    | (0.0)                       | **Keep** | Not circular; resolve the spec store-wide                   |
| `visual_salience`      | (0.0); render area if built | **Keep** | Not circular; redefine as importance concentration          |

---

## Part 3 — Proposed feature set

### 3.1 Design principles

1. **Prefer measuring the artifact over proxying it from text.** All three encodings exist at
   selection time; a heuristic estimate of a quantity the pipeline has already computed exactly is
   strictly worse information at the same cost.
2. **Discriminate _for_ HG and _for_ VC; let VS be the residual** (COLM §1.2.3 defines VS by
   absence).
3. **Represent the read-side context, not just the episode.** VC's advantage is budget-conditional
   (p = 0.3419 at ample budget), and the decision is read-side.
4. **No near-degenerate or collinear columns** — the failure mode found three times in Part 2.

### 3.2 The proposed features

| #   | Feature               | Source        | Captures                                       | Change                       |
| --- | --------------------- | ------------- | ---------------------------------------------- | ---------------------------- |
| 1   | `token_length`        | episode       | episode scale                                  | **unchanged**                |
| 2   | `budget_ratio`        | read context  | episode size against the read budget           | new                          |
| 3   | `semantic_dispersion` | turns + `v_j` | whether one centroid can represent the episode | replaces `topic_diversity`   |
| 4   | `fact_node_yield`     | `H_j`         | atomic, separately-addressable facts extracted | replaces `entity_count`      |
| 5   | `hyperedge_density`   | `H_j` + store | how strongly the episode binds into the store  | **kept**, spec resolved      |
| 6   | `hyperedge_time_span` | `H_j` + store | how far across time those bindings reach       | replaces `temporal_ordering` |
| 7   | `visual_salience`     | `I_j`         | concentration of importance on the canvas      | **kept**, redefined          |

**7 → 7 dimensions.** Three features survive (one unchanged, two with corrected definitions);
`cooccurrence_density` is removed outright and three are replaced.

---

**1. `token_length`** — `log1p(total tokens)`, unchanged. The numerator of the ratio that decides
whether a canvas is worth reading, and a bound on how much one VS vector is being asked to stand
for. _Basis:_ FluxMem `avg page length`; MemOCR Fig. 1(b).

**2. `budget_ratio`** — `log1p(total tokens / read_token_budget)`. The coordinate the current vector
most conspicuously lacks. MemOCR's advantage is monotone in budget tightness and vanishes into
statistical insignificance when the budget is ample (+0.9, p = 0.3419) **[verified]**.

_Caveat:_ at a **fixed** budget this is a monotone transform of feature 1 and adds nothing. It earns
its place only if training spans multiple budgets — which is how MemOCR itself trains
(budget-aware objectives over deliberately varied budgets). **Include it and sweep the budget during
label collection; if the project settles on one fixed budget, drop it rather than carry a redundant
column.** **[assumed — a judgement about the training protocol, not a paper claim]**

**3. `semantic_dispersion`** — mean pairwise cosine distance among per-turn embeddings. FluxMem's
`semantic complexity`, restored after COLM's paraphrase dropped it. The cleanest _negative_
predictor for VS: `v_j` collapses the episode to one point, which is faithful only when the turns
are semantically coherent. High dispersion means the centroid represents none of the content well.

_Constraint:_ compute per-turn embeddings in a **separate** pass. Do not mean-pool turn vectors to
produce `v_j` as an optimization — COLM specifies `v_j = Enc(summary(e_j))`, and a mean of turn
vectors is not that. ADR 0008 forbids exactly this kind of silent substitution.

**4. `fact_node_yield`** — `log1p` of the number of fact nodes the HG extractor produced. Replaces
the NER proxy with the real quantity: whether HyperMem's fact layer has anything to index for this
episode. HyperMem's single-hop result (96.08) is attributed to atomic fact nodes enabling precise
retrieval **[verified]**, so an episode yielding few facts has little to gain from that layer.

**5. `hyperedge_density`** — resolved **store-wide**: the fraction of this episode's nodes
participating in at least one hyperedge that also contains nodes from a _different_ episode.
Bounded [0,1]. This measures the thing HyperMem's multi-hop advantage is attributed to — binding
across episodes — rather than internal interconnection. An episode whose nodes join no cross-episode
hyperedge is topically isolated, and HG traversal offers it nothing VS does not.

**6. `hyperedge_time_span`** — `log1p` of the wall-clock spread between this episode and the
episodes sharing a hyperedge with it. Pairs with #5: density gives the _topical_ binding, span gives
the _scattering_. HyperMem's claim is "topically related episodes **scattered across time**"
**[verified]**, and FluxMem's `time span` is motivated as "Longer spans reduce effectiveness of pure
recency-based retrieval" **[verified]**. High density with a _small_ span describes content that VS
plus recency already serves; high density with a _large_ span is the case HG exists for.

**7. `visual_salience`** — redefined as the **share of the rendered canvas area occupied by the
high-visibility (crucial) region**, bounded [0,1], replacing pixel-area-over-budget. Directly
observable on `I_j`, so no proxy is needed.

Its predictive content is interactional, not monotone on its own: a _small_ salient share combined
with large `token_length` and tight `budget_ratio` is MemOCR Fig. 1(b)'s picture exactly — a small
crucial core inside a large auxiliary remainder, under pressure. A two-layer MLP can represent that
interaction; the feature does not need to be independently monotone in VC-goodness. **[inferred —
MemOCR does not define a salience scalar; this operationalizes its crucial/detail region split,
which Figure 5's oracle analysis treats as the canvas's defining property]**

### 3.3 Considered and rejected

| Candidate                                              | Why not                                                                                                                       |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `entity_density` (spaCy NER, normalized)               | Subsumed by `fact_node_yield`, which counts what the extractor actually produced rather than mentions that may yield no fact  |
| `multi_entity_rate` (≥3 entities per turn)             | A heuristic estimate of hyperedge formation; `hyperedge_density` measures the outcome directly, so the proxy adds noise only  |
| `store_affinity` (cosine to store episodes)            | Same — a proxy for cross-episode binding that `hyperedge_density` measures directly once `H_j` exists                         |
| `auxiliary_ratio` (entity/date/numeral-free sentences) | A heuristic for the crucial/auxiliary split that the rendered canvas exposes directly via `visual_salience`                   |
| `topic_transitions` (FluxMem Table 5)                  | EM-LLM segments at surprise boundaries, so within-episode topic shifts are suppressed by construction — low variance expected |
| `is_qna_pattern`, `decision_tree`, `entity_centric`    | Discourse-shape indicators for choosing a _logical layout_; HG/VC/VS differ in medium, not layout                             |
| `page_count`, `avg_page_length`                        | FluxMem's document-page notion has no counterpart in dialogue episodes; `token_length` covers scale                           |
| Query-type features (single-hop / multi-hop mix)       | No query exists at write time — see §5.2                                                                                      |

Four of these eight rejections are proxies I would have proposed under a lazy-encoding assumption.
Unconditional encoding makes them redundant: when the artifact exists, measure it.

### 3.4 Computational cost

FluxMem App. G's bar is "negligible overhead... executed once per interaction" **[verified]**.

| Feature | Marginal cost over today                                                                                                                                     |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 4, 5, 7 | **zero at the selector** — read off artifacts HETREP already built. Features 5 and 6 need the store-wide hyperedge index, which HG retrieval requires anyway |
| 6       | O(k) over the episodes sharing a hyperedge — timestamps are already on `EpisodicUnit`                                                                        |
| 1, 2    | free / O(1)                                                                                                                                                  |
| 3       | **the only real cost** — `n_turns` MiniLM passes per episode instead of 1                                                                                    |

Feature 3 dominates: for a 10–20 turn episode, 10–20 extra MiniLM forwards, single-digit
milliseconds batched on GPU. Negligible against the LLM extraction and VLM read that the same
episode already pays for unconditionally. Removing the spaCy NER pass (features 1–2 of the old
vector) may make the net change _negative_. **[assumed — not benchmarked]**

---

## Part 4 — Dependencies and open questions

### 4.1 Phase gating, stated plainly

Five of seven features read artifacts, so the vector is phase-gated:

| Phase | Available                                                        |
| ----- | ---------------------------------------------------------------- |
| 1     | `token_length`, `budget_ratio`, `semantic_dispersion`            |
| 2     | \+ `fact_node_yield`, `hyperedge_density`, `hyperedge_time_span` |
| 3     | \+ `visual_salience` — complete                                  |

This is not a regression relative to a text-only proxy set, because the _pipeline_ is gated the same
way: under unconditional encoding there is no configuration in which HG or VC is selectable but
unbuilt. Designing text proxies to dodge the gating would buy an earlier feature vector for a
selector that still cannot be trained (§4.2), at the cost of permanently worse inputs.

### 4.2 Training remains blocked on Phase 3

`hetrep/vc/stub.py:6-9` states it: with VC stubbed, COLM Eq. 6's label `f*_j = argmax_f r_j^f` is
**unmeasurable for VC**, since `r^VC` requires rendering and reading a real canvas. No feature set
fixes a missing arm.

Recommended staging keeps the architecture fixed. **The 3-class head and `FORMAT_ORDER = (HG, VC,
VS)` stay exactly as they are** — this is not a proposal to retarget the selector, which was raised
and rejected previously. What changes is only the *training set*: in Phase 2, labels can be derived
for episodes where HG or VS is reward-optimal, but no example can carry a VC label, because `r^VC`
does not exist. The VC logit therefore stays untrained until Phase 3, which is a property of the
data, not of the model.

The practical consequence to respect: a selector trained on that set will never predict VC, so its
accuracy is only interpretable as HG-vs-VS discrimination and must be reported that way. This lets
features 4–6 be validated before feature 7 arrives, rather than shipping seven untested columns at
once. **[assumed — a judgement about training sequence, not a paper's claim]**

### 4.3 Open questions

- **Write-time selection predicts a read-time payoff.** HG's benefit appears on multi-hop questions,
  VC's under tight budgets; neither is known when `f_j` is assigned. Eq. 6's labels absorb the query
  distribution implicitly, so a selector trained under one query mix may not transfer. Mitigating
  this considerably: because all three encodings persist (§0), a wrong `f_j` degrades the primary
  read path rather than losing the information — worth confirming that ITERRET can in fact fall back
  to a non-primary encoding, which is outside this document's scope.
- **`budget_ratio` needs a budget sweep** to be non-redundant (§3.2).
- **`visual_salience`'s redefinition presumes the VC encoder exposes region geometry.** MemOCR's
  drafter produces structured markdown with headings and highlights, so the crucial region is
  identifiable in principle, but whether the Phase 3 renderer surfaces it as metadata is an
  integration requirement, not a given.
- **`hyperedge_density`'s store-wide definition makes the feature depend on store state at
  encode time.** An episode encoded when the store is nearly empty scores low through no property of
  its own. Whether to recompute it as the store grows, or accept the time-of-encoding snapshot, is
  unresolved and affects reproducibility of any selector trained on it.
- **`SelectorConfig.hidden_dim = 16` vs FluxMem's stated hidden size 4** **[verified]**. Unrelated
  to features; noted so it is not lost.

### 4.4 Incidental defects found

Not in scope, recorded so they are not lost:

- `hetrep/vc/stub.py:11` cites `docs/adr/0003-episode-producer-seam-and-visual-salience.md`; the
  file on disk is `docs/adr/0003-memocr-fluxmem-integration.md`. Stale cross-reference.
- `hetrep/vs/encoder.py::_summarize` iterates turns with `if "user" in turn` — dict-style membership
  against `Turn`, a `slots=True` dataclass. Observed by reading; **not** confirmed by execution, so
  it may be dead or guarded on a path I did not trace.
