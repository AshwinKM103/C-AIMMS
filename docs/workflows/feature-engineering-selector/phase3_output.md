# Phase 3 Output: Selector MLP Feature Set

**Status**: Complete
**Date**: 2026-08-15
**Agent**: data-scientist [Sonnet]
**Validation script**: `phase3_features.py` (this directory) — run with the `caimms` conda env
(`numpy`, `pandas` only, no torch needed).

---

## Summary

- **Total features: M = 15** (within the ≤20 budget, at the middle of the 12–16 target).
- **Grouped**: Geometry & Scale (3) · Efficiency & Memory Trade-off (2) · Complementarity &
  Overlap (4) · Robustness & Stability (3) · Task-Relevant / Domain Fit (3).
- **Testability is three-way, not binary** — see [Part 0](#part-0-a-correction-and-a-load-bearing-design-decision)
  and the [testability table](#testability-matrix): (1) mechanically computable now _and_
  numerically meaningful, (2) mechanically computable now but running on synthetic
  placeholder data (formula validated, signal not), (3) genuinely uncomputable until an
  encoder ships (fixed constant today).
- **4 features were designed, computed, found redundant on this sample, and dropped** —
  documented in [Rejected/Merged Features](#rejected--merged-features-shown-not-hidden).
  This is reported as a finding, not hidden.

---

## Part 0: A Correction, and a Load-Bearing Design Decision

Two things surfaced during data inspection that change how the rest of this document should
be read. Evidence-discipline requires surfacing both rather than quietly working around them.

### 0.1 The sample data is synthetic for all three encoders, not just HG/VC

Phase 2 and the task brief describe `vs_embeddings.npy` as "real semantic embeddings." I
checked `docs/workflows/feature-engineering-selector/sample_data_generator.py:42-46` (in this
same directory) against the array on disk:

```python
def generate_vs_embedding(seed: int = 0, dim: int = 384) -> np.ndarray:
    np.random.seed(seed)
    embedding = np.random.randn(dim).astype(np.float32)   # <-- Gaussian noise, not SentenceTransformer
    embedding = embedding / np.linalg.norm(embedding)
    return embedding
```

`vs_embeddings.npy` is `np.random.randn` per episode, L2-normalized — not `all-MiniLM-L6-v2`
output. Verified: `np.linalg.norm(vs, axis=1)` is 1.0 for all 10 rows in the saved array,
consistent with this generator and inconsistent with `phase2_output.md`'s claim that VS
output is "not cosine-normalized at encode time." I flag the contradiction rather than
resolving it silently — it could mean the sample-data generator normalized for convenience,
or that `hetrep/vs/encoder.py`'s actual behavior differs from what Phase 2 documented; I did
not re-run the real encoder to settle it.

**Consequence for this document**: every validation number below (VS, HG, and VC alike)
demonstrates that a _formula runs correctly, is bounded as specified, and produces a
sensible-looking range on data of the right shape_. None of it demonstrates that a feature
_discriminates real episodes meaningfully_ — the VS numbers are exercising Gaussian-vector
geometry, not sentence semantics. This is a strictly weaker claim than "VS features are
tested," and I use that weaker, correct phrasing throughout rather than the stronger one Phase
1/2 implied.

The one partial exception: `hg_data/adjacency_*.npy` is _not_ pure noise — `generate_hg_graph`
(same file, lines 72–103) builds a fixed index-bucket hierarchy (node 0 = root, nodes 1–2 =
level 1, ...) plus ~10% random edges. This gives the HG sample graphs non-trivial, non-random
topology, which is why the HG-derived features below show more structured (and more
inter-correlated — see [Part 4](#part-4-correlation-matrix-and-what-it-taught-me)) behavior
than the VS/VC ones. I did **not** use the index-bucket structure as a stand-in for
Facts/Episodes/Topics typing (see `hg_hierarchy_depth` below) because that structure is an
artifact of this mock generator's construction, not a property real HyperMem output — or even
the _intended_ 3-layer design — would necessarily share.

### 0.2 `task_accuracy` cannot be a selector input feature — it is the label, not a feature

The task brief's worked examples (`vs_memory_efficiency = performance_score / 1.5KB`,
`accuracy_gap = max_performance - min_performance`) require a downstream task-accuracy score
_per encoding, per episode_. That score is only knowable after running the episode through
reasoning under each format and scoring the result — exactly what `fluxmem/supervision.py`'s
label-generation step already does (Algorithm 3: run under each structure, compute reward,
assign `f*_j = argmax_f r_j^f` as the **training label**). If a selector's _input_ feature
included that score, the selector would need the answer before making the decision the score
is meant to evaluate — target leakage, not a feature.

So: `task_accuracy` (and any literal `performance_score / footprint` ratio) belongs on the
**label side** of training (already architected, per `fluxmem/selector.py`'s
`StandardScaler → MLP → cross-entropy` pipeline against offline-derived labels), not in the
15 features below. What the reward function's _structure_ (`accuracy / memory_budget`) does
license as **input features** is the denominator — real, computable, per-episode memory
footprint — plus proxies for the numerator that don't require ground truth (information
density, structural richness, concentration). That is what Group 2 and Group 5 below actually
compute. I made this substitution explicit rather than quietly building the leaking version
and hoping it looked like it matched the brief.

---

## Feature Definitions

### Group 1 — Geometry & Scale (3 features)

| #   | Feature                   | Formula                                                                                                                        | Range             | Encoder | Why it matters                                                                                                                                                                                                                                                                                        |
| --- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `vs_energy_concentration` | Participation ratio of squared components, rescaled: `PR=(Σv_i²)²/(D·Σv_i⁴)`, feature `=(PR−1/D)/(1−1/D)`, D=384               | (0, 1]            | VS      | High = embedding energy spread across many dims (isotropic, generalizes well to nearest-neighbor search); low = a few dims dominate (peaky, possibly overfit to one lexical cue). Single-vector-computable — no corpus needed.                                                                        |
| 2   | `hg_hierarchy_depth`      | `(distinct abstraction levels present) / 3`                                                                                    | Fixed `1/3` today | HG      | Genuinely future-only: the 3-layer Facts/Episodes/Topics typing (Phase 2 spec) has no analog in the current no-op stub or in this flat sample graph — there is no per-node `layer` field to count. Constant until typed 3-layer output exists; included for design completeness, not for today's MLP. |
| 3   | `vc_layout_concentration` | `1 − H_norm(S)`, where `S`=channel-mean spatial map (7×7→49 cells), `H_norm`=Shannon entropy of `S/ΣS` normalized by `log(49)` | [0, 1]            | VC      | MemOCR's design intent is that crucial evidence occupies large, high-visibility regions (low spatial entropy); auxiliary detail is spread thin (high entropy). High concentration = layout is doing its job.                                                                                          |

### Group 2 — Efficiency & Memory Trade-off (2 features)

See [Part 0.2](#02-task_accuracy-cannot-be-a-selector-input-feature--it-is-the-label-not-a-feature)
for why this group excludes literal `accuracy/footprint` ratios. It also excludes fixed-size
formats' footprints as separate features — see [4.3](#43-why-group-2-has-2-features-not-3-a-fixed-vs-variable-size-argument).

**Memory budget = tokens used to store the encoding, not disk space.** An earlier draft of this
section measured footprint in disk KB (bytes on the wire / on disk for the serialized array).
That was the wrong unit: the actual constraint a selector trades off against is the model's
context/token budget, not a storage-bytes budget -- a selector choosing between VS/HG/VC pays in
prompt tokens, and disk-KB and token-count do not even preserve the _ordering_ between formats
(see below). Corrected 2026-08-15; full rationale in
[DESIGN_RATIONALE.md](DESIGN_RATIONALE.md#memory-budget--tokens-not-disk-space).

| #   | Feature                    | Formula                                                                                     | Range                                         | Encoder | Why it matters                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --- | -------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4   | `hg_cost_position`         | `(hg_footprint_tokens - vs_footprint_tokens) / (vc_footprint_tokens - vs_footprint_tokens)` | [0.02, 2.27] observed; unbounded in principle | HG      | Where HG's per-episode footprint sits between VS's fixed 384-token (cheap) and VC's fixed 1176-token (expensive) bookends. Directly reward-relevant: this _is_ the memory-budget side of `accuracy/memory_budget`, expressed per-episode. Note the observed range now straddles and exceeds 1.0 -- under token accounting HG can be _more_ expensive than VC for larger graphs, the reverse of the old KB-based ordering (see below). |
| 5   | `hg_information_per_token` | `hg_relational_richness / hg_footprint_tokens`                                              | [3.5e-05, 5.8e-04] observed                   | HG      | Confidence-weighted relational density normalized by HG's own footprint -- an information-per-token proxy that does _not_ require ground-truth accuracy. The one ratio in this set whose numerator and denominator are not both driven by the same underlying scalar (verified -- see [4.2](#42-hg_information_per_token-is-not-a-trivial-rescaling)).                                                                                |

`hg_footprint_tokens(adj) = N_nodes*50 + N_edges*30` -- **documented assumption**: 50 tokens/node
and 30 tokens/edge are placeholder text-token estimates for a typed node/edge rendered into the
model's context (node: id, content/summary, confidence, temporal/spatial tags, keywords; edge:
type, role, endpoints) -- the low end of the 50-100 tokens/node guidance range and the midpoint
of the 20-50 tokens/edge range, not measured against a real HyperMem-produced payload. Replace
with measured constants once the real HG encoder lands.

`vs_footprint_tokens = 384*1 = 384` tokens is architecture-fixed (1 token per embedding
dimension) and `vc_footprint_tokens = 7*7*24 = 1176` tokens is likewise architecture-fixed --
49 spatial cells (the fixed 7x7 VC grid) at 24 tokens/cell. The 24-tokens/cell constant is
itself derived, not arbitrary: the token-budget guidance's absolute range ("784-1600 tokens
typical") is anchored to a finer 28x28=784-patch grid at 1-2 tokens/patch; our fixed 7x7 grid is
coarser, so each of our cells covers a 4x4 block of that finer grid (28/7 = 4 per side, 16 fine
patches/cell) -> 16-32 tokens/cell, and 24 is the midpoint. Channel depth (512) is deliberately
excluded from the token count -- a token here represents one rendered/tokenized patch position,
not one raw float, so a deeper feature map at the same spatial grid costs the same tokens to
store, unlike the old KB accounting where every one of the 512 channels' raw float32 values
counted toward the footprint. This is the largest single driver of the ordering flip described
above: under KB, VC's raw (7,7,512) float32 tensor (98 KB) dwarfed HG's small JSON-ish payload
(~1-6 KB); under tokens, VC's _rendered_ representation (1176 tokens) is modest next to what a
node/edge-heavy HG graph costs to spell out as text (up to 2180 tokens on this sample, see
below).

### Group 3 — Complementarity & Overlap (4 features)

Naive cross-modal cosine similarity is not meaningful here: VS is 384-d, HG node features are
64-d, VC is a (7,7,512) map — comparing raw vectors across incompatible spaces would require an
arbitrary, currently-nonexistent learned projection, and any number produced would be noise
dressed as signal. I rejected that approach and instead compare each encoder's **spread**
statistic — how concentrated vs. distributed the encoder's information is across its own basis
units (reusing the un-flipped entropy/participation-ratio quantities from Group 1, before any
polarity flip). This measures _structural_ agreement (do both encoders show similarly
(de)concentrated information for this episode?), not semantic agreement (do they agree on
_what_ the important content is) — a real limitation, stated plainly in
[4.4](#44-what-structural-agreement-does-and-does-not-measure).

| #   | Feature                      | Formula             | Range                 | Why it matters                                                                                                                                                                                |
| --- | ---------------------------- | ------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6   | `vs_hg_structural_agreement` | `1 −                | vs_spread − hg_spread | `                                                                                                                                                                                             | [0, 1] | High = VS and HG concentrate information similarly → redundant signal; low = one finds structure the other doesn't → complementary. |
| 7   | `vs_vc_structural_agreement` | `1 −                | vs_spread − vc_spread | `                                                                                                                                                                                             | [0, 1] | Same, for VS/VC.                                                                                                                    |
| 8   | `hg_vc_structural_agreement` | `1 −                | hg_spread − vc_spread | `                                                                                                                                                                                             | [0, 1] | Same, for HG/VC.                                                                                                                    |
| 9   | `complementarity_score`      | `1 − mean(6, 7, 8)` | [0, 1]                | Aggregate: high = the three encoders are structurally distinct (selector's choice matters more); low = they overlap (selector's choice matters less). Requested explicitly in the task brief. |

### Group 4 — Robustness & Stability (3 features)

**The weakest-evidence group in this document, honestly stated.** The correct definition of
robustness is _input_-perturbation stability: paraphrase the episode's text, re-run the full
encoder pipeline, and measure how much the output moves. That requires the live encoder
(`SentenceTransformer`, the HG extractor, the MemOCR renderer) and the original episode text —
none of which this static `.npy` snapshot provides, for any of the three encoders, including
VS. What I computed instead is a weaker, output-space proxy: perturb the _saved array itself_
with small Gaussian noise and re-measure a derived statistic. This is a legitimate local
sensitivity measure but is **not** a substitute for true paraphrase-invariance testing — stated
here so the gap is not silently absorbed into "tested."

| #   | Feature                  | Formula                                                    | Range                                 | Encoder                                                    | Caveat                                                                                     |
| --- | ------------------------ | ---------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 10  | `vs_embedding_stability` | `1 − E[‖v̂ − (v+ε)/‖v+ε‖‖₂ / 2]`, ε~N(0, 0.01²I), 20 trials | [0, 1]                                | VS                                                         | Proxy for local Lipschitz sensitivity around the current point, not paraphrase robustness. |
| 11  | `hg_graph_stability`     | `1 − E[                                                    | richness(A) − richness(A+ε)           | ]`, ε~N(0, 0.02²) on existing edge weights only, 20 trials | [0, 1]                                                                                     | HG  | Uses `hg_relational_richness` (continuous, weight-sensitive) as the base statistic — **not** `hg_edge_density`, which I tried first and found saturates at a constant 1.0 for this noise level, because a count-based statistic only moves when noise flips an edge across the zero threshold (essentially never at σ=0.02 against these weights). Documented as a finding in [4.1](#41-a-metric-choice-bug-i-caught-count-based-statistics-dont-respond-to-small-perturbation). |
| 12  | `vc_layout_stability`    | `1 − E[                                                    | concentration(F) − concentration(F+ε) | ]`, ε~N(0, 0.02²), 10 trials                               | [0, 1]                                                                                     | VC  | Saturates near 1.0 on this sample (see [4.1](#41-a-metric-choice-bug-i-caught-count-based-statistics-dont-respond-to-small-perturbation)) — current VC noise sits at the spatial-entropy ceiling, so there's nowhere for a small perturbation to push it. Expect real variance once real rendered layouts (concentrated, off-ceiling) exist.                                                                                                                                     |

### Group 5 — Task-Relevant / Domain Fit (3 features)

| #   | Feature                   | Formula                                                    | Range                          | Encoder | Why it matters                                                                                                                                                                                                                                                                                                                           |
| --- | ------------------------- | ---------------------------------------------------------- | ------------------------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 13  | `vs_semantic_specificity` | `1 − max(cos(v, corpus_mean), 0)`                          | [0, 1]                         | VS      | High = embedding is distinctive relative to the batch centroid (VS's fine-grained similarity search will actually discriminate it); low = near-generic (VS may not separate it from other episodes — HG/VC's explicit structure may matter more). Requires a corpus/batch to define "generic," unlike the single-vector Group 1 feature. |
| 14  | `hg_relational_richness`  | `Σ(edge weights) / (N·(N−1))`                              | [0, 1]                         | HG      | Confidence-weighted relational density (Phase 2 emphasizes per-edge confidence scores as part of the intended design) — higher = richer, higher-confidence relational content, matching HG's stated strength (entity-centric, multi-hop reasoning).                                                                                      |
| 15  | `vc_layout_prominence`    | `(max(S) − mean(S)) / (max(S) + mean(S))`, `S`=spatial map | [−1, 1], observed [0.03, 0.06] | VC      | Peak-to-mean ratio of the spatial map — does one region stand out as "most important," matching MemOCR's stated design (crucial evidence gets prominent placement)? Low observed values on this sample correctly flag that the current placeholder data has no such standout region (it's noise).                                        |

---

## Validation on Sample Data

Computed with `phase3_features.py` on the 10 episodes in `/tmp/selector_sample_data/`
(`np.random.seed(42)` for the perturbation trials in Group 4; the underlying sample arrays are
whatever `sample_data_generator.py` produced, per [Part 0.1](#01-the-sample-data-is-synthetic-for-all-three-encoders-not-just-hgvc)).

### Fixed architecture constants

```
vs_footprint_tokens = 384.0 tokens  (384 dims × 1 token/dim, constant across all episodes)
vc_footprint_tokens = 1176.0 tokens (7×7 grid × 24 tokens/cell, constant across all episodes)
```

### Summary statistics (final 15 features)

| Feature                      | min     | max     | mean    | std                 |
| ---------------------------- | ------- | ------- | ------- | ------------------- |
| `vs_energy_concentration`    | 0.2388  | 0.3509  | 0.3031  | 0.0327              |
| `hg_hierarchy_depth`         | 0.3333  | 0.3333  | 0.3333  | ~0 (exact constant) |
| `vc_layout_concentration`    | 0.0001  | 0.0003  | 0.0002  | 5.5e-05             |
| `hg_cost_position`           | 0.0202  | 2.2677  | 1.0884  | 0.7396              |
| `hg_information_per_token`   | 3.5e-05 | 5.8e-04 | 1.8e-04 | 1.9e-04             |
| `vs_hg_structural_agreement` | 0.3118  | 0.4016  | 0.3604  | 0.0314              |
| `vs_vc_structural_agreement` | 0.2392  | 0.3512  | 0.3033  | 0.0326              |
| `hg_vc_structural_agreement` | 0.8935  | 0.9731  | 0.9429  | 0.0225              |
| `vs_embedding_stability`     | 0.9026  | 0.9074  | 0.9043  | 0.0013              |
| `hg_graph_stability`         | 0.9975  | 0.9998  | 0.9991  | 0.0008              |
| `vc_layout_stability`        | 1.0000  | 1.0000  | 1.0000  | ~0                  |
| `vs_semantic_specificity`    | 0.6555  | 0.7936  | 0.7045  | 0.0428              |
| `hg_relational_richness`     | 0.0743  | 0.2481  | 0.1348  | 0.0583              |
| `vc_layout_prominence`       | 0.0314  | 0.0568  | 0.0426  | 0.0090              |
| `complementarity_score`      | 0.4325  | 0.5072  | 0.4645  | 0.0218              |

### Zero/near-zero-variance features (std < 1e-6)

`hg_hierarchy_depth` (exact, by construction — future-only, as documented) and
`vc_layout_stability` (std ≈ 9.4e-7 — a ceiling-saturation effect from noise-level input, see
below). Both are expected, not bugs; both are called out per-feature above.

---

## Part 4: What Validation Actually Found (not just "ran without error")

### 4.1 A metric-choice bug I caught: count-based statistics don't respond to small perturbation

First draft of `hg_graph_stability` used `hg_edge_density` (a `count(nonzero) / (N(N-1))`
statistic) as the perturbed-and-remeasured base. Result: exactly 1.0 stability for every
episode, every trial — the perturbation (σ=0.02 Gaussian jitter on existing edge weights,
clipped to [0,1]) essentially never flips a weight across the zero threshold at this noise
level, so `count(nonzero)` never changes. That's not "perfectly stable," it's "measuring
something insensitive to the perturbation being applied." I switched the base statistic to
`hg_relational_richness` (continuous, weight-magnitude-sensitive) and got real per-episode
variance (std 0.0008, range 0.9975–0.9998 — small but non-degenerate). `vc_layout_stability`
has the same saturation shape for a different reason: the VC sample data is near-maximum-entropy
noise (see 4.1 below on `vc_layout_concentration`), so it's already sitting at the ceiling the
entropy statistic can reach — there's nowhere for a small perturbation to push it. This should
resolve once real rendered layouts (which are supposed to be _concentrated_, i.e., off the
entropy ceiling) exist; I flag it as a monitoring item for the Phase 3+ validation gate rather
than fix it further against synthetic noise.

### 4.2 `hg_information_per_token` is not a trivial rescaling

Sanity check (bottom of `phase3_features.py`): a naive `vs_memory_efficiency =
vs_energy_concentration / vs_footprint_tokens` correlates `r = 1.000000` with
`vs_energy_concentration` alone — exactly, because `vs_footprint_tokens` is
architecture-constant, so dividing by it is a pure rescaling that adds zero information for a
per-episode classifier. This is the concrete demonstration behind the [Group 2 sizing decision](#43-why-group-2-has-2-features-not-3-a-fixed-vs-variable-size-argument).
`hg_information_per_token`, by contrast, divides by `hg_footprint_tokens`, which _does_ vary
per-episode (it's driven by node/edge count, not richness) — so the ratio is not a pure
rescaling of either operand. It is, however, correlated with both `hg_relational_richness`
(r=0.966) and `hg_cost_position` (r=−0.845) on this sample (recomputed under the token-based
footprint; near-identical to the KB-based run's r=0.967/r=−0.847, since these correlations are
driven by the same underlying node/edge-count structure regardless of the per-unit constant) —
discussed as a probable structural (not purely small-N) effect in
[Part 4.5](#45-high-correlation-pairs-and-which-ones-i-trust).

### 4.3 Why Group 2 has 2 features, not 3: a fixed-vs-variable-size argument

VS and VC representations are architecture-fixed size (384-d always; 7×7×512 always), so their
footprints are the _same number on every episode_. Any "memory efficiency" feature built as
`{signal}/{footprint}` for VS or VC is, by construction, a scalar rescaling of `{signal}` alone
— confirmed exactly in 4.2. HG is the only one of the three whose representation size varies
with episode content (node/edge count), so it's the only encoder where a footprint-normalized
feature carries information beyond its own numerator. I considered padding this group to 3
features for template-symmetry with the other four groups, and rejected it: a third feature
here would necessarily be another transform of `hg_footprint_tokens` (I tried `hg_cost_share`,
`hg_memory_footprint_log_tokens` — both r > 0.97 with `hg_cost_position` under the token-based
footprint (0.982 and 0.970 respectively — down slightly from the KB-based run's r>0.98, but
still well above the 0.9 collinearity threshold used throughout this document), see
[Rejected/Merged](#rejected--merged-features-shown-not-hidden)). Reporting 2 honestly-distinct
features is better than reporting 3 that are secretly the same scalar.

### 4.4 What "structural agreement" does and does not measure

Group 3's features answer "do these two encoders concentrate information similarly for this
episode," not "do these two encoders agree on _what the important content is_." The latter —
real semantic overlap — needs aligned provenance (which HG node / which VC region corresponds
to which part of the episode VS embedded), which doesn't exist in this synthetic sample
(no shared index across the three arrays) and won't exist until the encoders are real. Once HG
and VC are implemented, the correct upgrade is a **round-trip check**: embed HG's
episode-node summary text (or VC's OCR'd rendered text) with the same VS encoder and compute a
real cosine similarity against `v_j` — genuinely comparable, same embedding space. I did not
attempt to fake that comparison with an arbitrary random projection now; a fabricated number
would be worse than an honestly-labeled proxy.

### 4.5 High-correlation pairs, and which ones I trust

At `|r| > 0.9` (N=10 — correlation estimates at this sample size have wide confidence
intervals; I distinguish what's _structurally_ expected to persist from what's plausibly a
small-N coincidence rather than asserting either without support):

| Pair                                                   | r      | Classification                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------ | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hg_information_per_token` ↔ `hg_relational_richness`  | 0.966  | **Structural, likely persists.** Both derive from the same weighted adjacency; on this generator, edge count scales roughly linearly with node count (`e ≈ 1.9n`), so richness (∝ 1/n-ish) and footprint (∝ n) move in tandem, and dividing one by the other doesn't fully decouple them. Real HyperMem output may or may not preserve this linear node/edge scaling — worth re-checking once real data lands, not assumed to hold. (Recomputed under the token-based footprint; r=0.967 under the old KB-based one — the correlation is driven by node/edge-count structure, not the per-unit constant, so it survives the unit change essentially unchanged.)                  |
| `hg_information_per_token` ↔ `hg_graph_stability`      | −0.977 | **Structural, mechanistically explained.** Larger graphs (more nodes/edges) average out per-edge perturbation noise more (a sample-size/CLT-shaped effect: `Δrichness ≈ σ√E / (N(N−1))` shrinks as N grows), so stability rises with graph size while richness falls with it (density dilutes as graphs grow) — two statistics both driven by the same underlying "episode size" confound, opposite directions.                                                                                                                                                                                                                                                                  |
| `vs_vc_structural_agreement` ↔ `complementarity_score` | −1.000 | **Placeholder artifact, not structural.** `vc_spread` has std 5.2e-5 versus `vs_spread`'s std 0.033 (≈600× smaller) — i.e., the VC sample data sits at a near-constant, near-maximum entropy value (consistent with it being unstructured Gaussian-then-ReLU-then-smoothed noise, not a real rendered layout). Any feature combining `vc_spread` with a real-varying operand collapses onto that operand. This is expected to break (decorrelate) once VC produces genuine per-episode layout variance — flagged as a re-validation gate, not a reason to drop the feature now, since the _formula_ is sound and the current degeneracy is a data problem, not a design problem. |

I dropped the first category's redundant siblings (`hg_edge_density`, `hg_cost_share`,
`hg_memory_footprint_log_tokens` — see below) because they measure the _same_ thing as a kept
feature through the _same_ mechanism. I kept `vs_vc_structural_agreement` and
`complementarity_score` despite r=−1.000 because their redundancy is diagnosed as a stub-data
artifact, not a mechanism that will necessarily persist — dropping a correctly-designed feature
because today's placeholder data can't yet exercise it would contradict the Option A design
mandate (design for intended future behavior).

---

## Rejected / Merged Features (shown, not hidden)

| Feature                                               | Formula                                                             | Why rejected                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hg_edge_density`                                     | `count(nonzero)/(N(N−1))`                                           | r=0.989 with `hg_relational_richness` (= density × mean_edge_weight; mean edge weight has low variance relative to density on this sample, verified directly: density range 0.114–0.300 vs. mean-weight range 0.63–0.87 — so richness inherits most of its variance from density). Also the metric whose count-based insensitivity broke `hg_graph_stability` (4.1). Superseded by `hg_relational_richness` (Group 5) and `hg_hierarchy_depth` (Group 1).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `format_footprint_spread`                             | `std([vs_tokens, hg_tokens, vc_tokens]) / mean(...)`, clipped [0,1] | Under the KB-based footprint this saturated at exactly 1.0 for all 10 episodes (`vc_footprint_kb`=98 KB dominated the mean/std by 1-2 orders of magnitude). That saturation claim no longer holds verbatim under tokens: recomputed with `vs_footprint_tokens`/`hg_footprint_tokens`/`vc_footprint_tokens`, this now ranges [0.40, 0.59] (std 0.075, no longer zero-variance) and correlates only r=0.25 with `hg_cost_position` — the two are no longer near-redundant. It remains excluded from the final 15 regardless: `hg_cost_position`'s bookend-position semantics (0=as cheap as VS, 1=as expensive as VC) are more directly interpretable than a three-way coefficient of variation, and reopening Group 2's feature count is a separate design decision from this units correction, out of scope here (would need its own ADR if pursued) — flagged rather than silently left on the old, now-false "saturates at 1.0" claim. |
| `hg_cost_share`                                       | `hg_tokens / (vs_tokens+hg_tokens+vc_tokens)`                       | r=0.982 with `hg_cost_position` under the token-based footprint (was r=0.9999 under KB) — both are monotonic reparameterizations of the same single scalar (`hg_footprint_tokens`) against the same two constants; still comfortably above the 0.9 collinearity threshold. Kept `hg_cost_position` for its more directly interpretable bookend semantics (0=as cheap as VS, 1=as expensive as VC).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `hg_memory_footprint_log_tokens`                      | `log1p(hg_footprint_tokens)`                                        | r=0.970 with `hg_cost_position` (was r≥0.985 under KB). Its raw (unbounded, log-scale) information is retained inside `hg_cost_position`'s formula as an intermediate value; not worth exposing separately as an MLP input.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| naive `vs_memory_efficiency` / `vc_memory_efficiency` | `signal / footprint_tokens`                                         | Exact rescalings of `signal` (r=1.000000, footprint is architecture-constant for VS/VC) — see [4.2](#42-hg_information_per_token-is-not-a-trivial-rescaling)/[4.3](#43-why-group-2-has-2-features-not-3-a-fixed-vs-variable-size-argument). This is also the leaking-template version rejected in [Part 0.2](#02-task_accuracy-cannot-be-a-selector-input-feature--it-is-the-label-not-a-feature).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

---

## Design Decisions

### Relationship to FluxMem's 12 / C-AIMMS's 7 features

Per `phase1_output.md`'s gap analysis: FluxMem's 12 features measure **conversation-level**
properties (entities, Q&A structure, topic transitions) extracted from dialogue text; this
Phase 3 set measures **encoder-output** properties (embedding geometry, graph structure,
spatial layout) extracted from `unit.embedding` / the HG structure / the VC render. They are
not comparable feature-for-feature and I did not attempt to force a 1:1 mapping. The one direct
relationship is at the placeholder boundary: C-AIMMS's current 7-feature selector reads
`hyperedge_density` and `visual_salience` as opaque scalars off the episode
(`fluxmem/features.py:extract_features`, lines ~160–161). This Phase 3 set is a superset
replacement for exactly those two placeholders:

- `hyperedge_density` (1 scalar) → `hg_hierarchy_depth`, `hg_cost_position`,
  `hg_information_per_token`, `hg_relational_richness`, `hg_graph_stability`,
  `vs_hg_structural_agreement`, `hg_vc_structural_agreement` (7 features)
- `visual_salience` (1 scalar) → `vc_layout_concentration`, `vc_layout_prominence`,
  `vc_layout_stability`, `vs_vc_structural_agreement`, `hg_vc_structural_agreement`,
  `complementarity_score` (6 features, 2 shared with the HG list above)

The other 5 features (`vs_energy_concentration`, `hg_cost_position` covered above,
`vs_embedding_stability`, `vs_semantic_specificity`, plus the shared `complementarity_score`)
add VS-side and cross-modal signal the current 7-feature selector has no analog for at all
(VS currently contributes no geometry feature — only the FluxMem-inherited
`entity_count`/`cooccurrence_density`/etc., which are conversation-level, not embedding-level).

### Testability Matrix

| Feature                      | Formula runs on sample data       | Numeric result is meaningful signal today                                                                                               |
| ---------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `vs_energy_concentration`    | Yes                               | No — sample VS is `np.random.randn`, not real embeddings ([0.1](#01-the-sample-data-is-synthetic-for-all-three-encoders-not-just-hgvc)) |
| `hg_hierarchy_depth`         | Yes (trivially — it's a constant) | No — no typed-layer field exists yet, by design                                                                                         |
| `vc_layout_concentration`    | Yes                               | No — sample VC is smoothed noise, not a rendered layout                                                                                 |
| `hg_cost_position`           | Yes                               | Partial — footprint arithmetic is real, but the per-node/edge token constants are documented assumptions, not measured                  |
| `hg_information_per_token`   | Yes                               | Partial — same caveat                                                                                                                   |
| `vs_hg_structural_agreement` | Yes                               | No — structural-agreement proxy on random data ([4.4](#44-what-structural-agreement-does-and-does-not-measure))                         |
| `vs_vc_structural_agreement` | Yes                               | No — see [4.5](#45-high-correlation-pairs-and-which-ones-i-trust) collinearity finding                                                  |
| `hg_vc_structural_agreement` | Yes                               | No                                                                                                                                      |
| `complementarity_score`      | Yes                               | No                                                                                                                                      |
| `vs_embedding_stability`     | Yes                               | No — output-space perturbation proxy, not input-text robustness ([Group 4 preamble](#group-4--robustness--stability-3-features))        |
| `hg_graph_stability`         | Yes                               | No                                                                                                                                      |
| `vc_layout_stability`        | Yes                               | No — saturated at ceiling ([4.1](#41-a-metric-choice-bug-i-caught-count-based-statistics-dont-respond-to-small-perturbation))           |
| `vs_semantic_specificity`    | Yes                               | No                                                                                                                                      |
| `hg_relational_richness`     | Yes                               | Partial — mechanically sound, exercised on a mock generator with plausible-looking but not extraction-derived weights                   |
| `vc_layout_prominence`       | Yes                               | No                                                                                                                                      |

**Every formula runs; none produces a trustworthy per-episode signal yet**, because every
sample array in this dataset is synthetic (Part 0.1). This is a stronger and more honest
statement than "VS is tested, HG/VC are not," which is what I initially assumed before
inspecting `sample_data_generator.py` and would have reported without checking.

---

## Feature Correlation Matrix

Full 15×15 matrix (Pearson, N=10) is printed by `phase3_features.py`; only pairs with `|r| >
0.9` are actionable at this sample size and all are discussed in [4.5](#45-high-correlation-pairs-and-which-ones-i-trust)
above. `hg_hierarchy_depth` correlates `NaN` with everything (exact constant, zero variance —
expected, not a bug).

---

## Next Steps for Implementation

1. **Extend `fluxmem/features.py`**: `FEATURE_NAMES` grows from 7 to `7 − 2 + 15 = 20` (drop
   the two placeholder scalars `hyperedge_density`/`visual_salience`, add the 15 here) — at the
   `M ≤ 20` ceiling exactly. `FEATURE_DIM` changes accordingly; this is a breaking change to
   any already-trained/pickled `FormatSelector` (per `.claude/rules/storage-invariants.md`,
   pickle artifacts are never committed anyway — retraining from scratch is the expected path,
   not a bug to route around).
2. **Extraction function shape**: one function per group (`extract_geometry_features`,
   `extract_efficiency_features`, `extract_overlap_features`, `extract_robustness_features`,
   `extract_task_features`), each taking the three raw encoder outputs
   (`vs_embedding: np.ndarray(384,)`, `hg: tuple[adjacency, node_features]`,
   `vc: np.ndarray(7,7,512)`) plus a `corpus_vs_mean` for `vs_semantic_specificity` — mirrors
   `phase3_features.py`'s functions directly, which are the reference implementation.
3. **Normalization**: 12 of 15 features are formula-bounded to [0,1] or a documented-small
   observed range already; `hg_cost_position` and `hg_information_per_token` are unbounded above
   in principle (footprint/richness could exceed the observed sample range once real HG data
   arrives -- and already does exceed it in the token-based run above, where `hg_cost_position`
   ranges up to 2.27 on this synthetic sample rather than staying near [0, 1]). Recommend feeding the full 15 through the existing `StandardScaler` (fit on train
   only — `fluxmem/selector.py`'s pattern already does this correctly) rather than hand-rolling
   a second normalization scheme; the bounded features don't need it but z-scoring them again is
   harmless, and the two unbounded ones need it.
4. **Corpus mean for `vs_semantic_specificity`**: unlike the other 14 (computable episode-by-
   episode), this one needs a running mean over previously-seen episodes' VS embeddings —
   requires either a batch context at feature-extraction time or an incrementally-updated
   running mean cached alongside the VS index. Flag for the implementer: this is the one
   feature in the set that is not a pure function of a single episode's encoder outputs.
5. **Validation gate before trusting any HG/VC number in production**: re-run
   `phase3_features.py`'s correlation-matrix step against real HyperMem/MemOCR output once
   those encoders exist, and specifically re-check the two "structural, likely persists" pairs
   in [4.5](#45-high-correlation-pairs-and-which-ones-i-trust) — if they hold up on real data
   (not just this N=10 synthetic sample), drop one of each pair for the production feature set;
   if they decorrelate, keep both as currently designed.
