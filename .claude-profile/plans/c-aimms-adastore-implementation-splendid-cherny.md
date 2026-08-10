# ADASTORE: three-tier hierarchy, format-adaptive selector, BMM-gated fusion

> **Fresh session? Start here.** This plan is self-contained — the *Equation reference card*
> below carries every formula needed, so **you do not need to reopen the PDFs** to implement.
> Execute Steps 0→8 in order; each step ends with tests passing (paste output, don't summarize),
> `codegraph sync`, and a conventional commit. **Step 5 is a hard gate**: ADR 0002 must be
> written and approved before any Step 6 code. Hard prohibition: **never read, import, or copy
> conventions from `LightMem/`** — it is a confirmed-wrong download that happens to contain an
> unrelated `src/fluxmem/`. The FluxMem *paper* in `docs/` is a primary source and is fine to read;
> the `LightMem/` *directory* is not.

## Context

`docs/COLM_Cognitive_AI_Memory_Architecture_Project.pdf` describes C-AIMMS, a four-component
memory framework (HETREP → ADASTORE → ITERRET → WORKMEM). None of it is implemented in this
repo: outside `LightMem/` (a separate, gitignored git repo — a confirmed-wrong download) and the
vendor toolkit clone, there is **zero Python** at the C-AIMMS root (verified: `find . -name
"*.py"` excluding those two trees returns nothing; no root `pyproject.toml`, no `tests/`).

This plan implements exactly two paragraphs of that paper — **§1.3.2** (three-tier memory
hierarchy) and **§1.3.3** (format-adaptive selector, offline supervision, BMM-gated fusion) — as
a new top-level `fluxmem/` package. The intended outcome is a tested, self-contained ADASTORE
that a later HETREP/ITERRET effort can plug into, without touching or importing `LightMem/`.

### The research finding that reshapes this plan

The kickoff framing treated the per-format reward `r^HG, r^VC, r^VS` as the single highest-risk
ambiguity, because COLM §1.3.3 only says "evaluate the downstream response quality and memory
utilization effectiveness." **That is under-specified in COLM but specified in its cited source.**
Reading `docs/Choosing How to Remember Adaptive Memory Structures for LLM Agents.pdf` (Lu et al.,
2026 — the FluxMem paper COLM cites for both the feature set and the BMM gate) yields:

| Item | Source | Content |
|---|---|---|
| Reward form | FluxMem Eq. 9 + App. D Alg. 2 | `r(s) = λ_judge·r_judge(s) + λ_mem·r_mem(s)`, `λ = 0.7 / 0.3`, `y = argmax_s r(s)` |
| Selector training | FluxMem App. D Alg. 3 | StandardScaler → softmax → cross-entropy → Adam → early stop on validation |
| Score normalization | FluxMem Eq. 17 | min–max to `(0,1)` with `ε` inset; `x_i ← 0.5` when `s_max = s_min` |
| BMM EM | FluxMem Eqs. 18–24 | quantile init (`q₀.₃`/`q₀.₇`), **log-space** E-step, responsibility-weighted moment M-step, `κ = μ(1−μ)/σ² − 1` |
| High-compat component | FluxMem Alg. 1 line 11 | `k* = argmax_k α_k/(α_k+β_k)` (larger Beta mean) |
| Merge decision | FluxMem Alg. 1 + Eq. 26 | threshold `τ` on posterior, **min-keep `m_min` top-k fallback**, merge into `argmax_{i∈I} x_i` |
| STIM eviction | FluxMem Eq. 5 | evict the `\|M\|−C` pages minimising `τ(p)` = last access time (true LRU, not FIFO) |
| LTSM eligibility | FluxMem Eq. 6 | `u(m) ≥ τ_u ∧ r(m) ≥ τ_r (∧ c(m) ≥ τ_c)` — third gate optional |

So the ADR narrows to concretizing `Judge(·)` and `MemUtil(·)` only. Everything else above is
citable, not invented. **`m_min` (min-keep) does not appear in COLM at all** — it is load-bearing
in Alg. 1 and would have been silently dropped without reading the source paper.

### Decisions taken (from your answers)

1. **Reward** — anchor on FluxMem Eq. 9 as paper-specified; ADR scopes only `Judge`/`MemUtil`.
2. **Features** — 7 scalars under the paper's 6 named classes; `d` derived from `FEATURE_NAMES`.
3. **Vector store** — FAISS-backed from day one (requires installing `faiss-cpu`, currently absent).
4. **MLP framework — you delegated; I choose PyTorch.** Reasoning: torch 2.8.0+cu128 is already
   installed, so it is not a *net* new dependency; the rest of C-AIMMS (frozen Qwen3 backbone
   §1.5, δ-mem OSAM, HETREP encoders) is unavoidably torch, so an sklearn selector would be the
   one component needing a rewrite at integration time; and an explicit loop makes Eq. 6's
   cross-entropy literal rather than hidden inside `MLPClassifier`. Cost is slower, seed-sensitive
   tests — mitigated by a tiny CPU-only net, fixed `torch.manual_seed`, and `torch.use_deterministic_algorithms(True)`.
   sklearn's `StandardScaler` is still used for Alg. 3 line 1, since the algorithm names it.
5. **NER** — real spaCy NER behind a Protocol, not a capitalization heuristic (see Step 4).

### Reading rule (governs every ambiguity below)

> **COLM is the specification. FluxMem fills COLM's *silences* only. Where the two *conflict*,
> COLM wins.**

COLM cites FluxMem as its source for both the feature classes and the BMM gate, and the two share
tier names, capacities, and equation forms — so FluxMem's appendices are the implementation spec
for sentences COLM compresses (that is how log-space EM, quantile init, and `m_min` were
recovered; **`m_min` appears nowhere in COLM but is load-bearing in FluxMem Alg. 1**). But COLM is
what we are implementing. Two consequences, both applied below: the optional confidence gate is
**deleted** (COLM omits it), and promotion gate 1 uses the **composite `U`** (COLM's letter), with
no config flag offering FluxMem's usage-only reading.

### Equation reference card

A fresh session should not need to reopen the PDFs. `§` = COLM
(`docs/COLM_Cognitive_AI_Memory_Architecture_Project.pdf`); `F` = FluxMem
(`docs/Choosing How to Remember Adaptive Memory Structures for LLM Agents.pdf`).

| Ref | Equation | Where |
|---|---|---|
| §Eq. 5 | `U(e_j) = w₁·c(e_j) + w₂·ℓ(e_j) + w₃·d(e_j)` — access frequency, interaction intensity, recency | `mtem.py` |
| §1.3.2 | Promotion: `U(e_j) ≥ τ_u ∧ r(e_j) ≥ τ_r` | `ltsm.py` |
| §1.3.2 | STIM capacity 4; evict to MTEM by LRU | `stim.py` |
| F Eq. 5 | `E_t = argmin_{P⊂M_t, |P|=|M_t|−C} Σ_{p∈P} τ(p)`, `τ(p)` = **last access time** ⇒ LRU | `stim.py` |
| §Eq. 6 | `L_sel(θ) = Σ_j ℓ(f_θ(x_j), f*_j)` — cross-entropy | `selector.py` |
| F Eq. 8 | `ŝ = argmax_s Softmax(g_θ(x))[s]` | `selector.py` |
| F Eq. 9 / Alg. 2 | `r(s) = λ_judge·r_judge(s) + λ_mem·r_mem(s)`, `λ = 0.7/0.3`; `y = argmax_s r(s)` | `supervision.py` |
| F Alg. 3 | StandardScaler → softmax → CE → Adam → early stop | `selector.py` |
| F Eq. 17 | `x_i = ε + (1−2ε)·(s_i−s_min)/(s_max−s_min)`; **`x_i = 0.5` if `s_max = s_min`** | `fusion.py` |
| §Eq. 7 / F Eq. 18 | `p(x) = π·Beta(x;α₁,β₁) + (1−π)·Beta(x;α₀,β₀)` | `fusion.py` |
| F Eqs. 20–21 | log-space E-step; `π_k = N_k/n`, `N_k = Σ_i r_ik` | `fusion.py` |
| F Eqs. 22–23 | `μ_k, σ²_k` responsibility-weighted; `κ = μ(1−μ)/σ² − 1`, `α = μκ`, `β = (1−μ)κ` | `fusion.py` |
| F Eq. 24 | `k* = argmax_k α_k/(α_k+β_k)` (larger Beta mean = high-compatibility) | `fusion.py` |
| F Eq. 25 | `g(x) = π_{k*}·Beta(x;α_{k*},β_{k*}) / Σ_k π_k·Beta(x;α_k,β_k)` | `fusion.py` |
| F Eq. 26 / Alg. 1 | `I = {i : g(x_i) ≥ τ}`; **if `\|I\| < m_min` then `I ← TopK({x_i}, m_min)`**; merge into `argmax_{i∈I} x_i`, else NEW | `fusion.py` |
| F Eq. 6 | `u(m) ≥ τ_u ∧ r(m) ≥ τ_r (∧ c(m) ≥ τ_c)` — **not implemented**; see Reading rule | — |

---

## Step 0 — Environment and packaging

- `pip install faiss-cpu pytest pytest-cov spacy && python -m spacy download en_core_web_sm`
  into `/mnt/ssd/users/durgesh/conda-envs/lightmem`.
  **Verified present**: numpy 2.2.6, sklearn 1.7.2, torch 2.8.0+cu128, pydantic 2.12.0, scipy
  1.15.3, pyyaml 6.0.3, nltk 3.9.2, transformers 4.57.0, Python 3.11.15.
  **Verified missing**: `faiss`, `pytest`, `spacy`.
- **Disk**: `/home` is at 97% (46G free), `/mnt/ssd` has 1.9T. The conda env already lives on
  `/mnt/ssd`, so `pip`/spaCy model installs land there. Do **not** let anything write to
  `~/nltk_data` or `~/.cache/huggingface` (CLAUDE.md: large artifacts never on `/home`).
- New root `pyproject.toml` (none exists). `requires-python = ">=3.10,<3.12"` per CLAUDE.md.
  Runtime deps pinned exactly per `.claude/rules/dependency-management.md`; `[project.optional-dependencies].dev`
  gets pytest/pytest-cov/ruff. Add `[tool.ruff]` and `[tool.pytest.ini_options]` here.
- **Name-collision check**: `LightMem/` is gitignored and is its own repo; it is never on
  `sys.path` from the C-AIMMS root, so root-level `fluxmem/` cannot shadow or be shadowed by
  `LightMem/src/fluxmem/`. Guard this with a test asserting `fluxmem.__file__` resolves under the
  repo root, so a stray `pip install -e LightMem/` can never silently swap the import.
- Branch from `main` as `feat/adastore-fluxmem` (current branch is a docs branch, and
  `docs/prompts/caimms-implementation-kickoff.md` has uncommitted edits — commit or stash first).
- After the package tree exists, run a **full `codegraph index`**, not `sync` — adding a top-level
  package is structural surgery per `.claude/rules/code-navigation.md`.

## Step 1 — `fluxmem/interfaces.py` + stub producer

The boundary fake everything else is tested against. No dependencies.

- `MemoryFormat(StrEnum)`: `HG`, `VC`, `VS` — opaque labels, no encoders (HETREP is a non-goal).
- `Turn`: `turn_id`, `user`, `assistant`, `timestamp`, `last_access`.
- `EpisodicUnit`: `episode_id`, `turns`, `primary_format`, `features`, `embedding`,
  `access_count`, `last_access`, `created_at`, plus opaque `hyperedge_density` /
  `visual_salience` carried in from a (not-implemented) HETREP.
- `VectorStore` Protocol: `add(ids, vectors)`, `search(query, k) -> list[tuple[str, float]]`, `__len__`.
- `EpisodeProducer` Protocol + `StubEpisodeProducer` — seeded, deterministic, generates episodes
  with controllable entity counts / topic spread / lengths so feature tests have known ground truth.
- `EntityExtractor` Protocol: `extract(text) -> list[Entity]` (`Entity` = surface form + label).
  Two implementations: `SpacyEntityExtractor` (loads `en_core_web_sm` **lazily**, once, at module
  level — loading per episode is a 10× slowdown) and `FakeEntityExtractor` (returns a scripted
  entity list per input, so feature tests are hermetic and never touch a model).

**Verify**: `/snapshot-test` on the stub producer output — golden snapshot pins the fake so later
refactors can't silently change every downstream fixture.

## Step 2 — `fluxmem/stim.py`

`ShortTermInteractionMemory`, capacity 4 (COLM §1.3.2; FluxMem confirms "STIM (capacity 4, LRU)").

- `OrderedDict` + `move_to_end`; `push(turn)`, `touch(turn_id)`, `evicted` returned to the caller
  rather than pushed to MTEM directly — keeps STIM dependency-free, as the build order requires.

**LRU vs. FIFO — implement LRU.** COLM §1.3.2 says "the **oldest** turn is evicted … via
**Least-Recently-Used (LRU)** policy," which names two different policies in one sentence: FIFO
evicts by *insertion* order ("oldest"), LRU evicts by *last-access* order. They coincide only if
turns are never re-read after insertion. Worked example — STIM holds `T1 T2 T3 T4`, something
reads `T1`, then `T5` arrives: FIFO evicts `T1`; LRU evicts `T2`, because `T1`'s last-access is now
the newest. FluxMem Eq. 5 settles it — it minimises `Σ τ(p)` where `τ(p)` is explicitly *"the last
access time of page p"*, i.e. LRU. So: LRU is the policy, "oldest" is loose prose for the common
case. Today it is moot (COLM Alg. 1's write phase never re-reads STIM, so nothing calls `touch()`
and LRU degenerates to FIFO); it starts to matter the moment a read path touches STIM — which
ITERRET, out of scope here, would do. `touch()` is the entire difference between the two policies.

**Tests**: fills to 4 without eviction; 5th push evicts exactly one; `touch` on the oldest changes
which turn is evicted (the test that distinguishes LRU from FIFO); eviction order under
interleaved access; capacity 1 and capacity 0 edges.

## Step 3 — `fluxmem/mtem.py` — utility score `U(e_j)`

`utility_score(episode, weights, now) -> float` as a **pure function**, Eq. 5:
`U = w₁·c(e) + w₂·ℓ(e) + w₃·d(e)`.

Neither paper defines `c`, `ℓ`, `d` concretely. Each concretization is a **modeling choice,
docstring-flagged as such**, and each is individually swappable:

| Term | Paper name | Field name in code | Implementation | Range |
|---|---|---|---|---|
| `c(e)` | access frequency | **`access_frequency`** | `access_count / (age_in_turns + 1)` | `(0,1]` |
| `ℓ(e)` | interaction intensity | `interaction_intensity` | mean tokens per turn ÷ a config'd reference length, clipped | `[0,1]` |
| `d(e)` | recency | `recency_decay` | `exp(-Δt / half_life)` | `(0,1]` |

Never name the first field `c`. COLM's `c(e_j)` = *access frequency* and FluxMem Eq. 6's `c(m)` =
*confidence* are the same glyph for unrelated quantities; conflating them would not crash, it
would silently change which episodes promote — the exact failure mode
`.claude/rules/evidence-discipline.md` targets. (The confidence gate is not implemented at all —
see Step 8 — so only one `c` exists here; leave a one-line comment recording the overload so a
future reader adding it cannot reintroduce the collision.)

All three normalized to `[0,1]` so `w₁,w₂,w₃` are comparable — itself a modeling choice, stated in
the docstring. `MidTermEpisodicMemory` then wraps storage: `add`, `get` (bumps `access_count` and
`last_access` — the write that makes `c` and `d` move), `merge(target_id, incoming)`,
`promotion_candidates(...)`, and a capacity cap (FluxMem: up to 2000 units).

**Tests**: each term in isolation with the other two weights zeroed; monotonicity (more accesses ⇒
higher `U`, more elapsed time ⇒ lower `U`); `w=(0,0,0)` ⇒ 0; `get` mutates access state; the
capacity cap evicts lowest-`U`.

**Then** `/diagram` the three-tier hierarchy into `docs/diagrams/` (Mermaid, per CLAUDE.md).

## Step 4 — `fluxmem/features.py`

```python
FEATURE_NAMES = ("entity_count", "cooccurrence_density", "temporal_ordering",
                 "topic_diversity", "token_length", "hyperedge_density", "visual_salience")
FEATURE_DIM = len(FEATURE_NAMES)   # 7 — never hardcode the 7
```

`extract_features(episode, entity_extractor) -> FeatureVector`, numpy-backed with `.to_array()`.
The extractor is **injected**, defaulting to `SpacyEntityExtractor`.

**Why real NER here.** FluxMem App. C argues features should be *"intentionally simple and
interpretable… lightweight… avoids introducing additional reasoning overhead into the control
layer."* That rules out an **LLM-based** extractor; it does not justify capitalization regex,
which fails on lowercase text, fires on sentence-initial words, and cannot type entities.
`entity_count` and `cooccurrence_density` are 2 of 7 features feeding the selector's *training
labels*, so noise there propagates into supervision. spaCy `en_core_web_sm` is real statistical
NER, ~10k words/s on CPU, deterministic, ~12MB. Rejected: `nltk.ne_chunk` (weaker, and its corpora
default to `~/nltk_data` on the 97%-full `/home`); `transformers` BERT-NER (~430MB, slow, makes
feature extraction model-dependent — keep as a drop-in upgrade behind the same Protocol if entity
quality ever becomes the bottleneck).

- `entity_count` — distinct spaCy entities, log1p-scaled.
- `cooccurrence_density` — entity pairs co-occurring within a turn ÷ possible pairs.
- `temporal_ordering` — fraction of adjacent turn pairs with monotonically increasing timestamps,
  blended with the rate of explicit temporal markers.
- `topic_diversity` — distinct-content-token ratio across turns.
- `token_length` — whitespace token count, log1p-scaled.
- `hyperedge_density`, `visual_salience` — read straight off the episode as **opaque placeholders**
  (default `0.0`); the HG/VC encoders that would produce them are explicit non-goals.

**Tests**: driven by `FakeEntityExtractor` (scripted entity lists ⇒ exact expected values, no model
load) — a 1-entity episode vs. a 10-entity one; every feature bounded and finite; empty-episode and
single-turn edges; `len(vector) == FEATURE_DIM`; placeholder passthrough is exact. One
`@pytest.mark.slow` integration test exercises the real `SpacyEntityExtractor` on a fixed sentence
so the model path is covered without slowing the default loop (`.claude/rules/testing.md`: mark
slow tests so they can be excluded).

## Step 5 — ADR: reward design (**gate — blocks Steps 6–7**)

`/adr` → `docs/adr/0002-format-selector-reward-design.md`, Nygard format, sequential.

- **Given (cited, not decided)**: `r(s) = 0.7·r_judge(s) + 0.3·r_mem(s)`, `y = argmax_s r(s)`
  (FluxMem Eq. 9, App. D Alg. 2). `λ` stays config-exposed.
- **Decided**: concrete `Judge(·)` and `MemUtil(·)`. Compare ≥2 candidates each, with tradeoffs:
  - `r_judge`: (a) exact-match / F1 against a reference answer — cheap, deterministic, testable
    offline, but brittle on free-form text; (b) LLM-as-judge on a 0–1 rubric — matches the paper's
    intent and its GPT-4.1/Qwen3 setup, but non-deterministic and requires an API call per
    (episode × format), i.e. 3× cost.
  - `r_mem`: (a) retrieval hit-rate (fraction of gold evidence surfaced); (b) hit-rate normalized
    by retrieved token budget — rewards *efficient* utilization, which is closer to "memory
    utilization effectiveness," but adds a tokenizer dependency to the reward.
  - Recommendation to be argued in the ADR, not pre-committed here.
- **Design for both**: `JudgeReward` and `MemUtilReward` are Protocols with deterministic fakes
  for tests, so no test ever needs an LLM call.
- Record which choice was made **and that it is a modeling choice, not a paper-specified fact** —
  training data downstream of the selector depends on it (`.claude/rules/evidence-discipline.md`).

## Step 6 — `fluxmem/selector.py` + `fluxmem/supervision.py`

> **Deviation from the layout in the kickoff prompt**: splitting the offline reward/labeling
> pipeline into `supervision.py` instead of folding it into `selector.py`. Rationale: it keeps
> both files under the 300-line cap (`.claude/rules/coding-style.md`), and it isolates the ADR'd
> modeling choice from the training mechanics so the reward can be revised without touching the
> MLP. Easy to veto in review — say so and it collapses back into `selector.py`.

`supervision.py` — FluxMem Alg. 2: `RewardConfig(lambda_judge=0.7, lambda_mem=0.3)`,
`JudgeReward`/`MemUtilReward` Protocols, `per_format_rewards(episode, runner) -> dict[MemoryFormat, float]`,
`label_episodes(episodes, runner) -> (X, y)` with `y = argmax_f r_f`. Ties broken deterministically
by a documented format order.

`selector.py` — FluxMem Alg. 3 / COLM Eq. 6:

- `nn.Sequential(Linear(FEATURE_DIM, hidden), ReLU, Linear(hidden, 3))` — the two-layer MLP.
- `StandardScaler` fit on train features only (never on validation — a leak the paper's Alg. 3
  line 1 doesn't spell out).
- Explicit loop: Adam, `F.cross_entropy` (Eq. 6 literally), minibatches, early stop on validation.
- `predict(x) -> MemoryFormat` via `argmax softmax` (FluxMem Eq. 8).
- `save`/`load` persisting scaler + `state_dict` together — loading one without the other yields
  silently wrong predictions, so they serialize as one artifact.

**Tests**: overfits a tiny separable synthetic set to ~100% train accuracy (the canonical
"training loop actually works" test); loss decreases monotonically over epochs on that set; early
stopping fires on a non-improving validation set; save→load round-trips to identical predictions;
scaler is not fit on validation; seeded runs reproduce exactly. `/track` logs params + metrics per
`experiment-discipline`.

## Step 7 — `fluxmem/fusion.py` (highest logic risk)

Implements FluxMem Eqs. 17–26 + Alg. 1 verbatim; COLM Eq. 7 is the two-component mixture.

1. `normalize_scores(s, eps)` — Eq. 17, **including** the `s_max == s_min ⇒ x_i = 0.5` branch.
2. `BetaMixtureModel`:
   - quantile init: `μ₀ ← q₀.₃`, `μ₁ ← q₀.₇`, fixed small initial variance → `(α,β)` by moment matching;
   - **log-space** E-step (Eq. 20) via `scipy.stats.beta.logpdf` + `logsumexp` — the paper calls
     out log-space explicitly for numerical stability near the boundaries;
   - M-step: `π_k = N_k/n` (Eq. 21); responsibility-weighted `μ_k, σ²_k` (Eq. 22); `κ = μ(1−μ)/σ² − 1`,
     `α = μκ`, `β = (1−μ)κ` (Eq. 23). **Guard `σ²→0` and `κ ≤ 0`** — both are reachable when one
     component collapses, and both produce invalid Beta parameters; clamp and document.
   - `high_compat_component` = `argmax_k α_k/(α_k+β_k)` (Alg. 1 line 11) — **not** hardcoded to
     index 1. COLM's Eq. 7 writes `π·Beta(α₁,β₁)` as if component 1 is always the high one; the
     source paper's Alg. 1 says to determine it empirically. We follow Alg. 1.
   - `posterior(x)` — Eq. 25.
3. `select_merge_target(incoming, candidates, scorer, cfg) -> MergeDecision` — Alg. 1:
   `I = {i : g(x_i) ≥ τ}`; **if `|I| < m_min`, `I ← TopK({x_i}, m_min)`** (Eq. 26); if `I` empty →
   `NEW`; else merge into `argmax_{i∈I} x_i`. Note in the docstring that with `m_min ≥ 1` the
   `I = ∅` branch is reachable only when the candidate set is empty — an Alg. 1 subtlety that is
   easy to implement as dead code by accident.

**Tests** (the reason this step is separate): recover `(π, α, β)` within tolerance from data
sampled off a known two-component Beta mixture at a fixed seed; log-space E-step does not
underflow at `x → 0⁺/1⁻`; Eq. 17's degenerate branch; `m_min` fallback fires when the threshold
retains too few; empty candidate set ⇒ `NEW`; single candidate; all-identical scores; component
labels are order-invariant (a fit that swaps component indices must not change the decision).

**Then**: `/logic-review` on `select_merge_target` and the M-step specifically — the only
nontrivial control flow in this scope — and `/diagram` the merge logic.

## Step 8 — `fluxmem/ltsm.py` — promotion gate + FAISS store

- `FaissVectorStore(VectorStore)` — `IndexFlatIP` over **L2-normalized** vectors (⇒ inner product
  = cosine), with an id↔row map since `IndexFlatIP` has no native string ids.
- `is_eligible(episode, weights, thresholds, now)` — exactly `U(e) ≥ τ_u ∧ r(e) ≥ τ_r`. Two
  conjuncts, no third. No config flag selecting an alternative reading.
- `promote(mtem, ltsm, ...)` — moves eligible episodes, embeds, writes to the store.

### Gate 1 is the composite `U`, not usage (the one real conflict between the papers)

COLM §1.3.2 writes `U(e_j) ≥ τ_u`, where capital `U` is Eq. 5's composite. FluxMem Eq. 6 writes
`u(m) ≥ τ_u` and defines `u(m)` as **usage alone**. Per the Reading rule, COLM wins. The
difference is **compensatory vs. non-compensatory**:

- *COLM (composite):* a rarely-accessed episode still promotes if it is long/intense and recent —
  `w₂ℓ + w₃d` compensate for low `c`. A heavily-accessed but thin, stale episode can *fail*.
- *FluxMem (usage-only):* access count alone decides; nothing compensates; intensity never
  affects promotion at all.

Two consequences that must survive into the code and the experiment record:

1. `is_eligible` takes `UtilityWeights`, so **promotion is coupled to `w₁,w₂,w₃`** — changing the
   weights changes which episodes promote. That coupling does not exist under FluxMem's reading.
   Record the weights alongside `τ_u` in every experiment record, or the promotion rate is not
   reproducible.
2. **`τ_u` is on a different scale** than FluxMem's: composite `U ∈ [0, w₁+w₂+w₃]` (given the
   `[0,1]` term normalization from Step 3) vs. raw usage counts `∈ [0,∞)`. FluxMem's tuned
   threshold values are **not portable** here.

### The confidence gate is not implemented

FluxMem Eq. 6's third conjunct `(∧ c(m) ≥ τ_c)` is parenthesised and called "optional" in its own
source, and **COLM omits it**. It is also unproducible here: confidence would have to come from a
fact extractor rating its own output, and ADASTORE's write path (COLM Alg. 1 lines 1–7: segment →
encode → select format → fuse → promote) has no step that emits one. Omitting it is coherent, not
an oversight.

Removing it is safe: it is a pure AND-conjunct, so dropping it makes promotion *strictly more
permissive* — it can never reject anything the two remaining gates accept. The only cost is that
LTSM may admit lower-confidence facts, against COLM's claim that LTSM stays "compact and
high-precision." **Known gap, not a silent one**: if LTSM precision measures poorly, this is the
first knob to revisit. Do not ship `τ_c` as dead disabled config — absent beats dead.

### `r(e_j)` and `d(e_j)` are both recency, deliberately

This is *not* a conflict between the papers — both do it (FluxMem's `U(s)` has `d(s)`=recency and
its Eq. 6 has `r(m)`=recency). It is inherent to the design, and the two roles differ:

- **`d(e_j)` inside `U` is soft and tradeable** — one addend of three. A slightly-stale episode
  with heavy access and high intensity still clears `τ_u`. `w₃` sets what staleness costs.
- **`r(e_j)` in the eligibility criterion is hard and non-negotiable** — a separate conjunct with
  no weight on it. No amount of access frequency rescues an episode past `τ_r`.

Belt-and-braces by design: the composite admits useful-but-aged episodes; the hard gate still
refuses to consolidate anything genuinely ancient. Collapsing them loses the second guarantee.
Both derive from the same `Δt = now − last_access`, differing only in the transform:

| Symbol | Role | Transform | Why this shape |
|---|---|---|---|
| `d(e_j)` | weighted addend | `exp(−Δt / half_life)` | smooth, no cliff — right for a term being *summed* against others |
| `r(e_j)` | hard threshold | `1 − min(Δt / max_age, 1)` | linear, so `τ_r = 0.8` reads as "touched within the most recent 20% of the horizon" |

Both are strictly decreasing in `Δt`, so they can never disagree in rank order — an episode cannot
be "recent" under one and "stale" under the other. **That invariant gets a property test.**

**Tests**: eligibility truth table over both thresholds (all four `U`/`r` pass-fail combinations);
boundary values are `≥`, not `>`; the rank-order invariant above (`d` and `r` never disagree on
which of two episodes is more recent); changing `w₁,w₂,w₃` changes the promotion set (pins the
composite-gate coupling); FAISS round-trip (add → search returns the same id at score ≈ 1.0);
vectors are actually L2-normalized before `add`; promotion removes from MTEM and appears in LTSM;
empty-store search.

**Note on `.claude/rules/storage-invariants.md`**: it is path-scoped and `configs/` will match it,
but its `dense_weight`/`bm25_weight` invariant describes FluxMem's hybrid retriever in `LightMem/`.
This package builds **no** BM25 fusion — retrieval (ITERRET) is an explicit non-goal — so there are
no fusion weights to record yet. State this explicitly in any experiment record rather than
leaving a reader to assume the invariant was overlooked.

---

## Files

```
pyproject.toml               # NEW — repo root has no Python packaging today
fluxmem/
  __init__.py  config.py  interfaces.py  stim.py  mtem.py  features.py
  supervision.py  selector.py  fusion.py  ltsm.py
tests/                       # mirrors fluxmem/ 1:1, + conftest.py
configs/default.yaml         # w1/w2/w3, tau_u, tau_r, tau, m_min, em_iters, eps, half_life,
                             # max_age, MLP hparams, lambda_judge/lambda_mem. No tau_c.
scripts/train_selector.py  scripts/run_write_phase.py     # thin CLIs; logic stays in fluxmem/
docs/adr/0002-format-selector-reward-design.md
docs/diagrams/adastore-*.md
```

`config.py` is pydantic v2 (`StimConfig`, `UtilityWeights`, `PromotionThresholds`, `FusionConfig`,
`SelectorConfig`, `RewardConfig` → `AdaStoreConfig`) + `load_config(path)` over `configs/default.yaml`.
Validation at the boundary per `.claude/rules/security.md`: weights ≥ 0, `τ ∈ (0,1)`, `m_min ≥ 0`,
`capacity ≥ 1`, `half_life > 0`, `max_age > 0`. Every public function/class gets a docstring naming
the equation/section it implements. Comments explain *why* only.

**Do not add**: `tau_c`, `promotion_gate`, or any other flag that re-opens a resolved paper
ambiguity. The Reading rule resolves them; config is for tuning, not for hedging.

## Per-step discipline

Each of steps 1–8: write tests alongside (`tdd-mastery` / `/tdd`; the stub producer is a boundary
fake, never the unit under test) → run pytest and **paste output** → `codegraph sync` → `/commit`
with a conventional message (`feat(fluxmem): …`). `/type-hints` and `/doc-gen` once a module's
interface stabilizes.

## Verification

```bash
conda activate /mnt/ssd/users/durgesh/conda-envs/lightmem
pip install faiss-cpu pytest pytest-cov spacy
python -m spacy download en_core_web_sm
pip install -e ".[dev]"

pytest tests/ -v -m "not slow"                              # per step: pytest tests/test_<mod>.py -v
pytest tests/ -v -m slow                                    # real spaCy model path
pytest tests/ --cov=fluxmem --cov-report=term-missing        # ≥80% line, ≥75% branch
ruff check fluxmem/ tests/ scripts/

python -c "import fluxmem, pathlib; \
  assert pathlib.Path(fluxmem.__file__).is_relative_to(pathlib.Path.cwd()), fluxmem.__file__"
python scripts/train_selector.py --config configs/default.yaml --dry-run
codegraph index && codegraph explore "fluxmem fusion merge decision"
```

End-to-end smoke (`scripts/run_write_phase.py`): stub producer → 6 turns into STIM → 2 evicted to
MTEM → features extracted → selector assigns a format → BMM fusion decides merge-vs-new → one
episode crosses both promotion thresholds and lands in the FAISS store. Assert the tier counts at
each stage.

Done criteria: all tests pass with output pasted (not summarized); coverage thresholds met; ADR
0002 written and approved before Step 6 code; diagrams for hierarchy / feature flow / merge logic;
`/logic-review` clean on `fusion.py`; `codegraph index` reflects the new package; and a closing
report separating **verified** from **assumed** per `.claude/rules/evidence-discipline.md`.

## Out of scope — will not be touched

HETREP (HG/VC/VS encoders), ITERRET, WORKMEM/δ-mem, surprise-based segmentation (§1.3.1), the six
external papers (HyperMem, MemOCR, δ-mem, MemR³, R²-Mem, MRAgent), and **any** file under
`LightMem/`. Note for the record: the FluxMem *paper* (`docs/Choosing How to Remember….pdf`) is a
primary source for this scope and was read; the `LightMem/` *directory* was not read and will not be.

## Verified vs. assumed (as of planning)

**Verified** — both PDFs read in full at the cited equations/algorithms (COLM pp. 1–7; FluxMem
Apps. B/C/D, pp. 14–18, plus §3.2–3.4, pp. 4–5); no Python outside `LightMem/` and the toolkit
clone; no root `pyproject.toml`/`tests/`; package presence/absence checked by importing inside the
conda env; disk usage on `/home` (97%) and `/mnt/ssd` (45%); `.gitignore` excludes `LightMem/` but
not a root `fluxmem/`; ADR 0001 read for house format.

**Assumed** — that FluxMem's appendices are the right source for sentences COLM compresses
(strongly supported: COLM cites Lu et al. for both the feature classes and the BMM gate, and the
equation forms match; the Reading rule bounds the risk by making COLM authoritative on conflicts);
that the `[0,1]` normalization of the three `U` terms is acceptable (a modeling choice — the
papers never state term ranges, and it is what makes `w₁,w₂,w₃` comparable and `τ_u` interpretable);
that no teammate is concurrently creating a root `fluxmem/`; that `faiss-cpu` and spaCy
`en_core_web_sm` install cleanly on this Python 3.11 env — **unverified until Step 0 runs**.
