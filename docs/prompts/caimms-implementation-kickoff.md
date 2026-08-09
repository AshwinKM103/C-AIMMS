# ADASTORE Implementation Kickoff — Optimized Prompt

Optimized via the `prompt-engineering` skill on 2026-08-09. Version history: v1 covered
the full C-AIMMS paper; v2 narrowed to FluxMem; v3 narrowed further and stopped trusting
`LightMem/`'s existing code. This version (v4) drops `LightMem/` entirely — that repo was
confirmed to be the wrong download and must not bias this prompt or the implementation —
and narrows scope to exactly one section of the COLM C-AIMMS paper: §1.3.2 (three-tier
hierarchy) and §1.3.3 (format-adaptive selector, offline supervision, BMM-gated fusion).
File a new version here if scope changes again rather than editing this one back and
forth.

---

## Prompt

You are implementing **ADASTORE's storage layer** — §1.3.2 and §1.3.3 of
`docs/COLM_Cognitive_AI_Memory_Architecture_Project.pdf` — and nothing else. Do not
reference, import from, or otherwise let `LightMem/` (any file in that git repo) inform
this implementation; it was confirmed to be the wrong downloaded repo and any resemblance
between it and this task is coincidental, not authoritative. Where you need something
FluxMem-shaped (e.g. the "same classes of features used by FluxMem" the paper mentions in
§1.3.3), source it from the paper's own description, not from `LightMem/`.

### Exact scope — implement only this

1. **Three-tier memory hierarchy** (§1.3.2):
   - **STIM**: fixed-capacity buffer of the 4 most recent raw dialogue turns.
     Capacity-triggered eviction moves the oldest turn to MTEM via LRU.
   - **MTEM**: stores structured episodic units `e_j`, each tagged with which of the
     three representations (HG/VC/VS) it is stored in. Utility score
     `U(e_j) = w1*c(e_j) + w2*l(e_j) + w3*d(e_j)` (Eq. 5) over access frequency,
     interaction intensity, and recency. High-utility episodes become LTSM candidates.
   - **LTSM**: consolidated semantic facts as dense embeddings in a vector store.
     Promotion eligibility: `U(e_j) >= tau_u AND r(e_j) >= tau_r` (an independent
     recency/relevance gate `r`, distinct from the `d(e_j)` term inside `U`).
2. **Format-adaptive selector** (§1.3.3):
   - Feature vector `x_j` per episode: named-entity count + co-occurrence density,
     degree of temporal ordering, topic diversity, total token length, plus hyperedge
     density and visual salience score (6 features total, per the paper).
   - Two-layer MLP `f_theta: R^d -> {HG, VC, VS}`, trained by minimizing cross-entropy
     (Eq. 6) against offline-derived supervision targets.
   - Offline supervision: for a held-out set, compute per-format rewards
     `r_HG_j, r_VC_j, r_VS_j` (downstream response quality + memory utilization
     effectiveness), target `f*_j = argmax_f r_f_j`.
3. **BMM-gated memory fusion** (§1.3.3, borrowed from
   `docs/Choosing How to Remember Adaptive Memory Structures for LLM Agents.pdf`):
   - Two-component Beta Mixture Model over normalized matching scores
     `x_i in (0,1)`: `p(x) = pi*Beta(x; a1,b1) + (1-pi)*Beta(x; a0,b0)`
     (Eq. 7), high- vs. low-compatibility components.
   - Online EM with responsibility-weighted moment updates for `(pi, a1, b1, a0, b0)`.
   - Merge decision: candidates whose posterior under the high-compatibility component
     exceeds threshold `tau` are merge targets; incoming episode merges into the
     best-scoring retained candidate, or is stored as a new MTEM unit if none qualify.

### Explicit non-goals

- No HETREP (hypergraph/visual-canvas/vector-store encoders themselves) — episodic
  units are consumed here as opaque objects that already carry HG/VC/VS content and a
  feature vector `x_j`; build a minimal fake/stub producer for testing, not a real
  encoder.
- No ITERRET (retrieval controller, experience bank, CTC traversal) and no WORKMEM
  (δ-mem/OSAM). Out of scope.
- No surprise-based episodic segmentation (§1.3.1) — assume episode boundaries are
  already given as input; don't build the segmenter.
- No `LightMem/` code, structure, or conventions as a reference.

### Before writing the offline-supervision and reward code

Per `.claude/rules/evidence-discipline.md`: "evaluate the downstream response quality and
memory utilization effectiveness of all three formats" (§1.3.3) is underspecified — the
paper doesn't define the reward function's exact form. This is the single highest-risk
part of the task. Write down at least two concrete candidate reward definitions (e.g.
task-accuracy-on-a-probe-question vs. a compression/retrieval-precision proxy) before
implementing either, state which you're building against, and flag it as a modeling
choice rather than a paper-specified fact. If it's non-obvious or hard to reverse once
training data depends on it, write it up as an ADR (`/adr`) before committing to it.

### Where this code lives

Location is not yet decided — do **not** default to `LightMem/`. Use `Plan` / `/plan` to
propose a location (new top-level package in this repo, e.g. `adastore/` or `src/adastore/`,
vs. elsewhere) as the first output of this task, and confirm before writing files.

### Build order

1. STIM (simplest, no dependencies — fixed buffer + LRU eviction).
2. MTEM's utility score `U(e_j)` (pure function, testable in isolation).
3. Episode feature vector `x_j` extraction (depends on episodes carrying HG/VC/VS
   content — build against the stub producer).
4. Format-adaptive selector: MLP + training loop, once reward/supervision-target
   generation (the flagged high-risk piece above) has a stated design.
5. BMM-gated fusion: Beta-Mixture EM estimator, then the merge-decision logic on top.
6. LTSM promotion gate wiring MTEM → LTSM using `U(e_j)` and `r(e_j)`.

Test-first or test-immediately-after per `.claude/rules/testing.md` (mock at
boundaries — the stub episode producer counts as a boundary fake, not the unit under
test) and the `tdd-mastery` skill / `/tdd`.

### Tool routing

- `Plan` agent / `/plan` first — location decision + the build order above.
- `ml-engineer` for the selector's MLP training loop and offline reward pipeline.
- `data-scientist` for the BMM/EM estimator (statistical fitting, not a generic ML
  training loop — different skillset).
- `python-engineer` + `python-best-practices` skill for STIM/MTEM/LTSM plumbing.
- `vector-database-engineer` for LTSM's dense-embedding vector store.
- `test-architect` + `tdd-mastery` for the fake-backed test suite per component.
- `codegraph explore`/serena per `.claude/rules/code-navigation.md` for any
  "does something like this already exist in this repo" check — not grep, and not a
  look at `LightMem/`.
- Storage: `.claude/rules/storage-invariants.md` applies once LTSM's vector store is
  live — record retrieval weights in experiment records if/when retrieval is exercised.
- `/verify` before declaring any component done — paste actual test output. `/commit`
  with conventional-commit messages, one commit per build-order step.

### Definition of done

STIM, MTEM's utility score, feature extraction, the format selector (with its reward
design documented, ADR'd if non-obvious), BMM-gated fusion, and the LTSM promotion gate
each have passing tests against fakes/stubs, output pasted not summarized. Chosen code
location is confirmed via `/plan` before implementation, not assumed. No file in
`LightMem/` was read as a reference, imported, or relied on. Report what was verified
versus assumed at each step, per `.claude/rules/evidence-discipline.md`.
