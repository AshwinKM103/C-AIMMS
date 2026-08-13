# 0008. COLM vs. component fidelity resolved in COLM's favour

## Status

Accepted — 2026-08-12. Governs `hetrep/vs/encoder.py` (Phase 1) and `hetrep/hg/propagation.py`
(Phase 2) before either lands. Sets precedent for any future divergence discovered in Phase 3.

## Context

HetRep is a composition of externally sourced components integrated to implement one paper —
COLM, the paper C-AIMMS is on a publication track to write about (`CLAUDE.md`: "Research prototype
... on a publication track (COLM)"). Auditing HyperMem for reuse (hetrepv2 plan §2.6, formalized
in ADR 0007) surfaced three places where the component's own implementation diverges from what
COLM specifies, none previously logged in ADR 0004's G1–G8 gap list because that ADR's scope was
HyperMem's fidelity to _its own_ paper (Yue et al., 2025), not to COLM.

| #   | COLM spec                                                                  | HyperMem's actual code                                                                                                                                                          |
| --- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | §1.2.3: embed with **all-MiniLM-L6-v2** (384-d)                            | **Qwen3-Embedding-4B** over HTTP, hardcoded — `embedding_provider.py:55-56` raises `ValueError` if the model name does not contain `"Qwen3"`                                    |
| 2   | §1.2.3: **ANN retrieval** (faiss), hybrid **BM25 + dense, RRF-fused**      | No faiss anywhere in `embedding_provider.py`; similarity is brute-force `np.dot(doc_vecs, query_vec)` (`embedding_provider.py:27-52`, `cosine_similarity`)                      |
| 3   | Eq. 3: `φ'(e_j) = (1/\|N(e_j)\|) Σ φ(e_k)` — **plain, unweighted average** | **Softmax-weighted sum** of neighbor embeddings, verified at `stage3_hypergraph_index.py:650-692`: `torch.softmax(weights_tensor, dim=0)` then a weighted sum, not a plain mean |

Each of these is individually defensible as an engineering choice in HyperMem's own paper — a
larger embedding model and learned edge weights can plausibly retrieve better than the COLM
baseline they'd replace. That is exactly the problem: **plausibly better is not the same claim as
COLM makes**, and if HetRep silently inherits HyperMem's choices, any number HetRep reports cannot
be attributed to COLM's method without a caveat that currently exists nowhere in the plan or the
code. A reviewer (or a co-author, or a future re-implementer) reading "HetRep implements COLM
§1.2.3" would reasonably expect all-MiniLM-L6-v2 and ANN+BM25 retrieval; getting Qwen3-4B and
brute-force dot products instead is a silent substitution, not a documented approximation — the
distinction `.claude/rules/evidence-discipline.md` insists on.

There is also a standing project decision this ADR is a direct instance of. The
`resolve-paper-ambiguities-dont-hedge` memory note records: "on conflicts follow the paper being
implemented; never park a settled question behind a config flag." COLM's spec for these three
choices is not ambiguous — §1.2.3 names a specific model and a specific retrieval strategy, and
Eq. 3 is a literal formula, not a design space. There is nothing here to resolve by experiment; it
is a question with a written answer in the cited paper, and the project's own standing rule says
to follow it, not to add a switch that quietly defaults to the component's variant.

## Decision

**COLM's specification is authoritative wherever it conflicts with a reused component's
implementation. Components are adapted to match COLM; COLM's design is never adapted to match a
component's convenience.**

| #   | Conflict            | HetRep's choice                                                                                         | Where it applies                                                                                                                                                                                                                                                                                                              |
| --- | ------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Embedding model     | **`all-MiniLM-L6-v2`, 384-d**, via `sentence-transformers`, loaded from `.env`'s `EMBEDDING_MODEL_PATH` | `hetrep/vs/encoder.py` (Phase 1) — this is the model the VS arm's `Embedder` Protocol implementation uses                                                                                                                                                                                                                     |
| 2   | Retrieval mechanism | **faiss ANN**, hybrid BM25+dense with RRF, as COLM specifies                                            | Retrieval is ITERRET's concern, formally out of HetRep's scope (ADR 0005) — this row records the design _target_ for whichever future ADR implements ITERRET, so that decision is not re-litigated when that time comes; `fluxmem/ltsm.py:93`'s existing `FaissVectorStore` is the storage-side half of this already in place |
| 3   | Propagation formula | **Plain unweighted average**, `φ'(e_j) = (1/\|N(e_j)\|) Σ φ(e_k)`, exactly as Eq. 3 states              | `hetrep/hg/propagation.py` (Phase 2)                                                                                                                                                                                                                                                                                          |

For row 2: because retrieval itself is out of HetRep's encoding scope (ADR 0005 draws that
boundary explicitly), this ADR does not implement ANN retrieval — it records that when ITERRET is
eventually implemented, it inherits ANN+BM25 as the target, not HyperMem's brute-force approach,
closing the question now rather than leaving it to be rediscovered.

### Why not preserve the component variants as configurable options

| Option                                                            | Description                                                                             | Tradeoff                                                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| (a) Config flag selecting COLM-spec vs. component-native behavior | e.g., `embedding_model: "minilm" \| "qwen3"`, `propagation: "average" \| "softmax"`     | Preserves optionality; but per the standing memory note, this is exactly the anti-pattern — "no config flag is added to preserve HyperMem's Qwen3 embeddings or its softmax propagation" — because it lets an unresolved fidelity question hide behind a default rather than being decided |
| (b) COLM-spec only, hardcoded (chosen)                            | `hetrep/vs/encoder.py` and `hetrep/hg/propagation.py` implement only the paper's choice | No hedge; if COLM's choice underperforms, that is a measured finding recorded in an experiment record, not a silent fallback to whichever variant happens to score better                                                                                                                  |

**Decision: (b).** Absent beats dead: a component variant that is not wired in at all is easier to
reason about than one wired in behind a flag that could silently be left at a non-COLM default. If
HG retrieval quality turns out to be sensitive to the propagation formula, that becomes a new,
explicit experiment — implement the softmax variant as a clearly-labeled alternative and run a
before/after comparison, following the same discipline ADR 0004 applied to HyperMem's own G4/G6
questions (weight fusion, per-level BM25) — not a quiet re-adoption of the value that happened to
already be in the vendored code.

### Interaction with Phase 1's "no summary field" approximation

Phase 1 (VS) has a separate, unrelated approximation: `EpisodicUnit` has no `summary` field, and
COLM's `v_j = Enc(summary(e_j))` needs one. The hetrepv2 plan's decision — concatenate turn text as
the summary input, documented as a stated approximation with `_summarize()` as the drop-in
replacement point — is _not_ an instance of this ADR's fidelity conflict. There is no component
whose implementation disagrees with COLM here; there is simply a field COLM's formula assumes and
`EpisodicUnit` does not yet have. That gap is recorded in ADR 0005's context and the hetrepv2 plan
§5, not resolved by this ADR, because it is a missing input, not a competing design choice between
COLM and a reused component.

## Consequences

### Positive

- **Every number HetRep reports can be attributed to COLM's method without a caveat.** A claim
  like "HetRep implements COLM §1.2.3" becomes literally true rather than true-with-an-asterisk.
- **No hedge to later discover and question.** There is no config default silently carrying a
  component's variant into results; anyone reading `hetrep/vs/encoder.py` or
  `hetrep/hg/propagation.py` sees exactly one behavior, matching the paper.
- **Divergences that do matter become explicit experiments, not implicit defaults.** If COLM's
  plain average underperforms HyperMem's softmax weighting on LoCoMo, that comparison is run and
  reported as a finding — precisely the "measure, don't assume" discipline ADR 0004 already
  established for HyperMem's own G4/G6 questions, applied here at the COLM/component boundary
  instead of within HyperMem alone.
- **Consistent with the standing project rule.** This ADR is the concrete instance of the
  `resolve-paper-ambiguities-dont-hedge` memory note, not a new policy — recording it here makes
  that policy auditable against a specific, cited decision rather than only living in memory.

### Negative

- **HyperMem's engineering improvements are not inherited by default.** Qwen3-Embedding-4B and
  softmax weighting may genuinely retrieve or propagate better than COLM's simpler choices — this
  ADR forgoes that potential improvement in the baseline HetRep configuration until a measured
  comparison justifies adopting it as an explicit, documented variant.
- **`all-MiniLM-L6-v2` is a smaller, older model than Qwen3-Embedding-4B.** Embedding quality on
  out-of-distribution dialogue may be worse; this is a known, accepted cost of matching COLM's
  spec exactly, not an oversight.

### Risks

- **Reviewers or collaborators may expect the "better" component variant and be surprised by the
  simpler COLM-spec choice.** Mitigation: this ADR is the citable record for why; link it from any
  results write-up that reports HG or VS numbers.
- **If COLM's own spec is later found to be a typo or an oversimplification in the source paper**,
  this ADR's resolution needs revisiting rather than being treated as permanently settled — COLM
  is the paper C-AIMMS is writing, and if a genuine correction is warranted, it changes the paper's
  own spec, which then flows back through this ADR rather than around it.
- **Row 2 (retrieval) records a target with no implementation yet.** Because ITERRET does not
  exist in this repo, there is nothing to verify against today beyond the citation itself; the
  risk is that whoever eventually implements ITERRET does not consult this ADR and re-derives
  (or worse, silently inherits HyperMem's brute-force approach by copying code without re-reading
  the spec). Mitigation: `fluxmem/ltsm.py:93`'s `FaissVectorStore` already exists as the
  storage-side half, which is a standing hint in the code itself, not only in this document.

## Related

- **ADR 0003** — established explicit-approximation-vs-silent-deviation as a standard
  (`visual_salience`'s pixel-occupancy proxy is a documented approximation, not a silent one); this
  ADR applies the same standard to a different kind of gap (a component's implementation choice
  vs. the paper's).
- **ADR 0004** — HyperMem's own Phase 1–3 fixes (G1–G8) are unaffected by this ADR: those fixes
  improve HyperMem's fidelity to _its own_ paper and remain in place inside `HyperMem/`; HetRep's
  reuse of `structure.py`/`build_hypergraph()` (ADR 0007) does not touch the extraction/retrieval
  stages those fixes live in.
- **ADR 0005** — the `EpisodeEncoder` seam whose VS and HG arms this ADR constrains.
- **ADR 0007** — the injection boundary that makes swapping extractor/embedder implementations
  possible without touching `hetrep/hg/`'s orchestration code, which is what makes a future
  measured-comparison experiment (if one is run) a drop-in change rather than a rewrite.
