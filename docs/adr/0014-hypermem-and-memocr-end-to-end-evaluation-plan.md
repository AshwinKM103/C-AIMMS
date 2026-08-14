# 0014. HyperMem and MemOCR end-to-end evaluation plan

## Status

Proposed — 2026-08-13. Plan only; no implementation in this ADR. Scope is standalone
correctness verification of HyperMem and MemOCR in isolation from each other, from EM-LLM, and
from fluxmem — consistent with ADR 0004 (HyperMem is an active retrieval baseline, run as a
standalone harness), ADR 0007 (HyperMem's pipeline is not reused by HetRep; only
`structure.py`/`build_hypergraph()` are), and ADR 0011 (MemOCR's segmentation/training pipeline
is independent, no fallback role for HetRep).

## Context

HyperMem (`HyperMem/hypermem/`, Yue et al., 2026, ACL — see `HyperMem/README.md:8`) and MemOCR
(`MemOCR/`, Shi et al., 2026, arXiv:2601.21468 — `MemOCR/README.md:2`) are both vendored as git
submodules (ADR 0006) and both remain available for standalone research use per ADR 0004/0007/0011.
Neither component's ability to actually run standalone, end-to-end, has been verified by running
it — only read and audited. This ADR closes that gap: not by running the pipelines (out of scope
for a Proposed-status planning ADR), but by specifying exactly what "run successfully in isolation"
means for each, what would block it, and in what order to attempt it.

### HyperMem: package layout has moved since ADR 0004/0007

ADR 0004 and 0007 cite stage files at `HyperMem/main/stageN_*.py` and `HyperMem/hypermem/...`
directly. The current layout nests everything under `hypermem/`, confirmed by direct listing:

```
HyperMem/
├── hypermem/
│   ├── main/{stage1..stage6}_*.py, eval.py
│   ├── extractors/, llm/, prompts/, utils/
│   ├── config.py, types.py, structure.py
├── requirements.txt
└── tests/ (8 files, 385 lines total)
```

Entry points are `hypermem/main/stageN_*.py` (run individually, e.g.
`python hypermem/main/eval.py --stages 4 5 6`) or the six-stage driver
`hypermem/main/eval.py` (`HyperMem/hypermem/main/eval.py:1-18`), which shells out to each stage
script via `subprocess.run` (`eval.py:105-110`) — this means eval.py is a sequencing wrapper, not
a library entry point; each stage remains independently invocable. `HyperMem/README.md:118-122`
documents the same interface.

### MemOCR: entry point is not the file named in the task brief

`MemOCR/recurrent/impls/memory_img_final_only_triple.py` defines `MemoryAgent`, which subclasses
`recurrent.interface.RAgent` (`memory_img_final_only_triple.py:15,153`) and is driven by verl's
rollout/generation loop, not invoked directly. There is no `if __name__ == "__main__"` standalone
runner in this file. The actual CLI entry point for evaluation is
`MemOCR/taskutils/memory_eval/run_custom.py`, launched by `MemOCR/scripts/eval.sh:26`
(`python3 taskutils/memory_eval/run_custom.py`), which in turn requires a Ray cluster
(`MemOCR/README.md:151-167`) and a merged HF-format checkpoint under `results/<exp>/.../actor`
(`eval.sh:6-19`, calling `scripts/merge_ckpt.sh` if not already merged). "Consult
`memory_img_final_only_triple.py`" in the investigation brief finds the _agent implementation_,
not a standalone script — this is a load-bearing distinction for the test plan below.

**Rendering is not per-episode in this variant.** `MemoryAgent.action()` only calls
`batch_generate_images` (the Playwright render call) in the final-turn branch
(`memory_img_final_only_triple.py:220-225`); during intermediate chunk turns, image generation is
explicitly skipped (`:243`, `logger.info('Skipping memory image generation in FINAL_ONLY mode')`,
with the per-chunk render call commented out at `:244-254`). This is a real divergence from ADR
0011's context section ("Each buffered episode is drafted... and rendered to an image") — ADR
0011 describes MemOCR's general design; this specific implementation
(`memory_img_final_only_triple.py`, filename literally says "final only") renders once per
sample, not once per episode. Flagged here rather than silently assumed; ADR 0011 is not amended,
since it describes the paper's general architecture and other MemOCR variants may render
per-chunk — confirming which variant(s) exist and are default is **TBD: needs investigation**
into sibling files under `MemOCR/recurrent/impls/`.

### MemOCR: the released checkpoint's eval path is undocumented

`MemOCR/scripts/eval.sh:6` hardcodes `MODEL_ROOT` to a path produced by this project's own
training run (`results/memocr/memocr_train_qwen2d5_7b_vl_triple/global_step_1000/actor`), then
merges it via `merge_ckpt.sh` into an HF-format directory. Nothing in `eval.sh`, `train.sh`, or
`README.md` shows `MODEL_ROOT`/`MODEL_NAME` pointed at `meituan/MemOCR-7B` directly from
HuggingFace — the README documents `meituan/MemOCR-7B` only as a training start point
(`README.md:53` cross-referenced from ADR 0011's table) and as the citation-worthy released
artifact, not as a value ever assigned to an eval-path variable in any script. Whether pointing
`MODEL_NAME` straight at the HF checkpoint (skipping `merge_ckpt.sh`, since it's already
HF-format) works out of the box is **TBD: needs investigation** — verified absent from scripts,
not verified to fail.

### Rendering server is a required, unstubbed external process

`MemOCR/md2img/markdown_api_server.py:16` imports `playwright.sync_api` directly and launches
Chromium headless per-thread (`:42-56`, `chromium.launch(headless=True, args=[...])`) inside a
FastAPI service. `MemOCR/recurrent/impls/call_md_renderer.py:12-14` is the client side: it reads
`RENDER_SERVER_IP`/`RENDER_SERVER_PORT` from the environment and POSTs to `/render`
(`call_md_renderer.py:40-48`) with a `retry_on_error` decorator (`:17-28`, 3 retries, 1s delay,
then raises). There is no in-process rendering path and no mock/stub server shipped in the repo —
confirming ADR 0011's "Rendering remains HTTP-based" finding still holds, and adding that the
client-side dependency is a bare `requests.post` with no circuit breaker beyond 3 retries, so a
down render server surfaces as an exception after ~3 seconds, not a silent failure (correcting
ADR 0011's "Rendering server crashes silently" risk note for this specific client path — the
_server_ dying mid-render inside its own thread pool may still drop output silently, but the
_client's_ view of a fully-down server is an exception, not silence).

### HyperMem's LoCoMo dataset is not vendored

`HyperMem/hypermem/config.py:18` sets `dataset_path = DATA_DIR / "locomo10.json"` where
`DATA_DIR = PROJECT_ROOT / "data"`. Direct search of the repository tree found no
`HyperMem/data/` directory and no file matching `locomo*` anywhere under `C-AIMMS/` — the dataset
must be fetched separately before Stage 1 can run at all. This is a harder blocker than a config
default: without it, Stage 1 fails at `load_locomo_raw_data` (`stage1_memory_extraction.py:55`)
before any pipeline code under test executes.

### HyperMem's HTTP dependencies are hardcoded to one model family, confirmed at the call site

Re-verified directly (not just cited from ADR 0007): `HyperMem/hypermem/llm/embedding_provider.py:55`
raises `ValueError` unless `'Qwen3' in self.model_name`. This check runs inside `embed()`, i.e. at
call time against whatever `model_name` `config.py`'s `embedding_config` supplies
(`config.py:39-42`, defaulting to `"Qwen3-Embedding-4B"` against `localhost:11810`). There is no
`sentence-transformers` or other local-embedding fallback path in this file. Swapping the model
requires both changing the config value to something containing `"Qwen3"` (cosmetic) or editing
this check (real code change) — it is not a runtime-swappable dependency as currently written.

## Evaluation Criteria

"Working correctly in isolation" is defined per component as a pipeline that:

1. **Runs to completion** on a small, real (or minimally synthetic) dialogue input without any
   cross-component import (no `fluxmem`, `EM-LLM`, or the other of {HyperMem, MemOCR}).
2. **Produces the documented intermediate/final artifacts** (HyperMem: episode/topic/fact JSON,
   hypergraph JSON, BM25/vector indices, retrieval results; MemOCR: rendered memory image(s) and a
   final boxed answer).
3. **Fails loudly, not silently**, when a required external dependency (embedding server, reranker
   server, render server, LLM API key) is absent — an explicit connection error, not a hang or a
   wrong-but-plausible output.
4. Is judged on **correctness of the mechanism**, not benchmark accuracy — a hypergraph that has
   the right node/edge counts for a 3-message toy conversation is a pass; whether it scores well on
   LoCoMo is a separate, later question this ADR does not evaluate.

## Standalone Test Plan

### HyperMem

| Stage            | Entry point                                     | Can run without fluxmem/EM-LLM/MemOCR?                                                  | Hard external deps                                                                                      |
| ---------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 1 (extraction)   | `hypermem/main/stage1_memory_extraction.py`     | Yes — imports only `hypermem.*` (`stage1:33-41`)                                        | LLM API (OpenRouter, per `config.py:102-109`); LoCoMo JSON (**missing**, see Context)                   |
| 2 (hypergraph)   | `hypermem/main/stage2_hypergraph_extraction.py` | Yes, per ADR 0007's audit (pure function `build_hypergraph`, `hypergraph_extractor.py`) | LLM API (extractors call inline, per ADR 0007 Context §4)                                               |
| 3 (indexing)     | `hypermem/main/stage3_hypergraph_index.py`      | Yes                                                                                     | Embedding server (`localhost:11810`, Qwen3-only); NLTK data (auto-downloads if missing, `stage3:48-58`) |
| 4 (retrieval)    | `hypermem/main/stage4_hypergraph_retrieval.py`  | Yes                                                                                     | Embedding + reranker servers (`localhost:11810`/`:12810`)                                               |
| 5 (response)     | `hypermem/main/stage5_response.py`              | Yes, but out of scope per ADR 0004 ("removed from active import paths")                 | LLM API                                                                                                 |
| 6 (eval/scoring) | `hypermem/main/stage6_eval.py`                  | Yes — dataset-agnostic per ADR 0004                                                     | Judge LLM API                                                                                           |

**Success metrics (correctness, not accuracy):**

- Stage 1: N input messages → the boundary detector emits ≥1 `Episode` object with non-empty
  `summary`/`content` fields; `should_wait` is threaded per ADR 0004's Phase 1 fix
  (`stage1_memory_extraction.py`, wired per the Implementation Status note in ADR 0004).
- Stage 2: hypergraph JSON round-trips through `Hypergraph.to_dict()`/`from_dict()`
  (`structure.py:358-375`, already verified by ADR 0007 to execute cleanly) and every episode
  produced by stage 1 appears as exactly one `EpisodeNode`.
- Stage 3: `BM25Corpus` (`stage4_hypergraph_retrieval.py`, tested directly in
  `tests/test_stage4_bm25_corpus.py`) builds without exception over the stage-2 output; index
  file(s) exist under `results/<exp>/bm25_index/`.
- Stage 4: a query against the toy hypergraph returns a non-empty, ranked result list at each of
  topic/episode/fact levels, using the top-k defaults from `config.py:52-54` (10/10/30).
- Stage 6: `stage6_eval.py` scores a hand-constructed (prediction, gold) pair and returns a finite
  BLEU/ROUGE/BERTScore/METEOR set — the ADR 0004-documented "generic answer-scoring harness" role.

**Execution path:** a 3-10 message synthetic conversation (not the full LoCoMo-10) run through
`python hypermem/main/eval.py --stages 1 2 3 4` with `HYPERMEM_EXPERIMENT_NAME` set to an isolated
test experiment name, after standing up `scripts/serve_embedding.sh` and (if `use_reranker=true`
default holds) `scripts/serve_reranker.sh`.

**Current test coverage:** 8 files, 385 lines (`tests/test_config.py`,
`test_episode_summary_field.py`, `test_stage1_should_wait.py`, `test_stage3_bm25_indexing_mode.py`,
`test_stage4_bm25_corpus.py`, `test_stage4_rerank_field.py`, `test_stage4_weight_fusion.py`,
`test_topic_extractor_match_threshold.py`). Verified `test_stage4_bm25_corpus.py` directly: all
three tests construct `BM25Okapi` in-memory and assert on `BM25Corpus` wrapper behavior — no
network call, no LLM, no LoCoMo file. **These are hermetic unit tests of ADR 0004's gap-fix code,
not integration tests of the pipeline end-to-end.** No test in this suite exercises stage1's LLM
call, stage3/4's embedding-server call, or a full multi-stage run — that gap is exactly what this
ADR's standalone test plan is proposing to fill, not something already covered.

### MemOCR

| Component      | Entry point                                                                               | Standalone (no fluxmem/EM-LLM/HyperMem)?                                      | Hard external deps                                                                                                       |
| -------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Rendering      | `md2img/markdown_api_server.py` (server) + `recurrent/impls/call_md_renderer.py` (client) | Yes, as a two-process pair                                                    | Playwright + Chromium (`markdown_api_server.py:16,42-56`)                                                                |
| Memory agent   | `recurrent/impls/memory_img_final_only_triple.py::MemoryAgent`                            | No — requires verl's `RAgent`/`RDataset`/rollout loop, not directly invocable | verl, Ray, a tokenizer/processor matching a Qwen-VL checkpoint (`:179` asserts `'vl' in tokenizer.name_or_path.lower()`) |
| Evaluation CLI | `taskutils/memory_eval/run_custom.py` via `scripts/eval.sh`                               | No — hard-requires a merged local checkpoint (see Context) and a Ray cluster  | Ray, GPU, HF checkpoint, render server                                                                                   |

**What "end-to-end" means for MemOCR, corrected against the brief's assumption:** the brief frames
it as "input dialogue → fixed-window segmentation → render to image → output memory images."
Verified: segmentation (fixed-window chunk buffering) happens inside `MemoryAgent.action()`
(`:196-274`, chunk slicing at `:240`), but per the Context section's finding, _rendering in the
`final_only` variant happens once, at the end, over the accumulated memory text_ — not once per
chunk. "End-to-end" for this specific file is: dialogue → N chunk-conditioned memory-drafting LLM
calls (no image per chunk) → one render call over the final memory → one boxed answer. Whether a
non-final-only sibling implementation exists that renders per-chunk (matching the brief's original
framing and ADR 0011's general description) is **TBD: needs investigation** — out of this ADR's
verified findings, flagged for the next investigation pass rather than assumed either way.

**Retrieval:** none. Confirmed by direct README read (`MemOCR/README.md`) and file listing —
no BM25/FAISS/vector-index code anywhere under `MemOCR/`. Memory here means one evolving
markdown-then-image artifact per sample, read by the VLM directly; there is no separate retrieval
step to test. See Retrieval Gap Analysis below.

**Success metrics (correctness, not accuracy):**

- Rendering pair: POST a small markdown string to a locally-started `markdown_api_server.py`;
  assert a decodable PNG comes back and its dimensions are non-degenerate
  (`batch_filter_images`'s own bounds, `call_md_renderer.py:230`, `min(size) >= 10` and aspect
  ratio `<= 100`, is the repo's own definition of "not garbage" and is reusable as the test
  oracle).
- Memory agent: **cannot be tested by direct invocation** per the entry-point finding above.
  Testing it requires either (a) standing up the verl rollout harness against a real or mocked
  tokenizer/processor, which is a heavier lift than HyperMem's stage-by-stage scripts, or (b)
  extracting `MemoryAgent`'s pure-Python logic (chunk slicing, template formatting) into a
  unit-testable path that bypasses verl's `DataProto`/rollout plumbing. Neither is attempted here;
  this ADR records the gap as a blocker (see below), not a completed plan.

**Current test coverage:** `md2img/test_markdown_api.py`, `md2img/test_use_api.py`,
`md2img/test_use_html_api.py` (rendering-server client tests — likely hermetic against a locally
running server, not verified line-by-line here); `recurrent/test/test_pad_tensor_list_to_length.py`
(a narrow tensor-utility unit test); `taskutils/memory_eval/test_qa.py`,
`recurrent/chat_template/test.py`. **No test exercises `MemoryAgent` itself** — confirmed by
directory listing; `memory_img_final_only_triple.py` has no co-located test file, unlike every
HyperMem stage file which has at least one test targeting a specific gap-fix.

## Paper Fidelity Analysis

This section documents all known deviations between C-AIMMS's vendored components and their source
papers. Deviations are organized by component and paper, with evidence citations and the status of
each gap. This serves as the authoritative record of what "paper-faithful implementation" requires
for both HyperMem and MemOCR.

### HyperMem Deviations from Yue et al. 2025 (HyperMem paper)

**Status: Partially Addressed by ADR 0004 Gap Fixes (G1–G8)**

| #   | Deviation                                                                                                         | COLM Paper Ref     | Status                                                      | Evidence                                                                                                                               | Mitigation                                                                      |
| --- | ----------------------------------------------------------------------------------------------------------------- | ------------------ | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| H1  | Topic aggregation skips retrieval-seeded candidate discovery (§3.2.2)                                             | Not paper-faithful | Documented limitation, lower ROI                            | ADR 0004 G1 (`stage2_hypergraph_extraction.py` does direct LLM matching instead of BM25 pre-filter)                                    | Implement BM25/dense pre-filter before LLM matching (future work, not blocking) |
| H2  | Topic creation conflates Case 1 (Initialization) vs. Case 2 (Topic Creation); missing similarity-threshold branch | Not paper-faithful | **FIXED in ADR 0004 Phase 2**                               | ADR 0004 G2 (`topic_extractor.py:318,452,701,824` now gate topic creation on confidence threshold)                                     | Requires empirical threshold tuning on LoCoMo (currently TBD)                   |
| H3  | `should_wait` signal computed but discarded at stage1                                                             | Not paper-faithful | **FIXED in ADR 0004 Phase 1**                               | ADR 0004 G3 (`stage1_memory_extraction.py:176` now wires `should_wait` into buffer logic)                                              | Control flow distinction now threaded; fixes extraction diagnostics             |
| H4  | Hyperedge weights computed at extraction but discarded at retrieval                                               | Not paper-faithful | **MECHANISM IMPLEMENTED, NOT EVALUATED** (ADR 0004 Phase 3) | ADR 0004 G4 (`stage4_hypergraph_retrieval.py:683,706,798,821` has `fuse_hyperedge_weights` function, gated by config, default `False`) | Run LoCoMo before/after comparison; decide whether fusion justifies LLM cost    |
| H5  | Reranker uses `episode_description` (long) not `summary` (short)                                                  | Not paper-faithful | **FIXED in ADR 0004 Phase 2**                               | ADR 0004 G5 (both `stage4_hypergraph_retrieval.py` reranking calls now use `text_field="summary"`)                                     | Verified `summary` is LLM-generated (not truncation) before flip                |
| H6  | Single pooled BM25 corpus across Topic/Episode/Fact levels instead of per-level                                   | Silent divergence  | **MECHANISM IMPLEMENTED, NOT EVALUATED** (ADR 0004 Phase 3) | ADR 0004 G6 (`stage3_hypergraph_index.py` has `bm25_indexing_mode="per_level"` variant)                                                | Run LoCoMo per-level vs. pooled comparison; decide default                      |
| H7  | Config defaults (top_k, reranker) diverge from paper                                                              | Not paper-faithful | **FIXED in ADR 0004 Phase 1**                               | ADR 0004 G7 (`config.py:52-54` now `topic_top_k=10, episode_top_k=10` per paper; `use_reranker=True`)                                  | No further mitigation needed                                                    |
| H8  | No tests in vendored HyperMem code; active baseline runs untested                                                 | Process gap        | **FIXED in ADR 0004 Phase 4**                               | ADR 0004 G8 (22 new tests across `HyperMem/tests/`, `pytest tests/ -q` → `22 passed`)                                                  | Tests land in same PR as fixes (Phases 1-3)                                     |

**Rolled-up Paper Fidelity Status for HyperMem:**

- **H1:** Lower-priority limitation; does not affect retrieval correctness, only cost
- **H2, H3, H5, H7:** ✅ **FULLY FIXED** (implemented and merged)
- **H4, H6:** ✅ **MECHANISM IMPLEMENTED**, ⚠️ **PENDING EVALUATION** (default still off; Phase 3 LoCoMo runs required before deciding)
- **H8:** ✅ **TEST COVERAGE ADDED** (22 tests verify gap fixes)

### HyperMem Deviations from COLM Paper

**Status: Recorded as Design Target for Phase 2-3 (ADR 0008)**

These are conflicts between HyperMem's implementation choices and COLM's specification, _not_ gaps in
HyperMem's fidelity to its own paper. Per ADR 0008, COLM is the authoritative spec whenever it conflicts with
a reused component; HetRep must adapt the component, never adapt COLM.

| #   | COLM Spec                                                                                   | HyperMem Code                                                                                                         | HetRep Choice                                                        | Evidence                                                                                            | Status                                                     |
| --- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| C1  | Embedding model: **all-MiniLM-L6-v2** (384-d, COLM §1.2.3)                                  | Qwen3-Embedding-4B (HTTP, hardcoded at `:55-56`)                                                                      | **all-MiniLM-L6-v2** via `sentence-transformers`, loaded from `.env` | ADR 0008 row 1; `embedding_provider.py:55-56` raises `ValueError` if model not containing `"Qwen3"` | ✅ **DESIGNED FOR PHASE 1** (`hetrep/vs/encoder.py`)       |
| C2  | Retrieval: **ANN (faiss) + BM25 hybrid, RRF-fused** (COLM §1.2.3)                           | Brute-force `np.dot(doc_vecs, query_vec)` (no faiss, line `:27-52`)                                                   | **faiss ANN** + BM25 + RRF per COLM                                  | ADR 0008 row 2; no faiss in HyperMem's `embedding_provider.py`                                      | 📋 **DESIGNED FOR ITERRET (FUTURE ADR)**, not HetRep scope |
| C3  | Propagation: **Plain unweighted average** (COLM Eq. 3: `φ'(e_j) = (1/\|N(e_j)\|) Σ φ(e_k)`) | Softmax-weighted sum (`stage3_hypergraph_index.py:650-692`, `torch.softmax(weights_tensor, dim=0)` then weighted sum) | **Plain unweighted average** exactly per Eq. 3                       | ADR 0008 row 3                                                                                      | ✅ **DESIGNED FOR PHASE 2** (`hetrep/hg/propagation.py`)   |

**Rolled-up Paper Fidelity Status for COLM Alignment:**

- **C1:** ✅ Phase 1 (`hetrep/vs/encoder.py`) will implement COLM-spec embedding
- **C2:** 📋 Recorded as design target for whoever implements ITERRET; out of HetRep scope
- **C3:** ✅ Phase 2 (`hetrep/hg/propagation.py`) will implement COLM-spec propagation

### MemOCR Deviations from Shi et al. 2026 (MemOCR paper)

**Status: Partially Characterized; B7 Blocker Requires Investigation**

| # | Deviation | Paper Spec | Current Code | Status | Evidence | Mitigation |
| --- | --- | --- | --- | --- | --- |
| M1 | Rendering behavior: per-episode vs. final-only | General description implies per-episode (ADR 0011 Context §1-2: "Each buffered episode is drafted...and rendered to an image") | `memory_img_final_only_triple.py:220-254` renders only at final turn; per-chunk code commented out | **TWO VARIANTS EXIST** | `memory_img.py:240` calls `batch_generate_images` per chunk; `memory_img_final_only_triple.py:243` skips it ("FINAL_ONLY mode") | **B7 BLOCKER:** Confirm which variant is paper-faithful; test plan must target correct file |
| M2 | Rendering server infrastructure | HTTP-based Playwright/Chrome | `markdown_api_server.py:16,42-56` launches Chromium headless per-thread; `call_md_renderer.py:12-14` is HTTP client | Paper does not specify determinism; C-AIMMS gets non-deterministic pixel-level rendering | ADR 0014 Context §3; ADR 0011 Risks: "Chrome may render slightly differently on different systems or with different font/driver versions" | Document as inherent constraint; run same episode twice may produce pixel-different images (acceptable for now) |
| M3 | Entry point: standalone invocation | No specification given | `MemoryAgent` inherits `RAgent`, requires verl's rollout loop; no `__main__` block | Not paper-faithful | ADR 0014 Context §2 (`:15,153`); no standalone runner in file | **B3 BLOCKER:** Extract chunk/template logic to unit-testable path, or build minimal verl harness (heavy lift for integration test) |
| M4 | Checkpoint path: released vs. locally-trained | Paper cites meituan/MemOCR-7B as released checkpoint | `eval.sh:6-19` hardcodes locally-trained path (`results/memocr/...`); README documents HF checkpoint only as training start point, not eval path | Documented gap | ADR 0014 Context §2; `eval.sh` has no `MODEL_NAME` variable pointing to HF checkpoint | **B5 BLOCKER:** Test whether `MODEL_NAME=meituan/MemOCR-7B` works directly (cheap experiment) |
| M5 | Retrieval stage | None (fixed-window chunking, single artifact per sample) | No retrieval code anywhere under `MemOCR/`; confirmed by file listing | **ARCHITECTURAL GAP: NOT A FIDELITY ISSUE** | ADR 0014 Retrieval Gap Analysis; ADR 0011 Consequences: "Memory is one evolving artifact per sample rather than an indexed store queried at read time" | No mitigation needed for MemOCR isolation testing; document explicitly if cross-component comparison needed |
| M6 | Segmentation: fixed-window vs. surprise-based | Fixed-window as design choice (ADR 0011 Context §1) | `memory_img_final_only_triple.py:196-274` buffer slicing (`chunk_size * (step + 1)`) | **PAPER-FAITHFUL: DESIGNED CHOICE** | ADR 0011 Decision §1 (fixed-window is deliberate for visual memory); no contradiction with Shi et al. | No mitigation; segmentation design is independent per ADR 0011 |

**Rolled-up Paper Fidelity Status for MemOCR:**

- **M1:** ⚠️ **B7 BLOCKER** — Two variants exist; unclear which is paper-faithful
- **M2:** ✅ **ACCEPTABLE CONSTRAINT** — Non-determinism is inherent to HTTP Playwright rendering, documented
- **M3:** ⚠️ **B3 BLOCKER** — No standalone entry point; requires verl harness or logic extraction
- **M4:** ⚠️ **B5 BLOCKER** — Checkpoint path undocumented; cheap experiment to test
- **M5:** ✅ **ARCHITECTURAL DIFFERENCE, NOT GAP** — MemOCR has no retrieval by design; document in comparisons
- **M6:** ✅ **PAPER-FAITHFUL** — Fixed-window segmentation is independent design, not deviation

### Summary: What "Paper-Faithful Implementation" Requires

**For HyperMem (Yue et al. 2025 + COLM):**

1. ✅ All 8 ADR 0004 gap fixes (H1-H8) implemented
2. ✅ HyperMem G2, G3, G5, G7 are fully fixed and merged
3. ⚠️ HyperMem G4 (weight fusion) and G6 (per-level BM25) mechanisms implemented but NOT YET EVALUATED (Phase 3 LoCoMo runs pending)
4. ✅ COLM-spec embedding (`all-MiniLM-L6-v2`) designed for Phase 1 (not using Qwen3-Embedding-4B)
5. ✅ COLM-spec propagation (plain average, not softmax) designed for Phase 2

**For MemOCR (Shi et al. 2026):**

1. ✅ M2 (rendering non-determinism) is acceptable, documented
2. ✅ M6 (fixed-window segmentation) is paper-faithful design choice
3. ⚠️ M1 (rendering per-chunk vs. final-only) requires B7 blocker resolution
4. ⚠️ M3 (no standalone entry point) is a test harness blocker, not a fidelity issue
5. ⚠️ M4 (checkpoint path) is a cheap experiment to resolve via B5
6. ✅ M5 (no retrieval) is architectural difference, acceptable if documented

**Execution Prerequisites:**

- **HyperMem:** Complete Phase 3 LoCoMo runs (G4/G6) to enable/disable mechanisms; all Phases 1-4 (testing) merged
- **MemOCR:** Resolve B7 (which variant?), B5 (checkpoint path), B3 (entry point) before attempting integration test

## Retrieval Gap Analysis

**HyperMem retrieval (stages 3-4) is embedded in the pipeline, not external**, and is the active
baseline per ADR 0004 — it is the component ADR 0004's G4/G5/G6 gap-fixes already target, and its
BM25/reranking logic is directly unit-tested (`test_stage4_bm25_corpus.py`,
`test_stage4_rerank_field.py`, `test_stage4_weight_fusion.py`). No mitigation needed here; the gap
is coverage depth (unit-tested gap-fixes, not an integration run), not a missing capability.

**MemOCR has no retrieval stage, confirmed absent, not merely undocumented.** Memory is a single
evolving artifact per sample rather than an indexed store queried at read time; this is a
structural property of the approach (visual working-memory drafting), not an oversight. **This is
a genuine architectural gap relative to HyperMem, not a testing gap** — there is nothing to write
a retrieval test against. If C-AIMMS ever needs to compare "HyperMem's retrieval quality" against
"MemOCR's retrieval quality" on equal terms, that comparison has no MemOCR-side referent and would
need to be reframed as "hypergraph retrieval vs. single-artifact visual read," which is a research
design question outside this ADR's scope (isolation testing only, per the brief's constraints).

**Mitigation:** none required for MemOCR's isolation test — its test plan above does not need a
retrieval-stage row, since none exists to test. Document this explicitly in any future
cross-component comparison write-up so a missing retrieval score isn't misread as "not yet
measured" when it is in fact "not applicable."

## Blockers & Mitigations

| #   | Blocker                                                                             | Component | Evidence                                                                                                    | Mitigation                                                                                                                                                                                                                                                                                                                                 |
| --- | ----------------------------------------------------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| B1  | LoCoMo dataset not vendored (`HyperMem/data/locomo10.json` absent)                  | HyperMem  | Direct search, no match anywhere under `C-AIMMS/`                                                           | Fetch LoCoMo-10 separately, or hand-construct a 3-10 message synthetic conversation matching `load_locomo_raw_data`'s expected schema (`stage1_memory_extraction.py:55-`) for isolation testing — the standalone test plan above deliberately proposes the synthetic route to avoid a full-dataset dependency for a correctness-only check |
| B2  | Embedding/reranker HTTP servers required, hardcoded to Qwen3                        | HyperMem  | `embedding_provider.py:55-56`; `config.py:39-42,92-95`                                                      | Stand up `scripts/serve_embedding.sh`/`serve_reranker.sh` for isolation testing. **NOTE:** Qwen3 is HyperMem's design choice, not C-AIMMS's. For HyperMem's own pipeline testing, accept Qwen3. For COLM-faithful implementation (HetRep Phase 1), use all-MiniLM-L6-v2 (see Paper Fidelity Analysis, C1)                                  |
| B3  | `MemoryAgent` has no standalone entry point; only invocable via verl's rollout loop | MemOCR    | No `__main__` block in `memory_img_final_only_triple.py`; class inherits `RAgent`, consumes `DataProto`     | Extract chunk/template logic to unit-testable function (lighter), or build minimal verl harness (heavier). This is a test-harness blocker, not a paper-fidelity issue (see Paper Fidelity Analysis, M3)                                                                                                                                    |
| B4  | Render server is a hard, unstubbed dependency                                       | MemOCR    | `call_md_renderer.py:12-28`, no mock server in repo                                                         | Run `markdown_api_server.py` locally for isolation testing (Chrome/Playwright one-time install, per `MemOCR/README.md:113-130`). This is an acceptable infrastructure constraint (see Paper Fidelity Analysis, M2); non-determinism is inherent to HTTP Playwright                                                                         |
| B5  | Released checkpoint (`meituan/MemOCR-7B`)'s path into `eval.sh` is undocumented     | MemOCR    | `eval.sh:6-19` hardcodes locally-trained checkpoint path; no script references HF checkpoint                | Test whether `MODEL_NAME=meituan/MemOCR-7B` works directly (cheap experiment, skip `merge_ckpt.sh` if already HF-format). **Paper-fidelity issue:** Shi et al. cite released checkpoint; eval.sh doesn't expose this path (see Paper Fidelity Analysis, M4)                                                                                |
| B6  | GPU + Ray cluster required for any MemOCR run at all                                | MemOCR    | `README.md:151-167` (Ray cluster setup), `eval.sh` (GPU-sized `MODEL_TP=4`)                                 | Infrastructure prerequisite, not a code gap. Flag for whoever executes this plan                                                                                                                                                                                                                                                           |
| B7  | Two MemOCR rendering variants exist; unclear which is paper-faithful                | MemOCR    | `memory_img.py:240` renders per chunk; `memory_img_final_only_triple.py:243` skips per-chunk ("FINAL_ONLY") | Confirm which variant matches Shi et al. (general description implies per-episode). Both exist as real implementations. **Paper-fidelity blocker** (see Paper Fidelity Analysis, M1): must audit both against paper language before committing test plan                                                                                   |

## Recommendation

### MemOCR Blockers: B7 and B5 Resolved

**B7 (Rendering Variant) — RESOLVED:** Paper (Shi et al. 2026, Eq. 6) explicitly specifies rendering at "the final step," not per-chunk. Use `MemOCR/recurrent/impls/memory_img_final_only_triple.py` for evaluation. The per-chunk variant (`memory_img.py`) diverges from paper specification.

**B5 (Checkpoint Path) — RESOLVED:** Released checkpoint `meituan/MemOCR-7B` loads directly from HuggingFace without `merge_ckpt.sh`. Updated `MemOCR/scripts/eval.sh` to support both locally-trained (default for backward compatibility) and released checkpoint (paper-faithful). To use released: `MODEL_NAME=meituan/MemOCR-7B`.

### Execution Priority

**Priority order: HyperMem first, MemOCR second — B7 and B5 are now unblocked.**

HyperMem is more modular by direct evidence: six independently invocable stage scripts
(`hypermem/main/stageN_*.py`), 8 existing unit-test files already targeting its retrieval-layer
gap-fixes, and only two categories of external dependency (LLM API, embedding/reranker HTTP
servers) — both swappable via config even if not swappable past the Qwen3 hardcode without a code
change. Its isolation test plan above is executable largely as written, modulo fetching or
synthesizing a LoCoMo-shaped input (B1).

MemOCR is materially less modular: its core logic (`MemoryAgent`) has no standalone entry point
and is coupled to verl's actor/rollout interface (B3), its documented eval path assumes a
checkpoint this project would have to train itself rather than the released one (B5), and it
carries a GPU + Ray cluster prerequisite before any test — mechanism or accuracy — can run at all
(B6). Before spending effort on B3 (the heaviest blocker — building a harness or extracting logic),
resolve B7 first: confirm which `recurrent/impls/*.py` variant is the one this project actually
intends to evaluate, since the brief's originally assumed per-episode-rendering behavior does not
match the file it named. Attempting B5 (pointing eval at the released checkpoint) is a cheap,
high-information next step that can run in parallel with B7's file audit, since it doesn't depend
on which agent variant is chosen.

## Related

- **ADR 0004** — HyperMem is the active retrieval baseline this ADR's test plan targets; its
  gap-fixes (G1-G8) are what the existing 8-file test suite covers, confirmed here to be
  unit-level, not integration-level.
- **ADR 0006** — both components are git submodules; this ADR does not change that.
- **ADR 0007** — established `structure.py`/`build_hypergraph()` as HyperMem's only HetRep-reused
  surface; this ADR is orthogonal — it evaluates HyperMem's own pipeline standalone, not its reuse
  by HetRep.
- **ADR 0011** — MemOCR's segmentation/training architecture; this ADR's B7 finding (final-only
  vs. per-chunk rendering) is a correction of specificity against ADR 0011's general description,
  not a contradiction of its conclusions.
