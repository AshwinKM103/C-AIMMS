# 0007. HG encoder reuses HyperMem's data model, not its pipeline

## Status

Accepted — 2026-08-13 (revised). Sets the reuse strategy for Phase 2 of the hetrepv2 plan (`hetrep/hg/`,
tasks T-11 through T-14). No `hetrep/hg/` code lands before this ADR; `hetrep/hg/adapter.py`
implements the decision below. Documents the three independent segmentation methods in the C-AIMMS
codebase and their intended uses.

## Context

HyperMem (`HyperMem/hypermem/`, Yue et al., 2025, arXiv:2604.08256 — vendored per ADR 0004 and now
a submodule per ADR 0006) is the only available implementation of a three-level hypergraph memory
structure (Topic → Episode → Fact nodes, hyperedges linking each level) matching the shape COLM's
HG format needs. It was audited module-by-module for reuse potential (hetrepv2 plan §2.6, 39 tool
calls, findings below); the audit's conclusion splits cleanly into "importable as-is" and "not
importable without dragging in machinery HetRep's HG arm does not need."

### Three Independent Segmentation Methods (Audit Finding)

C-AIMMS contains three separate episode segmentation methods in its vendored/submodule components, none of which are designed to fall back to each other:

| Method                      | Source                                                           | Algorithm                                                                                                     | Notes                                                                                                |
| --------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **EM-LLM (surprise-based)** | `EM-LLM/caimms_boundary_creator.py`, `similarity_refinement/`    | Tracks model-output surprise/KV-divergence from running window to detect episode boundaries                   | Standalone implementation; not presently integrated with HetRep or fluxmem                           |
| **HyperMem (LLM-based)**    | `HyperMem/stage1_memory_extraction.py:135–250`                   | Calls `ConvEpisodeExtractor.extract_episode()` per-message; uses LLM boundary detector + `should_wait` signal | Fully independent; built for HyperMem's own research pipeline                                        |
| **MemOCR (fixed-window)**   | `MemOCR/recurrent/impls/memory_img_final_only_triple.py:196–274` | Fixed-size chunk buffer (`chunk_size * (step + 1)`) on message arrivals                                       | Fixed-window chunking; combined with encoding (renders to canvas) in one `MemoryAgent.action()` call |

**Design consequence:** These are three parallel implementations that are independent of each other. No fallback chain or automatic switching between them is assumed. Each component remains usable for standalone research or ablation studies.

### What is directly reusable

| Module                                        | Contents                                                                 | Verified how                                                                                                                             |
| --------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `hypermem/structure.py`                       | `Hypergraph`, `FactNode`/`EpisodeNode`/`TopicNode`, both hyperedge types | `to_dict()`/`from_dict()` (`structure.py:358-375`) round-trip through plain JSON — executed directly, not inferred from reading the code |
| `hypermem/types.py`                           | `Episode`, `Fact`, `Topic` dataclasses                                   | Imports cleanly with no LLM, no network                                                                                                  |
| `hypermem/extractors/hypergraph_extractor.py` | `build_hypergraph(episodes)` — a **pure function**                       | Imports and runs standalone; no `sys.path.insert` dependency at call time                                                                |
| `hypermem/utils/datetime_utils.py`            | stdlib-only helpers                                                      | No external dependency                                                                                                                   |

This is a small, coherent subset: a data model plus one pure function that assembles it from a
list of already-extracted episodes. Nothing in this subset makes a network call, imports `verl`
or any GPU-touching library, or requires `sys.path.insert(0, ...)` gymnastics — the pattern every
`main/stageN_*.py` file uses (`stage1_memory_extraction.py:31`, repeated identically in stages
2–6) because HyperMem is not a pip-installable package (no `pyproject.toml`, no `setup.py`).

### What is not reusable without pulling in the whole pipeline

Four properties of HyperMem's architecture, each independently confirmed:

1. **LLM required at extraction time, not just query time.** `topic_extractor.py:318,452,701,824` and
   `fact_extractor.py:375,592` and `conv_episode_extractor.py:242,290` all call an LLM inline
   during extraction. There is no offline/deterministic extraction path — every fact and topic
   the hypergraph would contain is produced by a live model call, with no injection boundary
   between the extractor and the LLM client.
2. **Two HTTP servers, hardcoded to one model family.** Embedding and reranking are both served
   over HTTP (Qwen3-Embedding-4B on `:11810`, Qwen3-Reranker-4B on `:12810`), and
   `embedding_provider.py:55-56` raises `ValueError` if the configured model name does not contain
   `"Qwen3"` — verified directly: `if 'Qwen3' not in self.model_name: raise ValueError(...)`.
   There is no `sentence-transformers` fallback path to swap in.
3. **Batch-only, per-conversation.** `hypergraph_extractor.py:60-83` always constructs a fresh
   `Hypergraph()` and `stage2:852-871` writes one JSON file per conversation. There is no
   persisted, incremental-append path for adding episodes to an existing graph over time — which
   is exactly the access pattern HetRep's HG arm needs, since episodes arrive one at a time from
   the segmentation/encoding seam (ADR 0005), not as a pre-collected batch.
4. **LoCoMo glue is load-bearing in stages 1, 4, 5, 6** (`config.py:18`, `stage1:49-125`,
   `stage4:1211,1333`, `stage5:24-90,144-145`, `stage6:60,380-414`). Only stages 2 and 3 are
   already dataset-agnostic. Reusing "the pipeline" would mean reusing LoCoMo-specific glue that
   has nothing to do with encoding a `EpisodicUnit` for COLM.

Persistence in the full pipeline is also pickle-based (`stage3:255,347,516-519,828-835`),
disposable per-run and explicitly prohibited from being committed by
`.claude/rules/storage-invariants.md`. This is a separate reason the pipeline as a whole is the
wrong reuse unit even before the LLM/HTTP/batch issues above: its own persistence choice conflicts
with a hard project rule, whereas `Hypergraph.to_dict()`'s verified plain-JSON round-trip does not.

### Extractors specifically are not independently importable

`topic_extractor.py` and `fact_extractor.py` are not standalone modules that happen to also
support an LLM mode — the LLM call is inline and unconditional, with no Protocol or callback
boundary separating "decide what facts/topics exist" from "call an LLM to decide." Importing them
means importing their LLM client construction, which in turn assumes the two-HTTP-server
environment above. There is no partial import that gets the extraction logic without the network
dependency.

## Decision

**Reuse `structure.py` and `build_hypergraph()` directly, unmodified. Do not import or adapt
HyperMem's extractors, stages, or pipeline glue.**

`hetrep/hg/adapter.py` maps `EpisodicUnit` (fluxmem/HetRep's type) to HyperMem's `Episode`
(`hypermem/types.py`) and back, so that `build_hypergraph()` can be called against
HetRep-produced episodes without modification to either side's data model.

`hetrep/hg/encoder.py` defines the extraction boundary HyperMem lacks, as Protocols:

```python
class FactExtractor(Protocol):
    def extract(self, episode: Episode) -> list[Fact]: ...

class TopicExtractor(Protocol):
    def match_topics(self, episode: Episode, existing: list[Topic]) -> list[Topic]: ...
```

`HgEncoder.encode` (satisfying `EpisodeEncoder` from ADR 0005) calls the injected extractors, then
`build_hypergraph()` with their output, then `hetrep/hg/propagation.py` (COLM Eq. 3 — see ADR 0008) to compute `hyperedge_density`. **Do not import
`hypermem.extractors.topic_extractor` or `fact_extractor` directly** — same reasoning ADR 0003
used for MemOCR's render path: an LLM-calling, network-touching component is external relative to
`hetrep/hg/`'s own test suite, and injecting it behind a Protocol is what keeps that suite
hermetic (`.claude/rules/testing.md`).

### Why injection over adaptation

| Option                                                                             | Description                                                                         | Tradeoff                                                                                                                                                                            |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (a) Wrap HyperMem's extractors in an adapter, expose as `FactExtractor` Protocol   | Reuse proven extraction logic                                                       | Still hard-imports HyperMem's LLM clients and HTTP dependencies; tests can't run hermetically without mocking those servers; externally-facing HTTP makes testing non-deterministic |
| (b) Inject Protocols for extractors; plug HyperMem's in or write new ones (chosen) | Hermetic testing; extractors swappable; LLM/HTTP fully decoupled from HG data model | HyperMem's extractors are not pre-wrapped; someone must wire them behind the Protocols if they are to be used; requires understanding HyperMem's LLM client shape                   |

**Decision: (b).** Tests in `hetrep/hg/` use fake extractors that return scripted facts/topics; when HetRep ships, a separate `hetrep/hg/hypermem_adapter.py` (or integration docs) can show how to wire HyperMem's LLM-based extractors behind the Protocol if a caller chooses to use HyperMem's approach. That adapter lives in HetRep's repo, not entangled with HyperMem's pipeline structure.

## Consequences

### Positive

- **Minimal dependency on HyperMem.** Only structure.py + `build_hypergraph()`, both generic.
- **Testable in isolation.** hetrep/hg/ can inject fakes; no LLM required for unit tests.
- **Flexible extraction.** Different extractors can be swapped (HyperMem's, custom, or no-op for ablation).
- **Extensible.** MemOCR or other extractors can implement the same Protocols.

### Negative

- **Extraction not yet implemented in hetrep.** Phase 2 code must write the injection wiring (not trivial; HyperMem's extractors are complex).
- **HyperMem pipeline unused.** Stages 1–6, LoCoMo glue, HTTP servers are out of scope; they remain in HyperMem for reference but hetrep starts from scratch.

### Risks

- **Extractor Protocol mismatch.** If HyperMem's extractors evolve (e.g., signature change), the Protocol must be updated. Mitigation: Protocol is versioned in hetrep/hg/; HyperMem adapter goes in HyperMem repo (ADR 0006).
- **Build order dependency.** `build_hypergraph()` expects certain fields in Episode; if extractors don't populate them, build fails silently. Mitigation: assertions in hetrep/hg/encoder.py.
- **Three independent segmentation methods.** Documentation should be explicit that these are parallel implementations, each available for standalone use.

## Related

- **ADR 0003.** Dependency injection pattern (EntityExtractor in fluxmem).
- **ADR 0004.** HyperMem integration and reference implementation.
- **ADR 0005.** HETREP architecture; HG is Phase 2; segmentation/encoding boundary.
- **ADR 0006.** HyperMem becomes a submodule; structure.py is importable.
- **ADR 0011.** MemOCR segmentation and training architecture.
