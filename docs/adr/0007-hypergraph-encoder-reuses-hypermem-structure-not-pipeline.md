# 0007. HG encoder reuses HyperMem's data model, not its pipeline

## Status

Accepted — 2026-08-12. Sets the reuse strategy for Phase 2 of the hetrepv2 plan (`hetrep/hg/`,
tasks T-11 through T-14). No `hetrep/hg/` code lands before this ADR; `hetrep/hg/adapter.py`
implements the decision below.

## Context

HyperMem (`HyperMem/hypermem/`, Yue et al., 2025, arXiv:2604.08256 — vendored per ADR 0004 and now
a submodule per ADR 0006) is the only available implementation of a three-level hypergraph memory
structure (Topic → Episode → Fact nodes, hyperedges linking each level) matching the shape COLM's
HG format needs. It was audited module-by-module for reuse potential (hetrepv2 plan §2.6, 39 tool
calls, findings below); the audit's conclusion splits cleanly into "importable as-is" and "not
importable without dragging in machinery HetRep's HG arm does not need."

### What is directly reusable

| Module                                        | Contents                                                                 | Verified how                                                                                                                             |
| --------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `hypermem/structure.py`                       | `Hypergraph`, `FactNode`/`EpisodeNode`/`TopicNode`, both hyperedge types | `to_dict()`/`from_dict()` (`structure.py:358-375`) round-trip through plain JSON — executed directly, not inferred from reading the code |
| `hypermem/types.py`                           | `Episode`, `Fact`, `Topic` dataclasses                                   | Imports cleanly with no LLM, no network                                                                                                  |
| `hypermem/extractors/hypergraph_extractor.py` | `build_hypergraph(episodes)` — a pure function, no side effects          | Imports and runs standalone; no `sys.path.insert` dependency at call time                                                                |
| `hypermem/utils/datetime_utils.py`            | stdlib-only helpers                                                      | No external dependency                                                                                                                   |

This is a small, coherent subset: a data model plus one pure function that assembles it from a
list of already-extracted episodes. Nothing in this subset makes a network call, imports `verl`
or any GPU-touching library, or requires `sys.path.insert(0, ...)` gymnastics — the pattern every
`main/stageN_*.py` file uses (`stage1_memory_extraction.py:31`, repeated identically in stages
2–6) because HyperMem is not a pip-installable package (no `pyproject.toml`, no `setup.py`).

### What is not reusable without pulling in the whole pipeline

Four properties of HyperMem's architecture, each independently confirmed:

1. **LLM required at encode time, not just query time.** `topic_extractor.py:318,452,701,824` and
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

| Option                                                                     | Description                                                                            | Tradeoff                                                                                                                                                                         |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (a) Wrap HyperMem's extractors in an adapter that supplies the HTTP config | `hetrep/hg/` owns HyperMem's Qwen3-over-HTTP client setup internally                   | Every `hetrep/hg/` test needs two live HTTP servers or heavy mocking of them; also locks the HG arm to Qwen3, which ADR 0008 rejects on fidelity grounds independent of this ADR |
| (b) Inject `FactExtractor`/`TopicExtractor` Protocols (chosen)             | `hetrep/hg/encoder.py` is pure orchestration; caller supplies real extractors or fakes | One layer of indirection at the call site; keeps `hetrep/hg/` tests hermetic and lets extractor implementation vary independently of the encoder                                 |

**Decision: (b).** This is the same shape of tradeoff ADR 0003 resolved for MemOCR's renderer and
ADR 0002 resolved for `JudgeReward`/`MemUtilReward`: the concrete implementation that does real
work over a network is injected, never imported directly, so the package's own tests never require
live infrastructure. Concretely, HyperMem's own extractors _can_ satisfy `FactExtractor` and
`TopicExtractor` — nothing prevents wiring `hypermem.extractors.topic_extractor` in as one
concrete implementation at the call site where a real LLM endpoint is available — but `hetrep/hg/`
itself never imports it, so a unit test can supply a deterministic fake instead.

## Consequences

### Positive

- **Minimal, verified reuse surface.** Only `structure.py` and one pure function cross the
  boundary; both were executed, not merely read, during the audit that grounds this decision.
- **`hetrep/hg/` is testable with zero LLM calls and zero network dependency**, satisfying
  `.claude/rules/testing.md`'s hermeticity requirement for the default test loop.
- **Extraction is swappable per experiment.** A different LLM provider, a smaller/cheaper model,
  or a no-op extractor for ablation studies can all satisfy the same Protocols without touching
  `hetrep/hg/encoder.py` or `adapter.py`.
- **Incremental use, unlike the source pipeline.** Because `hetrep/hg/` calls
  `build_hypergraph()` per-unit rather than adopting HyperMem's per-conversation batch model, it
  matches the actual arrival pattern of encoded units from the `EpisodeEncoder` seam (ADR 0005).
- **No pickle.** Persistence goes through `Hypergraph.to_dict()` → JSON (ADR 0005 §3.3), sidestepping
  HyperMem's own pickle-based indexing and `.claude/rules/storage-invariants.md` entirely.

### Negative

- **Extraction logic is not reused, only its target data model.** `hetrep/hg/` starts from a blank
  page for fact/topic extraction quality — HyperMem's extractors embody real prompt engineering
  and multi-case decision logic (e.g., ADR 0004's G2 fix, the three-case topic-matching tree) that
  a from-scratch `TopicExtractor` implementation does not automatically inherit.
- **Two codebases now define overlapping concepts.** `hetrep/hg/adapter.py`'s `Episode` mapping
  must be kept in sync with `hypermem/types.py` if HyperMem's schema changes across submodule SHA
  bumps (ADR 0006).

### Risks

- **Protocol/implementation drift.** If a future submodule SHA bump changes HyperMem's `Episode`
  or `Fact` dataclass shape, `hetrep/hg/adapter.py` breaks silently unless a test pins the
  expected shape. Mitigation: `hetrep/tests/` includes a round-trip test against the actual
  submodule's `types.py`, not just a hand-written fixture, so a SHA bump that changes the schema
  fails a test rather than corrupting graphs silently.
- **`build_hypergraph()` has undocumented input assumptions.** It is a pure function but was
  audited for importability, not for its full input-validation contract; if it expects fields on
  `Episode` that a from-scratch `TopicExtractor`/`FactExtractor` pair does not populate, the
  failure mode could be a malformed graph rather than an exception. Mitigation: assertions in
  `hetrep/hg/encoder.py` on the fields `build_hypergraph()` is confirmed to read, checked against
  the actual function body before Phase 2 code lands, not assumed from the type signature alone.
- **Reimplementing extraction from scratch is nontrivial.** This is flagged as a negative above
  and repeated here as a risk to the Phase 2 timeline specifically: the hetrepv2 plan's own task
  list (T-12) treats extractor implementation as real, non-trivial work, not a thin wrapper.

## Related

- **ADR 0003** — established the injection-over-direct-import pattern (MemOCR's render path) that
  this decision extends to HyperMem's extractors.
- **ADR 0004** — HyperMem's Phase 1–3 gap fixes (G1–G8); those fixes live in the stages this ADR
  explicitly does not reuse, so they improve HyperMem's own retrieval baseline without affecting
  `hetrep/hg/`'s correctness.
- **ADR 0005** — `EpisodeEncoder` seam; `HgEncoder.encode` is one of its concrete implementations.
- **ADR 0006** — HyperMem's submodule conversion; `structure.py` and `build_hypergraph()` are
  imported from the pinned submodule SHA, not a separate vendored copy.
- **ADR 0008** — resolves the propagation-formula and embedding-model choices `hetrep/hg/`'s
  `propagation.py` and Phase 2's extractor implementations must follow.
