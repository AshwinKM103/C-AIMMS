---
paths:
  - "**/retriever/**"
  - "**/memory/**"
  - "**/memory_toolkits/**"
  - "**/experiments/**"
  - "**/configs/**"
---

# Storage Invariants

FluxMem is the only subsystem in `LightMem/` (see `docs/adr/0001-isolate-fluxmem.md` —
lightmem-core, EM²Mem, and StructMem, along with their Qdrant store and pickle-based
baseline layers, were deleted). FluxMem's store is **FAISS** (`faiss-cpu`, optional
extra), with a real weighted dense+BM25 fusion in `retrieval/semantic_retriever.py`.

## FAISS + BM25 retrieval

FluxMem's `semantic_retriever.py` combines dense FAISS similarity with a custom BM25
(TF-IDF, `k1=1.5`, `b=0.75`) via `dense_weight` / `bm25_weight` (default `1.0` / `0.5`).

- **Record `bm25_weight` and `dense_weight` in the experiment record**, not just in code.
  A retrieval metric is not reproducible without them.
- FluxMem has no pickle-based persistence (verified: no `pickle`/`.pkl` references in
  `src/fluxmem/`) — no cross-machine artifact-coupling risk here.
