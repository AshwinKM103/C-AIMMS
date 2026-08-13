---
paths:
  - "**/retriever/**"
  - "**/memory/**"
  - "**/memory_toolkits/**"
  - "**/experiments/**"
  - "**/configs/**"
---

# Storage Invariants

FluxMem — the sole active subsystem — has **no vector retriever**; retrieval (ITERRET) is an
explicit non-goal, see `fluxmem/ltsm.py`. `networkx` backs graph memory where used.

## Pickle

- **Never commit `*.pkl`** — gitignored, and it stays that way.
- Pickle artifacts are **coupled to the exact `torch`/`transformers`/numpy build** that wrote them.
  A teammate on a different environment cannot load yours. If an artifact needs to cross machines,
  serialize to a documented format (JSON/Parquet/safetensors) instead.
- Never `pickle.load` a file you did not produce.
