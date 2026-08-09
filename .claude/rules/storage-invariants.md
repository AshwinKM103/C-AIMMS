---
description: Qdrant, pickle, and experiment-artifact invariants for LightMem
globs: ["**/retriever/**", "**/memory/**", "**/memory_toolkits/**", "**/experiments/**", "**/configs/**"]
---

# Storage Invariants

LightMem's memory store is **Qdrant** (`qdrant-client==1.15.1`); `networkx` backs graph memory.
These rules exist because two defaults in the current code destroy data silently.

**Know which subsystem you are in — the retrieval stacks differ:**

| Subsystem | Dense | Sparse / hybrid |
|---|---|---|
| `lightmem` core | Qdrant (`factory/retriever/embeddingretriever/qdrant.py`) | **none** — `factory/retriever/contextretriever/bm25.py` is an empty stub (1 byte) |
| `fluxmem` | FAISS (`faiss-cpu`, optional extra) | **real** weighted dense+BM25 fusion in `retrieval/semantic_retriever.py` (`bm25_weight`, default 0.5) |
| `mem0` / `agentic_memory` baselines | vendored adapters | vendored `BM25Okapi` usage |

Do not record "BM25 hybrid enabled" for a lightmem-core run — that path does not exist yet.
Note the trap: `src/lightmem/configs/retriever/bm25.py` *is* populated (`on_disk`, `use_stemming`,
`lowercase`), so the configuration surface looks live while the retriever behind it is empty.

## Qdrant

**1. Never leave `QdrantConfig.path` at its default.**
The default is `/tmp/qdrant` — wiped on reboot, and on the wrong disk. Every store that matters
goes under `/mnt/ssd/users/durgesh/`. `/home` is small and near-full; large artifacts do not go there.

**2. Set `on_disk=True` for any store you intend to keep.**
`Qdrant.__init__` (`src/lightmem/factory/retriever/embeddingretriever/qdrant.py`) does this:

```python
if not params:
    params["path"] = config.path
    if not config.on_disk:
        if os.path.exists(config.path) and os.path.isdir(config.path):
            shutil.rmtree(config.path)      # <- silent, unrecoverable
```

`on_disk` defaults to `False`. So re-running against an existing path **deletes the previous
store before writing**. If you are re-running an experiment and expect to compare against the
previous index, you have already lost it.

**3. This has already bitten this project.** `memory_toolkits/memories/layers/naive_rag.py:147`
carries the comment: *"Key: enable Qdrant on_disk; otherwise 'file created but points=0' occurs
often"*. The LoCoMo harness (`experiments/locomo/`) sets `on_disk=True` explicitly for the same
reason. Follow that precedent everywhere.

**4. Snapshot before any destructive re-run.** Copy the store directory, or point the new run at
a fresh path keyed by run id. Never re-point two runs at one path.

**5. Record the retrieval config in the experiment record**, not just in code:
`collection_name`, `embedding_model_dims` **as actually used** — the config default is 1024, but
`experiments/locomo/retrievers.py` uses 384; do not assume — the embedding model path, and the
distance metric (`Distance.COSINE` unless `create_col` was called with something else). For a
FluxMem run also record `bm25_weight` / `dense_weight`. A metric is not reproducible without them.

## Pickle

Several memory layers persist via `pickle` (`memories/layers/{amem,full_context,memzero,langmem}.py`,
`em2mem/memory/visual/Memory.py`).

- **Never commit `*.pkl`** — gitignored, and it stays that way.
- Pickle artifacts are **coupled to the exact `torch`/`transformers`/numpy build** that wrote them.
  A teammate on a different environment cannot load yours. If an artifact needs to cross machines,
  serialize to a documented format (JSON/Parquet/safetensors) instead.
- Never `pickle.load` a file you did not produce.

## Baselines are not the active path

`src/lightmem/memory_toolkits/memories/layers/baselines/**` vendors mem0, langmem, and
agentic_memory — including ~20 vector-store adapters (pgvector, redis, milvus, weaviate,
elasticsearch, …) and a SQLite `SQLiteManager`. **None of these are the active retriever.**

- Do not "optimize" or refactor a baseline adapter thinking it is production code.
- Do not add a relational database to this project on the strength of those files existing.
- Changes to baselines only matter when they change a reported baseline number — and then they
  need to be called out explicitly in the experiment record, because they alter a comparison point.
