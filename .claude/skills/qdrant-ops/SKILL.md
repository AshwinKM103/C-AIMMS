---
name: qdrant-ops
description: Operating the Qdrant vector store in LightMem — collection lifecycle, payload filters, on-disk vs in-memory, snapshots before destructive re-runs, and the rmtree footgun. Use when touching src/lightmem/factory/retriever/**, configuring a memory store, debugging retrieval quality, or setting up an experiment that persists vectors.
---

# Qdrant Operations (LightMem)

The wrapper is `src/lightmem/factory/retriever/embeddingretriever/qdrant.py`; its config is
`src/lightmem/configs/retriever/embeddingretriever/qdrant.py`. Nothing in the toolkit's database
skills applies here — this is a vector store, not a relational one.

## Connection modes

`QdrantConfig` validates that you supply exactly one of three:

| Mode | Fields | When |
|---|---|---|
| Embedded (local) | `path` | default; single-process experiments |
| Server | `host` + `port` | shared store across runs/machines |
| Cloud | `url` + `api_key` | not used here; `api_key` must come from env, never a literal |

Defaults: `collection_name="lightmem"`, `embedding_model_dims=1024`, `path="/tmp/qdrant"`,
`on_disk=False`.

## The two things that delete your data

**1. `on_disk=False` + existing `path` → `shutil.rmtree(path)`** on construction. Read that again:
constructing the client wipes the directory. `on_disk` defaults to `False`.

**2. `reset()` = `delete_col()` + `create_col()`.** No confirmation, no backup.

Before any re-run that reuses a path:

```bash
cp -r /mnt/ssd/users/durgesh/qdrant/<run-id> /mnt/ssd/users/durgesh/qdrant/<run-id>.bak
```

Better: key the path by run id so two runs never share one directory.

## Collection lifecycle

```python
store.create_col(vector_size=1024, on_disk=True, distance=Distance.COSINE)
store.insert(vectors=[...], payloads=[{...}], ids=[...])   # ids optional -> generated
store.col_info(); store.list_cols()
store.delete_col()                                          # destructive
```

`create_col` takes `vector_size` explicitly — it must match `embedding_model_dims` and the actual
output width of the embedding model in `.env` (`EMBEDDING_MODEL_PATH`). **Dims vary by experiment**:
the config default is 1024, but `experiments/locomo/retrievers.py` passes 384. Read the harness,
do not assume. A mismatch surfaces as a
Qdrant dimension error at insert, or worse, as silently poor recall if you changed models without
recreating the collection.

## Filters

`_create_filter` accepts a flat dict and builds `Filter(must=[...])`:

```python
filters = {"session_id": "abc"}                  # -> FieldCondition(match=MatchValue)
filters = {"turn": {"gte": 10, "lte": 20}}       # -> FieldCondition(range=Range)
```

Only `must` (AND) is constructed, only `MatchValue` and `Range` are supported. `MatchAny` is
imported but unused — if you need OR-semantics or `should`, you are extending `_create_filter`,
not passing a cleverer dict.

Filterable fields must exist in the payload you inserted. Payload keys are your schema; write them
down in the experiment record.

## Retrieval surface

`search(query_vector, limit, filters, return_full)` — `return_full=True` gives id/score/payload/vector,
otherwise a reduced form. Also: `get`, `list`, `scroll` (paginated, `with_payload`), `update`
(payload-only, vector-only, or both), `delete`, `exists`.

## Checklist before an experiment run

- [ ] `path` under `/mnt/ssd/users/durgesh/`, **never** `/tmp/qdrant`, never under `/home`
- [ ] `on_disk=True` if the store must survive
- [ ] path unique per run, or previous store backed up
- [ ] `embedding_model_dims` matches the model actually loaded (1024 default, 384 in locomo)
- [ ] payload keys and filter fields recorded
- [ ] `collection_name`, dims, distance metric, embedding model path in the experiment record

See `.claude/rules/storage-invariants.md` for the enforced rules and
`.claude/skills/experiment-discipline/SKILL.md` for what a run record must contain.
