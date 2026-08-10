# Workflow: Add or modify a memory backend

## 1. Know which subsystem you are extending

| Subsystem | Dense | Sparse | Notes |
|---|---|---|---|
| `lightmem` core | Qdrant | none (stub) | the factory pattern lives here |
| `fluxmem` | FAISS | real BM25 fusion | `bm25_weight` / `dense_weight` |
| `em2mem` | VLM2Vec embeddings | — | multimodal + semantic graph |
| `baselines/**` | vendored mem0 / langmem / agentic_memory | — | **comparison points, not production** |

Extending a baseline changes a published comparison. Extending the core changes the system.
These are different acts with different review bars.

## 2. Follow the factory pattern
Core retrievers register through `src/lightmem/factory/retriever/…/factory.py`, with a matching
pydantic config under `src/lightmem/configs/retriever/…`. Read `qdrant.py` and its config as the
reference implementation before writing a new one — including how it validates that exactly one
connection mode is supplied.

## 3. Match the existing surface
A retriever is expected to provide: `create_col`, `insert`, `search`, `get`, `list`, `scroll`,
`update`, `delete`, `delete_col`, `col_info`, `list_cols`, `reset`, `exists`. Partial
implementations break `/compare` runs in non-obvious ways.

## 4. Do not repeat the rmtree footgun
If your backend has a destructive path, make it explicit and opt-in. Do not delete user data as a
side effect of construction.

## 5. Verify with serena, not grep
```
rename / find-referencing-symbols   # serena, LSP-accurate
codegraph impact <symbol>           # what a change reaches
```

## 6. Test and document
- Unit tests for the retriever surface; property-based tests for retrieval invariants
  (`testing-strategies` skill).
- Record the choice as an ADR if it changes architecture: `/adr`.
- Update `.claude/rules/storage-invariants.md` if you added a new store with its own defaults.

## Related
`qdrant-ops` skill · `vector-database-engineer`, `mcp-developer` agents · `/adr`
