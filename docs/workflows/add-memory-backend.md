# Workflow: Add or modify a memory backend

FluxMem is the only subsystem in `LightMem/` (`docs/adr/0001-isolate-fluxmem.md`).
lightmem-core's factory/registration pattern (`factory/retriever/*/factory.py`) was deleted
with it — FluxMem has no equivalent factory. Extension is by subclassing one of three
abstract base classes in `src/fluxmem/interfaces/`, not by registering into a factory.

## 1. Pick the interface you're extending

| Interface | Abstract base | Default implementation | Extension point |
|---|---|---|---|
| Vector store | `BaseVectorStore` (`interfaces/vectorstore.py`) | `FAISSVectorStore` | `add`, `search`, `delete`, `get_vector` |
| LLM | `BaseLLM` (`interfaces/llm.py`) | `OpenAILLM` | 6 abstract methods — read `llm.py` before implementing |
| Embedder | `BaseEmbedder` (`interfaces/embedder.py`) | `OpenAIEmbedder` | 3 abstract methods — read `embedder.py` before implementing |

FluxMem's retrieval fusion (`retrieval/semantic_retriever.py`, `dense_weight` /
`bm25_weight`) is not itself pluggable — it's a concrete class that consumes whatever
`BaseVectorStore` you construct it with.

## 2. Match the abstract surface exactly

Every `@abstractmethod` on the base class must be implemented with the same signature. A
partial implementation fails at instantiation (Python enforces this for `ABC` subclasses),
which is stricter than lightmem-core's factory pattern was — no silent partial registration
is possible here.

## 3. No destructive defaults

FluxMem has no equivalent of the old Qdrant `rmtree`-on-construct footgun (verified: no
`shutil.rmtree` anywhere in `src/fluxmem/`). Keep it that way — if a new backend has a
destructive path, make it explicit and opt-in, never a side effect of construction.

## 4. Verify with serena, not grep

```
find-referencing-symbols   # serena, LSP-accurate — who constructs this base class today
codegraph impact <symbol>  # what a change reaches
```

## 5. Test and document

- FluxMem has no test suite today (a pre-existing gap, not created by this workflow) — add
  unit tests for your new implementation regardless.
- Record the choice as an ADR if it changes architecture: `/adr`.
- Update `.claude/rules/storage-invariants.md` if the new backend has its own defaults worth
  recording in an experiment record.

## Related
`vector-database-engineer`, `mcp-developer` agents · `/adr`
