# C-AIMMS

Cognitive AI Memory Architecture. Research prototype for memory-augmented LLM agents, on a
publication track (COLM). Multi-person team; workflow is code → review → merge.

Depth lives in `.claude/rules/` (always-on standards), `docs/workflows/` (read on demand),
`docs/adr/` (decisions), and `docs/component-index.md` (every installed command/agent/skill/hook/
plugin/MCP server).

## Layout

| Path                           | What                                                                                           |
| ------------------------------ | ---------------------------------------------------------------------------------------------- |
| `LightMem/`                    | The research codebase. **Its own git repo** (fork of zjunlp/LightMem) — commit there, not here |
| `docs/`                        | ADRs, diagrams, workflows, analysis                                                            |
| `.claude/`                     | Team config: rules, commands, agents, skills, hooks                                            |
| `awesome-claude-code-toolkit/` | Vendor clone, source of copied components. Not tracked                                         |

## Stack

Python 3.10–3.11 · torch 2.8 · transformers 4.57 · sentence-transformers · **Qdrant** (vector store)
· networkx (graph memory) · llmlingua (compression) · openai / ollama / vllm backends · pydantic 2
· pytest · ruff. Three subsystems: `lightmem` (core), `fluxmem`, `em2mem`.

- Conda env: `conda activate /mnt/ssd/users/durgesh/conda-envs/lightmem`
- Local model paths live in `LightMem/.env` (gitignored): `LLMLINGUA_MODEL_PATH`,
  `EMBEDDING_MODEL_PATH`, `QWEN3_4B_INSTRUCT_PATH`, `LLAMA31_8B_INSTRUCT_PATH`
- **Large artifacts go under `/mnt/ssd/users/durgesh/`, never `/home`** (small disk, near-full)

## Storage invariants (the short version)

Full rules: `.claude/rules/storage-invariants.md`. The two that bite:

1. `QdrantConfig.path` defaults to `/tmp/qdrant`, and constructing the client with `on_disk=False`
   (**the default**) on an existing path calls `shutil.rmtree` on it. Re-running an experiment
   silently destroys the previous store.
2. BM25 hybrid exists in **FluxMem only**. `lightmem/factory/retriever/contextretriever/bm25.py`
   is an empty stub — do not record "BM25 enabled" for a lightmem-core run.

Never commit `*.pkl`; pickle artifacts are bound to the exact torch/transformers build that wrote them.

## Routing — where to go for what

| When you are…                    | Reach for                                                                                |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| starting a new experiment        | `docs/workflows/new-experiment.md` · `experiment-discipline` skill · `/track`            |
| comparing runs                   | `/compare` · `/evaluate-model` · `benchmarking-specialist` agent                         |
| a metric moved unexpectedly      | `docs/workflows/debug-a-metric.md` → `/bisect`, then `/logic-review`                     |
| slow code or a memory blowup     | `/profile`, `/profile-memory`, `/optimize` · `performance-engineer` agent                |
| a confusing bug with no test yet | `/debug`, `/trace` · `error-detective` agent                                             |
| touching Qdrant / persistence    | `qdrant-ops` skill · `.claude/rules/storage-invariants.md`                               |
| adding a memory backend          | `docs/workflows/add-memory-backend.md` · `vector-database-engineer` agent                |
| navigating unfamiliar code       | `codegraph explore <query>` or serena's symbol tools                                     |
| renaming a symbol repo-wide      | serena (LSP-accurate) — not grep-and-replace, not `/rename`                              |
| choosing a library               | `/compare-tools`, `/evaluate` → record the outcome with `/adr`                           |
| looking for prior work           | `docs/workflows/literature-sweep.md` · `/deep-dive` · AutoSearch · `academic-researcher` |
| reviewing / merging              | `docs/workflows/review-and-merge.md` · `/pr-review` · `/logic-review`                    |
| writing Python                   | `python-best-practices` skill · `python-engineer` agent                                  |
| prompts / RAG / retrieval design | `llm-integration`, `prompt-engineering` skills · `llm-architect` agent                   |
| ending a session                 | `/checkpoint` — claude-mem captures the rest automatically                               |
| anything not above               | `docs/component-index.md` — full catalog, all commands/agents/skills/hooks/plugins/MCP   |

Rules in `.claude/rules/` are always active — no need to invoke them.
`storage-invariants.md` is path-scoped to retriever/memory/config/experiment paths.

## Do not reach for these

Each was deliberately excluded. Re-adding them creates duplicate systems:

| Don't                                                                                                                             | Because                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `explore` plugin                                                                                                                  | CodeGraph is already indexed here (`codegraph explore`), and serena does symbols properly             |
| `claude-memory-kit`, `memory-bank` skills                                                                                         | claude-mem (installed, user scope) already owns cross-session memory                                  |
| `continuous-learning` skill                                                                                                       | task-observer (installed) already does pattern capture                                                |
| toolkit session/memory hooks (`session-start`, `context-loader`, `learning-log`, `pre-compact`, `session-end`, `suggest-compact`) | claude-mem + headroom already hold those events; adding these means two systems writing session state |
| `post-edit-check.js` hook                                                                                                         | duplicates `lint-fix.js` on the same write — stale diagnostics at 2× latency                          |
| `auto-test.js` hook                                                                                                               | would run pytest on save; `experiments/**` tests load 4B/8B models                                    |
| Postgres/Redis/ERD skills and agents                                                                                              | every relational path in the repo is vendored mem0 **baseline** code, not the active store            |

## Installed elsewhere (user scope, not this repo)

`claude-mem` (cross-session memory) · `headroom` (token compression proxy) · `task-observer`
(skill-opportunity capture) · `claude-code-setup` · `omniroute` (MCP router) · CodeGraph CLI.
See `LightMem/CLAUDE.md` for their operational detail. Adopted externals and their install
commands: `setup/install-external.sh`.

## Conventions

- Conventional commits, enforced by a hook. `exp:` is a valid type for experiment-only changes.
- Decisions become ADRs (`/adr`) — Nygard format, sequential, supersede rather than edit.
- Diagrams are Mermaid in `docs/diagrams/`, so they render and diff in PRs.
- Report what you verified and what you assumed (`.claude/rules/evidence-discipline.md`).
