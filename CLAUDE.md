# C-AIMMS

Cognitive AI Memory Architecture. Research prototype for memory-augmented LLM agents, on a
publication track (COLM). Multi-person team; workflow is code → review → merge.

This file is deliberately short. It says what the project _is_ and where to go next — depth lives
in `.claude/rules/` (always-on standards), `docs/workflows/` (read on demand), and `docs/adr/`
(decisions). Keep it under ~150 lines: it loads in full, every session.

## Layout

| Path                    | What                                                  |
| ----------------------- | ----------------------------------------------------- |
| `fluxmem/`              | Core method — flat STIM/MTEM/LTSM encoding pipeline   |
| `hetrep/`               | Heterogeneous encoding (VS/HG/VC) upstream of fluxmem |
| `HyperMem/` (submodule) | Hypergraph memory extraction (ADR 0004 fixes)         |
| `MemOCR/` (submodule)   | Visual episode extraction and rendering               |
| `EM-LLM/` (submodule)   | Surprise-based episode segmentation                   |
| `docs/`                 | ADRs, diagrams, workflows, analysis                   |
| `.claude/`              | Team config: rules, commands, agents, skills, hooks   |

## Stack

Python 3.11.15 (caimms env, canonical per ADR 0009) · torch >=2.7,<2.9 · sentence-transformers ·
faiss · spacy · scikit-learn · pydantic 2 · pytest · ruff.
Memory system = segmentation (EM-LLM) → encoding (HetRep: VS/HG/VC) → storage (fluxmem STIM/MTEM/LTSM).
Components are composable; Phase 1 (VS) needs no LLM; Phases 2–3 (HG/VC) need serving infrastructure.

**Setup:**

```bash
git clone --recursive https://github.com/AshwinKM103/C-AIMMS.git
cd C-AIMMS
conda activate caimms  # or: conda create -n caimms python=3.11.15 && conda activate caimms
pip install -e .
```

- Canonical env: `caimms` (Python 3.11.15); required for fluxmem + hetrep imports
- Local model paths: `.env` (gitignored): `EMBEDDING_MODEL_PATH` (all-MiniLM-L6-v2), etc.
- **Large artifacts go under `/mnt/ssd/users/durgesh/`, never `/home`** (small disk, near-full)
- NLTK data: `/mnt/ssd/users/durgesh/nltk_data/` (pre-fetched in setup)

## Storage invariants

Full rules: `.claude/rules/storage-invariants.md`.

Never commit `*.pkl`; pickle artifacts are bound to the exact torch/transformers build that wrote them.
Large artifacts: `/mnt/ssd/users/durgesh/`, never `/home` (small disk).

## Workflows

| Task                | Reach for                                                    |
| ------------------- | ------------------------------------------------------------ |
| New experiment      | `docs/workflows/new-experiment.md` · `experiment-discipline` |
| Compare runs        | `benchmarking-specialist` agent                              |
| Debug metric change | `docs/workflows/debug-a-metric.md` → `/logic-review`         |
| Unfamiliar code     | CodeGraph or serena symbol tools                             |
| Repo-wide rename    | serena (LSP-accurate)                                        |
| Design decision     | write an ADR (`/adr`)                                        |
| Literature review   | `academic-researcher` agent                                  |
| PR review           | `/pr-review` · `/logic-review`                               |
| End session         | `/checkpoint`                                                |

Rules in `.claude/rules/` are always active. Decisions go in `docs/adr/`.

## Installed elsewhere (user scope, not this repo)

`claude-mem` (cross-session memory) · `headroom` (token compression proxy) · `task-observer`
(skill-opportunity capture) · `claude-code-setup` · `omniroute` (MCP router) · CodeGraph CLI.
`pro-workflow` (user-scope; self-correction memory, persistent wikis, `/wrap-up`, `/develop`, `/doctor`).
See `docs/workflows/tooling.md` for their operational detail. Adopted externals and their install
commands: `setup/install-external.sh`.

## Conventions

- Conventional commits, enforced by a hook. `exp:` is a valid type for experiment-only changes.
  Do not add `Co-Authored-By` trailers; commits are attributed via git user config.
- Decisions become ADRs (`/adr`) — Nygard format, sequential, supersede rather than edit.
- Diagrams are Mermaid in `docs/diagrams/`, so they render and diff in PRs.
- Report what you verified and what you assumed (`.claude/rules/evidence-discipline.md`).
