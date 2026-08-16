# C-AIMMS

**Cognitive AI Memory Architecture** — a research prototype for memory-augmented LLM agents, on a
publication track (COLM). This repository implements a two-stage memory system: heterogeneous
**encoding** (`hetrep/`) feeding a flat **storage** pipeline (`fluxmem/`), plus reference
submodules used to port and validate individual components.

This is a multi-person research codebase, not a packaged product. Workflow is code → review →
merge; see [Contributing](#contributing) before opening a PR.

## Table of contents

- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Verifying the install](#verifying-the-install)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Development phases](#development-phases)
- [Storage invariants](#storage-invariants)
- [Documentation map](#documentation-map)
- [Contributing](#contributing)
- [License](#license)

## Architecture

```
Episode text ──▶ HetRep (encoding)  ──▶  FluxMem (storage)
                 VS → HG → VC             STIM → MTEM → LTSM
                 (fills EpisodicUnit      (three-tier hierarchy,
                  fields: embedding,       format-adaptive selector,
                  hyperedge_density,       BMM-gated fusion)
                  visual_salience)
```

- **`hetrep/`** — Heterogeneous encoding. Produces `(H_j, I_j, v_j)` per episode across three
  formats: **VS** (Vector Store, dense embeddings), **HG** (Hypergraph, node/edge structure), **VC**
  (Visual Canvas, rendered layout salience). Components are composable and phased by
  infrastructure cost: Phase 1 (VS) needs no LLM; Phases 2–3 (HG/VC) need serving infrastructure.
- **`fluxmem/`** — ADASTORE: a flat three-tier memory hierarchy (STIM/MTEM/LTSM), a
  format-adaptive selector (trained MLP), offline supervision for reward labeling, and BMM-gated
  fusion for merge-vs-new decisions. Implements COLM §1.3.2–1.3.3. Does **not** implement HETREP,
  ITERRET, or WORKMEM — those are separate concerns (see submodules below).

Design rationale for the encoding/storage split and the interface between the two packages lives
in [`docs/adr/0005-hetrep-sibling-package-and-episode-encoder-seam.md`](docs/adr/0005-hetrep-sibling-package-and-episode-encoder-seam.md).

## Repository layout

| Path                    | What                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| `fluxmem/`              | Core method — flat STIM/MTEM/LTSM encoding pipeline (installable package)                  |
| `hetrep/`               | Heterogeneous encoding (VS/HG/VC), upstream of fluxmem (installable package)               |
| `configs/default.yaml`  | Default `AdaStoreConfig` — tier capacities, utility weights, promotion thresholds          |
| `scripts/`              | Thin CLIs that wire `fluxmem/` pieces together (write-phase smoke test, selector training) |
| `tests/`                | pytest suite for `fluxmem/` and `hetrep/`                                                  |
| `docs/adr/`             | Architecture Decision Records (Nygard format, sequential)                                  |
| `docs/workflows/`       | Task-oriented guides (new experiment, debug a metric, review & merge)                      |
| `docs/diagrams/`        | Mermaid diagrams                                                                           |
| `HyperMem/` (submodule) | Reference for the HG arm's data model and `build_hypergraph()` (ADR 0007)                  |
| `MemOCR/` (submodule)   | Reference for the VC arm's drafting prompt (ADR 0005 §3.2, Phase 3)                        |
| `EM-LLM/` (submodule)   | Surprise-based episode segmentation, upstream of the `EpisodeEncoder` seam                 |
| `IterRet/` (submodule)  | Reference for hybrid ANN+BM25 retrieval (ITERRET), design target for a future ADR          |
| `.claude/`              | Team config: rules, commands, agents, skills, hooks (gitignored; not shipped)              |

Submodules are vendored reference material, not integrated dependencies — each is at a different
maturity level and is pulled in for a specific, narrow purpose documented in
[`docs/adr/0006-vendored-components-become-git-submodules.md`](docs/adr/0006-vendored-components-become-git-submodules.md).
Do not assume code in a submodule is exercised by `fluxmem/` or `hetrep/` unless an ADR says so.

## Requirements

- Python 3.11.15 in the canonical `caimms` conda environment (ADR 0009). `pyproject.toml` declares
  `requires-python = ">=3.10,<3.12"`.
- `torch>=2.7,<2.9`, `sentence-transformers`, `faiss-cpu`, `spacy` (+ `en_core_web_sm`),
  `scikit-learn`, `pydantic 2`, `PyYAML`, `scipy`, `numpy` — see `pyproject.toml` for exact ranges.
- `pytest`, `ruff` for development (`[project.optional-dependencies].dev`).
- Git with submodule support (`--recursive` clone, or `git submodule update --init --recursive`).

## Installation

```bash
git clone --recursive https://github.com/AshwinKM103/C-AIMMS.git
cd C-AIMMS
conda activate caimms  # or: conda create -n caimms python=3.11.15 && conda activate caimms
pip install -e .
```

- `pip install -e .` installs both `fluxmem` and `hetrep` (see `[tool.setuptools.packages.find]`
  in `pyproject.toml`) in editable mode, so `scripts/*.py` and `import fluxmem` / `import hetrep`
  resolve without needing `PYTHONPATH` tricks.
- Local model paths (e.g. `EMBEDDING_MODEL_PATH` for `all-MiniLM-L6-v2`) go in a gitignored `.env`
  at the repo root. There is no `.env.example` at the root yet — check the submodules
  (`HyperMem/.env.example`, `MemOCR/.env.example`) for the variable shapes they expect.
- spaCy's default extractor (`fluxmem/features.py`) needs the `en_core_web_sm` model:
  `python -m spacy download en_core_web_sm`.
- NLTK data (`punkt`, `stopwords`, `wordnet`, used by HyperMem's HG stage) should be pre-fetched to
  `/mnt/ssd/users/durgesh/nltk_data/`, not the default `~/nltk_data` — see
  [Storage invariants](#storage-invariants).

## Verifying the install

```bash
python -c "import fluxmem, hetrep; print('ok')"
pytest tests/ -q
```

Expected: `ok`, then all tests passing (257 tests as of this writing, a few seconds on CPU).

## Usage

The scripts in `scripts/` are thin CLIs — all logic lives in `fluxmem/`; each script only wires
config → data → pipeline. Run these after `pip install -e .`.

**Write-phase smoke test** — pushes synthetic turns through STIM → MTEM → feature extraction →
selector → BMM fusion → LTSM promotion, asserting tier counts at each stage:

```bash
python scripts/run_write_phase.py
```

```
STIM: 4 turns buffered, 2 evicted
MTEM: 2 episodes
features: extracted 2 7-dim vectors
selector: assigned format VS
fusion: decision=MERGE target=episode-from-episode-1-turn-0
LTSM: promoted 2 episode(s), store size=2
smoke test OK
```

**Train the format selector** against synthetic (`StubEpisodeProducer`) data — there is no live
generation/retrieval pipeline yet (ITERRET/WORKMEM are non-goals of this package), so `--dry-run`
is the only supported mode:

```bash
python scripts/train_selector.py --config configs/default.yaml --dry-run --n-episodes 40
```

```
episodes=40 train=32 val=8
final_train_loss=0.0002 stopped_epoch=200
```

## Configuration

`fluxmem.config.AdaStoreConfig` is loaded from YAML; `configs/default.yaml` is the reference
config. The load-bearing sections:

| Section           | Key fields                                                             | Notes                                                                                                  |
| ----------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `stim`            | `capacity`                                                             | Short-term interaction memory buffer size                                                              |
| `mtem`            | `capacity`, `reference_length`, `half_life`                            | Mid-term episodic memory eviction/decay                                                                |
| `utility_weights` | `w1` (access freq), `w2` (interaction intensity), `w3` (recency decay) | Genuinely load-bearing, placeholder values — not paper-derived; record alongside any experiment result |
| `promotion`       | `tau_u`, `tau_r`, `max_age`                                            | LTSM promotion thresholds                                                                              |
| `fusion`          | `tau`, `m_min`, `em_iters`, `eps`                                      | BMM-gated merge-vs-new decision                                                                        |
| `selector`        | `hidden_dim`, `learning_rate`, `max_epochs`                            | Format-selector MLP hyperparameters                                                                    |

Note: there is intentionally no `tau_c` field — the FluxMem Eq. 6 confidence gate is not
implemented (see `fluxmem/ltsm.py` module docstring). Do not add one speculatively.

## Testing

```bash
pytest tests/ -q                       # full suite
pytest tests/ -q -m "not slow"         # skip tests that exercise real spaCy models
```

Tests live under `tests/`, mirroring `fluxmem/` and `hetrep/` module names. `tests/conftest.py`
seeds `torch` deterministically for every test (autouse fixture). `tests/test_package.py` guards
against a shadowing `fluxmem` install resolving instead of the repo-root package — if that test
fails, check `pip show fluxmem` for a stray install ahead of the editable one on `sys.path`.

## Development phases

| Phase | Encoding arm | Needs                            | Status                        |
| ----- | ------------ | -------------------------------- | ----------------------------- |
| 1     | VS           | No LLM/servers                   | Implemented (`hetrep/vs/`)    |
| 2     | HG           | 1-server (embedding/LLM serving) | In progress (`hetrep/hg/`)    |
| 3     | VC           | 3-server                         | Stubbed (`hetrep/vc/stub.py`) |

Phases are sequenced by infrastructure cost (0-server → 1-server → 3-server), not by dependency —
components are composable and each arm can be developed independently.

## Storage invariants

- Never commit `*.pkl` — pickle artifacts are bound to the exact torch/transformers build that
  wrote them (`.gitignore` also blocks `*.pt`, `*.bin`, `*.safetensors`).
- Large artifacts (models, indices, run outputs) go under `/mnt/ssd/users/durgesh/`, never
  `/home` — the home disk is small and near-full.
- Full rules: [`.claude/rules/storage-invariants.md`](.claude/rules/storage-invariants.md) (loaded
  automatically in Claude Code sessions; read directly if you're not using it).

## Documentation map

| Need                           | Where                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| Why a decision was made        | `docs/adr/` — sequential, Nygard format, supersede rather than edit                  |
| Full component/tooling catalog | `docs/component-index.md`                                                            |
| Task-oriented how-tos          | `docs/workflows/` (new experiment, debug a metric, review & merge, literature sweep) |
| Diagrams                       | `docs/diagrams/` (Mermaid, renders and diffs in PRs)                                 |
| Paper-fidelity audits          | `docs/FluxMem-Fidelity-Audit-2026-08-13.md`, `docs/MemOCR-Paper-Alignment.md`        |

## Contributing

- Workflow is **code → review → merge**; see
  [`docs/workflows/review-and-merge.md`](docs/workflows/review-and-merge.md).
- Commits follow Conventional Commits (`type(scope): subject`); `exp:` is a valid type for
  experiment-only changes. This is hook-enforced in Claude Code sessions.
- Significant decisions become ADRs, not buried in a PR description.
- Run `pytest tests/ -q` and `ruff check` before opening a PR.
- For research code specifically: if a change is wrong, would any test fail? If not, that gap is
  the review comment, not the change itself.

## License

No license file is present in this repository. It is an internal, multi-person research
prototype; do not assume permission to redistribute or reuse outside the project team without
checking with the maintainers.
