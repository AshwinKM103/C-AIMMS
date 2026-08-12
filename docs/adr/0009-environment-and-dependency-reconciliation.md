# 0009. Environment and dependency-pin reconciliation

## Status

Accepted — 2026-08-12. Gates Phase 0 task T-01 of the hetrepv2 plan. Changes `pyproject.toml`
pins and the packages installed into the `caimms` conda environment; no source code changes.

## Context

Two conda environments exist on the development machine, and neither matched `pyproject.toml`'s
declared pins at the time this ADR was opened:

|                         | `caimms` (Python 3.11.15)             | `base` (Python 3.13.11) | `pyproject.toml` (declared)                              |
| ----------------------- | ------------------------------------- | ----------------------- | -------------------------------------------------------- |
| torch                   | 2.7.1                                 | 2.12.0                  | `==2.8.0`                                                |
| numpy                   | 2.2.6                                 | 2.2.6                   | `==2.2.6`                                                |
| pydantic                | 2.13.4                                | 2.11.10                 | `==2.12.0`                                               |
| scipy                   | 1.17.1                                | 1.17.0                  | `==1.15.3`                                               |
| faiss / spacy / sklearn | **absent**                            | present                 | required by `fluxmem/features.py`, `fluxmem/selector.py` |
| vllm / ray              | 0.10.1.1 / 2.56.1                     | 0.11.0 / 2.53.0         | not declared                                             |
| `import fluxmem`        | **fails** (`No module named 'faiss'`) | succeeds                | —                                                        |

Not one pin in `pyproject.toml` matched a live environment exactly. `pyproject.toml` also declares
`requires-python = ">=3.10,<3.12"`, which **excludes `base`** (Python 3.13.11) — the only
environment where `pytest tests/ -q` was confirmed passing (155 tests, run 2026-08-12). This is
the inverse of a healthy state: the environment the project's own metadata says is supported
cannot import the package, and the environment where the package works is explicitly outside the
declared support range.

`caimms` fails to import `fluxmem` for a concrete, checkable reason: `faiss`, `spacy`, and
`scikit-learn` are absent, and `fluxmem/features.py`'s default entity extractor and
`fluxmem/selector.py`'s `StandardScaler` depend on them. This is not a version-mismatch problem
alone; it is missing packages, on top of the version mismatches.

### Why this predates HetRep but blocks it specifically

`fluxmem`'s existing 155 tests pass today only in `base`, a global, project-agnostic interpreter —
running research-specific dependency pins against the user's global Python is itself the source of
the drift described above (any other tool the user installs into `base` can silently shift
`fluxmem`'s effective dependency versions). HetRep's Phase 1 (VS) needs `sentence-transformers`;
Phase 2 (HG) needs the vllm/ray serving stack for an LLM endpoint. Neither is declared anywhere.
Until an environment both satisfies `pyproject.toml`'s `requires-python` bound and actually
imports `fluxmem`, no HetRep phase can start — this is a hard precondition, not a nice-to-have
cleanup, which is why it is Phase 0's first task (T-01) rather than deferred alongside other
technical debt.

## Decision

### `caimms` is the canonical environment

| Option                                                                  | Description                                                                         | Tradeoff                                                                                                                                                                                                                                           |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (a) Adopt `base`, relax `requires-python` to admit 3.13                 | Stop fighting the environment where tests already pass                              | `base` is the user's global interpreter, shared across every project and tool on the machine; pinning research-specific dependencies (torch, vllm, ray) into it is the documented mechanism by which the current drift happened in the first place |
| (b) Repair `caimms` (chosen)                                            | Install missing packages, relax `pyproject.toml`'s exact pins to ranges matching it | `caimms` is the only environment inside the declared `>=3.10,<3.12` bound; it already carries vllm/ray/transformers, which Phases 2–3 need for LLM/embedding serving — repair cost is one-time, install-only                                       |
| (c) Create a third, fresh environment matching `pyproject.toml` exactly | Pin `torch==2.8.0` etc. precisely, build from scratch                               | Throws away `caimms`'s existing vllm 0.10.1.1 stack, which pins torch to 2.7.1 as a transitive dependency — a fresh env matching the declared `==2.8.0` would immediately break vllm, recreating the same conflict from the other direction        |

**Decision: (b).** `caimms` (Python 3.11.15) is declared canonical. Consequence of this choice:
Phase 1 work is blocked until T-01 completes, because `import fluxmem` fails in `caimms` today —
this is stated as an explicit consequence, not glossed over, because it is a real sequencing cost
the hetrepv2 plan's task graph (T-06 depends on T-01 via T-03/T-04) already accounts for.

### `pyproject.toml` pins relax from exact versions to ranges matching `caimms`

| Package  | Prior pin  | New pin      | Reason                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------- | ---------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| torch    | `==2.8.0`  | `>=2.7,<2.9` | vllm 0.10.1.1 (already in `caimms`) pins torch to 2.7.1 as a transitive dependency; MemOCR's own `setup.py:20,44` separately requires `torch==2.8.0`, while its `requirements.txt:1` says `2.7.1` — MemOCR's pins are internally inconsistent (confirmed by direct read, not assumed), so no single exact pin satisfies MemOCR's own two files simultaneously, let alone vllm too. A range is the only pin that is not contradicted by something already in scope. |
| pydantic | `==2.12.0` | `>=2.12,<3`  | `caimms` has 2.13.4 installed; the exact pin would force a downgrade for no stated reason                                                                                                                                                                                                                                                                                                                                                                          |
| scipy    | `==1.15.3` | `>=1.15,<2`  | `caimms` has 1.17.1 installed; same reasoning as pydantic                                                                                                                                                                                                                                                                                                                                                                                                          |
| numpy    | `==2.2.6`  | unchanged    | `caimms` matches exactly already — no change needed, recorded here for completeness                                                                                                                                                                                                                                                                                                                                                                                |

`fluxmem`'s own code was checked before accepting torch's range: it uses only `nn.Linear`, `Adam`,
and `cross_entropy` (`fluxmem/selector.py`) — none of which have breaking signature changes between
2.7.1 and 2.9 — so `torch>=2.7,<2.9` does not risk correctness for the code that exists today. This
is stated because relaxing a pin without checking what the pinned package is used for is exactly
the kind of unverified assumption `.claude/rules/evidence-discipline.md` warns against.

### Missing packages, installed to `caimms`

```
faiss-cpu==1.15.0
spacy==3.8.15  +  en_core_web_sm            # fluxmem/features.py default extractor
scikit-learn==1.7.2                          # fluxmem/selector.py StandardScaler
sentence-transformers                        # hetrep/vs (Phase 1)
rank-bm25>=0.2.2  nltk>=3.8  json-repair  python-dotenv  aiohttp   # HyperMem reuse (Phase 2)
pytest==9.1.1  pytest-cov==7.1.0  ruff==0.15.0
```

`pyproject.toml`'s dependency list gains `faiss-cpu`, `spacy`, and `sentence-transformers` as
first-class dependencies (not just installed ad hoc into `caimms`), so a fresh `pip install -e .`
reproduces the working state — the same reproducibility goal ADR 0006 pursues for the vendored
submodules applies equally to the interpreter environment.

### NLTK corpora pre-fetched off `/home`

`punkt`, `stopwords`, and `wordnet` auto-download on first HyperMem run
(`stage3_hypergraph_index.py:48-58`) to `~/nltk_data` by default. Per `CLAUDE.md`'s storage
invariant — "Large artifacts go under `/mnt/ssd/users/durgesh/`, never `/home`" — these are
pre-fetched during T-01 to `/mnt/ssd/users/durgesh/nltk_data/` explicitly, rather than left to
auto-download to the small `/home` disk the first time HG extraction runs.

### Determinism flag, not resolved by this ADR

`fluxmem/selector.py`'s docstring notes its tests are seed-sensitive, mitigated by
`torch.use_deterministic_algorithms(True, warn_only=True)` and a fixed `torch.manual_seed`. Moving
the effective torch version from `base`'s 2.12.0 (where the 155 passing tests were last confirmed)
to `caimms`'s 2.7.1 may shift floating-point behavior enough to change selector test expectations,
even with the seed fixed — deterministic-algorithm guarantees are not identical bit-for-bit across
major torch releases. **This ADR does not resolve that risk; it names the check that must happen
before the pins are trusted**: run the full suite in `caimms` immediately after T-01's installs,
before any HetRep code depends on the environment, and record any diff against the `base`-run
baseline rather than assuming none exists. This is exactly the "reports a number that is wrong"
failure mode `.claude/rules/evidence-discipline.md` targets, applied to test expectations rather
than experiment metrics.

## Consequences

### Positive

- **Phase 1 is unblocked.** Once T-01 completes, `caimms` both satisfies `requires-python` and
  successfully imports `fluxmem` and `hetrep`, closing the gap where neither property held
  simultaneously in any environment.
- **One canonical environment, stated once.** Contributors do not need to guess which of two
  environments to use; `CLAUDE.md` and the hetrepv2 plan both point at `caimms`.
- **Serving infrastructure is already present.** `caimms`'s existing vllm/ray stack means Phase 2
  (HG, needs one LLM endpoint) and Phase 3 (VC, needs three services) do not require a second
  environment setup pass — the environment chosen for Phase 0 already anticipates Phase 2–3 needs.
- **`pyproject.toml` becomes an accurate description of what the project actually needs**, rather
  than a set of pins that matched no live environment — `faiss-cpu`, `spacy`, and
  `sentence-transformers` move from "installed by convention, undocumented" to declared
  dependencies.

### Negative

- **`base` stops being a working environment for `fluxmem`.** Anyone who has been running tests in
  `base` (the environment where the 155-test baseline was actually last confirmed) loses that path
  once the project's canonical guidance shifts to `caimms`. This is an intended consequence, not
  an oversight — `base` was never meant to carry research-specific pins — but it is a real
  behavior change for anyone relying on it today.
- **Range pins are less exactly reproducible than exact pins.** `torch>=2.7,<2.9` admits both 2.7.1
  and 2.8.0; two installs at different times could resolve different concrete versions. If a
  specific published result needs bit-exact reproduction, the range alone is not sufficient — a
  lockfile or an explicit version record alongside that result is still needed.

### Risks

- **Torch downgrade (2.12.0 → 2.7.1) may lose performance or fix availability** relative to the
  version the 155-test baseline last ran under. Mitigation: re-run the full suite in `caimms`
  immediately post-install and record the result before treating the new pins as validated —
  per the determinism flag above, this is not optional.
- **MemOCR's internally inconsistent pins** (`requirements.txt` says `torch==2.7.1`, `numpy==2.2.6`;
  `setup.py` says `numpy<2.0.0` and `torch==2.8.0`) mean MemOCR itself may not import cleanly in
  `caimms` even after this ADR's changes — this ADR resolves `fluxmem`'s and `hetrep`'s pins, not
  MemOCR's; MemOCR remains a Phase 3 concern per ADR 0005's sequencing decision, and its own pin
  conflict is a separate, unresolved item to check before Phase 3 starts, not fixed here.
- **NLTK corpora path is user-specific.** `/mnt/ssd/users/durgesh/nltk_data/` works for the current
  single-user development setup; a multi-user or CI environment would need a different, documented
  path. Mitigation: this is flagged in `CLAUDE.md`'s environment section, not solved by this ADR.

## Related

- **ADR 0004** — HyperMem's dependency list (`rank-bm25`, `nltk`) is the source of the
  HyperMem-reuse packages installed here; HyperMem's own `requirements.txt` uses lower-bound
  ranges throughout, which is why it introduces no conflict with `fluxmem`'s pins.
- **ADR 0005** — `hetrep/`'s package entry in `pyproject.toml`
  (`[tool.setuptools.packages.find]`, `include = ["fluxmem*", "hetrep*"]`) depends on this ADR's
  environment being importable in the first place.
- **ADR 0006** — submodule conversion and this ADR's environment repair are both Phase 0
  prerequisites; neither depends on the other, but both must complete before Phase 1 code lands.
- **`CLAUDE.md`** — "Conda env: needs recreating under `/mnt/ssd/users/durgesh/conda-envs/`" and
  "Large artifacts... never `/home`" are the storage-invariant constraints this ADR's NLTK-corpora
  decision follows.
