# Plan: Isolate FluxMem — delete LightMem-core, EM²Mem, StructMem

**Status:** proposed, not yet approved for execution.
**Question this answers:** can we keep only FluxMem and delete the other three memory
methods from `LightMem/`? **Answer: yes, structurally clean.** Zero cross-imports were
found in either direction between `src/fluxmem/` and `src/lightmem/` / `src/em2mem/`.
FluxMem has its own vector store (FAISS), its own extras group in `pyproject.toml`, and
no dependency on any shared base class, config loader, or experiment harness.

This is a **research decision**, not a refactor — closing it out requires an ADR
(`.claude/rules/evidence-discipline.md`: "a research thread quietly becoming an
implementation thread" needs a decision in between). Phase 1 writes that ADR before any
deletion happens.

## Phase 0: Documentation Discovery — DONE (this session)

Findings, with sources:

1. **StructMem has no standalone code.** `find . -iname '*structmem*'` returns only
   `StructMem.md` and `figs/StructMem.png`. It is a _mode_ of LightMem-core, toggled via
   `extraction_mode: "event"` / `summary_retriever` config keys on
   `lightmem.memory.lightmem.LightMemory` (`StructMem.md:69-90`, `README.md:518`).
   Deleting it means deleting doc files only, no source tree.
2. **Zero cross-imports.** `grep -rn -E "from (lightmem|structmem|em2mem)|import (lightmem|structmem|em2mem)" src/fluxmem/` → no matches. Reverse check (lightmem/em2mem importing fluxmem) → no matches.
   `src/fluxmem/agent.py:14-25` imports only from itself (`.config`, `.graph.*`,
   `.interfaces.*`, `.retrieval.*`, `.stages.*`, `.metrics.pems`). Confirmed independently
   via `codegraph explore "src/fluxmem experiment runner main entrypoint"` — blast radius
   for `FluxMemAgent` (`src/fluxmem/agent.py:32`) shows exactly 1 caller, inside
   `src/fluxmem/__init__.py`, nothing outside the package.
3. **No shared code.** LightMem's vector store is Qdrant
   (`src/lightmem/factory/retriever/embeddingretriever/qdrant.py`); FluxMem's is FAISS
   (`src/fluxmem/interfaces/vectorstore.py`, own `BaseVectorStore`/`FAISSVectorStore`, no
   `qdrant` reference in that file). No shared base classes, utils, or config loader
   across any of the four.
4. **Build config**: one root `LightMem/pyproject.toml`. `fluxmem` already has its own
   optional-dependency extra (line 92: `fluxmem = ["openai>=1.0.0", "faiss-cpu>=1.7.0",
"scikit-learn>=1.0.0"]`, installed via `pip install -e ".[fluxmem]"` per
   `FluxMem.md:56`). No `structmem`/`em2mem` extras exist; EM²Mem instead has its own
   _separate_ `experiments/egolife/pyproject.toml` (`name = "lightmem-egolife"`).
5. **Tests**: the only test file in the repo is `LightMem/tests/test_sensory_memory.py`,
   which imports `lightmem.factory.memory_buffer.sensory_memory` — LightMem-only.
   **No test coverage exists for FluxMem at all**, so nothing is lost or preserved on
   that front; it should be flagged as a pre-existing gap, not created by this plan.
6. **No dispatcher exists.** No `__main__.py`, no CLI that branches by method name.
   `mcp/server.py:17` imports only `LightMemory` — LightMem-only entrypoint.
   **FluxMem has no experiment/runner script anywhere in the repo** (confirmed by
   codegraph — no callers of `FluxMemAgent` outside its own package, and a repo-wide
   grep for `"fluxmem"` outside `src/fluxmem/` and `FluxMem.md` returns zero hits). It
   is invoked only via the inline code samples in `FluxMem.md`. This closes the one open
   gap from the first pass: FluxMem does **not** piggyback on
   `experiments/{longmemeval,locomo,egolife}/` — those are LightMem-core and EM²Mem only.
7. **Doc/config blast radius** (non-code files that name a specific method):
   - `CLAUDE.md:23` — "Three subsystems: `lightmem` (core), `fluxmem`, `em2mem`." → becomes just fluxmem.
   - `CLAUDE.md:37-38` — storage-invariants callout half is lightmem-core-specific (dead once deleted), half is FluxMem (stays).
   - `CLAUDE.md:25-26,84-85` — conda env name `lightmem`, pointer to `LightMem/CLAUDE.md`, `setup/install-external.sh`.
   - `.claude/rules/storage-invariants.md` — almost entirely about LightMem-core's Qdrant/rmtree footgun (lines 12,19,23-25,34,48,58,64-65,75-77) plus two FluxMem-specific lines that stay valid (20, 60).
   - `.claude/skills/qdrant-ops/SKILL.md` — entirely LightMem-core-specific (Qdrant is not FluxMem's store). Likely dead weight after deletion — verify FluxMem never touches Qdrant before removing.
   - `.claude/skills/experiment-discipline/SKILL.md:33` — enumerates `lightmem | fluxmem | em2mem | baseline:<name>`; drop lightmem/em2mem. Its "where runs live" section (`experiments/{longmemeval,locomo,egolife}/`) is entirely LightMem-core/EM²Mem harness paths per finding 6 — needs a decision on whether FluxMem gets an equivalent harness or the section is just cut.
   - `docs/CLAUDE_TOOLKIT_ANALYSIS.md` — heaviest single hit; it's an audit _of_ LightMem's storage/toolkit layout with FluxMem/EM²Mem as siblings. Needs rewriting or archiving, not spot fixes.
   - `docs/workflows/add-memory-backend.md:7-9,16-17` — table rows per method; core guidance is anchored to `src/lightmem/factory/.../factory.py`'s factory pattern, which disappears. FluxMem has no equivalent factory/plugin pattern documented — this workflow effectively goes away or needs a new FluxMem-specific extension guide written from scratch.
   - `docs/workflows/review-and-merge.md:4`, `docs/related-work-agent-memory.md:5,82`, `.claude/rules/evidence-discipline.md:51` — pointers into `LightMem/` that will dangle.
   - `docs/component-index.md`, `docs/adr/` — zero hits; no cleanup needed (adr/ is empty — no prior decision was ever recorded for the four-method setup).
8. **Critical extraction hazard**: `FluxMem.md`, `figs/FluxMem.png`, and `src/fluxmem/`
   itself all currently live _inside_ `LightMem/`, the repo slated for deletion. **Do
   not `rm -rf LightMem/`** — FluxMem's own code and docs must be moved out first, or
   they're destroyed along with everything else.
9. **`LightMem/` is its own git repo** (`CLAUDE.md:14`: "commit there, not here"). Any
   deletion of files under `LightMem/` gets committed inside _that_ repo's history, not
   this outer repo's. Plan phases below are split accordingly.

**Confidence:** high on the code axis (exhaustive grep + codegraph cross-check, two
independent passes agree). **Gap still open**: neither pass checked non-Python runtime
assets (dataset JSON paths in configs, `.codegraph/`/`.serena/` caches) for indirect
FluxMem references — low risk given FluxMem's FAISS-only, self-contained design, but
worth a final `grep -r "dataset/"  src/fluxmem/` sanity check in Phase 2 before deleting
`dataset/`.

## Phase 1: Decision (ADR) — do this before any deletion

1. Write `docs/adr/0001-isolate-fluxmem.md` (Nygard format per `.claude/rules/documentation.md`):
   context = four methods coexist, FluxMem is the one on the active publication track;
   decision = delete lightmem-core, EM²Mem, StructMem docs, keep only FluxMem;
   consequences = loses lightmem-core's Qdrant path, LoCoMo/LongMemEval/EgoLife
   experiment harnesses, and the `mcp/server.py` entrypoint (all LightMem-core-only, per
   Phase 0 findings 3, 5, 6); FluxMem gains no new capability from this, only removes
   dead weight and repo-navigation cost.
2. Decide the two open questions the ADR needs to state explicitly:
   - Does FluxMem get its own experiment harness (a new `experiments/fluxmem/` mirroring
     the LoCoMo/LongMemEval structure), or does evaluation stay ad hoc via `FluxMem.md`'s
     inline examples? (Finding 6 — no harness exists today.)
   - Rename the conda env `lightmem` → `fluxmem`, or leave the name as a historical
     artifact? (Finding 7, `CLAUDE.md:25`.)
3. Get sign-off from the rest of the multi-person team before deleting — this repo's
   workflow is "code → review → merge" (`CLAUDE.md` header); a deletion this size is not
   a solo call.

**Verification:** ADR file exists, is sequential (0001, not colliding with an existing
one — confirmed `docs/adr/` is currently empty), and answers both open questions above.

## Phase 2: Extract FluxMem, then delete inside `LightMem/` (separate git repo)

Run from inside `LightMem/` — this commits to _that_ repo's history, per Phase 0 finding 9.

1. Sanity check first (closes the one open gap from Phase 0): `grep -rn "dataset/" src/fluxmem/ FluxMem.md` — confirm no path reference into the top-level `dataset/` loaders before deleting them.
2. Move `FluxMem.md` and `figs/FluxMem.png` out of the deletion path — they can stay at their current paths since only _sibling_ content is being deleted, not `src/fluxmem/`'s parent directories. No move is actually needed as long as deletion targets are scoped precisely to the list below — state this explicitly to whoever executes this phase so they don't `rm -rf` the whole tree.
3. Delete, scoped exactly to these paths (per Phase 0 findings 1, 3, 5, 6):
   - `src/lightmem/` and `src/lightmem.egg-info/`
   - `src/em2mem/` (including the vendored `VLM2Vec/` tree)
   - `StructMem.md`, `figs/StructMem.png`
   - `EM2Mem.md`, `figs/EM2Mem.png` (if present)
   - `mcp/server.py` (LightMem-only entrypoint)
   - `LightMem/tests/test_sensory_memory.py` (LightMem-only; no FluxMem test loss, per finding 5)
   - `experiments/egolife/`, `experiments/locomo/`, `experiments/longmemeval/`
   - `examples/run_lightmem_ollama.py`, `examples/run_lightmem_transformers.py`, `examples/run_lightmem_bm25.py`, `examples/run_llmlingua2_gpt.py`
   - `dataset/` (only after step 1's grep confirms no fluxmem reference)
4. Edit root `LightMem/pyproject.toml`:
   - Drop dependencies only needed by lightmem-core/em2mem from the bundled
     `[project.dependencies]` list (torch, transformers, qdrant-client, llmlingua, etc. —
     enumerate exactly by re-grepping imports in the deleted trees before trusting this
     from memory).
   - Keep the `fluxmem` extras block (line 92) as-is.
   - Revisit `[tool.setuptools.packages.find] where = ["src"]` (line 111-112) — it
     currently auto-discovers all three packages; after deletion it will only find
     `fluxmem`, which is correct, but verify no stale package metadata remains.
5. Update `LightMem/README.md` to drop the "LightMem Series" table rows for
   LightMem/StructMem/EM²Mem (keep FluxMem's row), and drop the "LightMem Series"
   external-repo section if it's no longer relevant.
6. Rewrite or delete `LightMem/CLAUDE.md` sections that describe lightmem-core's
   structure — check what's left after `src/lightmem/` is gone.
7. Commit inside the `LightMem/` repo with a conventional commit message
   (`.claude/rules/git-workflow.md`), e.g. `refactor: remove lightmem-core, em2mem, structmem; keep fluxmem only`.

**Verification:**

- `python -c "import fluxmem"` succeeds from a clean install of `pip install -e ".[fluxmem]"`.
- `grep -rn -E "lightmem|structmem|em2mem" src/fluxmem/` returns nothing.
- `pytest` (even though FluxMem has no tests today, confirm the suite doesn't error on
  the now-missing `test_sensory_memory.py` import path).
- `git status` inside `LightMem/` shows a clean tree after commit.

## Phase 3: Update the outer C-AIMMS repo's docs (separate commit, outer repo)

Run from `/home/durgesh/aditya/C-AIMMS` — this commits to the outer repo, not `LightMem/`.

1. `CLAUDE.md`:
   - Line 23: "Three subsystems: `lightmem` (core), `fluxmem`, `em2mem`." → "One subsystem: `fluxmem`."
   - Lines 37-38: cut the lightmem-core half of the storage-invariants callout, keep the FluxMem BM25 line.
   - Line 25: decide per Phase 1 step 2 whether to rename the conda env reference.
   - Lines 84-85: update or remove the `LightMem/CLAUDE.md` pointer if that file was substantially gutted in Phase 2 step 6.
2. `.claude/rules/storage-invariants.md`: strip everything specific to LightMem-core's
   Qdrant wrapper (findings list above: lines 12,19,23-25,34,48,58,64-65,75-77) and
   EM²Mem's pickle usage (line 64-65's `em2mem/memory/visual/Memory.py` mention);
   keep/expand the FluxMem FAISS+BM25 lines (20, 60) as the doc's new primary subject.
3. `.claude/skills/qdrant-ops/SKILL.md`: **decide, don't assume** — grep
   `src/fluxmem/` for any Qdrant usage one more time after Phase 2's deletion is final.
   If none, delete this skill entirely (Qdrant was lightmem-core-only, finding 7). If
   FluxMem ever gains a Qdrant backend later, this becomes a re-add, not a rewrite.
4. `.claude/skills/experiment-discipline/SKILL.md:33`: drop `lightmem | em2mem` from the
   subsystem enum, keep `fluxmem | baseline:<name>`. Rewrite the "where runs live"
   section per whatever was decided in Phase 1 step 2 about a FluxMem harness.
5. `docs/CLAUDE_TOOLKIT_ANALYSIS.md`: this is an audit report about a toolkit
   installation, not really about the memory methods per se — rewrite its LightMem-core-
   specific storage section (lines 31-44) or add a note that it now describes a deleted
   subsystem, rather than silently leaving stale claims.
6. `docs/workflows/add-memory-backend.md`: drop the lightmem/em2mem table rows; either
   rewrite the "how to add a backend" guidance around FluxMem's actual extension points
   (if any exist — check `src/fluxmem/interfaces/` for an abstract base a new backend
   would implement) or mark the workflow as needing a rewrite before next use.
7. `docs/workflows/review-and-merge.md:4`, `.claude/rules/evidence-discipline.md:51`:
   update the `LightMem/CLAUDE.md` pointers to reflect its post-deletion content.
8. `docs/related-work-agent-memory.md:5,82`: reframe the "related work" comparison
   points that currently cite LightMem-core's efficiency claims — decide whether FluxMem
   has an equivalent claim to cite instead, or cut the comparison.
9. Commit in the outer repo: `docs: update routing and rules after fluxmem isolation`.

**Verification:**

- `grep -rniE "lightmem-core|em2mem|structmem" CLAUDE.md .claude/ docs/` returns only
  historical/ADR mentions (the ADR from Phase 1 is expected to name them — everything
  else should not).
- Read `CLAUDE.md`'s routing table top to bottom and confirm every remaining row points
  at something that still exists.

## Phase 4: Final Verification

1. Fresh clone or fresh checkout test: `git clone` (or `git worktree add`) both repos,
   run `pip install -e "LightMem[fluxmem]"`, `python -c "from fluxmem import FluxMemAgent"`.
2. Confirm no dangling references: re-run the two Phase 0 discovery greps
   (`grep -rn -E "from (lightmem|structmem|em2mem)" LightMem/src/fluxmem/` and the
   doc-wide grep from Phase 3's verification step) — both should be clean except for the
   ADR.
3. Report what was verified vs. assumed, per `.claude/rules/evidence-discipline.md` —
   in particular, flag the Phase 1 decision about a FluxMem experiment harness as either
   "built" or "explicitly deferred," not left ambiguous.

## Anti-pattern guards

- Do not `rm -rf LightMem/` wholesale — `src/fluxmem/`, `FluxMem.md`, `figs/FluxMem.png`
  live inside it and must survive (Phase 0 finding 8).
- Do not assume FluxMem has a factory/plugin pattern for backends just because
  lightmem-core does — Phase 0 found no shared base classes; verify against
  `src/fluxmem/interfaces/` before writing new "add a backend" docs.
- Do not delete the `qdrant-ops` skill without the one-line grep check in Phase 3 step 3
  — cheap to verify, expensive to silently lose if wrong.
- Do not commit LightMem/ deletions and outer-repo doc edits in the same commit — they
  belong to two different git repositories (Phase 0 finding 9).
