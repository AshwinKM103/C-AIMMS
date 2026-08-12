# 0006. Vendored components become git submodules

## Status

Accepted — 2026-08-12. Gates Phase 0 task T-02 of the hetrepv2 plan (§7). No source code changes;
version-control structure only. Blocking check (below) must clear before conversion runs.

## Context

Three external codebases that HetRep's phases depend on are present at repo root as nested git
clones, each with its own `.git` directory and its own GitHub remote:

| Path        | Role                                                                 | Remote                                    | VCS state before this ADR                         |
| ----------- | -------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------- |
| `HyperMem/` | HG arm's data model and pure `build_hypergraph()` (ADR 0007)         | `github.com/AshwinKM103/HyperMem.git`     | Nested clone, untracked in C-AIMMS's `.gitignore` |
| `MemOCR/`   | Reference for VC arm's drafting prompt only (ADR 0005 §3.2, Phase 3) | `github.com/AshwinKM103/MemOCR.git`       | Nested clone, untracked                           |
| `EM-LLM/`   | Segmentation (surprise-based), upstream of the `EpisodeEncoder` seam | **none, at the time this ADR was opened** | No VCS at all; contains original C-AIMMS code     |

None of the three is listed in `.gitignore`. That is not a deliberate exclusion — it means `git
status` in the C-AIMMS root reports these directories as untracked content, and neither `git
clone` nor CI sees any of it. Three concrete consequences follow:

1. **CI cannot detect drift.** If `HyperMem/` is modified locally — as it was under ADR 0004's
   Phase 1–3 gap fixes — C-AIMMS's own `git status` shows clean at the top level. There is no
   commit recording which version of HyperMem produced a given experiment's numbers, which
   directly conflicts with `.claude/rules/evidence-discipline.md`'s requirement to report what
   produced a result.
2. **A fresh clone of C-AIMMS is incomplete.** `git clone github.com/AshwinKM103/C-AIMMS` fetches
   none of `HyperMem/`, `MemOCR/`, or `EM-LLM/` — running any HetRep phase requires manually
   discovering and cloning three more repositories at unspecified SHAs.
3. **Version history is incoherent across repos.** `HyperMem/` and `MemOCR/` are each independent
   git trees; there is no single commit in C-AIMMS's history that pins "which HyperMem SHA this
   experiment used." Rebasing or bisecting C-AIMMS's main branch cannot account for changes in the
   nested repos at all.

`EM-LLM/` is a sharper version of the same problem: it has no version control whatsoever, and it
contains 154 lines of original C-AIMMS code (`caimms_boundary_creator.py`), not vendored upstream
content. Without a `.git` directory or a remote, this file exists on exactly one disk. Any disk
failure, `rm -rf`, or environment rebuild loses it permanently — a materially worse risk than the
"untracked but recoverable from a remote" state of `HyperMem/` and `MemOCR/`.

### Why submodules, not a monorepo merge or a package dependency

| Option                                               | Description                                                           | Tradeoff                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (a) `git submodule` (chosen)                         | Pin each vendored repo to an exact SHA, tracked in `.gitmodules`      | Standard tooling, preserves each project's independent history and remote, `git clone --recursive` fully reproduces a checkout; submodule UX (detached HEAD, manual `update`) is well-known but real friction                                                                                         |
| (b) Squash-merge into `C-AIMMS` as plain directories | Copy the content in, drop the nested `.git`s, commit as regular files | Simplest clone story, but discards HyperMem's and MemOCR's own commit history and their independent GitHub remotes — they stop being separately maintainable/publishable projects                                                                                                                     |
| (c) `pip install` from each repo as a dependency     | Package each as an installable library, pin via `pyproject.toml`      | Correct long-term shape for genuinely stable dependencies, but none of the three is packaged (§2.5/§2.6 of the hetrepv2 plan: no `pyproject.toml`, `sys.path.insert` per stage, no `__init__.py` in most of MemOCR) — packaging is a separate, larger undertaking than fixing the version-control gap |

**Decision: (a), submodules.** They are the minimal change that closes the CI-visibility and
clone-completeness gaps without requiring HyperMem or MemOCR to first become well-formed Python
packages — that packaging work is real (ADR 0007 addresses the HG arm's specific reuse strategy;
MemOCR is not being packaged at all per ADR 0005's §3.2 stub decision) but is out of scope for a
version-control fix. Option (b) is rejected specifically because HyperMem and EM-LLM both have
active, separate GitHub remotes this project depends on continuing to exist as separate,
citable/forkable projects — HyperMem in particular is cited as `arXiv:2604.08256` (Yue et al., 2025) in ADR 0004 and should remain independently attributable.

## Decision

**Convert `HyperMem/`, `MemOCR/`, and `EM-LLM/` to git submodules at their current paths**, in the
order given because `EM-LLM/` cannot become a submodule until it has a remote to point at.

### T-02a — give `EM-LLM/` a remote first

```
cd EM-LLM && git init && git add -A && git commit -m "chore: initialize em-llm as standalone module"
# create github.com/AshwinKM103/EM-LLM.git
git remote add origin https://github.com/AshwinKM103/EM-LLM.git && git push -u origin main
```

This step is a hard precondition for T-02b — a submodule cannot be added pointing at a URL that
does not yet exist.

### T-02b — convert all three, in order

For each of `HyperMem`, `MemOCR`, `EM-LLM`:

1. **Blocking check: confirm the nested repo is pushed.** `git -C <dir> status --short` must be
   empty. This check is not procedural boilerplate — see the Risks section below for what it
   guards against.
2. Move the working copy aside (`mv HyperMem _HyperMem.bak`), so `git submodule add` can clone
   fresh rather than adopt an existing directory with potential local-only state.
3. `git submodule add https://github.com/AshwinKM103/<name>.git <name>` — this records the current
   remote HEAD SHA in the new `.gitmodules` file and stages the submodule commit reference.
4. Diff the fresh clone against the moved-aside backup to confirm nothing was lost, then remove
   the backup.

### T-02c — commit and document

Commit `.gitmodules` and the three submodule references in one commit. Remove `HyperMem/`,
`MemOCR/`, `EM-LLM/` from `.gitignore` (they were never explicitly listed, but any glob covering
them must be narrowed). Add the clone instruction to `CLAUDE.md`:
`git clone --recursive <url>`, or `git submodule update --init --recursive` after a plain clone.

### Blocking check before any of this runs

`HyperMem/tests/` contains the 22 C-AIMMS-authored tests written to validate ADR 0004's Phase 1–3
gap fixes (G2, G4, G5, G6, G7 — see that ADR's Implementation Status). Those tests, and the source
changes they cover, are **untracked inside the nested `HyperMem` repository** at the time this ADR
was opened. If T-02b runs before they are committed and pushed to
`github.com/AshwinKM103/HyperMem.git`, moving the directory aside destroys them — there is no
remote copy to fall back to. This is called out explicitly, not left implicit in "confirm status
is clean," because it is exactly the kind of loss `.claude/rules/git-workflow.md`'s "never commit
secrets... never commit large binaries" caution doesn't cover but should: uncommitted work with no
remote is unrecoverable by definition, and a submodule conversion is precisely the operation that
discards it silently if skipped.

## Consequences

### Positive

- **Version-control integrity.** Every experiment result can now cite an exact `HyperMem`/`MemOCR`/
  `EM-LLM` SHA via `git submodule status`, closing the gap `.claude/rules/evidence-discipline.md`
  flags — no more "which version produced this number" ambiguity.
- **CI/CD visibility.** Submodule SHA bumps appear as ordinary diffs in C-AIMMS's commit history;
  a PR that updates `HyperMem`'s pinned SHA is now a reviewable, visible change instead of an
  invisible local-disk mutation.
- **Reproducible clones.** `git clone --recursive` (or `git submodule update --init --recursive`)
  produces a working checkout without manual out-of-band steps.
- **`EM-LLM`'s original code is no longer single-disk.** Giving it a remote is a strict safety
  improvement independent of the submodule conversion itself.
- **ADR 0004's work becomes permanent.** The 22 tests and the Phase 1–3 fixes move from "exists on
  one disk in an untracked nested repo" to "pinned SHA in C-AIMMS's tracked history."

### Negative

- **Submodule UX has real friction.** Contributors must run `git submodule update --recursive`
  after every pull that touches a submodule pointer, and forgetting silently leaves a stale
  checkout rather than erroring — the CLAUDE.md clone instructions mitigate but do not eliminate
  this.
- **Detached-HEAD workflow inside submodules.** Making a change inside `HyperMem/` now requires
  checking out a branch explicitly before committing, rather than committing directly as one could
  when it was an independent working copy — a small but real added step for anyone actively
  developing HyperMem-side fixes (as ADR 0004's Phase 3 LoCoMo evaluation work will require).
- **Repo size grows modestly.** `.git` history gains the submodule pointer commits (small; actual
  vendored content stays in the submodules' own histories, not duplicated into C-AIMMS's).

### Risks

- **`EM-LLM` remote creation fails or is delayed.** T-02a is a hard precondition for T-02b; if the
  GitHub remote cannot be created (permissions, naming collision), the entire conversion for
  `EM-LLM` blocks. Mitigation: this is a manual, one-time user action, not automatable, and should
  be done first and confirmed before any of T-02b begins for any of the three repos.
- **Untracked `HyperMem/tests/` is destroyed by a premature `mv`.** Covered in detail above; the
  mitigation is the blocking check itself, run and confirmed clean before T-02b, not skipped under
  time pressure.
- **Submodule SHA drifts from what a paper draft cites.** Once HetRep phases start reporting
  numbers, the pinned SHA at the time of a run must be recorded alongside the result — this ADR
  makes that possible but does not itself enforce it; the experiment-record discipline in
  `docs/workflows/new-experiment.md` and the `experiment-discipline` skill are the actual
  enforcement points.

## Related

- **ADR 0004** — the HyperMem Phase 1–3 gap fixes and the 22 tests this conversion must not lose.
- **ADR 0005** — HetRep's dependency direction; Phase 2 (HG) and Phase 3 (VC) depend on
  `HyperMem`/`MemOCR` being reachable at a pinned SHA once they become submodules.
- **ADR 0007** — HyperMem's structure module is what the HG arm actually imports post-conversion;
  the injection boundary there is unaffected by whether HyperMem is a nested clone or a submodule.
