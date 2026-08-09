# Workflow: Review and merge

The team loop is code → review → merge. Codex independently reviews everything Claude produces in
`LightMem/` (see `LightMem/CLAUDE.md`) — write for that reader.

## Before opening a PR
1. `/logic-review` on anything that computes a metric, mutates memory state, or touches retrieval.
   Ruff and mypy passing is not evidence of correctness here.
2. Tests actually run — paste the output, don't summarise it.
3. `/secrets-scan` if the diff touches configs or `.env` handling. The secret-scanner hook catches
   writes in-session; this catches what predates it.
4. Conventional commit messages (hook-enforced). `exp:` for experiment-only changes.
5. `/pr-create` — summary, test plan, and what a reviewer should look at hardest.

## Reviewing
- `.claude/rules/code-review.md` is the shared checklist.
- `/pr-review <n>` for structured feedback; `code-reviewer` agent for a deeper security/perf pass.
- For research code specifically, ask: **if this is wrong, would any test fail?** If not, that is
  the review comment.
- Check whether a baseline under `baselines/**` was touched. If so, every number in that column
  needs re-stating.

## Merging
- Direct pushes to `main`/`master` prompt for confirmation; force-pushes to them are blocked
  outright by `pre-push-check.js`. Use a feature branch and a PR.
- `/changelog` after merge; `/release` when tagging a version behind a paper result — a tagged
  commit is what makes a reported number reproducible.

## Related
`/pr-review`, `/pr-create`, `/logic-review` · `code-reviewer`, `test-architect` agents
