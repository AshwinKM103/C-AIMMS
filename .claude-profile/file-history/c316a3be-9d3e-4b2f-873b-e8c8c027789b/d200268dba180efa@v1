# Component Index

Full catalog of the installed Claude Code stack. `CLAUDE.md`'s routing table stays short by
design (loads in full every session); this file is the depth layer it points to — read on demand,
not preloaded. Excludes `react-specialist`, `typescript-specialist`, `react-patterns`,
`typescript-advanced` — no frontend or TypeScript code exists in this repo.

## Commands (`.claude/commands/`)

| Command             | Use for                                                              |
| ------------------- | -------------------------------------------------------------------- |
| `/adr`              | Write an ADR for a significant decision                              |
| `/analyze-failures` | Root-cause test failures, find patterns, suggest targeted fixes      |
| `/audit`            | Security audit across common vulnerability categories                |
| `/bisect`           | `git bisect` to find the commit that introduced a bug                |
| `/build-regex`      | Build a regex pattern from a natural-language description            |
| `/changelog`        | Generate a changelog since the last tag                              |
| `/checkpoint`       | Save a session checkpoint                                            |
| `/ci-pipeline`      | Generate a GitHub Actions config                                     |
| `/cleanup`          | Remove dead code, unused imports, unreachable branches               |
| `/commit`           | Draft a conventional commit message from staged changes              |
| `/compare`          | Compare ML experiment runs side-by-side                              |
| `/compare-models`   | Compare multiple ML models to select the best performer              |
| `/compare-tools`    | Compare developer tools for a selection decision                     |
| `/dashboard`        | Generate a monitoring dashboard config for project metrics           |
| `/dead-code`        | Find and remove dead code                                            |
| `/debug`            | Systematic debugging: symptoms → hypotheses → tests                  |
| `/dependency-audit` | Audit dependencies for vulnerabilities and staleness                 |
| `/design-review`    | Structured review of a module/feature/system                         |
| `/diagram`          | Generate a Mermaid diagram                                           |
| `/doc-gen`          | Generate documentation for a scope                                   |
| `/dockerfile`       | Generate an optimized Dockerfile                                     |
| `/env-setup`        | Set up env config files from templates, with validation              |
| `/env-validate`     | Validate env config against template and runtime requirements        |
| `/evaluate`         | Score a developer tool against structured criteria                   |
| `/evaluate-model`   | Evaluate ML model performance                                        |
| `/extract`          | Extract a function/module into its own unit                          |
| `/find-leaks`       | Detect and diagnose memory leaks                                     |
| `/fix-issue`        | Fix a GitHub issue end to end: branch, implement, PR                 |
| `/integration-test` | Generate integration tests                                           |
| `/memory-bank`      | Update CLAUDE.md with session learnings                              |
| `/onboard`          | Generate an onboarding guide                                         |
| `/optimize`         | Apply targeted performance optimizations                             |
| `/orchestrate`      | Break a complex task into coordinated sub-tasks                      |
| `/organize`         | Sort, group, clean up import statements                              |
| `/plan`             | Structured implementation plan                                       |
| `/pr-create`        | Create a PR with a structured description                            |
| `/pr-review`        | Review a PR by number                                                |
| `/profile`          | Find performance bottlenecks and scalability concerns                |
| `/profile-memory`   | Capture and analyze memory usage                                     |
| `/refactor`         | Systematic refactor of a code area                                   |
| `/refactor-py`      | Python-specific refactor for clarity and idiom                       |
| `/release`          | Tagged release with generated notes                                  |
| `/rename`           | Rename a symbol repo-wide — prefer serena instead, see routing table |
| `/report`           | Project analytics report: code quality, velocity, health             |
| `/secrets-scan`     | Scan for leaked secrets/keys/tokens                                  |
| `/simplify`         | Simplify code for readability                                        |
| `/snapshot-test`    | Generate snapshot tests                                              |
| `/tdd`              | Start a TDD red-green-refactor cycle                                 |
| `/test-coverage`    | Find coverage gaps, generate tests                                   |
| `/test-fix`         | Diagnose and fix failing tests                                       |
| `/test-regex`       | Test a regex against sample inputs and edge cases                    |
| `/trace`            | Trace an execution path through the codebase                         |
| `/track`            | Log an experiment's params/metrics/artifacts                         |
| `/type-hints`       | Add type hints to Python code                                        |
| `/update-codemap`   | Refresh the project codemap                                          |
| `/verify`           | Second-pass verification of changes before commit                    |
| `/worktree`         | Set up git worktrees for parallel branches                           |
| `/wrap-up`          | End-of-session summary and memory update                             |

Also from installed plugins: `/logic-review`, `/logic-explain`, `/logic-diff`, `/logic-locate`,
`/logic-health`, `/logic-fix-all` (logic-lens) · claude-mem's own command set (`/mem-search`,
`/make-plan`, `/do`, `/checkpoint` overlap — see `claude-mem:how-it-works`).

## Agents (`.claude/agents/`)

| Agent                      | Use for                                                         |
| -------------------------- | --------------------------------------------------------------- |
| `academic-researcher`      | Literature reviews, citation analysis, research synthesis       |
| `ai-engineer`              | Model API integration, RAG pipelines, embedding strategies      |
| `autoresearch-agent`       | Automated experiment optimization via tree search               |
| `benchmarking-specialist`  | Benchmarks, comparative evaluations, measurement methodology    |
| `code-reviewer`            | Pattern/anti-pattern, security, performance, readability review |
| `context-manager`          | Context window optimization, progressive loading                |
| `data-engineer`            | ETL/ELT pipelines, warehousing, orchestration                   |
| `data-researcher`          | Data analysis, pattern recognition, statistical interpretation  |
| `data-scientist`           | Statistical analysis, hypothesis testing, EDA in Python         |
| `data-visualization`       | Dashboards and charts (Matplotlib/Plotly/D3/Chart.js)           |
| `dependency-manager`       | Dependency audits, updates, lockfile integrity                  |
| `documentation-engineer`   | API references, guides, ADRs                                    |
| `dx-optimizer`             | Tooling ergonomics, workflow friction reduction                 |
| `error-detective`          | Stack trace analysis, root cause identification                 |
| `git-workflow-manager`     | Branching strategy, CI integration patterns                     |
| `knowledge-synthesizer`    | Cross-source synthesis, knowledge graphs                        |
| `llm-architect`            | LLM system design, fine-tuning, eval frameworks                 |
| `mcp-developer`            | MCP server/tool development                                     |
| `ml-engineer`              | Training pipelines, feature engineering, deployment             |
| `mlops-engineer`           | Model serving, monitoring, A/B testing, CI/CD for models        |
| `multi-agent-coordinator`  | Parallel agent execution, dependency management                 |
| `nlp-engineer`             | Text processing, embeddings, classification, NER                |
| `performance-engineer`     | Profiling, memory analysis, load testing                        |
| `performance-monitor`      | Agent execution monitoring, token usage tracking                |
| `prompt-engineer`          | Chain-of-thought, few-shot, structured-output prompting         |
| `python-engineer`          | Python 3.12+, typing, async, pydantic, packaging                |
| `refactoring-specialist`   | Dead code removal, abstraction extraction                       |
| `research-analyst`         | Systematic literature review, evidence synthesis                |
| `search-specialist`        | Advanced search, source evaluation, synthesis                   |
| `technical-writer`         | Polished technical documentation                                |
| `technology-scout`         | Build-vs-buy, vendor/tool adoption analysis                     |
| `test-architect`           | Unit/integration/e2e strategy, property-based, mutation testing |
| `vector-database-engineer` | Embedding pipelines, FAISS/Qdrant/Pinecone/Weaviate             |

## Skills (`.claude/skills/`)

| Skill                      | Use for                                                                     |
| -------------------------- | --------------------------------------------------------------------------- |
| `ci-cd-pipelines`          | GitHub Actions/GitLab CI patterns, deployment automation                    |
| `data-engineering`         | ETL, warehousing, Spark, data quality validation                            |
| `deep-dive`                | Claude-native deep research, DAG query planning                             |
| `docker-best-practices`    | Multi-stage builds, compose patterns, image security                        |
| `experiment-discipline`    | This project's `/track` + `/compare` conventions and storage specifics      |
| `git-advanced`             | Worktrees, bisect, interactive rebase, recovery                             |
| `llm-integration`          | API usage, streaming, function calling, RAG, cost optimization              |
| `manage-skills`            | Discover/create/edit/toggle skills across agent tools                       |
| `mcp-development`          | MCP tool design, resource endpoints, transport config                       |
| `monitoring-observability` | OpenTelemetry, Prometheus, Grafana, structured logging                      |
| `prompt-engineering`       | Structured prompts, chain-of-thought, few-shot, system prompts              |
| `python-best-practices`    | Modern type hints, dataclasses, async, packaging, testing                   |
| `qdrant-ops`               | Qdrant collection lifecycle, filters, on-disk/in-memory, the rmtree footgun |
| `tdd-mastery`              | Red-Green-Refactor cycle                                                    |
| `testing-strategies`       | Contract/snapshot/mutation/property-based testing                           |

## Hooks (`.claude/settings.json` → `.claude/hooks/scripts/`)

| Hook                | Event                                         | Does                                                 |
| ------------------- | --------------------------------------------- | ---------------------------------------------------- |
| `secret-scanner.js` | PreToolUse: Write/Edit/MultiEdit/NotebookEdit | Blocks writes containing API keys/tokens/credentials |
| `commit-guard.js`   | PreToolUse: `git commit`                      | Blocks non-conventional commit messages              |
| `pre-push-check.js` | PreToolUse: `git push`                        | Blocks force-push to `main`                          |
| `lint-fix.js`       | PostToolUse: Write/Edit/MultiEdit             | Runs ruff after every write                          |

## Plugins (user scope, `claude plugin list`)

| Plugin              | Does                                                                |
| ------------------- | ------------------------------------------------------------------- |
| `claude-mem`        | Cross-session memory (SQLite + Chroma), worker daemon               |
| `headroom`          | Token compression proxy                                             |
| `claudebase`        | Git-backed `.claude/` sync across machines, secret-blocking on push |
| `logic-lens`        | Execution-tracing logic review (`/logic-review` etc.)               |
| `claude-hud`        | Terminal statusline: context, quota, active subagents               |
| `notify`            | Desktop notification on task completion                             |
| `claude-code-setup` | Automation recommender for hooks/agents/skills/MCP                  |

## MCP servers (`.mcp.json`, project-scoped)

| Server       | Provides                                                               |
| ------------ | ---------------------------------------------------------------------- |
| `filesystem` | Read/write project files, datasets, experiment records                 |
| `github`     | Issues, PRs, branches, releases (needs `GITHUB_PERSONAL_ACCESS_TOKEN`) |
| `memory`     | Knowledge graph for research findings across sessions                  |
| `serena`     | LSP-backed symbol search and safe symbol-level edits                   |

## Rules (`.claude/rules/`, always active)

`agents.md` · `code-review.md` · `coding-style.md` · `dependency-management.md` ·
`documentation.md` · `error-handling.md` · `evidence-discipline.md` · `git-workflow.md` ·
`monitoring.md` · `naming.md` · `security.md` · `storage-invariants.md` (path-scoped) ·
`testing.md`
