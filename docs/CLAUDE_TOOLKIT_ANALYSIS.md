# Claude Code Toolkit — Component Analysis for C-AIMMS

**Scope:** all 405 local components in `awesome-claude-code-toolkit/` **plus** all 344 third-party
entries linked from its README — 749 components reviewed.
**Project:** C-AIMMS — Cognitive AI Memory Architecture (Python, LLM-memory research, multi-person
team, GitHub-backed, COLM track).
**Revised:** 2026-08-09 (second pass). Supersedes the first pass, which excluded external
components, did not know a Claude Code stack was already installed, and guessed at storage.

---

## 0. What this project actually is

Grounded in the repo, not the brief:

| Signal | Evidence | Consequence |
|---|---|---|
| LLM-memory research on a publication track | `LightMem/` (arXiv 2510.18866), `EM2Mem.md` / `FluxMem.md` / `StructMem.md`, `docs/COLM_*.pdf` | LLM/RAG/embedding/eval tooling scores high; REST/GraphQL/microservice tooling scores zero |
| Python library + MCP server | `pyproject.toml`, `LightMem/mcp/`, `src/`, `tests/` | Python + MCP tooling in; Node-based `backend-developer` agent out as wrong language |
| Benchmark-driven | `experiments/{longmemeval,locomo,egolife}/` | Experiment tracking, model evaluation, Jupyter MCP, benchmarking agents in |
| Constrained hardware | `/home` near-full; models under `/mnt/ssd/users/durgesh/`; torch 2.8 + 4B/8B locals | Anything holding models resident or spending background tokens is a real cost, not a footnote |
| Existing Claude Code stack | `LightMem/CLAUDE.md`: claude-mem, headroom, task-observer, claude-code-setup, omniroute, CodeGraph | Roughly a dozen "obvious" picks are duplicates |
| Linux | `uname`: Linux 6.17 | ~35 macOS/Windows-only external tools eliminated outright |

**A note on the toolkit's own numbers.** Its README advertises "176+ plugins, 135 agents". Most of
the surplus is links to third-party repos, not shipped files. The clone contains **405 installable
components**; the other **344** are external links. This pass covers both, separately.

---

## 1. Storage — what LightMem actually uses

The first pass guessed. This is what is in the code.

| Layer | Reality | Location |
|---|---|---|
| **Primary vector store** | **Qdrant**, `qdrant-client==1.15.1` pinned | `src/lightmem/factory/retriever/embeddingretriever/qdrant.py` |
| Config defaults | `collection_name="lightmem"`, `embedding_model_dims=1024`, `path="/tmp/qdrant"`, `on_disk=False` | `src/lightmem/configs/retriever/embeddingretriever/qdrant.py` |
| Graph memory | `networkx` | `lightmem/memory/graph.py`, `fluxmem/graph/`, `em2mem/memory/semantic_graph/` |
| Layer persistence | `pickle` | `memories/layers/{amem,full_context,memzero,langmem}.py`, `em2mem/memory/visual/Memory.py` |
| Sparse / hybrid | **FluxMem only** — weighted dense+BM25 fusion (`bm25_weight`, default 0.5) | `src/fluxmem/retrieval/semantic_retriever.py` |
| Sparse in lightmem core | **does not exist** — `factory/retriever/contextretriever/bm25.py` is an empty 1-byte stub | — |
| Optional | FAISS (`fluxmem` extra), ChromaDB (agentic_memory baseline) | `pyproject.toml` |
| Vendored, inactive | SQLite `SQLiteManager` + ~20 mem0 adapters (pgvector, redis, milvus, weaviate…) | `baselines/mem0/**` — **no `sqlite3` in lightmem core at all** |

### Two defaults that destroy data

```python
# Qdrant.__init__
if not params:
    params["path"] = config.path              # default "/tmp/qdrant"
    if not config.on_disk:                    # default False
        if os.path.exists(config.path) and os.path.isdir(config.path):
            shutil.rmtree(config.path)        # silent, unrecoverable
```

Re-running an experiment against an existing path **deletes the previous store before writing**.
Combined with the `/tmp` default — ephemeral, and on the disk you were told to avoid — this is the
most likely cause of an unexplained recall collapse. `reset()` is the same hazard by another name
(`delete_col()` + `create_col()`).

Encoded as `.claude/rules/storage-invariants.md`, path-scoped to retriever/memory/config/experiment
paths so it costs nothing on unrelated work.

### Database skills — revised verdict

Still rejected: `postgres-optimization`, `redis-patterns`, `database-optimizer`, `query-optimizer`,
`schema-designer`, `database.md`. Not because "no database" — because **every relational, pgvector,
and redis path in this repo is vendored mem0 baseline code, not the active retriever**. Optimising
them would be work on a comparison point, not on the system.

Conditional: `database-optimization` applies only if you run the mem0 SQLite-history baselines.

**The real gap:** no toolkit component covers vector-store operations. Filled with a project-local
`qdrant-ops` skill (collection lifecycle, the `_create_filter` surface — `MatchValue` and `Range`
only, `must`-only, `MatchAny` imported but unused — on-disk vs in-memory, snapshot before
destructive re-runs).

---

## 2. Reconciling with the already-installed stack

`~/.claude/settings.json` has **no hooks**; every hook comes from a plugin, and they stack at
runtime. The first pass would have added 16 more.

| Event | Already firing | First pass proposed | Installed |
|---|---|---|---|
| SessionStart | claude-mem, headroom | session-start, context-loader | **none** |
| PreToolUse | claude-mem, headroom | secret-scanner, commit-guard, pre-push-check, smart-approve, block-dev-server | **3** |
| PostToolUse | claude-mem | lint-fix, auto-test, type-check, suggest-compact | **1** |
| Stop / SessionEnd / PreCompact | claude-mem | stop-check, session-end, learning-log, pre-compact | **none** |

**Six reversals from the first pass:**

| Reversed | Because |
|---|---|
| `claude-memory-kit` skill | claude-mem already owns cross-session memory |
| `continuous-learning` skill | task-observer already does pattern capture |
| `explore` plugin | CodeGraph is indexed here; serena does symbols properly |
| 12 session/memory/test hooks | claude-mem + headroom hold those events |
| `post-edit-check.js` | duplicates `lint-fix.js` on the same write — stale diagnostics at 2× latency |
| `auto-test.js` | would run pytest on save; `experiments/**` tests load 4B/8B models |

### The toolkit's hooks are broken as shipped

Discovered while installing them. Every script reads
`JSON.parse(process.argv[2] || "{}")` — but Claude Code delivers hook input on **stdin**. With no
argv[2] they parse `{}`, find no `file_path`, and `process.exit(0)`. **All 20 silently no-op.**

Four further defects in the four we wanted:

1. `secret-scanner.js` reads the file **from disk** (`fs.readFileSync(filePath)`). On a `Write`
   that is the *old* content — so the secret being written is never seen. Fixed to scan
   `tool_input.content` / `new_string` / `edits[].new_string`.
2. Field paths are wrong: `input.file_path` should be `input.tool_input.file_path`.
3. `commit-guard.js` emits top-level `{"decision":"block"}` — the PostToolUse shape. PreToolUse
   requires `hookSpecificOutput.permissionDecision`, so the block was ignored.
4. `pre-push-check.js` matched `-f` as a substring, flagging any branch containing "-f"
   (e.g. `feat-fluxmem`).

All four were rewritten against the documented contract, attribution retained, and **tested** —
see §8. Two project-specific additions: `sk-*` / `hf_*` key patterns (this project holds OpenAI,
DeepSeek and HuggingFace credentials) and an `exp:` conventional-commit type for experiment-only
changes.

---

## 3. Local toolkit — 405 components

Selected **176** (43.5%). Condensed by function; the reasoning that changed is in §1–2.

| Area | Selected | Rejected | Notes on the calls that matter |
|---|---|---|---|
| **Python dev** | `python-expert`, `python-best-practices`, `python-engineer`, `refactor-engine`, `refactoring-specialist`, `dead-code-finder`, `import-organizer`, `bug-detective`, **`debug-session`**, `error-detective`, `double-check`, `perf-profiler`, `memory-profiler`, `dependency-manager` ×2, `env-manager`, `regex-builder` | `backend-architect`, `backend-developer` (Node — wrong language), all REST/OpenAPI/contract tooling, `complexity-reducer` (redundant with `refactor-engine`), 7 DB components, 23 non-Python language agents | `debug-session`'s `/bisect` is the highest-value debugging tool here: bisect on the *metric*. `memory-profiler` confirmed to support Python `tracemalloc` |
| **Experiments & metrics** | `experiment-tracker`, `model-evaluator`, `mlops-engineer`, `ml-engineer`, `benchmarking-specialist`, `data-visualization`, `analytics-reporter`, `test-results-analyzer`, `data-science.json`, `llm-cost.json` | `monitoring-setup` (Prometheus/Grafana — no SLO), `finance-tracker` (tracks engineer-hours, not API spend), `feature-engineer` (tabular framing), `growth-engineer` (product A/B, different meaning) | `experiment-tracker` is prompt-scaffolding, **not an MLflow/W&B client**. Kept explicitly as a discipline layer — seeds, dataset version, never-overwrite |
| **LLM / research** | `llm-integration`, `prompt-engineering`, `ai-prompt-lab`, `rag-builder`, `embedding-manager`, `vector-database-engineer`, `llm-architect`, `ai-engineer`, `nlp-engineer`, `data-scientist`, `data-engineer`, `autoresearch-agent`, `academic-researcher`, `research-analyst`, `data-researcher`, `search-specialist`, `technology-scout`, `deep-dive`, `tool-evaluator`, `research.json`, `research.md` context | `prompt-optimizer` (redundant — `ai-prompt-lab` adds a test loop), 5 market/IP/threat research agents | `vector-database-engineer` confirmed by the Qdrant/FAISS/Chroma reality |
| **Git & GitHub** | `smart-commit`, `changelog-gen`, `release-manager`, `git-flow`, `update-branch`, `create-worktrees`, `git-advanced`, `git-workflow-manager`, `github-issue-manager`, `fix-github-issue`, `fix-pr`, `ci-debugger`, `ci-cd-pipelines`, 7 git commands, `git-workflow.md` | `commit-commands` (auto-push fights `pre-push-check` and review-before-merge), `changelog-writer`, `linear-helper` (team uses GitHub), `slack-notifier` (token setup for a team already on GitHub) | |
| **Review & testing** | `pr-reviewer`, `code-review-assistant`, `code-reviewer`, `test-writer`, `test-architect`, `test-results-analyzer`, `tdd-mastery`, `testing-strategies`, 5 test commands, `security-guidance`, 3 security commands | `code-guardian` (three-in-one, `/review` collision), `unit-test-generator`, `mutation-tester` (compute), `load-tester`/`performance-monitor`/`api-benchmarker` (RPS ≠ retrieval quality), 5 QA agents, `security-hardening` | `testing-strategies` earns its place for **property-based testing** of retrieval invariants |
| **Docs & diagrams** | `adr-writer`, `doc-forge`, `code-explainer`, `readme-generator`, `onboarding-guide`, `documentation-engineer`, `technical-writer`, `code-architect`, `/diagram`, `data-visualization`, `vision-specialist`, 4 doc commands | `codebase-documenter` (mass auto-commenting = review churn), `api-reference`/`/api-docs`, `schema-designer` (ERDs describe relational schemas), `block-md-creation.js` (fights root-level design docs) | Mermaid into `docs/diagrams/` — text-diffable, renders in PRs |
| **Team & memory** | `/memory-bank`, `/checkpoint`, `/wrap-up`, `context-manager`, `knowledge-synthesizer`, `performance-monitor`, `multi-agent-coordinator`, `dx-optimizer`, `discuss`, `plan`, `ultrathink`, `manage-skills`, `agents.md`, 4 contexts | `claude-memory-kit`, `continuous-learning` (**reversed** — see §2), `phoenix-prime-starter`, `task-coordinator`, `workflow-director`, `sprint-prioritizer` + 5 PM agents (team-size mismatch) | |
| **Infra / frontend / domain** | `docker-helper`, `docker-best-practices`, `/dockerfile`, `/ci-pipeline`; frontend set held conditional | all 11 infrastructure agents, all 15 specialized-domain agents, cloud/K8s/mobile plugins, accessibility ×7 | Largest single rejection block: 26 agents with zero overlap with a Python research library |

**Rules:** 11 installed (+ 2 written). **Commands:** 34. **Agents:** 35. **Skills:** 15 (+ 2 written).
**Hooks:** 4 (rewritten). **MCP:** 5 configs merged into 8 servers. **Contexts:** 4. **Templates:** 3.

---

## 4. External catalog — 344 third-party entries

The first pass excluded these as "links, not files". They were re-examined by **fetching each
candidate's documentation directly** rather than judging from the README's one-line blurbs. Seven
clusters were compared head-to-head instead of block-rejected.

### 4.1 Adopted (10)

| Component | Why it wins | Cost / dependencies |
|---|---|---|
| **claudebase** | Syncs `.mcp.json` + `.claude/{settings,agents,commands,skills,hooks,rules,agent-memory}` to a private GitHub repo. `shared/` base + `profiles/<name>/` overlay; blocks API keys / PEM / Bearer on push; blocks push if another machine pushed since your last sync; keeps 10 versions | `gh`, `jq`, `git`, `bash`. Never syncs conversations, logs, or `settings.local.json` |
| **logic-lens** | Behavioural review by execution tracing: Premises → Trace → Divergence → Trigger → Remedy, risk codes L1–L9 (state mutation, control-flow escape, callee-contract mismatch, async races). Full Python | zero deps, optional `.logic-lens.yaml` |
| **serena** | Real **LSP language servers**: `find_symbol`, `find_referencing_symbols`, type hierarchy, and symbol-level **edits** — rename / move / inline / safe-delete. 40+ languages | `uv tool install -p 3.13 serena-agent`. **MCP server — zero hooks** |
| **AutoSearch** | 40 channels including **arXiv, PubMed, OpenAlex, DBLP**; deduped, ranked, every result carries a source URL; most channels keyless | `npx autosearch-ai`, `autosearch doctor` |
| **Prism Scanner** | Security-scans agent skills / plugins / MCP servers: 39+ rules, AST taint tracking, A–F grade | `pip install prism-scanner`. **Run before the other installs** |
| **ccusage** | Offline spend and quota totals from local JSONL, zero API calls | 11.5k★, `npx ccusage` |
| **cc-cost** | Answers a *different* question: prompt-cache hit rate, tool-call distribution, top expensive turns, `--diagnose` naming the three cost drivers | single-file Python 3.9+, zero deps |
| **PRISM** | Answers a *third* question: CLAUDE.md adherence and re-read cost (a real finding of theirs: 6738% re-read cost), attention-curve decay, emits **diff-format CLAUDE.md fixes** | `pip install prism-cc`, Py3.11+, read-only |
| **claude-hud** | Terminal statusline via the native API: context-window bar, quota burn, active subagents, todo %, branch, token speed | Node 18+, Linux confirmed, **zero hooks** |
| **notify** | Desktop notification on completion — one of the few with explicit Linux `notify-send` support | duration filtering, git context |

### 4.2 Adopted as lessons — ported into rules, not installed (6)

| Source | What was taken |
|---|---|
| **claude-agentic-coding-playbook** | Hooks beat instructions (>95% vs 50–90% compliance); the research loop Question→Collect→Synthesize→Close alongside Explore→Plan→Code→Verify→Commit; two-hypothesis minimum; numbered observations with sources. Its 35+ hooks and `node:sqlite` knowledge DB would collide with claude-mem, so nothing was installed |
| **GateGuard** | Verify the data schema against real records before writing code that assumes it. Their evidence is weak — **N=3, self-scored, admitted bias, +2.0 not the advertised +2.25** — but the failure it names is exactly how a silent metric bug is born, and the check is free |
| **obey** | The Stop-hook completion checklist, as a rule. Its 17 hooks are genuinely cheap (~10ms PreToolUse) but it keeps a parallel rule store at `~/.config/obey/`, fighting git-backed config |
| **paul-graham-skills** | `distrust-the-surface`, `mark-what-you-don't-know`, `change-method-not-goal` — epistemics that matter more in research than in product code |
| **claude-code-blueprint** | **Model tiering** (Opus/Sonnet/Haiku per agent — a direct cost lever) and **path-scoped rules** (a rule loads only on matching globs). Explicitly "a template to copy from", zero deps |
| **spartan-ai-toolkit** / **GAAI** | Spartan's 3-layer memory (Index always-loaded → Topics on-demand → Transcripts grep-only) is independent validation of the modular CLAUDE.md chosen here. GAAI contributes backlog-as-contract and the discovery≠delivery split; both are markdown+YAML, git-committed, no SDK |

### 4.3 Evaluate later, with triggers (7)

`ccpm` (only if parallel task execution becomes the bottleneck — it wants 5–10 parallel tasks and
upfront PRDs) · `Bouncer` (independent Gemini quality gate via Stop hook — mirrors the existing
"Codex reviews everything" practice) · `claude-code-sessions` (session **diff**, orphaned-task
detection, `/session-resume` — things claude-mem does not do, but 11 more skills) · `cc-inspect`
(audit the installed stack) · `skills-janitor` / `pulser` (skill hygiene as the set grows) ·
`docker-claude-code` (sandbox for trialling third-party plugins; pairs with Prism Scanner) ·
`agent-dotfiles` (**only if** teammates use Cursor/Copilot — see §4.5).

### 4.4 The seven cluster comparisons

#### Memory (13 systems) → claude-mem + git-committed files

| System | Storage | Team-shareable | Reported | Runtime cost |
|---|---|---|---|---|
| **claude-mem** *(installed)* | `~/.claude-mem/` SQLite+FTS5+Chroma, **user-global** | ❌ **none documented** | — | 5 hooks, already paid |
| **Cortex** | SQLite default; Postgres+pgvector for teams | ⚠️ shared Postgres, never git | LongMemEval **98.2% R@10 / 0.915 MRR**, LoCoMo, BEAM; 2 arXiv papers | 9 hooks; embedding model **+ FlashRank cross-encoder resident per session**; consolidation every 6h; **the autonomous wiki worker is a Claude agent on a schedule — recurring token spend** |
| **axme-code** | `.axme-code/` markdown+YAML **in repo** | ✅ if un-gitignored | LongMemEval 89.2% E2E / 97.8% R@5; ToolEmu 100% | 2 hooks — but **alpha, 14★, 1 fork** |
| **knowledge-graph** | `.kg/` + per-module CLAUDE.md | ✅ | — | bash+jq, ~4 hooks |

Cortex wins on measured retrieval. It was still rejected: recurring background LLM spend competes
with the experiment budget these cost tools exist to protect, two models sit resident alongside
torch 2.8 and the 4B/8B locals on a box whose `/home` is near-full, and it *still* cannot git-commit
memories. Its degraded mode disables vector search — removing exactly what made it win.

**claude-mem's own docs confirm it is user-global with no team sharing**, so the team layer is
plain git-committed files: ADRs, rules, workflows, run records. That is the architecture axme-code
and GAAI both converge on, at zero runtime cost.

Cortex and axme-code are instead tracked as **paper baselines** — both publish on LongMemEval and
LoCoMo, the benchmarks already in `experiments/`. See `docs/related-work-agent-memory.md`.

#### Codebase index/search (6) → keep CodeGraph, **add serena**

| Tool | Method | Local | Verdict |
|---|---|---|---|
| CodeGraph *(installed)* | AST + knowledge graph, CLI | ✅ | keep |
| **serena** | LSP servers; symbol-level **edits** | ✅ | **add** — does what CodeGraph does *plus* accurate rename/move/inline. MCP, no hooks |
| claude-context | BM25+dense, AST chunking | ❌ | reject — **requires Milvus/Zilliz Cloud + an OpenAI embeddings key**. A paid cloud vector DB, for a team running its own Qdrant |
| reporecall | tree-sitter, local MiniLM, call-graph | ✅ | reject — best-evidenced claim in the whole survey (75.4% fewer tokens; 1,306-file repo, 30 pre-registered queries, no model calls) but adds 2 hooks and overlaps CodeGraph |

#### Hook-heavy enforcement (9) → none installed, two ideas ported
See §4.2. `VibeGuard` (13 hooks / 88 rules), `cc-discipline`, `SpecLock`, `cc-agents-md`, and the
temporal plugins add no capability the 4 guards + rules do not cover, at real per-call cost.

#### Cost/session analytics (13) → three installed
`ccusage` / `cc-cost` / `PRISM` answer *how much*, *why this session*, and *is the CLAUDE.md
working*. PRISM specifically earns its slot because this setup ships a routing CLAUDE.md — it
measures whether that file is being re-read wastefully. The other 10 are redundant.

#### HUDs (5) → claude-hud
A HUD is **not a GUI** — it renders through the native statusline API, in the terminal, at zero
hook cost. Only one statusline can be active, so claude-hud excludes cc-hud, craft-statusline,
cc-tempo, and codachi.

#### Config/routing managers (9) → claudebase covers it
`claude-snapshot` (tar.gz export/restore) is redundant against claudebase's git history + profiles.
`claude-overlay`, `cc-switch`, `claude-code-router`, `gemini-bridge`, the AWS gateway and
`systemprompt-template` are redundant against **omniroute**, already installed. `LynxPrompt` is a
hosted platform for the same job.

**The one genuine gap: `agent-dotfiles`** — it syncs rules and skills *cross-tool* (Cursor, Copilot,
Codex), a different axis from claudebase's cross-machine sync. Adopt only if a teammate uses a
non-Claude agent. It handles rules and skills only — not hooks, not MCP.

#### Opinionated full setups (17) → patterns salvaged, nothing installed
See §4.2. Spartan's docs do not confirm a non-overwriting install; blueprint explicitly warns not
to run Claude Code inside its repo.

### 4.5 Rejected blocks (~320)

| Block | ~n | Reason |
|---|---|---|
| macOS/Windows-only GUIs, statuslines, menu-bar apps | 35 | **Wrong platform** — this is Linux |
| Domain verticals (SEO ×6, marketing/social ×12, PM/product ×8, finance/crypto ×5, healthcare, embedded, packaging, gaming, life-science) | ~50 | Out of scope |
| Multi-agent orchestration (vibe-kanban, KANBAII, amux, Watchfire, ORCH, Ruflo, nexus-agents, myclaude, oh-my-claudecode, paco, production-grade, great_cto, Axiom, AuraKit, wshobson/agents, idea-factory, claude-code-templates) | 17 | Team-size mismatch / too complex |
| Opinionated full setups | 17 | Would overwrite this configuration |
| Memory systems | 13 | claude-mem + git chosen; Cortex on runtime cost |
| Cost dashboards beyond the three | 10 | Redundant |
| Hook-heavy enforcement | 9 | Hook budget; two ideas ported as rules |
| Config/routing managers | 8 | claudebase + omniroute |
| Codebase index/search | 4 | CodeGraph + serena |
| Notification/cosmetic | ~12 | Out of scope (except `notify`) |
| Discovery registries, misc utilities | ~20 | Marginal |

---

## 5. Documentation architecture

`CLAUDE.md` loads **in full, every session** — the toolkit's own `/memory-bank` caps it at 200
lines for that reason, and PRISM exists because re-read cost is a measurable, frequently enormous
waste. So depth does not go there.

```
CLAUDE.md              89 lines — stack, storage invariants, ROUTING TABLE, "do not reach for"
.claude/rules/         13 files — always-on standards; storage-invariants is path-scoped
docs/workflows/         5 recipes — read only when the task matches
docs/adr/                        — decisions, Nygard format, supersede rather than edit
```

The load-bearing part is the **routing table**: intent → destination. "A metric moved unexpectedly"
→ `docs/workflows/debug-a-metric.md` → `/bisect`, then `/logic-review`. Paired with a **"do not
reach for"** table naming each excluded overlap and why, so the six reversals in §2 are not quietly
re-added by the next person.

This is the 3-layer pattern spartan-ai-toolkit arrived at independently: index always loaded,
topics on demand, archive grep-only.

---

## 6. Statistics

| Population | Reviewed | In use | Rate |
|---|---|---|---|
| Local toolkit | 405 | 176 | 43.5% |
| External catalog | 344 | 16 (10 installed + 6 ported as lessons) | 4.7% |
| **Combined** | **749** | **192** | **25.6%** |

External also carries 7 "evaluate later" with explicit triggers. The local rate is high because the
toolkit is generic and cheap to adopt; the external rate is low because most of that catalog is
platform-specific, domain-specific, or duplicates something already installed.

---

## 7. Critical questions

**Which components support GitHub config storage?** All of it — everything selected is plain files
in `.claude/`, `docs/`, and `.mcp.json`, committed to the umbrella repo and reviewed in PRs.
claudebase adds cross-machine sync with profiles and secret scanning. The `github` MCP server
closes the issue/PR loop.

**How many integrate with experiment tracking?** 14, in three tiers: core (`experiment-tracker`,
`model-evaluator`, `mlops-engineer`), feeders (`perf-profiler`, `memory-profiler`,
`analytics-reporter`, `test-results-analyzer`, `benchmarking-specialist`, `data-visualization`),
and persistence/reasoning (`data-science.json` MCP, `llm-cost.json`, `/checkpoint`, the
`experiment-discipline` skill, run records in git). Caveat unchanged: `experiment-tracker` is a
convention layer, not a tracking-server client.

**Hook conflicts?** Yes, and they drove the design. Beyond the four in the first pass, the decisive
one is that claude-mem and headroom already occupy SessionStart, PreToolUse, PostToolUse and Stop —
so only 4 guards were installed, none on session events. And the shipped scripts were broken (§2).

**Installation order?** `CLAUDE.md` → rules → MCP → commands/agents/skills → hooks → **Prism
Scanner** → other externals. Prism Scanner before the rest is the point: it scans the third-party
code you are about to let run in your session.

**Team-wide shared configuration?** The umbrella repo's `.claude/` (PR-reviewed), `CLAUDE.md` +
routing table, 13 rules, `docs/workflows/`, `docs/adr/`, run records under `experiments/runs/`, and
claudebase for personal cross-machine layers.

---

## 8. What was built, and what was verified

### Installed in this repo

```
CLAUDE.md                    89 lines, routing table + "do not reach for"
.claude/rules/               13  (11 copied + storage-invariants + evidence-discipline)
.claude/commands/            34
.claude/agents/              35
.claude/skills/              17  (15 copied + qdrant-ops + experiment-discipline)
.claude/hooks/scripts/        4  (rewritten against the documented stdin contract)
.claude/settings.json            hook registration, ${CLAUDE_PROJECT_DIR}-relative
.mcp.json                     8  servers, ${ENV} references only, no literal tokens
docs/workflows/               5  recipes
docs/related-work-agent-memory.md
setup/install-external.sh        the 10 adopted externals, Prism Scanner first
```

### Verified, with results

| Check | Result |
|---|---|
| `secret-scanner` on a pending Write containing `sk-proj-…` | **DENY** — "line 1: OpenAI/Anthropic-style key" |
| `secret-scanner` on `os.environ["OPENAI_API_KEY"]` | silent, exit 0 |
| `commit-guard` on `"Fixed the thing."` | **DENY** — 3 findings (no type, trailing period, capitalised) |
| `commit-guard` on `"feat(retriever): add on-disk qdrant persistence"` | silent, exit 0 |
| `pre-push-check` on `git push --force origin main` | **DENY** |
| `pre-push-check` on `git push origin HEAD:main` | **ASK** (refspec destination parsed correctly) |
| `pre-push-check` on `git push origin feat-fluxmem` | allowed |
| `lint-fix` on a messy `.py` | ruff 0.15.0 fixed formatting **and removed the unused import** |
| `.mcp.json` | valid JSON, 8 servers, zero literal credentials |

`pre-push-check` initially flagged `feat-fluxmem` because it fell back to the *current* branch
instead of reading the explicit refspec — caught by the test above and fixed before commit.

### Not executed, and why

- ~~No GitHub repo created~~ — **done.** `AshwinKM103/C-AIMMS`, private, pushed.
- ~~No external tools installed~~ — **done**, in isolated environments:
  `~/.venvs/prism-scanner`, `~/.venvs/cc-tools` (prism-cc + cc-cost — see the collision note
  below), `uv tool install serena-agent`, plus `ccusage`/`autosearch-ai` via `npx`. The lightmem
  conda env was never touched.
- ~~Prism Scanner has not graded anything yet~~ — **done, see below.**

### A real bug found while installing: prism-cc and prism-scanner collide

Both PyPI packages install a top-level module named `prism` with *different* CLI entry points
(`prism.cli:app` vs `prism.cli:main`). Installing both into one venv silently merges their
`site-packages/prism/` directories — confirmed by inspecting the merged package, which contained
files from both (`scanner.py`/`rules/` alongside `dashboard.py`/`advisor.py`), with whichever
installed last winning the `prism` command. **Neither tool is safe to trust from a shared venv.**
Fixed: separate venvs (`~/.venvs/prism-scanner`, `~/.venvs/cc-tools`) with thin wrapper scripts at
`setup/bin/{prism,prism-scanner,cc-cost}` so both are callable without the collision.

### A wrong command caught before it shipped: serena's MCP entry

`.mcp.json`'s serena entry used `--context ide-assistant`, invented from memory while writing the
first draft — that context does not exist (`serena context list`: `agent`, `claude-code`, `codex`,
`ide`, `desktop-app`, …). Fixed to `claude-code`, the context built for this exact tool, and
verified by starting the server directly: it loads 52 tools and activates the `lightmem` project
correctly.

### Plugin installs — non-interactive after all, and security-scanned

The earlier assumption that `/plugin install` needs an interactive session was wrong — the `claude
plugin` CLI installs and enables marketplace plugins non-interactively. All four adopted plugins
(claudebase, logic-lens, claude-hud, notify) are installed and enabled.

Prism Scanner (step 0, as designed) graded all four **F (Critical) × 3, C (Caution) × 1** — a
19-finding total across CRITICAL/HIGH/MEDIUM severities. Every one was manually verified against
source and found to be a false positive: pattern matching without taint context (a doc string or a
test-grading regex containing a suspicious substring) and, most notably, a unicode *sanitizer* in
claude-hud flagged for containing the injection characters it strips. claude-hud's `credential_read`
finding was checked further than the rest — confirmed zero network primitives and zero runtime
dependencies anywhere in its source, so even the flagged local auth-display read has no path to
exfiltrate anything. Full finding-by-finding writeup and raw scanner JSON:
`docs/security-scans/README.md`.

### Config audit — run against the real code

Auditing `LightMem/` for the two footguns produced three findings:

1. **The team has already been bitten by this.**
   `memory_toolkits/memories/layers/naive_rag.py:147` carries the comment: *"Key: enable Qdrant
   on_disk; otherwise 'file created but points=0' occurs often"*. The LoCoMo harness
   (`experiments/locomo/`) sets `on_disk=True` explicitly in three places. The rule is not
   hypothetical — it documents a lesson already paid for, so the next person does not re-learn it.
2. **`embedding_model_dims` is not always 1024.** `experiments/locomo/retrievers.py:26` uses
   **384**. The rule and the `qdrant-ops` skill were corrected: record the dims actually used,
   never assume the default. A collection built at one width and queried at another degrades
   silently rather than erroring.
3. **`/tmp/qdrant` appears only as a config default**, never hardcoded in a harness — the exposure
   is anything that constructs `QdrantConfig` without overriding `path`, not existing experiments.

A fourth trap worth naming: `src/lightmem/configs/retriever/bm25.py` is fully populated
(`on_disk`, `use_stemming`, `lowercase`) while the retriever it configures is an empty file. The
configuration surface looks live; the implementation is not there.

### Open items

1. Run `setup/install-external.sh` (Prism Scanner first).
2. `gh repo create <org>/C-AIMMS --private --source=.` and push.
3. `/sync-setup` in claudebase with profile `c-aimms`.
4. ~~Grep the experiment configs for the Qdrant footguns~~ — **done, see below.**
5. Add a pointer line in `LightMem/CLAUDE.md` to this umbrella config. (Left untouched otherwise —
   it correctly documents the user-scope stack.)
6. `gh extension install yahsan2/gh-sub-issue` — installed. Only load-bearing if `ccpm` is later
   adopted (currently "evaluate later," not installed).
7. **Project-scoped `.mcp.json` servers require one-time interactive approval** — this is documented,
   intentional behavior (`claude mcp list` shows `⏸ Pending approval (run claude to approve)` for
   all 8 until someone runs `claude` interactively in this project and accepts). This is *why* they
   don't appear in the VS Code extension's `/mcp` panel — pending servers aren't shown there at all,
   only connected/failed/needs-auth ones. Approve by running `claude` (CLI or a fresh VS Code
   session) in this project once. To pre-approve for the whole team without the per-person prompt,
   set `enableAllProjectMcpServers: true` in `.claude/settings.json` — not done here, since that
   trades away the human-review step the approval gate exists for; left as your call.
8. `github` MCP server warns `Missing environment variables: GITHUB_PERSONAL_ACCESS_TOKEN` — expected,
   the `.mcp.json` value is `${GITHUB_PERSONAL_ACCESS_TOKEN}` by design (never a literal). Export it
   in your shell before that server will connect.
9. **claude-mem's worker had never actually run on this machine.** Its `SessionStart` hook
   (matcher `startup|clear|compact`) starts a background daemon via `bun-runner.js` ->
   `worker-service.cjs --daemon`, and Bun was not installed anywhere on this system — confirmed
   (`command -v bun`: nothing; no `~/.bun/`; no shell-rc reference). The hook fails
   non-blocking, so sessions started fine and nothing looked wrong, but `~/.claude-mem/` (the
   SQLite+Chroma store) did not exist until Bun was installed and the hook run end-to-end just now.
   **This changes a claim in §2/§C of this analysis**: claude-mem was described as "already
   installed... already owns cross-session memory," which was true for "installed" and false for
   "owns" — it had never captured anything. Fixed: `curl -fsSL https://bun.sh/install | bash`,
   added to both `~/.bashrc` (the installer's default) and `~/.profile` (for GUI-launched VS Code,
   which does not reliably source `.bashrc` — see the VS Code PATH note above), then verified by
   running the hook's exact command by hand: `{"continue":true,"status":"ready"}`, and confirmed the
   daemon process (`bun .../worker-service.cjs --daemon`) is now actually resident.
10. **The `autosearch` MCP entry was broken and has been removed.** `autosearch-ai`'s installer
   (npx step, §5) auto-registers a project-scoped MCP server via `~/.claude.json`, separate from
   this repo's `.mcp.json`. It failed to connect: its own dependency resolution pulled in a package
   named `mcp` at version `2.0.0` with no `fastmcp` submodule — inconsistent with the real MCP
   Python SDK's API, which `autosearch`'s own code imports (`from mcp.server.fastmcp import
   FastMCP`). This is a bug in `autosearch-ai`'s packaging, not in this repo's config. The
   `autosearch` CLI itself is unaffected and confirmed working (27/40 channels ready, arXiv/PubMed/
   OpenAlex/DBLP all keyless-ready) — only its bundled MCP server was broken. Removed via
   `claude mcp remove autosearch` rather than hand-patched, since patching a third party's
   dependency pin isn't something to maintain here.
7. `serena project create` also generated `LightMem/.serena/{project.yml,project.local.yml}`.
   `project.yml` is meant to be versioned (per its own header comment); left uncommitted since
   `LightMem/` is a separate repo with its own review process — a decision for whoever next
   commits there, not made unilaterally here.
