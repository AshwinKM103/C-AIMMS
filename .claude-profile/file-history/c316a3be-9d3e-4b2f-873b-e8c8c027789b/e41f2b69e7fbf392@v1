# Related Work — Agent Memory Systems

Surveying the Claude Code ecosystem for tooling surfaced ~13 persistent-memory systems. Several
are not tools we would install — they are **prior art for C-AIMMS**, and two publish numbers on
the exact benchmarks in `LightMem/experiments/`.

**All figures below are self-reported by their authors. We have reproduced none of them.**
Treat them as claims to verify, not as a leaderboard.

## Systems reporting on our benchmarks

### Cortex (`cdeust/Cortex`, MIT)
Neuroscience-grounded memory: thermodynamic heat decay (`e^(-λt)`, computed at read time), a
predictive-coding write gate that stores only prediction errors, reconsolidation on retrieval,
emotional-valence encoding, pattern separation, active forgetting. 36 cited mechanisms, 97-reference
bibliography, two arXiv-ready papers ("Thermodynamic Memory vs. Flat-Importance Stores",
"Stage-Aware Context Assembly").

Retrieval fuses five signals — vector similarity, BM25, trigram, heat, recency — reranked by a
cross-encoder. Hierarchical "stage-aware context assembly" partitions long conversations by
temporal stage.

| Benchmark | Reported |
|---|---|
| LongMemEval (n=500) | **98.2% Recall@10, 0.915 MRR** |
| BEAM (10M tokens) | 0.471 MRR temporal-stage detection, vs 0.429 with oracle topic labels |
| Long-context stability | flat retrieval degrades 0.500 → 0.466 MRR (500K → 1M tokens); assembler holds +0.07 |
| LoCoMo | n=1986, reported |

**Why it matters to us:** it ships a reproducible harness — per-mechanism ablation campaigns with
code SHAs, dirty flags, manifests, and per-row JSON in `benchmarks/results/`. That is a directly
usable comparison point, and the ablation methodology is worth borrowing regardless.

**Why we did not install it:** 9 lifecycle hooks; embedding model **and** FlashRank cross-encoder
resident per session; consolidation every 6h; and its "autonomous wiki cycle" is a Claude agent on
a schedule — recurring background token spend. Its degraded mode disables vector search, removing
what makes it competitive. Team sharing needs a shared PostgreSQL, not git.

### axme-code (`AxmeAI/axme-code`, MIT, alpha)
Local-first project memory in `.axme-code/` as human-readable markdown + YAML: an "oracle"
(stack/structure/patterns/glossary), decisions with **enforce levels** (required / advisory /
optional), feedback and pattern memory, safety rules, backlog, per-session metadata, and an
append-only worklog. A background auditor re-reads the transcript after session close to recover
memories the agent forgot to save.

| Benchmark | Reported |
|---|---|
| LongMemEval E2E | **89.2%** (vs Mastra 84.23–94.87%, Zep 71.2%, Mem0 49.0%) |
| LongMemEval R@5 | 97.8% |
| ToolEmu safety | 100% accuracy, 0% false-positive rate |
| Token efficiency | ~10K tokens per correct answer; 2 LLM calls/question (reader + judge) vs "dozens" |

**Why it matters to us:** its storage architecture — plain markdown/YAML in the repo, decisions
carrying enforcement levels — is the design our own `.claude/rules/` + `docs/adr/` layer converges
on independently. Its Mastra/Zep/Mem0 comparison row is a ready-made baseline table.

**Why we did not install it:** alpha, 14 stars, 1 fork. Too early to make a team dependency.

## Other systems in the space

| System | Approach | Notable |
|---|---|---|
| **claude-mem** (installed here) | SQLite + FTS5 + Chroma at `~/.claude-mem/`, AI-compressed observations, 5 hooks, progressive disclosure (~10× token savings) | 35.9k★. **User-global; no repo or team sharing** — this is why our team layer is git-committed files |
| **Hindsight** (Vectorize) | Biomimetic retain / recall / reflect, 4 parallel retrieval strategies | self-hosted or cloud, MIT |
| **MUSE** (myths-labs) | Pure-Markdown memory OS, 48 skills, 8 roles with permission isolation | no database |
| **knowledge-graph** (hilyfux) | Git-native: `.kg/events.jsonl`, per-module `CLAUDE.md` nodes with `@` references, co-change inference | bash + jq, ~3ms/event, claims ~500–900 token baseline |
| **claude-supermemory** | Hosted persistent memory, user-profile injection | 2.3k★ |
| **memory-bank** (Nagendhra-web) | 3-tier progressive loading, branch-aware, session diffing | claims 60–80% token reduction |
| **cog** (marciopuga) | Plain-text CLAUDE.md conventions, self-reflection, foresight | zero dependencies |
| **claude-memory-bridge** | Cross-project namespaces, shared layer between global and project memory | MCP, no database |
| **reporecall** | Tree-sitter AST (22 langs), hybrid keyword+vector, call-graph traversal, local MiniLM embeddings | measured 75.4% fewer tokens than reading whole files (1,306-file repo, 30 pre-registered queries, no model calls) |

## Observations worth carrying into the paper

1. **Nobody in this cohort git-commits their memory.** Cortex explicitly declines to; claude-mem is
   user-global; Hindsight and supermemory are services. axme-code and knowledge-graph put files in
   the repo but gitignore them by default. Team-shared, reviewable agent memory is an open gap.
2. **Reported LongMemEval numbers are not comparable across systems** — different retrieval budgets,
   different reader models, different judge protocols. Cortex reports R@10, axme-code reports R@5
   and E2E. Any comparison table we publish has to normalise this or say plainly that it cannot.
3. **Token-per-correct-answer is becoming a reported axis**, not just accuracy (axme-code's ~10K/answer,
   reporecall's 75.4% reduction, Cortex's stage-assembly efficiency). Consistent with LightMem's own
   efficiency framing — and a reason to record token cost in every run record.
4. **Retrieval-quality claims cluster far above end-to-end claims** (97–98% R@k vs 89% E2E). The gap
   between "retrieved the right memory" and "answered correctly" is where the interesting work is.

## Provenance
Compiled 2026-08 while auditing `awesome-claude-code-toolkit` (344 external entries) for tooling.
Claims taken from each project's own documentation, fetched directly. Nothing here is independently
verified.
