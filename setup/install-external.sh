#!/usr/bin/env bash
# Install the external (third-party) components adopted for C-AIMMS.
#
# Nothing here runs automatically. Read it, then run it, or copy individual lines.
# Every one of these executes code inside your Claude Code session or shell, which is
# why step 0 is a security scan and why nothing is piped straight from curl to bash.
#
# IMPORTANT: activate the project env first if you want these in it, and be aware that
#   pip installs below go into whatever Python is active:
#     conda activate /mnt/ssd/users/durgesh/conda-envs/lightmem
#   The lightmem env has pinned, load-bearing versions (torch 2.8, transformers 4.57).
#   Prefer pipx / a separate tools env for the CLIs so you cannot disturb those pins.

set -euo pipefail

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

say "0. Security-scan third-party agent code BEFORE trusting it"
# Prism Scanner: 39+ rules, AST taint tracking, A-F grading for skills/plugins/MCP servers.
pipx install prism-scanner || pip install --user prism-scanner
# Then, for each candidate before/after install:
#   prism-scanner scan ~/.claude/plugins/<name>
# Record the grade in docs/CLAUDE_TOOLKIT_ANALYSIS.md. Do not skip this for the plugins below.

say "1. claudebase - config backup/sync to a private GitHub repo"
# Requires: gh (authenticated), jq, git, bash.
#   claude plugin marketplace add rohithzr/claudebase
#   claude plugin install claudebase@rohithzr
# Then, inside Claude Code:
#   /sync-setup                       # creates the private repo, first push
#   /sync-profiles list               # use profile: c-aimms
#   /sync-status                      # local vs remote
# Syncs .mcp.json and .claude/{settings,agents,commands,skills,hooks,rules,agent-memory}.
# Blocks API keys / PEM / Bearer tokens on push. Blocks push if another machine pushed since
# your last sync. Keeps the last 10 versions. Never syncs conversations, logs, or settings.local.json.

say "2. logic-lens - behavioural code review (L1-L9), full Python support"
#   claude plugin marketplace add hyhmrright/logic-lens
#   claude plugin install logic-lens@logic-lens-marketplace
# Optional .logic-lens.yaml to disable risk codes or ignore globs. Zero dependencies.
# Use on anything that computes a metric or mutates memory state.

say "3. serena - LSP-backed symbol search and safe symbol-level edits"
# Already referenced in .mcp.json; this installs the binary it calls.
uv tool install -p 3.13 serena-agent
#   serena init      # run once, from the repo root
# Gives rename / move / inline / safe-delete / find-referencing-symbols across Python via a real
# language server. Use it instead of grep-and-replace for repo-wide renames.

say "4. AutoSearch - literature search across arXiv / PubMed / OpenAlex / DBLP"
#   npx autosearch-ai
#   autosearch doctor     # verify channel health; most channels need no API key

say "5. Cost and session analytics - three tools, three different questions"
# ccusage  : how much have we spent?  (offline, reads local JSONL, zero API calls)
#   npx ccusage
# cc-cost  : why was THIS session expensive? (prompt-cache hit rate, tool distribution, --diagnose)
pipx install cc-cost || pip install --user cc-cost
#   cc-cost --diagnose
# PRISM    : is our CLAUDE.md actually working? (adherence, re-read cost, diff-format fixes)
pipx install prism-cc || pip install --user prism-cc
#   prism advise            # read-only; only `prism advise --apply` ever writes

say "6. claude-hud - terminal statusline (context %, quota, subagents, todos)"
#   claude plugin marketplace add jarrodwatts/claude-hud
#   claude plugin install claude-hud
#   /claude-hud:setup
# Renders through the native statusline API - no hooks, no GUI. Only ONE statusline can be
# active, so this excludes cc-hud / craft-statusline / cc-tempo / codachi.

say "7. notify - desktop notification when a long run finishes (Linux notify-send)"
#   claude plugin marketplace add ApurvBazari/claude-plugins
#   claude plugin install notify
# Duration filtering suppresses noisy short events; extracts repo + branch context.

cat <<'NOTE'

Deliberately NOT installed
--------------------------
  Cortex, axme-code, knowledge-graph, Hindsight, MUSE   -> claude-mem + git-committed files.
      Cortex additionally keeps an embedding model and cross-encoder resident per session and
      runs a scheduled Claude agent that spends tokens. Tracked as related work instead:
      docs/related-work-agent-memory.md
  claude-context, reporecall, codebase-graph            -> CodeGraph (indexed) + serena.
      claude-context also requires Milvus/Zilliz Cloud plus an OpenAI embeddings key.
  obey, VibeGuard, GateGuard, cc-discipline, SpecLock   -> hook budget. Two of their ideas are
      ported into .claude/rules/evidence-discipline.md instead.
  SuperClaude, Claudify, claude-forge, spartan, GAAI    -> would overwrite this setup. Four of
      their patterns are salvaged (model tiering, path-scoped rules, 3-layer memory,
      backlog-as-contract) without installing anything.
  ~35 macOS/Windows GUIs                                -> wrong platform; this is Linux.

Full reasoning: docs/CLAUDE_TOOLKIT_ANALYSIS.md
NOTE
