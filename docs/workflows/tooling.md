# CLAUDE.md

Guidance for Claude Code when working in this repository (`LightMem`).

## Review process

**Codex reviews everything Claude produces in this repo.** Write code and
commits as if a second, independent reviewer will read them closely:
- Prefer clear, self-explanatory code over cleverness; don't rely on context
  that only exists in this conversation.
- Explain non-obvious decisions in commit messages, not just in chat.
- Don't leave partial/half-finished changes, TODOs without context, or
  unexplained workarounds — Codex will flag these.
- If Codex leaves review comments, treat them as you would a human
  reviewer's: address them or explain why not, don't silently dismiss them.

## Installed plugins

Five Claude Code plugins are installed (user scope, so available in every
session, not just this repo). Verify with `claude plugin list`.

### claude-mem (`claude-mem@thedotmack`)
Persistent cross-session memory. Captures what happens during a session,
compresses it, and injects relevant context back into future sessions
automatically via hooks (`SessionStart`, `PreToolUse`, `PostToolUse`, `Stop`) —
no manual invocation needed for the core behavior. Useful skills to invoke by
name when needed:
- `mem-search` — search stored memories by keyword/semantic similarity.
- `timeline-report`, `standup`, `weekly-digests` — summarize past work.
- `make-plan`, `pathfinder`, `smart-explore` — planning/exploration aids that
  draw on stored context.
Restart the session after install/updates for memory injection to take effect.

### headroom (`headroom@headroom-marketplace`)
Compresses tool outputs, logs, files, and RAG chunks before they reach the
model (fewer tokens, same answers). The plugin itself is just two hooks
(`SessionStart`, `PreToolUse`) that run `headroom init hook ensure` to make
sure a local Headroom runtime is up — the actual engine is the `headroom` CLI,
installed in an isolated venv at `~/.venvs/headroom` and symlinked to
`~/.local/bin/headroom` (kept out of the conda base env deliberately — do not
`pip install headroom-ai` into a shared env). When active, Headroom runs a
local proxy that intercepts Claude Code's API traffic (`ANTHROPIC_BASE_URL`
rewritten to `127.0.0.1`) to compress it before forwarding to Anthropic.
Check `headroom status` if compression seems inactive.

### claude-code-setup (`claude-code-setup@claude-plugins-official`)
Read-only meta-skill: analyzes this codebase and recommends Claude Code
automations (hooks, skills, MCP servers, subagents, slash commands) tailored
to it. Invoke with prompts like "recommend automations for this project" or
"what hooks should I use?" when setting up new workflows here. It does not
modify anything itself — treat its output as suggestions to implement
manually.

### task-observer (skill, not a marketplace plugin — `~/.claude/skills/task-observer/`)
From `rebelytics/one-skill-to-rule-them-all` — installed manually since this
project has no `.claude-plugin/marketplace.json` to hook into `/plugin`.
Watches multi-step work sessions for patterns worth turning into reusable
skills: repeated corrections, workflow insights, methodology worth keeping.
Its own description asks to be invoked at the start of every task-oriented
session, but description-matching alone isn't reliable — so: **at the start
of any multi-step task in this repo, invoke the `task-observer` skill
explicitly** rather than assuming it will trigger on its own. Observations
feed into periodic reviews (see its `references/weekly-review.md`) where you
decide what becomes a new skill.

### omniroute (MCP server, not a plugin)
`diegosouzapw/OmniRoute` — a local AI gateway/router (160+ providers, token
compression, auto-fallback), installed via `npm install -g omniroute` and
wired in as an MCP server (`claude mcp add-server omniroute --type http --url
http://localhost:20128/api/mcp/stream`) rather than through `/plugin`, since
it isn't a Claude Code plugin. Run `omniroute` to start its local server
(dashboard at `http://localhost:20128`) before its MCP tools will respond.
⚠️ Several unrelated GitHub repos also named "omniroute" are scam/token-bait
clones with documented security issues — only `diegosouzapw/OmniRoute` (npm
package `omniroute`, confirmed via `npm view omniroute repository`) is the
one installed here.

## CodeGraph (no Claude Code plugin exists for this — CLI/MCP tool only)

This repo has a CodeGraph index (`.codegraph/`, `codegraph.db`) built with the
`codegraph` CLI (`~/.local/bin/codegraph`) — a standalone code-intelligence
and knowledge-graph tool, not a Claude Code plugin (checked the official
marketplace; nothing matches beyond `coderabbit`, which merely mentions
"codegraph relationships" as a term, unrelated to this tool).

Useful commands when navigating or reasoning about impact in this codebase:
- `codegraph query <search>` — search for symbols.
- `codegraph explore <query...>` — relevant symbols' source + call paths in
  one shot.
- `codegraph node <name>` — one symbol's source + caller/callee trail.
- `codegraph callers <symbol>` / `codegraph callees <symbol>` — call graph
  in either direction.
- `codegraph impact <symbol>` — what's affected by changing a symbol.
- `codegraph affected [files...]` — test files affected by changed source.
- `codegraph sync` — refresh the index after making changes (index can go
  stale; re-run before relying on it for a large refactor).

CodeGraph also exposes an MCP server (`codegraph install`) that isn't wired
into this Claude Code session yet — prefer the CLI commands above unless the
MCP tools get installed explicitly.

## Local models and environment

- Conda env: `/mnt/ssd/users/durgesh/conda-envs/lightmem` (Python 3.11,
  `pip install -e .` done). Activate with:
  `conda activate /mnt/ssd/users/durgesh/conda-envs/lightmem`
- Local model paths are in `.env` at the repo root (gitignored):
  `LLMLINGUA_MODEL_PATH`, `EMBEDDING_MODEL_PATH`, `QWEN3_4B_INSTRUCT_PATH`,
  `LLAMA31_8B_INSTRUCT_PATH`.
- All large models/checkpoints live under `/mnt/ssd/users/durgesh/`, not
  `/home` — `/home` is on a smaller, near-full disk; don't download large
  artifacts there.
