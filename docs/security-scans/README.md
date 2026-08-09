# Plugin security scans

Prism Scanner (`pip install prism-scanner`, run via `setup/bin/prism-scanner`) results for the
four externally-installed plugins, with manual verification of every CRITICAL/HIGH finding —
scanner output is a starting point, not a verdict.

## Verdict

| Plugin | Scanner grade | Manual verdict | Findings verified |
|---|---|---|---|
| claudebase | F (Critical) | **safe to run** | 1 critical, 1 high — both false positives |
| logic-lens | F (Critical) | **safe to run** | 1 critical, 1 high — both false positives |
| claude-hud | F (Critical) | **safe to run** | 16 high, 2 medium — all false positives |
| notify | C (Caution) | **safe to run** | 1 medium — false positive |

**Every CRITICAL/HIGH finding across all four plugins was a false positive.** Full source review
below. Raw scanner JSON kept alongside for audit: `claudebase.json`, `logic-lens.json`,
`claude-hud.json`, `notify.json`.

## claudebase (`rohithzr/claudebase`)

| Finding | Location | What it actually is |
|---|---|---|
| CRITICAL P10 "Agent manipulation: urgency pressure" | `skills/sync-pull/SKILL.md:36` | Documentation for a `--yes`/`-y` CLI flag ("Skip confirmation prompt (auto-approve)") — the same pattern as `npx --yes`, not embedded instructions |
| HIGH S12 "Unsafe deserialization via pickle.load()" | `tests/bats/docs/list-links.py:3` | Test helper that loads Sphinx's own local `build/doctrees/environment.pickle` to check doc links. Not attacker-controlled input; scanner itself notes "Data source taint: unknown" |

**Operational check:** hooks are `SessionStart` (`diff-config.sh --quiet \|\| true`) and
`SessionEnd` (`sync-push.sh --auto \|\| true`). `sync-push.sh` calls `check_gh` first and exits 1
if unconfigured — confirmed the auto-push hook is a no-op until `/sync-setup` is deliberately run.

## logic-lens (`hyhmrright/logic-lens`)

| Finding | Location | What it actually is |
|---|---|---|
| CRITICAL P10 "Agent manipulation: urgency pressure" | `evals/real-world/probes/002-exit-code-swallowed/fixed.sh:14` | A code comment in an eval-harness worker function: `"skip check"` refers to its own idempotent-resume logic ("Emptiness is guarded at write time below, not by the skip check") |
| HIGH S13 "Persistence mechanism reference" | `scripts/grade-iteration.py:733` | A **grading regex** (`systemctl restart`) used to check whether the tool's own eval output correctly identified a bug in a *test fixture*. Not code that installs a systemd service |

**Operational check:** one hook, `SessionStart` (matcher `startup\|clear\|compact`), loads skill
instructions. No persistence mechanism exists anywhere in the actual runtime code.

## claude-hud (`jarrodwatts/claude-hud`)

| Finding | Location | What it actually is |
|---|---|---|
| 3× HIGH P5 "Unicode/zero-width prompt injection" | `src/render/index.ts:89`, `src/utils/sanitize.ts:12`, `tests/extra-cmd.test.js:30` | `sanitize.ts`'s docstring: "Strip ANSI escape sequences... and bidi overrides from an untrusted string... to prevent escape injection." This code **removes** the exact characters the scanner flagged — it is the defense, not the attack. `render/index.ts:89` computes terminal grapheme cell-width (legitimate rendering logic) |
| 2× MEDIUM P3 "Hex/Unicode escape obfuscation" | `tests/render.test.js:4055-4056` | Test assertions checking that rendered output does **not** contain bidi/zero-width characters (`assert.doesNotMatch(...)`) |
| HIGH M1 "Unexpected capability for git skill: credential_read" | `src/auth.ts` | Reads the `oauthAccount` block from `~/.claude.json` to display plan/account in the statusline (e.g. "Claude Max 20x"). Checks `ANTHROPIC_API_KEY` **presence only**, never its value. Defensively coded: sanitizes all values, caches with `0o600`/`0o700` permissions, atomic write-then-rename, `O_NOFOLLOW`. **Confirmed zero network primitives anywhere in `src/` and zero runtime dependencies in `package.json`** — this plugin cannot exfiltrate anything even in principle |
| 16× HIGH M7 "Publishable package contains source map file" | `dist/*.js.map` (120 files) | Standard TypeScript build artifacts (`.js.map`/`.d.ts.map`). A packaging-hygiene nag, inflated to 16 separate HIGH findings by counting each file plus category totals — not a vulnerability |

**Operational check:** no `hooks.json` — confirmed zero hooks, matching the plugin's advertised
design (native statusline API only).

## notify (`ApurvBazari/claude-plugins` → `notify`)

| Finding | Location | What it actually is |
|---|---|---|
| MEDIUM P8 "IOC match: GitHub raw shell script download" | `scripts/install-notifier.sh:26` | The official Homebrew install URL, printed inside a quoted `echo` as a copy-paste instruction *for the user* if Homebrew is missing (`# shellcheck disable=SC2016 # intentional: literal copy-paste instruction for user`) — never piped into a shell by the script. This whole branch is also gated on `[[ "$PLATFORM" = "macos" ]]`; **on this Linux machine it never executes** |

**Operational check:** no `hooks.json` — `install-notifier.sh` is invoked only via
`skills/setup/SKILL.md`, i.e. on explicit request, never automatically.

## Method

`prism-scanner scan <path> --show-trace --verbose` for the finding locations and evidence, then
each finding's surrounding source read directly (not taking the scanner's label at face value).
Two false-positive shapes recurred: **pattern matching without taint context** (a doc string or a
grading regex containing a suspicious substring) and **detection code mistaken for the attack it
detects** (a sanitizer flagged for containing the characters it strips).
