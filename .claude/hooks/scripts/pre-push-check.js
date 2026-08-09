#!/usr/bin/env node
/**
 * PreToolUse(Bash) — guard git push.
 *
 * Adapted from awesome-claude-code-toolkit hooks/scripts/pre-push-check.js (Apache-2.0),
 * fixed to read stdin / tool_input.command and to emit PreToolUse decisions. The original
 * also matched "-f" as a substring, so any push whose branch name contained "-f" (e.g.
 * "feat-fluxmem") was flagged; this matches real flags only.
 *
 * Policy: force-push to a protected branch is denied outright; other force pushes and
 * direct pushes to a protected branch escalate to the user rather than being silently allowed.
 */
const { execFileSync } = require("child_process");

const PROTECTED = ["main", "master"];

function decide(decision, reason) {
  process.stdout.write(
    JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: decision,
        permissionDecisionReason: reason,
      },
    })
  );
  process.exit(0);
}

function currentBranch(cwd) {
  try {
    return execFileSync("git", ["branch", "--show-current"], { cwd, stdio: "pipe" }).toString().trim();
  } catch {
    return "";
  }
}

/**
 * Resolve which branch this push actually lands on.
 *   git push                      -> current branch
 *   git push origin              -> current branch (token is the remote)
 *   git push origin feat-x       -> feat-x
 *   git push origin HEAD:main    -> main (destination side of the refspec)
 */
function pushTarget(command, cwd) {
  const after = command.split(/\bpush\b/)[1] || "";
  const tokens = after
    .split(/\s+/)
    .filter(Boolean)
    .filter((t) => !t.startsWith("-"));

  if (tokens.length < 2) return currentBranch(cwd);

  const ref = tokens[1];
  const dest = ref.includes(":") ? ref.split(":").pop() : ref;
  return dest.replace(/^refs\/heads\//, "");
}

function main(raw) {
  let input;
  try {
    input = JSON.parse(raw || "{}");
  } catch {
    process.exit(0);
  }

  const command = ((input.tool_input || {}).command || "").trim();
  if (!/\bgit\b[^\n]*\bpush\b/.test(command)) process.exit(0);

  const cwd = input.cwd || process.cwd();
  const isForce = /(^|\s)(-f|--force|--force-with-lease)(\s|$)/.test(command);
  const target = pushTarget(command, cwd);
  const hittingProtected = PROTECTED.includes(target);

  if (isForce && hittingProtected) {
    decide(
      "deny",
      `Force-push to "${target}" is blocked. This rewrites history other people have pulled.\n` +
        `Open a PR from a feature branch instead, or force-push to a feature branch.`
    );
  }
  if (isForce) {
    decide("ask", `Force-push to "${target}" rewrites remote history. Confirm this branch is not shared.`);
  }
  if (hittingProtected) {
    decide(
      "ask",
      `Pushing directly to "${target}" bypasses review. The team workflow is code -> review -> merge; ` +
        `confirm this is intentional.`
    );
  }

  process.exit(0);
}

let raw = "";
process.stdin.on("data", (c) => (raw += c));
process.stdin.on("end", () => main(raw));
