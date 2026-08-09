#!/usr/bin/env node
/**
 * PreToolUse(Bash) — enforce Conventional Commits.
 *
 * Adapted from awesome-claude-code-toolkit hooks/scripts/commit-guard.js (Apache-2.0),
 * fixed to read stdin, to look at tool_input.command, and to emit the PreToolUse
 * hookSpecificOutput format (the original's top-level {"decision":"block"} is the
 * PostToolUse shape and is ignored on PreToolUse).
 *
 * This is load-bearing: /changelog and release-manager parse conventional commits.
 */
const TYPES = "feat|fix|docs|style|refactor|perf|test|chore|ci|build|revert|exp";
const CONVENTIONAL = new RegExp(`^(${TYPES})(\\([^)]+\\))?!?: .+`);

function extractMessages(command) {
  // -m "msg" / -m 'msg' / --message="msg", possibly repeated.
  const out = [];
  const re = /(?:-m|--message)[= ]\s*(["'])([\s\S]*?)\1/g;
  let m;
  while ((m = re.exec(command)) !== null) out.push(m[2]);
  return out;
}

function main(raw) {
  let input;
  try {
    input = JSON.parse(raw || "{}");
  } catch {
    process.exit(0);
  }

  const command = ((input.tool_input || {}).command || "").trim();
  if (!/\bgit\b[^\n]*\bcommit\b/.test(command)) process.exit(0);
  // Amending metadata or committing with an editor/file — nothing to validate here.
  if (/--amend\b/.test(command) && !/-m|--message/.test(command)) process.exit(0);
  if (/-F\b|--file\b/.test(command)) process.exit(0);

  const messages = extractMessages(command);
  if (messages.length === 0) process.exit(0);

  const subject = messages[0].split("\n")[0];
  const errors = [];

  if (!CONVENTIONAL.test(subject)) {
    errors.push(`does not match "type(scope): description" — type must be one of ${TYPES.replace(/\|/g, ", ")}`);
  }
  if (subject.length > 72) errors.push(`subject is ${subject.length} chars (max 72)`);
  if (subject.endsWith(".")) errors.push("subject must not end with a period");

  const desc = subject.replace(new RegExp(`^(${TYPES})(\\([^)]+\\))?!?: `), "");
  if (desc && desc[0] === desc[0].toUpperCase() && /[A-Za-z]/.test(desc[0])) {
    errors.push("description should start lowercase");
  }

  if (errors.length === 0) process.exit(0);

  process.stdout.write(
    JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason:
          `Commit message rejected: "${subject}"\n` +
          errors.map((e) => `  - ${e}`).join("\n") +
          `\nExample: feat(retriever): add on-disk qdrant persistence` +
          `\n(use type "exp" for experiment-only changes)`,
      },
    })
  );
  process.exit(0);
}

let raw = "";
process.stdin.on("data", (c) => (raw += c));
process.stdin.on("end", () => main(raw));
