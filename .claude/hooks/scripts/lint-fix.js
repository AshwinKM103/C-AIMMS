#!/usr/bin/env node
/**
 * PostToolUse(Write|Edit|MultiEdit) — autofix lint on the file just written.
 *
 * Adapted from awesome-claude-code-toolkit hooks/scripts/lint-fix.js (Apache-2.0),
 * fixed to read stdin / tool_input.file_path. Never blocks: a formatter failing must
 * not stop the session, so every path exits 0.
 *
 * Python is the priority path (ruff). The toolkit's separate post-edit-check.js was
 * deliberately NOT installed — it runs `ruff check` on the same write, duplicating work
 * and reporting diagnostics computed before this autofix runs.
 */
const { execFileSync } = require("child_process");
const path = require("path");
const fs = require("fs");

const FIXERS = {
  ".py": [
    ["ruff", ["check", "--fix", "--quiet"]],
    ["ruff", ["format", "--quiet"]],
  ],
  ".ts": [["npx", ["prettier", "--write", "--log-level", "warn"]]],
  ".tsx": [["npx", ["prettier", "--write", "--log-level", "warn"]]],
  ".json": [["npx", ["prettier", "--write", "--log-level", "warn"]]],
  ".md": [["npx", ["prettier", "--write", "--log-level", "warn"]]],
};

function main(raw) {
  let input;
  try {
    input = JSON.parse(raw || "{}");
  } catch {
    process.exit(0);
  }

  const filePath = (input.tool_input || {}).file_path || "";
  if (!filePath) process.exit(0);

  const cwd = input.cwd || process.cwd();
  // Never touch files outside the project.
  if (!path.resolve(filePath).startsWith(path.resolve(cwd))) process.exit(0);
  if (!fs.existsSync(filePath)) process.exit(0);

  const fixers = FIXERS[path.extname(filePath).toLowerCase()];
  if (!fixers) process.exit(0);

  const applied = [];
  for (const [cmd, args] of fixers) {
    try {
      execFileSync(cmd, [...args, filePath], { stdio: "pipe", timeout: 15000, cwd });
      applied.push(cmd);
    } catch {
      // Tool missing, or it exited non-zero on remaining unfixable findings. Either way,
      // this is advisory — keep going.
    }
  }

  if (applied.length) {
    process.stdout.write(`lint-fix: ${applied.join(" + ")} applied to ${path.basename(filePath)}`);
  }
  process.exit(0);
}

let raw = "";
process.stdin.on("data", (c) => (raw += c));
process.stdin.on("end", () => main(raw));
