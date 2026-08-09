#!/usr/bin/env node
/**
 * PreToolUse(Write|Edit|MultiEdit|NotebookEdit) — block writes containing secrets.
 *
 * Adapted from awesome-claude-code-toolkit hooks/scripts/secret-scanner.js (Apache-2.0).
 * Three fixes over the original:
 *   1. Reads hook input from stdin, not process.argv[2] (the original always parsed
 *      "{}" and exited immediately, so it never scanned anything).
 *   2. Scans the *pending* content from tool_input, not the file on disk. The original
 *      read the existing file, so a brand-new secret being written was never seen.
 *   3. Emits hookSpecificOutput.permissionDecision, the format PreToolUse honors.
 */
const path = require("path");

const PATTERNS = [
  { name: "AWS Access Key", regex: /AKIA[0-9A-Z]{16}/ },
  { name: "AWS Secret Key", regex: /aws_secret_access_key\s*=\s*["']?[A-Za-z0-9/+=]{40}/i },
  { name: "GitHub Token", regex: /(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}/ },
  { name: "Private Key", regex: /-----BEGIN (RSA|EC|OPENSSH|PGP) PRIVATE KEY-----/ },
  { name: "Generic API Key", regex: /api[_-]?key\s*[:=]\s*["'][a-zA-Z0-9]{20,}["']/i },
  { name: "Slack Token", regex: /xox[bpors]-[0-9a-zA-Z-]{10,}/ },
  { name: "Database URL with password", regex: /(postgres|mysql|mongodb|redis):\/\/[^:\s]+:[^@\s]+@/ },
  { name: "JWT", regex: /eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}/ },
  // C-AIMMS-specific: this project holds OpenAI/DeepSeek/Anthropic and HF credentials.
  { name: "OpenAI/Anthropic-style key", regex: /\b(sk-ant-|sk-proj-|sk-)[A-Za-z0-9_-]{20,}/ },
  { name: "HuggingFace token", regex: /\bhf_[A-Za-z0-9]{30,}/ },
];

// Files where a placeholder is expected and a match is not a leak.
const ALLOWLIST = [/\.env\.example$/, /(^|\/)secret-scanner\.js$/, /(^|\/)CLAUDE_TOOLKIT_ANALYSIS\.md$/];

function pendingContent(toolName, ti) {
  if (!ti) return "";
  switch (toolName) {
    case "Write":
      return ti.content || "";
    case "Edit":
      return ti.new_string || "";
    case "MultiEdit":
      return (ti.edits || []).map((e) => e.new_string || "").join("\n");
    case "NotebookEdit":
      return ti.new_source || "";
    default:
      return ti.content || ti.new_string || "";
  }
}

function main(raw) {
  let input;
  try {
    input = JSON.parse(raw || "{}");
  } catch {
    process.exit(0);
  }

  const ti = input.tool_input || {};
  const filePath = ti.file_path || ti.notebook_path || "";
  if (!filePath) process.exit(0);
  if (ALLOWLIST.some((re) => re.test(filePath))) process.exit(0);

  const content = pendingContent(input.tool_name, ti);
  if (!content) process.exit(0);

  const lines = content.split("\n");
  const findings = [];
  for (const { name, regex } of PATTERNS) {
    for (let i = 0; i < lines.length; i++) {
      if (regex.test(lines[i])) findings.push(`line ${i + 1}: ${name}`);
    }
  }

  if (findings.length === 0) process.exit(0);

  const reason =
    `Blocked: potential secret in ${path.basename(filePath)}\n` +
    findings.map((f) => `  - ${f}`).join("\n") +
    `\nPut the value in .env (gitignored) and read it via os.environ, or use a ` +
    `\${ENV_VAR} reference if this is a config file.`;

  process.stdout.write(
    JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: reason,
      },
    })
  );
  process.exit(0);
}

let raw = "";
process.stdin.on("data", (c) => (raw += c));
process.stdin.on("end", () => main(raw));
