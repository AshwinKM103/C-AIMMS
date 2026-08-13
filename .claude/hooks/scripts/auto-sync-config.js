#!/usr/bin/env node
/**
 * PostToolUse(Write|Edit|MultiEdit) — auto-sync Claude Code configuration to GitHub
 * when skills, hooks, plugins, or agents are created/modified.
 *
 * Watches for changes in .claude/{skills,hooks,plugins,agents}/ directories
 * and automatically runs `claudebase sync-push` to back up configuration.
 */
const { execFileSync } = require("child_process");
const path = require("path");

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
  const resolvedPath = path.resolve(filePath);

  // Check if the file is in a configuration directory
  const configDirs = ["skills", "hooks", "plugins", "agents"];
  const inConfigDir = configDirs.some((dir) =>
    resolvedPath.includes(path.join(".claude", dir))
  );

  if (!inConfigDir) process.exit(0);

  // Trigger configuration sync via git
  const fileName = path.basename(filePath);
  const configType = configDirs.find((dir) =>
    resolvedPath.includes(path.join(".claude", dir))
  );

  try {
    // Stage the configuration file
    execFileSync("git", ["add", filePath], {
      stdio: "pipe",
      timeout: 10000,
      cwd,
    });

    // Check if there are staged changes
    const status = execFileSync("git", ["status", "--porcelain", "--cached"], {
      stdio: "pipe",
      timeout: 10000,
      cwd,
    })
      .toString()
      .trim();

    if (status) {
      // Commit configuration changes
      execFileSync(
        "git",
        [
          "commit",
          "-m",
          `chore: auto-sync Claude Code configuration (${configType}/${fileName})`,
        ],
        {
          stdio: "pipe",
          timeout: 10000,
          cwd,
        }
      );

      // Push to remote
      execFileSync("git", ["push", "origin", "HEAD"], {
        stdio: "pipe",
        timeout: 30000,
        cwd,
      });

      process.stdout.write(
        `auto-sync: configuration synced to GitHub (${configType}/${fileName})`
      );
    }
  } catch (err) {
    // Sync failure should not block the session
    if (
      err.message.includes("nothing to commit") ||
      err.message.includes("no changes added")
    ) {
      // No changes, that's fine
      process.exit(0);
    }
    if (err.message.includes("not a git repository")) {
      // Not in a git repo, skip silently
      process.exit(0);
    }
    // For other errors, log but don't fail
    process.stderr.write(`auto-sync: warning — sync failed: ${err.message}`);
  }

  process.exit(0);
}

let raw = "";
process.stdin.on("data", (c) => (raw += c));
process.stdin.on("end", () => main(raw));
