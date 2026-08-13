# Auto-Sync Configuration Hook

## Overview

The `auto-sync-config.js` hook automatically commits and pushes Claude Code configuration changes to GitHub whenever you create or modify:

- Skills (`.claude/skills/`)
- Hooks (`.claude/hooks/`)
- Plugins (`.claude/plugins/`)
- Agents (`.claude/agents/`)

This ensures your Claude Code configuration stays synchronized with your GitHub repository without manual intervention.

## How It Works

1. **Trigger**: When you use `Write`, `Edit`, or `MultiEdit` tools on any file in the configuration directories
2. **Detection**: The hook detects if the file path is in `.claude/{skills,hooks,plugins,agents}/`
3. **Sync**: If a configuration file was modified:
   - Stages the file with `git add`
   - Commits with message: `chore: auto-sync Claude Code configuration (type/filename)`
   - Pushes to the remote repository: `git push origin HEAD`
4. **Safety**: Errors are logged but never block your session

## Example Scenarios

### Creating a New Skill

When you create `.claude/skills/my-skill/skill.md`:

- The hook detects it's in the skills directory
- Stages and commits the change
- Pushes to remote
- Displays: `auto-sync: configuration synced to GitHub (skills/skill.md)`

### Modifying an Existing Hook

When you edit `.claude/hooks/scripts/my-hook.js`:

- The hook detects it's in the hooks directory
- Commits and pushes the update
- Configuration stays in sync with your repository

### Adding a Plugin

When you write `.claude/plugins/my-plugin/config.json`:

- The hook stages and commits the plugin configuration
- Automatically backs up to GitHub

## Configurable via settings.json

The hook is registered in `.claude/settings.json` under `PostToolUse`:

```json
{
  "matcher": "Write|Edit|MultiEdit",
  "hooks": [
    {
      "type": "command",
      "command": "node ${CLAUDE_PROJECT_DIR}/.claude/hooks/scripts/auto-sync-config.js",
      "timeout": 30
    }
  ]
}
```

## Disabling Auto-Sync

To temporarily disable auto-sync, you can:

1. **Comment out the hook** in `.claude/settings.json`
2. **Delete the hook script** if you no longer need it
3. **Use local settings** in `.claude/settings.local.json` to override

## Error Handling

The hook gracefully handles several error scenarios:

| Error                | Behavior                                    |
| -------------------- | ------------------------------------------- |
| No changes to commit | Silently exits (nothing to sync)            |
| Not a git repository | Silently exits                              |
| Push fails           | Logs warning but doesn't block your session |
| Missing git config   | Logs warning but doesn't block your session |

## Git Commit Messages

All auto-sync commits follow the conventional commit format:

```
chore: auto-sync Claude Code configuration (type/filename)
```

Where `type` is one of: `skills`, `hooks`, `plugins`, `agents`.

## Requirements

- Git must be configured with `user.name` and `user.email`
- Remote repository must be configured (typically `origin`)
- Write access to the remote repository

## See Also

- `.claude/settings.json` — Hook configuration
- `.claude/hooks/scripts/lint-fix.js` — Similar hook for auto-formatting
- Conventional commits — Message format reference
