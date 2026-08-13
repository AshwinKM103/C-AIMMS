# Documentation Workflows

When writing or updating project documentation, use these tools:

## Available Tools

### Plugins (installed via `/plugin install` in Claude Code)

- **`codebase-documenter`** — Auto-document entire codebase with inline comments and API docs. Best for generating comprehensive docstrings across multiple files.
- **`doc-forge`** — Generate documentation, API docs, and README maintenance. Good for end-to-end documentation generation.
- **`readme-generator`** — Smart README generation from project analysis. Use for auto-generating project READMEs.
- **`onboarding-guide`** — Create onboarding documentation for new developers. Use for developer onboarding docs.

### Commands (available as `/doc-gen`, `/api-docs`, `/onboard`)

Located in `.claude/commands/documentation/`:

- `/doc-gen` — Generate documentation from code
- `/api-docs` — Generate API docs from route handlers
- `/onboard` — Create onboarding guide for new devs

### Rules (always active)

- `.claude/rules/documentation.md` — Documentation style and conventions (inline comments, JSDoc/docstrings, README structure)

## Common Workflows

### Adding docstrings to a module

```
1. Use python-best-practices skill for style guidance
2. Either manually write docstrings (following .claude/rules/documentation.md)
3. Or use codebase-documenter plugin for auto-generation
```

### Generating API documentation

```
1. Use /api-docs command for route/endpoint documentation
2. Or use doc-forge plugin for comprehensive API doc generation
3. Result: OpenAPI/Swagger compatible docs
```

### Creating onboarding docs

```
1. Use /onboard command for quick onboarding guide
2. Or use onboarding-guide plugin for comprehensive setup
3. Result: New developer setup and navigation guide
```

### Updating README

```
1. Use readme-generator plugin to analyze project and generate README
2. Or manually update using documentation.md guidelines
3. Result: Well-structured README with installation, usage, contribution sections
```

## Notes

- All plugins must be installed via `/plugin install claude-code-toolkit@<name>` in Claude Code
- Documentation rules in `.claude/rules/documentation.md` are always active
- When in doubt, follow the evidence discipline in `.claude/rules/evidence-discipline.md` — show what you verified about the code before documenting
