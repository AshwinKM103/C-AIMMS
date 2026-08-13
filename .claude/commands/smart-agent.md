---
name: smart-agent
description: Invoke an agent with automatic model cost optimization (read-only agents use Haiku, reasoning agents use Opus)
examples:
  - smart-agent academic-researcher "summarize recent advances in transformers"
  - smart-agent code-reviewer "review the changes in src/fluxmem"
  - FORCE_MODEL=opus smart-agent search-specialist "find all references to FAISS"
  - VERBOSE=1 smart-agent Explore "quick search for embeddings"
---

**smart-agent** wraps the Agent tool with automatic model selection based on agent type and cost optimization.

## Model Assignment Strategy

| Agent Type | Model | Rationale |
|---|---|---|
| **Read-only** (academic-researcher, code-reviewer, data-researcher, error-detective, knowledge-synthesizer, performance-engineer, research-analyst, search-specialist, technology-scout) | Haiku 4.5 | Sufficient for analysis, research synthesis, and evaluation — no complex reasoning required. ~10× cheaper than Opus. |
| **General-purpose** (most agents) | Sonnet 5 | Best balance of cost and capability for code generation, refactoring, design. |
| **Heavy reasoning** (llm-architect, ml-engineer, autoresearch-agent, multi-agent-coordinator) | Opus 5 | Complex architectural decisions, experiment design, orchestration — benefit from maximum reasoning depth. |

## Usage

```bash
# Use model-router to detect recommended model for an agent
node .claude/hooks/scripts/model-router.js academic-researcher
# Output: claude-haiku-4.5

# Check reasoning
VERBOSE=1 node .claude/hooks/scripts/model-router.js code-reviewer
# Output: [model-router] code-reviewer (read-only) → claude-haiku-4.5
```

## Override Model

```bash
# Force a specific model (e.g., for complex queries to a normally-cheap agent)
FORCE_MODEL=opus node .claude/hooks/scripts/model-router.js academic-researcher
# Output: claude-opus-5
```

## Cost Savings Example

For 10 subagent invocations across a session:
- **Before**: 10 agents × Opus ($0.015/1K tokens) = $0.15 baseline
- **After**: 3 Haiku ($0.0008/1K) + 5 Sonnet ($0.003/1K) + 2 Opus ($0.015/1K) ≈ **60–70% savings**

## Agent Model Assignments

Verified by `grep "^model:" .claude/agents/*.md`:
- **9 Haiku agents** (read-only): code-reviewer, academic-researcher, data-researcher, error-detective, knowledge-synthesizer, performance-engineer, research-analyst, search-specialist, technology-scout
- **22 Sonnet-5 agents** (general): ai-engineer, code-review, context-manager, data-engineer, data-scientist, data-visualization, dependency-manager, documentation-engineer, dx-optimizer, general-purpose, mcp-developer, nlp-engineer, and others
- **4 Opus agents** (reasoning): autoresearch-agent, llm-architect, ml-engineer, multi-agent-coordinator
