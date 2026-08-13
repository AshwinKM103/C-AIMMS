#!/usr/bin/env node
/**
 * Model Router — Automatic cost optimization for subagents.
 *
 * Agents that are read-only or primarily for exploration/search (no write operations)
 * are automatically downgraded from claude-opus-5 to claude-haiku-4.5 to reduce costs.
 *
 * To use: before invoking an agent, run:
 *   node .claude/hooks/scripts/model-router.js <agent-name>
 *
 * Environment variables:
 *   FORCE_MODEL: override detected model (e.g., FORCE_MODEL=sonnet-5 forces Sonnet)
 *   VERBOSE: log reasoning for model selection
 */

const fs = require("fs");
const path = require("path");

// Agents that only read and don't write — safe to downgrade to Haiku.
const READ_ONLY_AGENTS = new Set([
  "Explore", // Quick code search, no modifications
  "academic-researcher", // Literature review and synthesis
  "code-reviewer", // Reviews code, does not write
  "data-researcher", // Data analysis and findings
  "error-detective", // Debugging and diagnostics
  "performance-engineer", // Profiling and analysis
  "research-analyst", // Research synthesis
  "search-specialist", // Information retrieval
  "technology-scout", // Evaluation and recommendations
  "knowledge-synthesizer", // Synthesizes information
]);

// Agents that are complex reasoning tasks — benefit from Opus/Sonnet High.
const HEAVY_REASONING_AGENTS = new Set([
  "Plan", // Architecture and design planning
  "llm-architect", // LLM system design
  "ml-engineer", // Model architecture decisions
  "autoresearch-agent", // Complex experiment design
  "multi-agent-coordinator", // Orchestration logic
]);

function detectAgentType(agentName) {
  if (READ_ONLY_AGENTS.has(agentName)) {
    return "read-only";
  }
  if (HEAVY_REASONING_AGENTS.has(agentName)) {
    return "heavy-reasoning";
  }
  return "general";
}

function recommendModel(agentName) {
  const override = process.env.FORCE_MODEL;
  if (override) {
    if (process.env.VERBOSE) {
      console.error(`[model-router] FORCE_MODEL=${override}`);
    }
    return override;
  }

  const agentType = detectAgentType(agentName);
  let recommended;

  switch (agentType) {
    case "read-only":
      recommended = "claude-haiku-4.5"; // Cheapest, sufficient for analysis
      break;
    case "heavy-reasoning":
      recommended = "claude-opus-5"; // Best reasoning for complex tasks
      break;
    case "general":
    default:
      recommended = "claude-sonnet-5"; // Balanced default
      break;
  }

  if (process.env.VERBOSE) {
    console.error(
      `[model-router] ${agentName} (${agentType}) → ${recommended}`
    );
  }

  return recommended;
}

function main() {
  const agentName = process.argv[2];
  if (!agentName) {
    console.error("Usage: model-router.js <agent-name>");
    process.exit(1);
  }

  const model = recommendModel(agentName);
  console.log(model);
}

if (require.main === module) {
  main();
}

module.exports = { detectAgentType, recommendModel, READ_ONLY_AGENTS, HEAVY_REASONING_AGENTS };
