# Claude Code Toolkit Analysis Prompt

## For C-AIMMS Research Prototype Project

### Project Context

**Tech Stack:**

- Backend: Python
- Frontend: TypeScript (if needed)
- Type: Research prototype with experiments & metrics
- Team: Multi-person collaboration
- Workflow: Code → Review → Merge

**GitHub Integration Needs (All three):**

- Manage issues & pull requests automatically
- Push/pull code workflows
- Store configs in GitHub repo

**Automation Preferences:**

- Proactive hooks (auto-run on events)
- Manual commands (invoke when needed)
- Both welcome

**Key Requirements:**

- Track experiments & metrics
- Visual diagram documentation
- Remember context between sessions
- Track decisions & architecture choices
- Maintain living runbook/documentation
- Team-friendly configuration

---

### Task: Identify ALL Useful Components

Go through the awesome-claude-code-toolkit at `/home/durgesh/aditya/C-AIMMS/awesome-claude-code-toolkit/` and identify **every** plugin, skill, hook, command, and agent useful for this project. **No minimization—if it's useful, include it.**

---

## Phase 1: Read & Categorize

### Step 1: Read the Full README

Read `/home/durgesh/aditya/C-AIMMS/awesome-claude-code-toolkit/README.md` completely.

- Understand all sections
- Note featured plugins & their descriptions
- Identify all categories

### Step 2: Systematically Explore Each Category

For **EACH plugin/skill/hook/command/agent**, read its documentation to understand:

- What it does
- How it could help this specific project
- Any dependencies or conflicts

#### A. Python Backend & Development

**Look for plugins/skills that:**

- Python code generation, refactoring, debugging
- Backend frameworks (Flask, Django, FastAPI, etc.)
- Testing & test generation
- Code quality & linting
- Database design & optimization
- Architecture patterns

**Read:** Python-related plugins in `/plugins/` directory

#### B. Frontend & TypeScript

**Look for plugins/skills that:**

- TypeScript/JavaScript code generation
- React/Vue/other framework patterns
- Frontend testing
- Component generation
- CSS/styling helpers

**Read:** Frontend-related plugins

#### C. Git & GitHub Workflow Automation

**Look for plugins/skills that:**

- Automatic commit generation (key example)
- PR management & review automation
- Issue triage & automation
- Branch management
- Changelog generation
- Commit message generation
- GitHub Actions integration

**Read:** Git/GitHub plugins & hooks

#### D. Code Review & Testing

**Look for plugins/skills that:**

- Automated code review
- Test generation
- Test execution
- Coverage tracking
- Debugging helpers

**Read:** Review/testing plugins

#### E. Documentation & Knowledge Management

**Look for plugins/skills that:**

- Auto-generate documentation
- Architecture Decision Records (ADR)
- Runbook/wiki generation
- Code explainers
- Documentation from code

**Read:** Documentation plugins

#### F. Experiment Tracking & Metrics

**Look for plugins/skills that:**

- Experiment tracking (MLflow, Weights & Biases, etc.)
- Metrics collection
- Performance monitoring
- Cost tracking
- Quality metrics
- A/B testing

**Read:** Experiment/observability plugins

#### G. Visual Documentation & Diagrams

**Look for plugins/skills that:**

- Diagram generation (flowcharts, architecture diagrams, ERDs)
- Visual documentation
- Architecture visualization
- Data flow diagrams

**Read:** Diagram/visualization plugins

#### H. Team Collaboration & Memory

**Look for plugins/skills that:**

- Shared context/memory across team members
- Decision tracking
- Meeting notes/runbooks
- Configuration sharing
- GitHub-backed storage

**Read:** Collaboration/memory plugins

#### I. Research & Data Science

**Look for plugins/skills that:**

- Data analysis helpers
- Research workflow tools
- Reproducibility helpers
- Notebook integration
- Data visualization

**Read:** Research/data science plugins

#### J. Hooks (Automation Triggers)

**Read:** `/hooks/hooks.json` and `/hooks/scripts/`

**Look for hooks that:**

- Trigger on git events (commit, push, PR)
- Trigger on file changes
- Trigger before/after code execution
- Enable learning from past actions
- Track metrics automatically

#### K. Commands (Slash Commands)

**Read:** All `/commands/` directories

**Look for commands matching the above categories**

#### L. Agents

**Read:** `/agents/` directory

**Look for specialized agents that could help with:**

- Code review
- Architecture decisions
- Testing
- Documentation
- Research workflows

#### M. Rules & Templates

**Read:** `/rules/` and `/templates/`

**Look for rules/templates relevant to:**

- Python best practices
- Team collaboration
- Experiment tracking
- Documentation standards

#### N. MCP Configs

**Read:** `/mcp-configs/`

**Look for configs that integrate:**

- Data science tools
- Research platforms
- GitHub (for config storage)
- Observability tools

---

## Phase 2: Evaluate & Document ALL Components

### Step 3: For EACH Component (Selected or Rejected)

You must document **every** component you encounter with clear decision reasoning.

#### For SELECTED Components (✅):

```
**[Category Name]**

1. **[Plugin/Skill Name]** ✅ SELECTED
   - Type: Plugin / Skill / Hook / Command / Agent / Rule / Template / MCP
   - Description: [What it does - 1-2 sentences]
   - Why Useful: [Specific way it helps this project]
   - Dependencies: [Other components it needs]
   - Trigger: [Hook trigger / Command name / When it runs]
```

**Questions to answer for each:**

- Does it save time/reduce manual work?
- Does it fit the research prototype + team context?
- Does it help with GitHub integration?
- Does it support experiment tracking, documentation, or visualization?
- Are there conflicts with other useful components?

#### For REJECTED Components (❌):

Document components that don't fit your project with clear reasoning:

```
1. **[Plugin/Skill Name]** ❌ NOT SELECTED
   - Type: Plugin / Skill / Hook / Command / Agent / Rule / Template / MCP
   - Description: [What it does - 1-2 sentences]
   - Why NOT Useful: [Specific reason it doesn't fit]
   - Rejection Category: [See list below]
   - Conflicts With: [If it conflicts with a selected component, name it]
   - Alternative Selected: [If you chose something else for same purpose, name it]
```

#### Rejection Categories (pick the most applicable):

- **Out of Scope** — Doesn't apply to Python research prototype (e.g., mobile, Kubernetes, blockchain, etc.)
- **Specialized Domain** — Only useful for domains not in your project (e.g., pure DevOps if you're not doing DevOps)
- **Redundant** — Another selected component already does this better
- **Enterprise Overkill** — Built for large teams/enterprises; you need something simpler
- **Setup Cost Too High** — Configuration/maintenance overhead exceeds benefit for a research project
- **Conflicts with Priority** — Clashes with a more critical selected component
- **Deprecated/Unmaintained** — No longer actively maintained or outdated
- **Too Complex** — High learning curve; simpler alternatives exist with same result
- **Too Expensive** — Resource/token/cost overhead not justified for prototype
- **Missing Critical Feature** — Can't do what you need it to do
- **Wrong Tool for Job** — Designed for different use case (e.g., framework-specific if you're framework-agnostic)
- **Team Size Mismatch** — Assumes different team size than you have

---

## Phase 3: Summarize Findings

### Step 4: Create Final Summary with Both Selected & Rejected

Create a comprehensive summary organized by **function** showing what you selected AND rejected with reasoning:

```markdown
# Components Analysis Summary

## Python Backend Development

### Selected ✅
- [List all selected Python-related plugins/skills with 1-line rationale]

### Rejected ❌
- [List rejected Python plugins with rejection reason]

## Frontend & TypeScript

### Selected ✅
- [List all selected frontend plugins/skills]

### Rejected ❌
- [List rejected frontend plugins with rejection reason]

## Git & GitHub Automation

### Selected ✅
- [List all selected git/GitHub plugins/skills/hooks]

### Rejected ❌
- [List rejected git/GitHub components with rejection reason]

## Code Review & Testing

### Selected ✅
- [List all selected review/testing plugins/skills]

### Rejected ❌
- [List rejected review/testing components with rejection reason]

## Documentation & Knowledge

### Selected ✅
- [List all selected documentation plugins/skills]

### Rejected ❌
- [List rejected documentation components with rejection reason]

## Experiment Tracking & Metrics

### Selected ✅
- [List all selected experiment/metrics plugins/skills]

### Rejected ❌
- [List rejected experiment/metrics components with rejection reason]

## Visual Documentation & Diagrams

### Selected ✅
- [List all selected diagram/visualization plugins/skills]

### Rejected ❌
- [List rejected diagram/visualization components with rejection reason]

## Team Collaboration & Memory

### Selected ✅
- [List all selected collaboration plugins/skills/hooks]

### Rejected ❌
- [List rejected collaboration components with rejection reason]

## Research & Data Science

### Selected ✅
- [List all selected research plugins/skills]

### Rejected ❌
- [List rejected research components with rejection reason]

## Automation Hooks

### Selected ✅
- [List all selected hooks with their triggers]

### Rejected ❌
- [List rejected hooks with rejection reason]

## Useful Commands

### Selected ✅
- [List useful slash commands by category]

### Rejected ❌
- [List rejected commands with rejection reason]

## Useful Agents

### Selected ✅
- [List selected specialized agents]

### Rejected ❌
- [List rejected agents with rejection reason]

## Rules & Templates

### Selected ✅
- [List selected rules and templates]

### Rejected ❌
- [List rejected rules/templates with rejection reason]

## MCP Integrations

### Selected ✅
- [List selected MCP configs]

### Rejected ❌
- [List rejected MCP configs with rejection reason]

---

## Key Decisions & Tradeoffs

[For major rejections or when multiple similar tools exist, explain the decision made]

### Example:
- **Rejected 5 diagram tools in favor of [Selected Tool]** because: [specific reasons]
- **Rejected enterprise-grade experiment tracking for**: [chose simpler alternative instead] because: [reasons]

---

## Installation & Setup Notes

[Any dependencies, ordering requirements, or setup notes]

## Summary Statistics

- **Total Components Reviewed:** [Number]
- **Total Components Selected:** [Number] ✅
- **Total Components Rejected:** [Number] ❌
- **Selection Rate:** [Percentage]
```

---

## Critical Questions to Answer

As you go through, answer these:

1. **GitHub Config Storage:** Which plugins support storing configs in a GitHub repo?
2. **Experiment Integration:** How many plugins can integrate with your experiment tracking?
3. **Hook Conflicts:** Are there hooks that might conflict with each other?
4. **Installation Order:** Are there dependencies (some plugins need others first)?
5. **Team Setup:** Which components enable team-wide shared configuration?

---

## What You're Looking For

- **Universally useful things** (like the commit skill example—saves manual work)
- **Research-specific tools** (experiment tracking, reproducibility)
- **Team collaboration** (shared context, decision tracking)
- **Automation** (hooks that run proactively)
- **Documentation** (diagrams, runbooks, architecture tracking)
- **GitHub integration** (config storage, issue/PR automation)

**If it's useful for ANY of the above, include it. No filtering for "minimal set"—maximize utility.**

---

## Deliverable

A comprehensive summary organized by function showing:

### Selected Components ✅

- All useful plugins, skills, hooks, commands, agents, rules, templates, MCPs
- 1-2 sentence description of each
- Why it's useful for this specific project
- Dependencies and setup notes
- Installation order if dependencies exist

### Rejected Components ❌

- All rejected/non-useful components
- 1-2 sentence description of what they do
- Specific reason(s) why they weren't selected (using rejection categories)
- Any conflicts with selected components
- What you selected instead (if applicable)

### Summary Statistics

- Total components reviewed
- Total components selected
- Total components rejected
- Selection rate (% of toolkit that applies to your project)
- Key tradeoff decisions explained

This dual approach ensures:

- **Transparency:** Why you chose what you chose
- **Decision Audit Trail:** Future team members understand the reasoning
- **Avoidance of Missing Tools:** Explicit rejection prevents "we forgot about X"
- **Easier Updates:** Clear criteria for evaluating new additions later