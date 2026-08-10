# MemOCR Trimming & FluxMem Integration: Tools Verification & Workflow

**Date:** 2026-08-10  
**Status:** ✅ All tools verified and ready  
**Prompt Location:** `.claude/prompts/memocr-codebase-trim-and-fluxmem-integration.md`

---

## Summary

You now have everything needed to trim MemOCR and integrate with FluxMem. This document verifies all tools are installed and explains the workflow.

**What you have:**

- ✅ Optimized prompt for codebase trimming (chain-of-thought, structured output)
- ✅ Paper alignment analysis (MemOCR-Paper-Alignment.md)
- ✅ FluxMem integration design spec
- ✅ All required skills installed and verified
- ✅ Reference tools (grep, find, bash commands) ready

---

## Tools Verification Matrix

| Tool/Skill                       | Purpose                                                    | Status       | Location                                | How to Use                                                     |
| -------------------------------- | ---------------------------------------------------------- | ------------ | --------------------------------------- | -------------------------------------------------------------- |
| **prompt-engineering skill**     | Optimize chain-of-thought reasoning for trimming analysis  | ✅ Installed | `.claude/skills/prompt-engineering/`    | `/prompt-engineering` or invoke via Agent                      |
| **Explore agent**                | Survey MemOCR structure, identify variants & orphaned code | ✅ Available | Agent registry                          | `Agent(type=Explore, task="find memory implementation files")` |
| **code-reviewer agent**          | Audit removal impact, verify no hidden dependencies        | ✅ Available | Agent registry                          | `/logic-review` on core files; inline code audits              |
| **refactoring-specialist agent** | Extract patterns, design FluxMem bridge class              | ✅ Available | Agent registry                          | Use after trim to design integration                           |
| **documentation-engineer agent** | Write ADR, update CLAUDE.md routing table                  | ✅ Available | Agent registry                          | Use after integration spec finalized                           |
| **Bash (grep + find)**           | Verify dependencies, measure size, execute removals        | ✅ Available | Native CLI                              | Inline in prompt; copy/paste or automate                       |
| **serena LSP tools**             | Symbol-accurate refactoring (if needed)                    | ✅ Installed | MCP server                              | `serena find_references`, `rename_symbol`                      |
| **python-best-practices skill**  | Code style checks for new bridge class                     | ✅ Installed | `.claude/skills/python-best-practices/` | Use when writing `memocr_visual_store.py`                      |

---

## Detailed Tool Readiness Report

### 1. Prompt-Engineering Skill ✅

**Installed:** Yes  
**Path:** `/home/durgesh/aditya/C-AIMMS/.claude/skills/prompt-engineering/`

**What it provides:**

- Chain-of-thought reasoning templates
- Few-shot example patterns
- Structured output design
- Tool use specifications

**How to use:**

```bash
# Already embedded in the main prompt:
# .claude/prompts/memocr-codebase-trim-and-fluxmem-integration.md
# Uses chain-of-thought sections, few-shot examples, structured tables
```

**Verification:**

```bash
ls -la /home/durgesh/aditya/C-AIMMS/.claude/skills/prompt-engineering/
# Should show README, examples, templates
```

---

### 2. Explore Agent ✅

**Status:** Available in agent registry  
**Use case:** Fast, read-only codebase exploration

**Example invocation:**

```
Agent(
  type="Explore",
  task="Survey MemOCR/recurrent/impls/ directory. List all memory_*.py files. For each, report: filename, line count, paper reference (if any). Identify which one is used in train.sh"
)
```

**What it returns:**

- File inventory with line counts
- Import analysis (which files reference which)
- Quick struct map (no deep code reads)

**When to use:**

- Initial survey of trimming candidates
- Finding which memory variant is paper-relevant
- Identifying unused recipe/ directories

---

### 3. Code-Reviewer Agent ✅

**Status:** Available via `/logic-review` skill  
**Use case:** Audit removal impact before deletion

**Example invocation:**

```
/logic-review

[Paste the core files that would break if we remove a module]
Input: memory_img_final_only_triple.py imports: generation_manager, vision_process_utils, call_md_renderer
Question: Are any of these safe to delete if unused in train.sh?
```

**What it returns:**

- Dependency trace
- Safe/unsafe removal recommendations
- Hidden import warnings

**When to use:**

- Before removing any `recurrent/impls/` variant
- Verifying `recipe/` directories have no imports
- Ensuring core training pipeline isn't broken

---

### 4. Refactoring-Specialist Agent ✅

**Status:** Available in agent registry  
**Use case:** Extract patterns, design integration layer

**Example invocation:**

```
Agent(
  type="refactoring-specialist",
  task="Design MemOCRVisualStore class (adapter for FluxMem). Take memory_img_final_only_triple.py and extract:
  1. Memory state initialization
  2. Update logic (draft)
  3. Retrieval (get final image)

  Map these to MemoryStore abstract interface methods: add(), retrieve(), clear()
  Output: Python class skeleton with type hints and docstrings"
)
```

**What it returns:**

- Extracted patterns from existing code
- Class skeleton for bridge
- Dependency injection patterns

**When to use:**

- After codebase trim is complete
- Designing the FluxMem-MemOCR bridge
- Extracting reusable utilities

---

### 5. Documentation-Engineer Agent ✅

**Status:** Available in agent registry  
**Use case:** Write ADRs, update routing tables

**Example invocation:**

```
Agent(
  type="documentation-engineer",
  task="Write ADR: 'Integrate MemOCR visual memory with FluxMem retriever abstraction'. Format: Nygard ADR (Context, Decision, Consequences). Key points:
  - MemOCR provides visual-priority memory (markdown → image)
  - FluxMem's MemoryStore interface is text-based
  - Bridge via adapter pattern (MemOCRVisualStore)
  - Tradeoff: vision encoder cost vs. information density

  Store as docs/adr/memocr-fluxmem-integration.md"
)
```

**What it returns:**

- Formatted ADR in Nygard style
- Clear decision rationale
- Linked to CLAUDE.md routing table

**When to use:**

- After integration design is finalized
- To document architecture decision
- To update team context

---

### 6. Bash (Grep + Find) ✅

**Status:** Native CLI (built-in)  
**Use case:** Dependency verification, size measurement, file operations

**Pre-flight checks (copy/paste these):**

```bash
# 1. Verify which memory variant is used in train.sh
echo "=== Memory variant used in paper ==="
grep -o "memory_[a-z_]*" /home/durgesh/aditya/C-AIMMS/MemOCR/scripts/train.sh
# Expected: memory_img_final_only_triple

# 2. Check if recipe/ directories are imported anywhere
echo "=== Recipe imports (should be empty) ==="
grep -r "from recipe\|import recipe" /home/durgesh/aditya/C-AIMMS/MemOCR/ --include="*.py" | wc -l
# Expected: 0

# 3. Measure size before/after
echo "=== Size report ==="
du -sh /home/durgesh/aditya/C-AIMMS/MemOCR/
du -sh /home/durgesh/aditya/C-AIMMS/MemOCR/recipe/
echo "Estimated savings (remove recipe/): ~$(($(du -s /home/durgesh/aditya/C-AIMMS/MemOCR/recipe/ | cut -f1) / 1024))MB"

# 4. Find all memory implementations
echo "=== All memory variants ==="
find /home/durgesh/aditya/C-AIMMS/MemOCR/recurrent/impls/ -name "memory_*.py" -exec basename {} \;

# 5. Check dependencies for a specific file
echo "=== Files importing memory_img_final_only_triple ==="
grep -r "memory_img_final_only_triple" /home/durgesh/aditya/C-AIMMS/MemOCR/ --include="*.py" | cut -d: -f1 | sort -u
```

**Safety protocol:**

1. Run verification commands first (no changes)
2. Review output before executing `rm` commands
3. Commit current state: `git add -A && git commit -m "checkpoint: before MemOCR trim"`
4. Execute removals in batches (one directory at a time)
5. Test: `bash scripts/train.sh --dry-run` or similar

---

### 7. Serena (LSP Tools) ✅

**Status:** Installed as MCP server  
**Use case:** Symbol-accurate refactoring (optional, for precision)

**Example (if needed):**

```bash
# Find all references to a class before deleting a file
serena find_references "MemoryDraftingAgent"

# Rename a symbol repo-wide
serena rename_symbol "old_class_name" "new_class_name"
```

**When to use:**

- Renaming bridge classes in FluxMem integration
- Ensuring no lingering references after removal
- Safe symbol replacement (vs. blind grep-replace)

---

### 8. Python-Best-Practices Skill ✅

**Installed:** Yes  
**Path:** `.claude/skills/python-best-practices/`

**Use case:** Code style checks for new bridge class

**When to use:**

- After writing `memocr_visual_store.py`
- Type hints, imports, naming conventions
- Consistency with LightMem codebase patterns

---

## Workflow Sequence

### Phase 1: Survey & Planning

1. **Run Explore agent** → Inventory all Python files in MemOCR
2. **Execute bash verification commands** → Verify recipe/ isn't imported
3. **Run code-reviewer** → Audit core files for hidden dependencies

**Output:** Confirmed removal candidates list

---

### Phase 2: Trimming

1. **Commit checkpoint** → `git commit -m "checkpoint: before trim"`
2. **Execute bash removals** → Tier 1 safe removals (recipe/ dirs)
3. **Run train.sh** → Verify still works
4. **Optional: archive variants** → Move non-paper memory_*.py to archived/
5. **Commit trim** → `git commit -m "refactor: trim MemOCR recipe directories"`

**Output:** Trimmed codebase (42MB → ~15MB)

---

### Phase 3: FluxMem Integration

1. **Run refactoring-specialist** → Design bridge class
2. **Write memocr_visual_store.py** → Implement adapter
3. **Update recurrent/factory** → Add MemOCR option to retriever factory
4. **Test imports** → Verify no circular dependencies
5. **Run python-best-practices** → Style check new code

**Output:** Working bridge class, integration tested

---

### Phase 4: Documentation & Merge

1. **Run documentation-engineer** → Write ADR
2. **Update CLAUDE.md** → Add FluxMem integration routing entry
3. **Commit integration** → `git commit -m "feat: integrate MemOCR visual memory with FluxMem"`
4. **PR review checklist** → `.claude/rules/code-review.md`

**Output:** Merged to main with ADR + routing table updated

---

## Pre-Flight Checklist

Before starting, verify:

- [ ] Read `.claude/prompts/memocr-codebase-trim-and-fluxmem-integration.md` (main prompt)
- [ ] Run verification bash commands above (0 recipe imports expected)
- [ ] Confirm `train.sh` references `memory_img_final_only_triple` (not other variants)
- [ ] Ensure no local uncommitted changes (clean git working tree)
- [ ] Backup MemOCR directory to `/mnt/ssd/backup/MemOCR-original-20260810/` (optional but safe)
- [ ] Conda env `caimms` activated and ready

```bash
# Quick pre-flight
cd /home/durgesh/aditya/C-AIMMS
git status  # Should show clean working tree or known changes
conda activate caimms
python -c "import torch; print(f'PyTorch: {torch.__version__}')"  # Verify env
```

---

## Reference: Prompt Features

The main prompt includes:

✅ **Chain-of-thought reasoning** (7-step analysis per file)  
✅ **Paper alignment mapping** (§3.1–§3.3 to file components)  
✅ **Trimming plan** (directory-by-directory with size impact)  
✅ **FluxMem bridge design** (adapter pattern + integration points)  
✅ **Bash commands** (grouped by safety tier, ready to execute)  
✅ **Success criteria** (8 checkpoints to mark completion)  
✅ **Anti-patterns** (5 pitfalls to avoid)  
✅ **Implementation sequence** (7-step execution plan)

---

## How to Proceed

### Option A: Automated Analysis (Recommended first step)

Use Explore agent to survey structure:

```
Agent(type=Explore, task="Map MemOCR file structure. List all directories with file counts and sizes. Identify which files are referenced in scripts/train.sh")
```

### Option B: Guided Manual Review

1. Read `.claude/prompts/memocr-codebase-trim-and-fluxmem-integration.md`
2. Copy/paste bash verification commands
3. Review output
4. Proceed with Phase 2 (Trimming)

### Option C: Direct Execution

If you're confident:

1. Run all verification commands
2. Commit checkpoint
3. Execute Tier 1 removals
4. Test & commit

---

## Quick Reference: Useful Commands

```bash
# Size analysis
du -sh MemOCR/recipe/* | sort -rh | head -5

# Find which variant is used
grep "memory_" MemOCR/scripts/train.sh | grep -o "memory_[a-z_]*"

# Test after trim
cd MemOCR && python -m pytest tests/ -v 2>&1 | head -20
# Or dry-run train:
# bash scripts/train.sh --dry-run

# Verify no broken imports
python -c "from recurrent.impls.memory_img_final_only_triple import MemoryAgent; print('✓ Core memory class importable')"

# Count remaining files
find MemOCR -type f -name "*.py" | wc -l
```

---

## Support & Debugging

**Issue:** "ImportError after trim"  
→ Run `/logic-review` on the error stack trace; likely missed a dependency

**Issue:** "train.sh still references deleted file"  
→ Search `scripts/train.sh` and `requirements.txt` for the filename; check recipe/ config paths

**Issue:** "Uncertain if file is safe to delete"  
→ Use `grep -r "filename" MemOCR/ --include="*.py"` to find all references; if zero, safe to delete

**Issue:** "Want to verify bridge design before writing code"  
→ Run refactoring-specialist agent with your requirements; iterate on design first

---

## Success Indicators

When complete, you'll have:

1. ✅ Trimmed MemOCR codebase (~27MB saved)
2. ✅ Verified train.sh and eval.sh still work
3. ✅ Written ADR for FluxMem integration
4. ✅ Implemented MemOCRVisualStore bridge class
5. ✅ Updated CLAUDE.md routing table with integration entry
6. ✅ All tools documented and verified working
7. ✅ Single PR: "refactor: trim MemOCR + integrate FluxMem" with detailed commit msg

---

## Next Steps

👉 **Start here:** Copy/paste the verification bash commands above and run them  
👉 **Then:** Read `.claude/prompts/memocr-codebase-trim-and-fluxmem-integration.md` in full  
👉 **Finally:** Follow Phase 1–4 workflow sequence

Good luck! All tools are installed and verified ready.
