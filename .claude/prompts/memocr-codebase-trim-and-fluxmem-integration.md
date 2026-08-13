# MemOCR Codebase Trimming & FluxMem Integration Prompt

**Optimized with:** prompt-engineering skill · Chain-of-thought · Structured output · Paper-driven scope

**Scope:** Trim MemOCR (42MB, 515 files) to essential components for COLM paper reproduction, then integrate memory abstraction with FluxMem.

**Output format:** Markdown table (keep/trim/integrate) + bash rm/mv commands + FluxMem bridge design doc

---

## System Prompt

You are a research codebase minimizer and systems integration architect. Your role:

1. **Identify essential components** — map paper sections (§3.1–§3.3) to minimal file set
2. **Flag optional utilities** — locate test utilities, experimental recipes, data loaders that support paper but aren't core
3. **Plan trimming** — produce concrete shell commands (rm, mv) with justification per file/directory
4. **Design FluxMem bridge** — specify an `EpisodeProducer` adapter connecting MemOCR visual memory to fluxmem's encoding boundary (`fluxmem/interfaces.py`); fluxmem has no retriever to bridge into (retrieval/ITERRET is explicitly out of scope, see `fluxmem/ltsm.py`)
5. **Verify no loss** — ensure trimmed codebase still runs paper's training, evaluation, and baseline scripts

**Scope:** Core MemOCR (memory drafting, visual rendering, GRPO training) + minimal evaluation harness + FluxMem integration points.

**Out of scope:** Data download scripts, deployment configs, experimental recipes not in paper.

**Constraints:**

- Never delete `.git`, README, LICENSE, pyproject.toml, requirements.txt
- Preserve all of `md2img/`, `recurrent/impls/*_triple.py`, `verl/trainer/ppo/ray_trainer.py`
- Keep `scripts/train.sh`, `scripts/eval.sh`, essential taskutils (not all recipes)
- Output must include concrete bash commands grouped by category (safe removals, optional removals, integration)
- Flag any uncertain removals (ask before deleting)

---

## Chain-of-Thought Reasoning Process

When analyzing each directory/file:

1. **Map to paper spec** — Which paper section (§3.1, §3.2, §3.3, §4.1) does this serve?
   - §3.1 = Memory Drafting (recurrent/impls/*_triple.py, prompts)
   - §3.2 = Visual Rendering (md2img/, vision_process_utils.py)
   - §3.3 = GRPO Training (verl/trainer/ppo/ray_trainer.py, generation_manager.py)
   - §4.1 = Experimental Setup (scripts/, taskutils/memory_eval/)

2. **Check Paper-Alignment.md** — Is it ✓ Implemented, ⚠ Partial, or ✗ Missing?
   - Implemented components are essential
   - Partial components: keep the working subset, trim stubs
   - Missing components: safe to delete (unless needed for baselines)

3. **Classify file category**:
   - **Core:** Must keep (no workaround exists)
   - **Support:** Strongly recommended (reduces duplication in integration)
   - **Optional:** Nice to have; can move to `archived/` or delete
   - **Experimental:** Recipe variants, hyperparameter sweeps, not in paper; candidate for removal
   - **Tooling:** Scripts, utilities, configs; evaluate per use case

4. **Check dependencies** — What imports this file? Would removing it break core training/eval?
   - Use `grep -r "from.*X import\|import.*X"` to trace

5. **Assess file size impact** — Is it ≥10% of total size? Prioritize large experimental recipes.

6. **Propose action** — Specific action (rm, mv to archived/, keep + integrate)

7. **Justify with numbers** — File count reduction, size saved, integration complexity.

---

## Core Paper Architecture (Reference)

| Paper Section | Component          | Critical Files                                                                                                   | Optional Files                                                         | Experimental Recipes                                                             |
| ------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| §3.1          | Memory Drafting    | `recurrent/impls/memory_img_final_only_triple.py`, `recurrent/generation_manager.py`, prompts                    | `recurrent/impls/call_md_renderer.py` (for rendering), async utilities | Other `memory_*_*.py` variants in `recurrent/impls/`                             |
| §3.2          | Visual Rendering   | `md2img/markdown_api_server.py`, `recurrent/vision_process_utils.py`, `taskutils/memory_eval/utils/memocr_md.py` | CSS templates, HTML utilities                                          | Deprecated rendering backends (html_api_server.py if unused)                     |
| §3.3          | GRPO Training      | `verl/trainer/ppo/ray_trainer.py`, `verl/base_config.py`                                                         | GRPO advantage computation helpers                                     | Alternative PPO/RL algorithms (not in paper)                                     |
| §4.1          | Experimental Setup | `scripts/train.sh`, `scripts/eval.sh`, `taskutils/memory_eval/run_custom.py`                                     | `taskutils/memory_data/process_test.py`                                | `recipe/` subdirectories (18 variants), non-HotpotQA baselines, ablation studies |

---

## Few-Shot Examples

### Example 1: Safe Removal (Experimental Recipe)

**File:** `recipe/deepeyes/` (directory, ~2MB)

**Analysis:**

1. **Map to paper:** Not mentioned in paper; appears to be prior experiment tracking method
2. **Check alignment:** Not in Paper-Alignment.md; marked ✗ Missing
3. **Classify:** Experimental
4. **Dependencies:** `grep -r "from.*deepeyes\|import.*deepeyes" MemOCR/` → 0 results
5. **Size impact:** ~2MB / 42MB ≈ 5% savings
6. **Action:** Remove

**Recommendation:**

```bash
rm -rf MemOCR/recipe/deepeyes/
# Justification: Experiment tracking method not in paper; no imports; 5% size reduction
```

---

### Example 2: Conditional Keep (Alternative Memory Implementation)

**File:** `recurrent/impls/memory_img_final_only_double.py` (if exists)

**Analysis:**

1. **Map to paper:** Could be ablation or prior iteration
2. **Check alignment:** Not in final Paper-Alignment.md (which focuses on `memory_img_final_only_triple.py`)
3. **Classify:** Optional / Experimental variant
4. **Dependencies:** Check if referenced in `train.sh` → if yes, keep; if no, candidate for removal
5. **Action:** Move to `archived/` or remove if not used in train.sh

**Recommendation:**

```bash
# If NOT referenced in train.sh:
mkdir -p MemOCR/archived/recurrent_impls_variants/
mv MemOCR/recurrent/impls/memory_img_final_only_double.py MemOCR/archived/
# Justification: Variant not used in paper; move to archive for reference
```

---

### Example 3: Integrate with FluxMem (Encoding Boundary, Not Retrieval)

**File:** `recurrent/impls/memory_img_final_only_triple.py` (core, `MemoryAgent`, ~300 lines)

**Revision note (2026-08-10):** The version of this example that shipped with
Step 4 targeted `LightMem/lightmem/factory/retriever/` and a
`MemoryStore.add()/retrieve()` interface. Neither exists: `LightMem/` was
removed (commit `50d99be`), and the real `fluxmem/` package explicitly scopes
retrieval (ITERRET) out — see `fluxmem/ltsm.py`'s module docstring. The
correct integration point is `fluxmem.interfaces.EpisodeProducer`, the
Protocol fluxmem already calls "Boundary fake for HETREP (segmentation +
encoding), a non-goal here." MemOCR is a real encoder for the visual format,
so it concretizes that boundary instead of adding a competing one.

**Analysis:**

1. **Map to paper:** §3.1 Memory Drafting (MemOCR side) → COLM §1.3.2 HETREP boundary (FluxMem side)
2. **Classify:** Core
3. **Integration point:** `MemoryAgent` manages persistent Markdown memory state (`self.memory`, decoded per-batch-item text) and rendering goes through `md2img/markdown_api_server.py:_markdown_to_image_sync` (Markdown → HTML → Chromium → PNG).
   - FluxMem's boundary for "turn something external into memory units" is `EpisodeProducer.produce(count) -> list[EpisodicUnit]` (`fluxmem/interfaces.py`), today only satisfied by the test-only `StubEpisodeProducer`.
   - MemOCR memory is visual (rendered image), matching `MemoryFormat.VC` exactly.
   - No `add()`/`retrieve()`/query surface exists or should be added — MTEM/LTSM (downstream of `EpisodeProducer`) are what score, promote, and eventually serve episodes; MemOCR does not need to know about either.

**Recommendation:**

```python
# File: fluxmem/memocr_episodes.py (NEW)

from __future__ import annotations

from fluxmem.interfaces import EpisodicUnit, MemoryFormat, Turn


class MemOCREpisodeProducer:
    """Adapter: MemOCR's rendered Markdown memory -> fluxmem EpisodicUnit (COLM Sec 1.3.2).

    Concretizes `fluxmem.interfaces.EpisodeProducer` for the visual-canvas
    format. Wraps a `MemoryAgent` (recurrent/impls/memory_img_final_only_triple.py)
    plus the md2img render call; does not touch MTEM/LTSM/selector/fusion --
    those consume `EpisodicUnit` the same way regardless of producer.
    """

    def __init__(self, memory_agent, render_fn):
        self._agent = memory_agent  # MemoryAgent instance (owns self.memory)
        self._render_fn = render_fn  # e.g. md2img.markdown_api_server._markdown_to_image_sync

    def produce(self, count: int) -> list[EpisodicUnit]:
        """Renders up to `count` pending Markdown memory snapshots into EpisodicUnits."""
        snapshots = self._agent.pending_memory_snapshots(count)  # decoded Markdown text, batched
        return [self._to_episode(i, md) for i, md in enumerate(snapshots)]

    def _to_episode(self, index: int, markdown_text: str) -> EpisodicUnit:
        image_bytes = self._render_fn(markdown_text)
        # visual_salience is a modeling choice -- neither paper defines it
        # concretely; documented in the ADR, not hedged behind a config flag.
        return EpisodicUnit(
            episode_id=f"memocr-episode-{index}",
            turns=[],  # MemOCR's memory is already a compacted summary, not raw turns
            primary_format=MemoryFormat.VC,
            visual_salience=_estimate_visual_salience(image_bytes),
        )
```

**Action:** Create the `EpisodeProducer` adapter; document the format mapping and the `visual_salience` modeling choice in an ADR. Do not create a `MemoryStore`, factory registration, or any retrieval surface.

---

## Directory Trimming Plan

| Directory    | Size  | Files          | Status       | Action                              | Justification                                                                                                                   |
| ------------ | ----- | -------------- | ------------ | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `md2img/`    | ~1MB  | 4 Python files | Essential    | **Keep**                            | Core visual rendering (§3.2)                                                                                                    |
| `recurrent/` | ~2MB  | ~20 files      | Mixed        | **Keep core, trim variants**        | Keep `memory_img_final_only_triple.py`, `generation_manager.py`, `vision_process_utils.py`; move other memory_*.py to archived/ |
| `verl/`      | ~3MB  | ~30 files      | Essential    | **Keep**                            | GRPO training infrastructure (§3.3)                                                                                             |
| `taskutils/` | ~1MB  | ~15 files      | Mixed        | **Keep memory_eval, trim optional** | Keep `memory_eval/run_custom.py`, `memory_data/process_test.py`; remove non-HotpotQA loaders if space-constrained               |
| `recipe/`    | ~10MB | 18 dirs        | Experimental | **Move to archived/**               | 18 experimental recipe variants; none in paper; each is independent experiment tracker                                          |
| `scripts/`   | ~50KB | 4 files        | Mixed        | **Keep train.sh, eval.sh**          | Keep `train.sh`, `eval.sh`; optional: `read_result.py`, `model_merger.py`                                                       |
| `docs/`      | ~1MB  | Varies         | Reference    | **Keep**                            | Architecture, README, license                                                                                                   |

**Overall Impact:**

- **Before:** 42MB, 515 files
- **After (aggressive trim):** ~15MB, 150–180 files
- **Savings:** ~27MB (64% reduction), 335+ files removed

---

## Search Commands (Implementation Reference)

```bash
# 1. Identify all memory implementation variants
grep -r "class.*Memory" MemOCR/recurrent/impls/ --include="*.py" -l

# 2. Check which memory class is imported in train.sh
grep -o "memory_[a-z_]*" MemOCR/scripts/train.sh

# 3. Find all dependencies on recipe/ modules
grep -r "from recipe\|import recipe" MemOCR/ --include="*.py"

# 4. List directories by size (find largest candidates for removal)
du -sh MemOCR/recipe/* | sort -rh | head -10

# 5. Check if experimental files are referenced
grep -r "deepeyes\|collabllm\|dapo\|entropy" MemOCR/ --include="*.py" | wc -l

# 6. Trace imports for core training
grep -r "from verl\|from recurrent\|from md2img" MemOCR/scripts/train.sh

# 7. Identify orphaned modules (no imports)
for f in MemOCR/recurrent/impls/*.py; do
  name=$(basename "$f" .py)
  echo -n "$name: "
  grep -r "import.*$name\|from.*$name" MemOCR/ --include="*.py" | wc -l
done
```

---

## Expected Output Template

### Part 1: Summary Table

```markdown
| Directory/File                                  | Size | Type       | Paper Use      | Status    | Action   | Priority          |
| ----------------------------------------------- | ---- | ---------- | -------------- | --------- | -------- | ----------------- |
| md2img/                                         | 1MB  | Core       | §3.2 Rendering | Essential | **Keep** | 0 (do not touch)  |
| recurrent/impls/memory_img_final_only_triple.py | 20KB | Core       | §3.1 Drafting  | Essential | **Keep** | 0                 |
| recipe/deepeyes/                                | 2MB  | Experiment | None           | Removable | `rm -rf` | 1 (high priority) |
| ...                                             | ...  | ...        | ...            | ...       | ...      | ...               |
```

### Part 2: Bash Commands (Grouped by Safety)

```bash
### TIER 1: Safe removals (verified unused, not in paper)
rm -rf MemOCR/recipe/deepeyes/
rm -rf MemOCR/recipe/collabllm/
rm -rf MemOCR/recipe/dapo/
# ... (18 recipe dirs total, ~10MB saved)

### TIER 2: Optional removals (check dependencies first)
# Before running these, verify no scripts reference them:
# grep -r "genrm_remote\|minicpmo" MemOCR/scripts/
rm -rf MemOCR/recipe/genrm_remote/
# ... (only if safe)

### TIER 3: Archive variants (keep for reference, move to archived/)
mkdir -p MemOCR/archived/recurrent_impls/
mv MemOCR/recurrent/impls/memory_img_final_only_double.py MemOCR/archived/recurrent_impls/ 2>/dev/null || true
# (only if other memory_*.py variants exist and are unused)
```

### Part 3: FluxMem Integration Design

````markdown
## FluxMem Integration

**Corrected 2026-08-10:** `LightMem/` no longer exists (removed in `50d99be`),
and the real `fluxmem/` package (repo root) has no retriever/`MemoryStore`
abstraction — retrieval (ITERRET) is explicitly out of scope per
`fluxmem/ltsm.py`. The bridge below targets `fluxmem.interfaces.EpisodeProducer`
instead, the encoding-side boundary that already exists.

### Bridge Class (New File)

**Location:** `fluxmem/memocr_episodes.py`

**Purpose:** Adapt MemOCR's rendered Markdown memory to fluxmem's `EpisodeProducer`
boundary, producing `EpisodicUnit`s with `primary_format=MemoryFormat.VC`.

**Key Method:**

- `produce(count) -> list[EpisodicUnit]` → Renders pending MemOCR Markdown
  memory snapshots (via `md2img/markdown_api_server.py`) into `EpisodicUnit`s.

### Wiring (No Factory — None Exists)

`fluxmem/` has no factory/registry pattern; `StubEpisodeProducer` (test-only,
`fluxmem/interfaces.py`) and `MemOCREpisodeProducer` are both constructed
directly by whatever assembles the STIM→MTEM pipeline:

```python
from fluxmem.memocr_episodes import MemOCREpisodeProducer

producer = MemOCREpisodeProducer(memory_agent=agent, render_fn=_markdown_to_image_sync)
episodes = producer.produce(count=batch_size)  # -> list[EpisodicUnit], feeds MTEM
```

### ADR (Architecture Decision Record)

**File:** `docs/adr/NNNN-memocr-fluxmem-integration.md`

**Rationale:** MemOCR provides a visual-priority encoder for the `MemoryFormat.VC`
episodes fluxmem's `EpisodeProducer` boundary expects; no retrieval surface
needs to exist for this, since fluxmem does not implement one.

**Tradeoff:** Vision encoder/render cost (Markdown → image) vs. information
density gain; `visual_salience` is a modeling choice with no citation in
either paper, stated explicitly rather than assumed.

````

---

## Tools/Skills to Use

| Tool | When | Command |
| --- | --- | --- |
| **Explore agent** | Survey recurrent/ and recipe/ structure, identify orphan modules | `Agent(type=Explore, task="find all memory implementation files")` |
| **code-reviewer** | Audit removal impact: check for hidden dependencies, cross-repo imports | `/logic-review` on core files before removal |
| **refactoring-specialist** | Design the `EpisodeProducer` adapter (`fluxmem/memocr_episodes.py`) | Use after trim, targeting `fluxmem.interfaces.EpisodeProducer` |
| **documentation-engineer** | Write ADR for FluxMem integration, update CLAUDE.md routing table | `/adr` for integration decision |
| **Bash (find + grep)** | Verify no code references deleted files, measure size savings | Inline commands above |

---

## Anti-Patterns (Avoid These)

- ✗ **Delete entire `recipe/` without checking if any scripts reference it** — Check `train.sh`, `eval.sh` first
- ✗ **Remove `recurrent/impls/memory_*.py` variants without understanding which one paper uses** — Verify against `train.sh:103` (model selection)
- ✗ **Skip FluxMem integration spec** — Without an `EpisodeProducer` adapter, MemOCR and fluxmem remain disconnected
- ✗ **Assume all taskutils are optional** — Some (e.g., `process_test.py`) are required for eval
- ✗ **Delete without committing current state** — Risk losing reference if removal breaks things
- ✗ **Reintroduce a retriever/`MemoryStore` abstraction** — `fluxmem/` explicitly scopes retrieval (ITERRET) out; the bridge is an `EpisodeProducer`, not a query surface

---

## Success Criteria

- [ ] All 18 recipe/ directories audited (either removed or justified for keep)
- [ ] Bash commands provided for safe removals (tested locally or verified safe)
- [ ] Trimmed codebase still runs: `bash scripts/train.sh` (or test subset)
- [ ] All paper-required files identified and marked "Keep"
- [ ] `EpisodeProducer` bridge design documented (`fluxmem/memocr_episodes.py`, integration points clear)
- [ ] ADR written for integration decision
- [ ] `CLAUDE.md` routing table updated: "Using MemOCR as a visual `EpisodeProducer` for fluxmem" → `.claude/prompts/memocr-codebase-trim-and-fluxmem-integration.md`
- [ ] Final size reported: from 42MB → X MB; file count X → Y

---

## Implementation Sequence

1. **Survey** (Explore agent) — Map file → paper component, identify variants
2. **Verify** (grep commands) — Check no code references deletions
3. **Trim** (bash rm) — Remove safe candidates; move optional to archived/
4. **Test** (local run) — Verify train.sh and eval.sh still work
5. **Integrate** (refactoring-specialist) — Create `MemOCREpisodeProducer` (`fluxmem/memocr_episodes.py`)
6. **Document** (documentation-engineer) — Write ADR, update routing table
7. **Commit** — Single PR: "refactor: trim MemOCR codebase for COLM paper, integrate fluxmem `EpisodeProducer`" with detailed commit msg
```
