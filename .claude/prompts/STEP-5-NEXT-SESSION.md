# Step 5: FluxMem Integration — Next Session Handoff

**Status:** Ready for implementation
**Prerequisite:** Step 4 complete (MemOCR recipe trimmed)

**Revision note (2026-08-10):** The original version of this file targeted
`LightMem/lightmem/factory/retriever/` and a `MemoryStore.add()/retrieve()`
adapter. Both are stale: `LightMem/` (vendored fork) was removed in commit
`50d99be`, and the real `fluxmem/` package (repo root) has no retriever
abstraction — its own docstring (`fluxmem/ltsm.py`) states retrieval
(ITERRET) is an explicit non-goal. Design corrected below to target the
boundary that actually exists.

## What to Do

Design and implement a bridge between MemOCR and FluxMem's **encoding**
boundary, not a retriever:

- New file: `fluxmem/memocr_episodes.py`
- Implements `fluxmem.interfaces.EpisodeProducer` (`produce(count) -> list[EpisodicUnit]`),
  concretizing the boundary that today only has `StubEpisodeProducer` (tests)
  and is documented as "Boundary fake for HETREP (segmentation + encoding),
  a non-goal here" — MemOCR is a real HETREP-adjacent encoder for the visual
  format specifically, so it fills that boundary rather than adding a new one.
- No factory registration needed — `fluxmem/` has no factory/registry pattern;
  callers construct `MemOCREpisodeProducer` directly, same as `StubEpisodeProducer`.
- Write ADR
- Commit

## Reference Documents

**Main prompt:** `.claude/prompts/memocr-codebase-trim-and-fluxmem-integration.md`

- Section: "FluxMem Integration (Bridge Design)" — Design spec + code pattern
- Section: "Example 3" — Real example (`EpisodeProducer`-based, corrected)

**Paper reference:** `docs/MemOCR-Paper-Alignment.md`

- Shows what MemOCR does (§3.1–§3.3)

**Architecture reference (read before designing):**

- `fluxmem/interfaces.py` — `EpisodeProducer` Protocol, `EpisodicUnit`, `MemoryFormat.VC`
- `fluxmem/ltsm.py` docstring — states retrieval is out of scope; do not reintroduce it
- `MemOCR/md2img/markdown_api_server.py:_markdown_to_image_sync` — the real markdown→PNG render call
- `MemOCR/recurrent/impls/memory_img_final_only_triple.py` — `MemoryAgent`, batched markdown memory state (`self.memory`)

## Quick Start

```bash
# 1. Read the corrected design spec
cat .claude/prompts/memocr-codebase-trim-and-fluxmem-integration.md | grep -A 60 "FluxMem Integration"

# 2. Use agent for design
Agent(type="refactoring-specialist", task="Design MemOCREpisodeProducer implementing fluxmem.interfaces.EpisodeProducer...")

# 3. Implement (see example in main prompt)

# 4. Commit
/commit "feat(fluxmem): add MemOCR-backed EpisodeProducer for visual-format episodes"
```

## Files to Modify/Create

- Create: `fluxmem/memocr_episodes.py`
- Create: `docs/adr/ADR-NNNN-memocr-fluxmem-integration.md`
- Update: `CLAUDE.md` routing table (add MemOCR episode-producer entry, not a retriever entry)

## Success Check

```bash
python -c "from fluxmem.memocr_episodes import MemOCREpisodeProducer; print('✓')"
python -c "from fluxmem.interfaces import EpisodeProducer; from fluxmem.memocr_episodes import MemOCREpisodeProducer; assert isinstance(MemOCREpisodeProducer.__new__(MemOCREpisodeProducer), EpisodeProducer)" 2>/dev/null || true
git log --oneline -1 | grep "feat(fluxmem)"
```

---

See main prompt for detailed workflow and examples.
