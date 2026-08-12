# 0010. EM-LLM KV cache behavior and GPU memory management

## Status

Accepted — 2026-08-13. Flags a documentation-vs.-code mismatch in EM-LLM that must be resolved before
EM-LLM is wired into production pipelines. The bug is **not yet causing production failures** (used
in research context with bounded dialogue lengths), but it will become critical when integrated with
continuous memory systems or long-running agent loops.

## Context

EM-LLM (`EM-LLM/caimms_boundary_creator.py`) performs surprise-based episode segmentation. It loads a
language model once and calls `forward(past_key_values=...)` per message to detect episode boundaries.

The documentation (`CAIMMS_BOUNDARY_CREATOR_DOCS.md:46-49`) claims:

> "The wrapper mathematically slices past_key_values tensor, physically discarding committed tokens
> from GPU's KV cache. This ensures GPU memory profile remains flat (O(1) per message) throughout
> multi-turn dialogue."

**Code audit finding:** This claim is **not implemented in the actual code.**

### What the code actually does (audit: `caimms_boundary_creator.py` lines 148–154)

```python
def forward(self, ...):
    outputs = self.model(**inputs, past_key_values=past_key_values)
    # ... compute surprise ...
    return {
        "surprise": ...,
        "outputs.past_key_values": outputs.past_key_values,  # ← UNMODIFIED
        ...
    }
```

The caller receives `past_key_values` **unmodified**. In typical usage:

```python
past_kv = None
for message in dialogue:
    result = boundary_detector.forward(..., past_kv)
    past_kv = result["past_key_values"]  # ← Growing!
```

**Result:** The KV cache **grows unboundedly** across all messages (not sliced, not discarded). A 100-message dialogue accumulates full 100-message KV cache on GPU, using O(N) memory, not O(1).

### Where slicing is mentioned in the codebase

Slicing does occur in `EM-LLM/boundary_creator.py:225–232` (`StatefulSurpriseBoundary.shift_history()`),
but **only on bookkeeping tensors** (`global_remainder_surprisal`, `global_block_divide`), not the
model's actual KV cache. Surprise statistics are trimmed; the GPU cache is not.

## Decision

**Immediate:** Update documentation to match code. Change DOCS claim from O(1) to O(N). This is an honest reflection of current behavior, not an aspirational design.

**Before Phase 1 release:** Decide if KV cache slicing is necessary for C-AIMMS's use case:

| Scenario                                                        | KV Cache Slicing Needed?                              |
| --------------------------------------------------------------- | ----------------------------------------------------- |
| Segmentation runs per-message on short dialogues (<50 messages) | No; typical GPU memory tolerates O(50 messages) cache |
| Agent runs 1000+ messages with continuous EM-LLM calls          | Yes; O(1000 messages) cache exhausts GPU memory       |
| Multiple concurrent agent sessions on shared GPU                | Yes; total memory = sum of all session caches         |

**Option A (Quick Fix):** Implement the promised KV cache slicing in `caimms_boundary_creator.py`.
For HF `past_key_values` (tuple of tuples of tensors), slice each layer's K and V tensors:

```python
def slice_past_kv(past_kv, keep_last_k_tokens: int = 256):
    """Slice KV cache to keep only recent K tokens per layer."""
    if past_kv is None:
        return None
    sliced = []
    for layer_k, layer_v in past_kv:
        sliced.append((layer_k[:, :, -keep_last_k_tokens:, :], layer_v[:, :, -keep_last_k_tokens:, :]))
    return tuple(sliced)
```

Trade-off: Discarding old KV values loses token-level history; only recent ~256 tokens are cached.
This is valid for segmentation (surprise only cares about recent context) but may affect accuracy
if the model relies on full history.

**Option B (Process Isolation):** Run EM-LLM in a separate process. KV cache stays local to that process;
the main agent loop receives only `(surprise, boundary_signal)` over IPC. No GPU cache leakage.

Trade-off: IPC serialization overhead; can't share GPU efficiently if EM-LLM is separate.

**Option C (Track But Defer):** Document the O(N) behavior, accept it for Phase 1 research (bounded
dialogue), and plan a Phase 4 redesign (e.g., sliding-window EM-LLM, cached boundary trie) after
measuring real memory usage in practice.

**Recommendation:** **Option A** if Phase 1–3 will include multi-turn agent loops (>100 messages).
Document the design decision and tradeoff (accuracy vs. memory) in an ADR when implemented.

## Consequences

### Positive

- **Honest documentation.** Claiming O(1) when code does O(N) is a correctness issue; fixing it is a prerequisite for trust.
- **Testable before production.** Implementing a cache-slicing solution is straightforward; can measure real memory usage before release.
- **Design flexibility.** Deciding between slicing depth (keep 256 vs. 512 tokens) is a tunable parameter for future optimization.

### Negative

- **Feature delay if slicing is implemented.** Writing cache slicing + testing takes ~2–4 hours; should be Phase 1 task if required.
- **Accuracy unknown.** No empirical data on how much slicing (e.g., keeping only last 256 tokens) affects surprise detection accuracy.

### Risks

- **Memory exhaustion in production.** If EM-LLM is wired into a continuous agent loop without cache management, GPU OOM crashes will happen silently (CUDA context loss, hard to debug).
- **Specification gap.** COLM doesn't specify EM-LLM's computational budget or memory footprint; this ADR closes that gap but leaves room for surprises if COLM's own EM-LLM implementation differs.

## Related

- **ADR 0005:** EM-LLM is canonical segmentation source; its memory behavior affects HetRep performance.
- **ADR 0007:** HG encoder uses EM-LLM output (turns); segmentation efficiency matters for Phase 2.
