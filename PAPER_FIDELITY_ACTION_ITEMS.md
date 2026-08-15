# C-AIMMS Paper Fidelity: Action Items Complete ✅

**Date**: 2026-08-14  
**Status**: All three tasks completed

---

## Summary of Changes

### 1. ✅ Flagged MemOCR Rendering Strategy (Canonical Implementation)

**Files Modified**:

- `MemOCR/recurrent/impls/memory_img_final_only_triple.py` — Added module docstring marking this as **CANONICAL** implementation
- `MemOCR/recurrent/impls/memory_img.py` — Added module docstring marking this as **ALTERNATIVE/VARIANT**

**Key Message**:

```
CANONICAL IMPLEMENTATION for the MemOCR paper (Shi et al., 2026):
- Uses FINAL-ONLY rendering (memory drafted at each step, rendered only at final step)
- This matches the paper's described workflow
- Differs from memory_img.py (per-episode rendering variant)
```

**Impact**: Future developers will immediately see which implementation matches the paper.

---

### 2. ✅ Resolved HyperMem Config Discrepancy (CONFIG_DEFAULTS.md consolidated)

**Original Issue**: Earlier investigation created `HyperMem/CONFIG_DEFAULTS.md` and
`HyperMem/REPRODUCIBILITY_GUIDE.md` to document a discrepancy between the README's recommended
configuration (k^T=15, k^E=25, reranker=true) and the code's actual defaults (k^T=10, k^E=10).

**Resolution** (2026-08-15): The paper-recommended values are now the actual code defaults:

- `HyperMem/hypermem/config/constants.py:62-63` — Updated `DEFAULT_TOPIC_TOP_K` to 15 and
  `DEFAULT_EPISODE_TOP_K` to 25
- `HyperMem/hypermem/config/config.yaml:59-60` — Updated the YAML defaults to match
- `HyperMem/scripts/run_eval.sh:10,12-13` — Updated the eval script defaults and changed
  `HYPERMEM_USE_RERANKER` default from `false` to `true`
- `HyperMem/tests/test_config.py:38-40` — Updated test expectations to assert the new defaults
- `HyperMem/README.md` — Clarified that these are now the defaults, not optional overrides

**Paper Defaults Now (Consolidated to CONFIGURATION.md)**: See `HyperMem/docs/CONFIGURATION.md#paper-defaults`

| Parameter   | Value | Code Location |
| ----------- | ----- | -------------- |
| λ (alpha)   | 0.5   | constants.py:52 |
| k^T         | 15    | constants.py:62 |
| k^E         | 25    | constants.py:63 |
| k^F         | 30    | constants.py:64 |
| Reranker    | true  | config.yaml:57 |
| Aggregation | sum   | config/__init__.py:~105 |

**Consolidation**: The two investigative .md files (`CONFIG_DEFAULTS.md`, `REPRODUCIBILITY_GUIDE.md`)
have been deleted. Their essential content is now inlined in `HyperMem/docs/CONFIGURATION.md` as the
[Paper Defaults](#paper-defaults) section. The discrepancy has been resolved by updating code
defaults to match the paper values.

---

## Quick Reference

### For HyperMem Users

**Default behavior now matches paper specifications** (k^T=15, k^E=25, reranker=true).
No special environment variable setup needed for paper replication.

To customize parameters, see `HyperMem/docs/CONFIGURATION.md` and the `Paper Defaults` table within it.

### For MemOCR Users

1. **Which implementation to use?**
   - Use: `recurrent/impls/memory_img_final_only_triple.py` (canonical, matches paper)
   - Marked in code with clear docstring

2. **Why two implementations?**
   - Canonical: final-only rendering (matches paper)
   - Alternative: per-episode rendering (experimental variant)

---

## Verification Checklist

### HyperMem Reproducibility ✅

- [x] λ = 0.5 (verified in constants.py:52)
- [x] aggregation = "sum" (verified in config/__init__.py:~105)
- [x] RRF k = 60 (verified in stage4_hypergraph_retrieval.py:54)
- [x] k^T=15, k^E=25 (now code defaults in constants.py:62-63)
- [x] use_reranker=true (now code default in config.yaml:57 and run_eval.sh:10)
- [x] Configuration guide created with exact file locations (`CONFIGURATION.md#paper-defaults`)

### MemOCR Rendering Strategy ✅

- [x] Canonical implementation identified (final-only)
- [x] Alternative variant marked (per-episode)
- [x] Clear docstrings added to both files
- [x] Paper reference added to code

---

---

## Files Modified

1. `HyperMem/hypermem/config/constants.py` — Updated `DEFAULT_TOPIC_TOP_K` (15) and `DEFAULT_EPISODE_TOP_K` (25)
2. `HyperMem/hypermem/config/config.yaml` — Updated YAML defaults to match
3. `HyperMem/scripts/run_eval.sh` — Updated defaults and reranker to true
4. `HyperMem/tests/test_config.py` — Updated test expectations
5. `HyperMem/README.md` — Clarified that paper values are now defaults
6. `HyperMem/docs/CONFIGURATION.md` — Added "Paper Defaults" section, consolidated from deleted docs
7. `HyperMem/docs/ARCHITECTURE.md` — Updated reference from `CONFIG_DEFAULTS.md` to `CONFIGURATION.md#paper-defaults`
8. `MemOCR/recurrent/impls/memory_img_final_only_triple.py` — Added canonical implementation docstring
9. `MemOCR/recurrent/impls/memory_img.py` — Added alternative implementation note

## Files Deleted

1. `HyperMem/CONFIG_DEFAULTS.md` — Content consolidated into `CONFIGURATION.md#paper-defaults`
2. `HyperMem/REPRODUCIBILITY_GUIDE.md` — Discrepancy resolved; content no longer needed

---

## Next Steps

1. Use `MemOCR/recurrent/impls/memory_img_final_only_triple.py` as canonical implementation
2. Run HyperMem with default settings to replicate paper (no env var overrides needed)
3. Refer to `HyperMem/docs/CONFIGURATION.md` for tuning and advanced configuration

### For Documentation

1. These guides should be referenced in main README for onboarding
2. Add to CI/CD: parameter validation to warn if non-paper values detected
3. Archive experiment configs with saved runs for future reference

---

**All action items complete. See linked guides for detailed information.**
