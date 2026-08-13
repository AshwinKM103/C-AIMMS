# 0011. MemOCR segmentation and training architecture

## Status

Accepted — 2026-08-13. Documents MemOCR's independent segmentation and training pipeline, clarifying
that MemOCR's segmentation is fixed-window based (not surprise-based). MemOCR remains available
for standalone visual memory workloads and for fine-tuning experiments in Phase 3 if the
Qwen2.5-VL-7B checkpoint proves insufficient.

## Context

MemOCR (`MemOCR/recurrent/impls/memory_img_final_only_triple.py`) is a visual memory agent that:

1. **Segments via fixed-window chunking:** `MemoryAgent.action()` lines 196–274 buffer incoming messages
   into fixed-size chunks (`chunk_size * (step + 1)`). No surprise detection, no adaptive boundaries —
   just: "when buffer reaches N messages, emit an episode and render to image."

2. **Encodes by rendering:** Each buffered episode is drafted as markdown memory (via LLM prompt,
   `memory_img_final_only_triple.py:122–141`) and rendered to an image via HTTP POST to a
   `md2img/markdown_api_server.py` (Playwright + Chrome headless, line 50–57).

3. **Trains via GRPO (reinforcement learning):** `scripts/train.sh` uses verl's GRPO algorithm with
   `MemoryAgent` as the actor. Rewards are `subsampled_qa_reward_weight` + `gap_fill_reward_weight`
   (lines 106–107), meaning the agent learns which memory drafts improve QA and gap-filling accuracy.

**Segmentation design:** Fixed-window is not adaptive; it does not detect semantic episode boundaries.
This is a deliberate design choice for visual memory — fixed windows may align better with
user interaction patterns in visual rendering contexts. MemOCR remains available for standalone
visual memory experiments and ablation studies.

## Decision

### MemOCR is independent; segmentation choice remains fixed-window

MemOCR's fixed-window segmentation is an independent design choice and remains available for:

1. **Standalone visual memory experiments** (MemOCR as a self-contained system, not integrated with
   HetRep or fluxmem).
2. **Ablation studies** (comparing surprise-based vs. fixed-window in Phase 3 VC if needed).
3. **Fine-tuning benchmarks** (if MemOCR's releases checkpoint underperforms on C-AIMMS tasks).

MemOCR does **not** have a segmentation fallback role for HetRep; this is stated explicitly to
prevent accidental coupling.

### MemOCR supports fine-tuning and training

| Aspect                                 | Capability                                                   | Evidence                                                                                             |
| -------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| **Base model**                         | Qwen2.5-VL-7B-Instruct                                       | MemOCR/README.md:225, train.sh:38 default `MODEL_PATH`                                               |
| **Released checkpoint**                | meituan/MemOCR-7B (fine-tuned from base)                     | MemOCR/README.md:10,21; this is the shipped weight                                                   |
| **Training framework**                 | verl (GRPO, not SFT)                                         | train.sh:11 `verl.trainer.main_ppo`, algorithm.adv_estimator=grpo                                    |
| **Custom data support**                | Yes, swappable parquet train/val files                       | train.sh lines 125, 127: `data.train_files=$TRAIN_PATH`, `data.val_files=$VAL_PATH`                  |
| **Training can start from checkpoint** | Yes, `meituan/MemOCR-7B` or any HF model                     | train.sh line 38: `MODEL_PATH="..."` can point to any HF-compatible checkpoint                       |
| **Separate inference checkpoint**      | Yes, via verl's rollout_ref split                            | verl's standard RLHF pattern: trainable actor weights updated, separate rollout model for generation |
| **Generic SFT support**                | Yes, via verl's inherited `verl/trainer/fsdp_sft_trainer.py` | Not MemOCR-specific; available in vendored verl, can be used for SFT pretraining before GRPO         |

**Consequence:** If Phase 3 finds that Qwen2.5-VL-7B-Instruct (base) or meituan/MemOCR-7B (released)
underperforms on C-AIMMS's visual salience or layout-aware memory tasks, it is feasible to:

1. Run SFT on C-AIMMS episodes to adapt the base model to the domain.
2. Fine-tune via GRPO with LoCoMo-generated (or human) rewards.
3. A/B test the fine-tuned checkpoint against the released one.

This requires compute (GPU + training time) and data (labeled episodes), but is within verl's
capabilities.

### Rendering remains HTTP-based; determinism tradeoff

MemOCR's rendering (`md2img/markdown_api_server.py:50–57`) runs Playwright + Chrome headless on
localhost:9000. This means:

- **Non-deterministic layout rendering.** Chrome may render slightly differently on different
  systems or with different font/driver versions. Running the same episode twice may produce
  pixel-level different images.
- **Requires Chrome/Playwright runtime.** Phase 3 infrastructure setup must install Chromium binary
  (`playwright install chrome`).
- **Performance cost.** Rendering ~100 episodes per run takes wall-clock time (I/O + process startup).

These are **not bugs**, just design constraints documented here for Phase 3 planning.

## Consequences

### Positive

- **MemOCR is research-ready.** Training framework is proven (verl GRPO); fine-tuning is supported;
  released checkpoint (meituan/MemOCR-7B) is publicly available.
- **Independent implementation.** MemOCR's segmentation design (fixed-window) is standalone. Can be
  evaluated separately or used in ablation studies.
- **Flexible checkpoint management.** Can experiment with base model, released checkpoint, or
  fine-tuned variants independently.

### Negative

- **Chrome rendering complexity.** Requires system-level Chromium setup; introduces non-determinism;
  slower than in-process Python rendering would be.
- **Training compute cost.** GRPO with MemOCR-7B on GPU is expensive; Phase 3 experiments may need
  access to high-end hardware (A100/H100).

### Risks

- **Qwen2.5-VL-7B-Instruct base model may be insufficient.** Meituan's GRPO training is not
  reproducible without their original reward data. If MemOCR's released checkpoint alone doesn't
  work well for C-AIMMS tasks, fine-tuning from scratch is necessary.
- **Rendering server crashes silently.** If markdown_api_server.py dies mid-run, MemOCR silently
  drops the render (no exception). This is a latent bug; should add timeout + logging before Phase 3.

## Related

- **ADR 0005:** HetRep architecture and EpisodeEncoder interface.
- **ADR 0003:** MemOCREpisodeProducer sets `turns=[]` because MemOCR's encoding is visual, not turn-based.
- **ADR 0008:** COLM's VC arm specifies layout-aware rendering; MemOCR is an implementation candidate.
