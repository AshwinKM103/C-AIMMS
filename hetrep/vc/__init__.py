"""Visual Canvas (VC) encoding: rendered markdown + layout-aware salience.

COLM §1.2.2: render episode as markdown layout, extract salience via layout structure.

Phase 3: Needs 3 servers (LLM drafter + Chromium renderer + MemOCR VLM).
Phase 1–2: Stubbed to visual_salience=0.0, keeping FEATURE_DIM=7 and selector
head intact per ADR 0003 user decision (memocr-integration-handoff.md).
"""
