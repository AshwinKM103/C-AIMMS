"""VisualCanvasStub: Phase 1–2 no-op returning visual_salience=0.0.

Real VC encoder deferred to Phase 3 (blocked on Chromium server + MemOCR checkpoint).

Until Phase 3: visual_salience stays constant across episodes. This is a **stated
approximation** (not a hidden bug) per ADR 0003 user decision. With VC stubbed,
COLM Eq. 6's label f*_j = argmax_f r_j^f is **unmeasurable** for VC, so the selector
is wired and tested but **not trained**; no selector accuracy metric reported
until Phase 3 unblocks it.

See docs/adr/0003-episode-producer-seam-and-visual-salience.md.
"""

from fluxmem.interfaces import EpisodicUnit


class VisualCanvasStub:
    """Phase 1–2 pass-through: sets visual_salience=0.0 only."""

    def encode(self, unit: EpisodicUnit) -> EpisodicUnit:
        """Return unit with visual_salience=0.0 (Phase 1–2 placeholder).

        Args:
            unit: EpisodicUnit.

        Returns:
            Same unit with visual_salience=0.0.
        """
        unit.visual_salience = 0.0
        return unit
