"""HetRepEncoder: orchestrates three-format encoding (VS, HG, VC)."""

from fluxmem.interfaces import EpisodicUnit


class HetRepEncoder:
    """Enrich an EpisodicUnit with heterogeneous representations.

    Per COLM Algorithm 1 line 4: "Encode e_j into (H_j, I_j, v_j) via HETREP".

    Phases:
    - Phase 1 (VS): Vector Store embeddings from all-MiniLM-L6-v2.
    - Phase 2 (HG): Hypergraph structure from LLM extractors.
    - Phase 3 (VC): Visual Canvas rendered markdown + salience.

    Each arm is independently injectable to allow ablation and testing.
    """

    def __init__(self, vs_encoder=None, hg_encoder=None, vc_encoder=None):
        """Initialize encoder with optional arm implementations.

        Args:
            vs_encoder: EpisodeEncoder implementing Vector Store encoding (Phase 1).
            hg_encoder: EpisodeEncoder implementing Hypergraph encoding (Phase 2).
            vc_encoder: EpisodeEncoder implementing Visual Canvas encoding (Phase 3).

        If any arm is None, that format is skipped.
        """
        self.vs_encoder = vs_encoder
        self.hg_encoder = hg_encoder
        self.vc_encoder = vc_encoder

    def encode(self, unit: EpisodicUnit) -> EpisodicUnit:
        """Encode a segmented unit in place (COLM §1.2, Algorithm 1 line 4).

        Populates embedding (VS), hyperedge_density (HG), visual_salience (VC).
        Leaves turns and primary_format untouched; selector decides format downstream.

        Args:
            unit: EpisodicUnit from segmentation stage (EM-LLM).

        Returns:
            Same unit, enriched with encodings.
        """
        if self.vs_encoder is not None:
            unit = self.vs_encoder.encode(unit)
        if self.hg_encoder is not None:
            unit = self.hg_encoder.encode(unit)
        if self.vc_encoder is not None:
            unit = self.vc_encoder.encode(unit)
        return unit
