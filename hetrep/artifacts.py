"""HetRepArtifact: serializable handles for heterogeneous encodings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class HetRepArtifact:
    """Container for serialized heterogeneous representations of an episode.

    Per COLM Eq. 1: M_T = {(H_j, I_j, v_j, f_j)}.
    This class holds the three encodings and their on-disk paths.

    Attributes:
        episode_id: Unique episode identifier.
        vs_path: Path to Vector Store embedding (.npy file).
        hg_path: Path to Hypergraph structure (JSON file).
        vc_path: Path to Visual Canvas rendering (PNG file).
        vc_sidecar_path: Path to VC metadata (JSON sidecar).
    """

    episode_id: str
    vs_path: Path | None = None
    hg_path: Path | None = None
    vc_path: Path | None = None
    vc_sidecar_path: Path | None = None

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict for manifest."""
        return {
            "episode_id": self.episode_id,
            "vs_path": str(self.vs_path) if self.vs_path else None,
            "hg_path": str(self.hg_path) if self.hg_path else None,
            "vc_path": str(self.vc_path) if self.vc_path else None,
            "vc_sidecar_path": str(self.vc_sidecar_path) if self.vc_sidecar_path else None,
        }

    @staticmethod
    def from_dict(data: dict) -> "HetRepArtifact":
        """Deserialize from JSON dict."""
        return HetRepArtifact(
            episode_id=data["episode_id"],
            vs_path=Path(data["vs_path"]) if data.get("vs_path") else None,
            hg_path=Path(data["hg_path"]) if data.get("hg_path") else None,
            vc_path=Path(data["vc_path"]) if data.get("vc_path") else None,
            vc_sidecar_path=Path(data["vc_sidecar_path"]) if data.get("vc_sidecar_path") else None,
        )
