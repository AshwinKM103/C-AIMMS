"""HetRepStore: on-disk layout and manifest management (no pickle).

Artifacts persisted as:
- VS: .npy per shard + JSON manifest (indices rebuilt on load).
- HG: JSON via Hypergraph.to_dict() (round-trip verified).
- VC: PNG + JSON sidecar.

All under /mnt/ssd/users/durgesh/, never /home (disk space).
No .pkl files anywhere per storage-invariants.md.
"""

import json
from pathlib import Path

from hetrep.artifacts import HetRepArtifact


class HetRepStore:
    """Manage on-disk storage and manifest for heterogeneous encodings."""

    def __init__(self, root_dir: Path):
        """Initialize store rooted at the given directory.

        Args:
            root_dir: Base directory for all artifacts (/mnt/ssd/users/durgesh/hetrep/).
        """
        self.root_dir = Path(root_dir)
        self.manifest_path = self.root_dir / "manifest.json"
        self.vs_dir = self.root_dir / "vs"
        self.hg_dir = self.root_dir / "hg"
        self.vc_dir = self.root_dir / "vc"

        # Create directories
        for d in [self.vs_dir, self.hg_dir, self.vc_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save_artifact(self, artifact: HetRepArtifact) -> None:
        """Write artifact to manifest (paths assumed already persisted by encoders)."""
        manifest = self._load_manifest()
        manifest[artifact.episode_id] = artifact.to_dict()
        self._save_manifest(manifest)

    def load_artifact(self, episode_id: str) -> HetRepArtifact | None:
        """Retrieve artifact record from manifest."""
        manifest = self._load_manifest()
        if episode_id in manifest:
            return HetRepArtifact.from_dict(manifest[episode_id])
        return None

    def _load_manifest(self) -> dict:
        """Load manifest JSON; return {} if not yet created."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self, manifest: dict) -> None:
        """Write manifest JSON."""
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
