"""HetRep: Heterogeneous Representation encoding for episodic memory.

Three-format encoding pipeline producing (H_j, I_j, v_j) per episode:
- VS (Vector Store): all-MiniLM-L6-v2 embeddings for dense retrieval.
- HG (Hypergraph): node and edge structures for graph-based retrieval.
- VC (Visual Canvas): rendered markdown + layout-aware salience for visual reasoning.

Per COLM §1.2, upstream of fluxmem; populates EpisodicUnit fields:
- embedding: VS encoder output
- hyperedge_density: HG encoder output
- visual_salience: VC encoder output (stubbed to 0.0 in Phase 1–2)

Phases sequenced by infrastructure cost (0-server → 1-server → 3-server).
"""

__version__ = "0.1.0"
