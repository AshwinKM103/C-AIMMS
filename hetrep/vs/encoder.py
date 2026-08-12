"""VectorStoreEncoder: all-MiniLM-L6-v2 embeddings (Phase 1).

COLM §1.2.3: v_j = Enc(summary(e_j)) via all-MiniLM-L6-v2 (384-d).

Phase 1 (no servers needed) gates on sentence-transformers and faiss.
Embeddings are stored as .npy per shard, indices rebuilt on load.
"""

from fluxmem.interfaces import EpisodicUnit


class VectorStoreEncoder:
    """Encode episodes into 384-d dense vectors via all-MiniLM-L6-v2.

    Phases:
    - Phase 1: Implemented. Populates unit.embedding.
    - Later: Integrate with fluxmem's FaissVectorStore for hybrid retrieval.
    """

    def __init__(self, embedder=None):
        """Initialize encoder with optional injected embedder (for testing).

        Args:
            embedder: Callable (text: str) -> np.ndarray(384,). If None,
                      loads all-MiniLM-L6-v2 on first encode.
        """
        self.embedder = embedder
        self._model = None

    def encode(self, unit: EpisodicUnit) -> EpisodicUnit:
        """Populate unit.embedding by embedding episode text.

        Summary text: concatenates unit.turns as temporary approximation.
        COLM says v_j = Enc(summary(e_j)), but EpisodicUnit has no summary.

        Alternative (rejected for Phase 1): LLM-generate summary → reintroduces
        LLM dependency Phase 1 exists to avoid. If VS recall is poor,
        this is the first thing to revisit (drop-in change behind _summarize()).

        Args:
            unit: EpisodicUnit with turns populated.

        Returns:
            Same unit with embedding set.
        """
        summary = self._summarize(unit)
        if self.embedder is not None:
            unit.embedding = self.embedder(summary)
        else:
            # Phase 1: lazy load model on first call.
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            unit.embedding = self._model.encode(summary, convert_to_numpy=True)
        return unit

    @staticmethod
    def _summarize(unit: EpisodicUnit) -> str:
        """Concatenate turn text as summary input (Phase 1 approximation).

        COLM §1.2.3: v_j = Enc(summary(e_j)). We approximate summary by
        joining turn text; a better approach (LLM compression) is deferred.
        """
        if not unit.turns:
            return ""
        return " ".join(turn.get("text", "") for turn in unit.turns)
