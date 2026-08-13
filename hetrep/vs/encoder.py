"""VectorStoreEncoder: all-MiniLM-L6-v2 embeddings (Phase 1).

COLM §1.2.3: v_j = Enc(summary(e_j)) via all-MiniLM-L6-v2 (384-d).

Phase 1 (no servers needed) gates on sentence-transformers and faiss.
Embeddings stored as .npy per shard; indices rebuilt on load (no pickle).
"""

from typing import Protocol

import numpy as np

from fluxmem.interfaces import EpisodicUnit


class Embedder(Protocol):
    """Dependency injection boundary for embeddings (testing vs. real model)."""

    def __call__(self, text: str) -> np.ndarray:
        """Embed text to 384-d vector.

        Args:
            text: Input text.

        Returns:
            np.ndarray shape (384,).
        """
        ...


class VectorStoreEncoder:
    """Encode episodes into 384-d dense vectors via all-MiniLM-L6-v2.

    Satisfies EpisodeEncoder Protocol (ADR 0005).

    Phase 1: Implemented. Populates unit.embedding from turn text.
    Later: Integrate with fluxmem's FaissVectorStore for hybrid retrieval.
    """

    def __init__(self, embedder: Embedder | None = None):
        """Initialize encoder with optional injected embedder (for testing).

        Args:
            embedder: Callable (text: str) -> np.ndarray(384,). If None,
                      loads all-MiniLM-L6-v2 on first encode.
        """
        self.embedder = embedder
        self._model = None

    def encode(self, unit: EpisodicUnit) -> EpisodicUnit:
        """Populate unit.embedding via all-MiniLM-L6-v2 (COLM §1.2.3).

        Summary text: concatenates unit.turns (Phase 1 approximation).
        COLM: v_j = Enc(summary(e_j)). EpisodicUnit has no summary field.

        Alternative rejected for Phase 1: LLM-generate summary. This would
        reintroduce the LLM dependency Phase 1 exists to avoid. If VS recall
        is poor, this is the first thing to revisit (drop-in behind _summarize).

        Args:
            unit: EpisodicUnit with turns populated (from segmentation stage).

        Returns:
            Same unit with embedding set to 384-d vector.
        """
        summary = self._summarize(unit)
        if self.embedder is not None:
            unit.embedding = self.embedder(summary)
        else:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            unit.embedding = self._model.encode(summary, convert_to_numpy=True)
        return unit

    @staticmethod
    def _summarize(unit: EpisodicUnit) -> str:
        """Concatenate turn text as summary (Phase 1 approximation).

        COLM §1.2.3: v_j = Enc(summary(e_j)). We join turn text; a better
        approach (LLM compression) is deferred. If this causes poor recall,
        revisit by replacing this function body (it is a drop-in change).
        """
        if not unit.turns:
            return ""
        parts = []
        for turn in unit.turns:
            if "user" in turn and turn["user"]:
                parts.append(turn["user"])
            if "assistant" in turn and turn["assistant"]:
                parts.append(turn["assistant"])
        return " ".join(parts)
