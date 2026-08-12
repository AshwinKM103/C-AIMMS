"""HypergraphEncoder: node and edge extraction from episodes (Phase 2).

COLM §1.2.1: extract nodes (facts, episodes, topics) and hyperedges.
Extractors injected behind Protocols to avoid hard imports of HyperMem.extractors
(which require live LLM providers and HTTP servers at import time).

Per ADR 0007: reuse HyperMem's structure.py data model, not its pipeline.
"""

from fluxmem.interfaces import EpisodicUnit


class HypergraphEncoder:
    """Extract and store hypergraph structure per episode.

    Phase 2 requires:
    - FactExtractor Protocol: LLM-based fact extraction.
    - TopicExtractor Protocol: LLM-based topic matching and creation.
    - (Episodes are implicitly the input unit.)

    Outputs:
    - Hypergraph(nodes, hyperedges) persisted as JSON via to_dict().
    - hyperedge_density (number of incident edges / possible edges, clipped [0,1]).
    """

    def __init__(self, fact_extractor=None, topic_extractor=None):
        """Initialize encoder with optional injected extractors.

        Args:
            fact_extractor: Callable(episode) -> List[Fact]. If None, no facts extracted.
            topic_extractor: Callable(episode, existing_topics) -> List[Topic].
                           If None, no topics extracted.
        """
        self.fact_extractor = fact_extractor
        self.topic_extractor = topic_extractor

    def encode(self, unit: EpisodicUnit) -> EpisodicUnit:
        """Extract hypergraph structure and populate unit.hyperedge_density.

        Phase 2 stub: no-op. Implemented in Phase 2.

        Args:
            unit: EpisodicUnit with turns populated.

        Returns:
            Same unit with hyperedge_density populated (Phase 2).
        """
        # Phase 2 implementation will:
        # 1. Call fact_extractor, topic_extractor on unit.
        # 2. Build HyperMem.structure.Hypergraph.
        # 3. Compute hyperedge_density = |incident_edges| / |possible_edges|.
        # 4. Persist to HG JSON file.
        # 5. Set unit.hyperedge_density.
        return unit
