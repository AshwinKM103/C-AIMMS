"""Hypergraph (HG) encoding: node and edge structures for graph-based retrieval.

COLM §1.2.1: extract nodes (topics, facts, episodes) and hyperedges.
Injected extractors allow reuse of HyperMem.extractors logic without importing
the full LoCoMo pipeline (which requires live LLM and HTTP servers).

Phase 2: Needs one LLM endpoint for extraction.
"""
