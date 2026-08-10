# ADASTORE three-tier memory hierarchy

Implements COLM §1.3.2. STIM buffers raw turns; capacity overflow evicts by
LRU into MTEM. MTEM ranks episodic units by utility `U(e_j)`; episodes that
clear both the utility gate (`U >= tau_u`) and the recency gate (`r >= tau_r`,
LTSM's own hard gate) are promoted into LTSM's vector store. See
`fluxmem/stim.py`, `fluxmem/mtem.py`, `fluxmem/ltsm.py`.

```mermaid
flowchart TD
    subgraph STIM["STIM — capacity 4"]
        T1["Turn buffer (OrderedDict)"]
    end

    subgraph MTEM["MTEM — capacity 2000"]
        E1["EpisodicUnit store"]
        U["utility_score(e, weights, now)\nU = w1*c + w2*l + w3*d"]
    end

    subgraph LTSM["LTSM — FAISS vector store"]
        V1["is_eligible(e): U >= tau_u AND r >= tau_r"]
        V2["FaissVectorStore (IndexFlatIP, L2-normalized)"]
    end

    Turn["new dialogue turn"] --> T1
    T1 -- "capacity exceeded\n(LRU eviction, push() returns evicted turn)" --> E1
    E1 --> U
    U -- "promotion_candidates(tau_u)" --> V1
    V1 -- "eligible" --> V2
    V1 -- "ineligible" --> E1

    style STIM fill:#e8f0fe,stroke:#4285f4
    style MTEM fill:#fef7e0,stroke:#f9ab00
    style LTSM fill:#e6f4ea,stroke:#34a853
```

## Notes

- STIM's eviction is LRU (`touch()` matters once a read path re-reads STIM,
  out of scope here), not FIFO -- see `fluxmem/stim.py` docstring.
- The utility gate (`U`, composite of three weighted terms) and the recency
  gate (`r`, a separate hard threshold) are deliberately distinct despite
  both deriving from the same `now - last_access` -- `d(e_j)` inside `U` is a
  soft, tradeable addend; `r(e_j)` in the LTSM gate is a hard, non-negotiable
  conjunct. See `fluxmem/ltsm.py` (Step 8) for the full rationale.
