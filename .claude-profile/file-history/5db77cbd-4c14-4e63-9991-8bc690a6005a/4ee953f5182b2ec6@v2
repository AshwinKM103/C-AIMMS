# FluxMem: Paper vs. Code Gap Analysis

## Context

The user wants to compare the FluxMem implementation living in `LightMem/src/fluxmem/`
against "the paper" to identify what's missing from the code. The target paper is
`docs/Choosing How to Remember Adaptive Memory Structures for LLM Agents.pdf`
(Lu, Wu, Liu, Xu, Li, Wang, Hu, Ding, Sun, Lu, Zhang — preprint, Feb 2026), which
introduces a framework it names **FluxMem**.

**Critical discovery, already verified by reading both the paper and the code directly:**
the paper's FluxMem and the repo's FluxMem appear to be two unrelated systems that
happen to share a name.

| | Paper (`Choosing How to Remember...pdf`) | Code (`LightMem/src/fluxmem/`) |
|---|---|---|
| Core abstraction | Three-tier memory hierarchy: STIM → MTEM → LTSM | Heterogeneous typed graph: `SemanticNode`, `EpisodicNode`, `ProceduralNode` |
| Central mechanism | Per-turn selection among **Linear / Graph / Hierarchical** structures via a trained MLP classifier (§3.4) | Fixed three-**stage** connectivity pipeline: formation → refinement → consolidation (`stages/stage1_formation.py`, `stage2_refinement.py`, `stage3_consolidation.py`) |
| Fusion/dedup | Beta-Mixture-Model gate (§3.5) over similarity scores, with posterior threshold τ_BMM and `m_min` | No BMM gate anywhere in `agent.py`/`config.py` — no fusion decision at all |
| Convergence signal | None described | **PEMS** metric (`metrics/pems.py`), convergence threshold `pems_threshold=0.01`, cited to a *different* paper (Fang et al., EMNLP 2026) inside `FluxMem.md` |
| Eviction policy | STIM capacity 4, LRU eviction to MTEM (§3.2) | No STIM concept; no capacity-bound short-term buffer |
| Structure selector | 2-layer MLP, 12-dim input, trained offline on interaction-derived reward labels (§3.4, confirmed again in the sibling COLM paper's §1.3.3) | No selector/classifier module anywhere in the package |
| Evaluation | LoCoMo + PERSONAMEM, F1/BLEU-1/ROUGE-L/accuracy, ablations (w/o Linear, w/o Graph, w/o Hierarchy, w/o BMM), τ_BMM sensitivity sweep | No eval harness in `fluxmem/`; only a `PEMSCalculator` |
| Retrieval fusion weights | Dense + BM25 via reciprocal rank fusion (no numeric weights stated) | `dense_weight=1.0`, `bm25_weight=0.5`, `llm_weight=0.3` — a linear weighted sum, not RRF |
| Attribution | Lu et al., ICML 2026 | `FluxMem.md`'s own citation block: Fang et al., EMNLP 2026 — a **different author list and venue** |

So essentially every named component in the paper's Methodology (§3.1–§3.5) — STIM,
MTEM, LTSM, the three interchangeable structure types, the learned selector, the BMM
gate — is absent from the code, and every component the code has (typed graph nodes,
GroundEdge/DistillEdge/StepLinkEdge, the 3-stage online/online/offline pipeline, PEMS)
is absent from the paper. This isn't a partial implementation gap; it's two designs.

Note also: the project's own COLM submission
(`docs/COLM_Cognitive_AI_Memory_Architecture_Project.pdf`) cites "FluxMem (Lu et al.,
2026)" and explicitly extends its selector features "used by FluxMem" — i.e. the COLM
paper's authors believe the *Lu et al.* FluxMem (STIM/MTEM/LTSM one) is the baseline
being built on. If the codebase intends to implement that FluxMem, it currently
doesn't.

## Goal of this task

Produce a written gap-analysis report (research workflow: this closes with a document,
not a code change — per `.claude/rules/evidence-discipline.md`'s dev-vs-research
distinction) that:

1. States the naming collision plainly, with the evidence table above.
2. For each paper component (§3.1–§3.5, plus §4 eval setup), gives an explicit
   Present / Absent / Partial verdict against the code, citing file:line.
3. Lists what would need to be built for the code to actually match the paper's
   FluxMem, ordered as: (a) STIM buffer + LRU eviction, (b) MTEM with the three
   swappable structure representations (Linear/Graph/Hierarchical), (c) the MLP
   structure selector + offline reward-derived training loop, (d) LTSM eligibility
   pruning, (e) the BMM fusion gate, (f) a LoCoMo/PERSONAMEM eval harness matching
   the paper's metrics.
4. Flags the citation mismatch (`FluxMem.md` credits Fang et al. EMNLP 2026, not Lu et
   al. ICML 2026) as something the user should resolve/confirm before this goes
   further, since it affects what "matching the paper" even means.

## Steps

1. Re-confirm remaining unread code: `graph/nodes.py`, `graph/edges.py`,
   `graph/memory_graph.py`, `retrieval/*.py`, `metrics/pems.py`, `interfaces/*.py` —
   quick reads to fill in file:line citations for the Present/Absent table (most
   already characterized via `agent.py`/`config.py`/`FluxMem.md`, but component-level
   citations need exact anchors).
2. Re-read paper §3.2–§3.5 and §4 closely (already extracted, pages 3-8) to pull exact
   parameter names/values (STIM capacity=4, τ_BMM=0.5, m_min=1, selector hidden
   size, 12-dim feature vector) for the citation table.
3. Write the report to `docs/fluxmem-paper-vs-code-gap-analysis.md` (new file, follows
   the project's existing `docs/` convention for analysis docs like
   `docs/related-work-agent-memory.md`).
4. Do not touch `LightMem/` code — it's a separate git repo per root `CLAUDE.md`
   ("commit there, not here"), and this task is analysis, not implementation.

## Verification

- Every claim in the report traces to either a page/section of the paper or a
  `file:line` in the code — no unverified assertions (evidence-discipline rule).
- User reviews the report and decides: (a) rename/re-scope the code's FluxMem to
  avoid the collision, (b) build the paper's FluxMem as a new module, or (c) update
  the COLM paper's citation if it meant the EMNLP FluxMem all along. This plan does
  not decide that — it surfaces the choice.
