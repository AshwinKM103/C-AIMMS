# Phase 2 Output: Encoder Output Characterization & Semantics

**Status**: ✅ Complete  
**Date**: 2026-08-14  
**Agents**: Explore [Haiku], ml-engineer [Sonnet]

---

## Executive Summary

**Critical Finding**: VC encoder output from MemOCR paper is **NOT a scalar float**, but a **rendered visual image** with layout-guided information density. This changes feature design significantly.

**Current Implementation Status**:

- ✅ **VS (Vector Semantic)**: Fully implemented, produces 384-dim embeddings
- 🔨 **HG (Hierarchical Graph)**: Phase 2 stub (no-op, returns constant 0.0)
- 🔨 **VC (Visual Canvas)**: Phase 2 stub (incorrect design; should output rendered image, not scalar)

**Design Decision**: Phase 3 will design features for **intended future behavior** (Option A), understanding that HG/VC features will be currently untestable (zero variance) until encoders are implemented.

---

## Part A: VS (Vector Semantic) - Implemented ✅

### Output Specification

- **Shape**: (384,) dense vector
- **Type**: numpy.ndarray, float32
- **Normalization**: Raw sentence-transformer output (not cosine-normalized at encode time; L2 normalization applied only in FAISS at retrieval)
- **Model**: all-MiniLM-L6-v2 (384-dim sentence embedding)
- **Memory**: ~1.5 KB per episode

### Implementation

- **File**: `hetrep/vs/encoder.py`
- **Input**: EpisodicUnit with concatenated turn text (Phase 1 approximation; full summary deferred)
- **Processing**: `SentenceTransformer("all-MiniLM-L6-v2").encode(summary, convert_to_numpy=True)`
- **Output field**: `unit.embedding`

### Semantics (ml-engineer interpretation)

**What it encodes**: Dense semantic content of the episode's dialogue. Captures semantic similarity via inner product, but loses explicit structure (entities, relations, temporal ordering).

**Strengths**:

- Lightweight (fixed 384-d regardless of episode length)
- O(1) comparison cost once FAISS-indexed
- Deterministic given fixed model checkpoint
- No LLM/GPU serving needed (runs locally)

**Limitations**:

- No explicit entity/relation information
- Phase 1 "summary" is verbatim turn concatenation (no LLM compression yet)
- Effective context window not verified (sentence-transformers may truncate long episodes silently)
- No temporal or hierarchical structure exposed

**Best for**: Dense semantic retrieval via FAISS (nearest-neighbor search).

**Reasoning capability**: Good for semantic similarity matching; poor for entity-centric or multi-hop queries.

**Memory efficiency**: Very high (fixed 384-d, ~1.5 KB).

---

## Part B: HG (Hierarchical Graph) - Phase 2 Stub 🔨

### Current Status: No-Op Stub

- **Implementation**: `hetrep/hg/encoder.py:54` returns `unit` unchanged (does nothing)
- **Output field**: `unit.hyperedge_density` = 0.0 (constant, for all episodes)
- **Variance**: ZERO (untestable in Phase 3 feature engineering)

### Intended Design (from HyperMem Paper & ADR 0007)

**Intended Output**: Three-layer hypergraph structure (JSON-serializable via Pydantic)

```
Layer 1 (Facts):
  - FactNode: content, confidence, temporal/spatial tags, keywords
  - FactHyperedge: typed relations connecting fact groups

Layer 2 (Episodes):
  - EpisodeNode: participants, summary, type, keywords
  - EpisodeHyperedge: roles (initiating, developing, climax, key_moment, etc.)
  - Coherence scores: float [0, 1]

Layer 3 (Topics):
  - TopicNode: title, summary, episode_ids, keywords
```

**Intended Extraction Pipeline** (currently unimplemented):

1. Episode → FactExtractor (LLM-backed) → extract facts with confidence scores
2. Facts → FactHyperedges (typed relations)
3. Episode metadata → EpisodeNode
4. Facts/Episodes → TopicExtractor → topics with hierarchy
5. Compute `hyperedge_density = |incident_edges| / |possible_edges|` [0, 1]

**Serialization**: JSON via `.to_dict()` method (human-readable, diff-friendly)

### Semantics (ml-engineer interpretation of intended design)

**What it encodes** (when implemented):

- Typed entities (facts, episodes, topics) across three abstraction levels
- Higher-order relations via hyperedges (not just pairwise)
- Explicit confidence scores and role annotations
- Temporal/spatial metadata for facts
- Episode coherence scores

**Strengths** (of the design, once implemented):

- Explicit, structured representation of relational information
- Supports entity-centric and multi-hop queries
- Three-level hierarchy enables coarse-to-fine reasoning
- Types and roles provide semantic clarity
- Hyperedge representation captures group relations, not just pairwise

**Limitations** (even when implemented):

- LLM-dependent extraction (cost, latency, quality variance)
- Variable size (unbounded node/edge count, episodic)
- Requires graph traversal for queries (not O(1) lookup)
- JSON serialization overhead vs. dense representations
- Quality depends on FactExtractor/TopicExtractor quality

**Best for**: Entity-centric and relational multi-hop reasoning; structured knowledge representation.

**Reasoning capability**: Excellent for relational reasoning (by design), once implemented.

**Memory efficiency**: Low as raw structure (unbounded JSON); the derived scalar feature (hyperedge_density) is maximally efficient (1 float).

### Current Implication for Phase 3

⚠️ **HG currently produces ZERO VARIANCE** (hyperedge_density = 0.0 for all episodes)

- Any feature based on HG discriminating episodes = constant signal
- **Can be designed NOW for future implementation**
- **Will be untestable until Phase 2 encoder lands**

---

## Part C: VC (Visual Canvas) - Phase 2 Stub, Incorrect Design 🔨

### Current Implementation (Wrong)

- **File**: `hetrep/vc/stub.py`
- **Current behavior**: Sets `unit.visual_salience = 0.0` unconditionally
- **Status**: Complete stub (constant for all episodes)
- **Variance**: ZERO (untestable)

### Correct Design (from MemOCR Paper)

**Paper Reference**: Shi et al. (2026), "MemOCR: Layout-Aware Visual Memory for Efficient Long-Horizon Reasoning"

**Intended Output**: Rendered visual memory image V_T (Equation 6, Section 3.2)

```
Input: Rich-text memory M_t^RT
  (formatted text with headers, bold, indentation, structure)
  ↓
Renderer R (lightweight, no LLM)
  ↓
Output: V_T = 2D rendered image/canvas
  - Visual representation of memory with layout-guided importance
  - Crucial evidence: larger scale + high-visibility regions
  - Auxiliary details: smaller scale + lower-priority regions
  - Resolution control: downsampling to fit visual token budget
```

**Intended Processing Pipeline** (from MemOCR Section 3.2):

1. Memory drafting (text domain): M_t^RT with formatting
2. Rendering (vision domain): R(M_t^RT) → V_T (2D image)
3. Vision encoding: Extract visual/spatial features from V_T
4. Agent reasoning: Use V_T as visual working context

**Output Characteristics**:

- **Shape**: Variable (depends on rendered layout), but typically 400-800 pixels wide × 300-600 pixels tall (based on budget)
- **Type**: 2D image (numpy array or PIL Image, typically uint8 RGB)
- **Information**: Layout-stratified information density
  - High-visibility regions: most important information
  - Low-visibility regions: auxiliary details
- **Visual tokens**: Measured by patch-count (28×28 patches, per MemOCR budget accounting)

**Key Property** (from paper):

- "Crucial evidence is rendered with larger scale and placing it in high-visibility regions"
- "Auxiliary details are written in lower-priority regions"
- Layout encodes information importance, not just content

### Why ADR 0003's Scalar Approach Is Wrong

ADR 0003 specifies: `visual_salience = rendered_pixel_area / configured_patch_budget`

**Problem**: This is **pixel occupancy**, not visual semantics.

- Two renders with identical pixel budget but very different content → same score
- Loses all information about layout, importance, spatial structure
- "A large blank render would score as highly as a dense one" (even ADR 0003 admits this)

**Correct Approach**: VC outputs a **rendered canvas** that can be:

1. Used directly by vision models for reasoning
2. Processed to extract spatial/visual features (layout prominence, spatial coherence, etc.)
3. Analyzed for layout properties (what fraction of high-visibility regions is filled?, etc.)

### Semantics (ml-engineer interpretation of correct design)

**What it encodes** (when implemented correctly):

- Visual/spatial representation of memory with layout-guided importance
- High-visibility regions = critical information (layout-encoded priority)
- Low-visibility regions = auxiliary details
- Spatial structure and layout patterns
- Information density stratification (not uniform)

**Strengths** (of correct design):

- Layout-aware importance encoding (visual cues guide attention)
- Multimodal grounding (visual + textual information)
- Spatial reasoning capability (layout, prominence, structure)
- Natural for vision-capable agents

**Limitations** (even when implemented correctly):

- Rendering cost (Chromium server needed per MemOCR pipeline)
- Variable output size (depends on rendering resolution)
- Quality depends on rendering fidelity and formatting
- Requires vision encoder for feature extraction
- No explicit entity/relation structure (unlike HG)

**Best for**: Visual/spatial reasoning; multimodal memory representation; layout-aware information density.

**Reasoning capability**: Good for spatial/layout reasoning (by design); complement to text-based reasoning.

**Memory efficiency**: Moderate (rendered image is largest of three formats, but separate from feature vector).

### Current Implication for Phase 3

⚠️ **VC currently produces ZERO VARIANCE** (visual_salience = 0.0 for all episodes)

- **Scalar float approach is incorrect** per MemOCR paper
- **Should output rendered image**, not scalar
- **Can be designed NOW for future implementation** (rendered canvas output)
- **Will be untestable until Phase 3 encoder (MemOCR + Chromium) lands**

---

## Part D: Comparative Analysis

### Implementation Status vs. Semantics

| Encoder | Implemented? | Output Type               | Current Signal | Design Quality                     | Testability (Phase 3)        |
| ------- | ------------ | ------------------------- | -------------- | ---------------------------------- | ---------------------------- |
| **VS**  | ✅ Yes       | 384-d dense vector        | ✓ Varying      | ✓ Good                             | ✓ Testable now               |
| **HG**  | ❌ No (stub) | 3-layer hypergraph (JSON) | ✗ Constant 0.0 | ✓ Good                             | ❌ Zero variance             |
| **VC**  | ❌ No (stub) | 2D rendered image         | ✗ Constant 0.0 | ❌ Wrong (scalar instead of image) | ❌ Zero variance + incorrect |

### Information Content (What Each Preserves/Loses)

| Dimension            | VS                    | HG                    | VC                   |
| -------------------- | --------------------- | --------------------- | -------------------- |
| **Semantic content** | ✓ Preserved           | ✗ Lost                | ✓ Preserved (visual) |
| **Entities**         | ✗ Hidden in embedding | ✓ Explicit nodes      | ✓ Visual prominence  |
| **Relations**        | ✗ Hidden in embedding | ✓ Explicit hyperedges | ✓ Spatial structure  |
| **Temporal info**    | ✗ Implicit            | ✓ Explicit tags       | ✓ Visual layout      |
| **Hierarchy**        | ✗ Flat                | ✓ Three layers        | ✓ Layout regions     |
| **Visual structure** | ✗ None                | ✗ None                | ✓ Layout-aware       |

### Reasoning-Pattern Suitability (Design Intentions)

| Pattern                  | VS           | HG           | VC           |
| ------------------------ | ------------ | ------------ | ------------ |
| Semantic similarity      | ✓✓ Excellent | ✗ Poor       | ✓ Good       |
| Entity-centric queries   | ✗ Poor       | ✓✓ Excellent | ✗ Poor       |
| Multi-hop reasoning      | ✗ Poor       | ✓✓ Excellent | ✗ Poor       |
| Relational reasoning     | ✗ Poor       | ✓✓ Excellent | ✗ Poor       |
| Spatial/layout reasoning | ✗ None       | ✗ None       | ✓✓ Excellent |
| Visual grounding         | ✗ None       | ✗ None       | ✓✓ Excellent |

---

## Part E: Feature Engineering Implications (Phase 3)

### Design Constraint: Option A (Future Intended Behavior)

Phase 3 data-scientist will design features for:

1. **VS-derived features** (testable now; real varying signals)
2. **HG-intended features** (designed for future implementation; currently untestable)
3. **VC-intended features** (designed for correct canvas output; currently untestable)

### Feature Categories to Design

**From VS (testable)**:

- Embedding dimensionality
- Embedding isotropy (are all directions equally used?)
- Embedding-to-embedding redundancy (if needed)
- Nearest-neighbor retrieval precision

**From HG (designed for future; untestable now)**:

- Graph node density (facts/episodes per unit)
- Graph edge density (relations per node)
- Hierarchy depth and breadth
- Hyperedge types and distribution
- Confidence/coherence score distributions

**From VC (designed for correct canvas; untestable now)**:

- Canvas rendering size (pixels, aspect ratio)
- Information density stratification (high-visibility vs. low-visibility region fill)
- Layout prominence scores (what fraction of high-salience regions are used?)
- Spatial coherence and structure metrics

### What NOT to Design

❌ **Scalar pixel occupancy** (ADR 0003 approach) — this loses semantic information
❌ **Features based on current 0.0 constants** — they have zero variance
✅ **Features that measure intended output properties** — even if untestable now

---

## Part F: Critical Decision for Phase 3

**Two-Hypothesis Situation (Evidence Discipline)**:

**Hypothesis A** (chosen): Design features for **intended future HG/VC behavior**

- ✅ Features are semantically meaningful and will activate when encoders land
- ❌ Currently untestable (zero variance on stub outputs)
- ⏳ Becomes operational in Phase 3+ once HG/VC encoders implemented

**Hypothesis B** (rejected): Design features for **current VS-only**

- ✅ Immediately testable
- ❌ Excludes HG/VC reasoning capabilities
- 🔄 Requires redesign later when encoders land

**Your decision**: Option A (features for future)

---

## Part G: Files for Reference

**MemOCR Paper Reference**:

- File: `/home/durgesh/aditya/C-AIMMS/docs/cited-papers/MemOCR Layout-Aware Visual Memory for Efficient Long-Horizon Reasoning-with-annotations.pdf`
- Key sections: Section 2.1 (Raw History Memory), Section 3.1 (Memory Drafting), Section 3.2 (Memory Reading in Vision Domain), Equation 6 (Rendered memory image)

**Current Implementation Files**:

- VS: `hetrep/vs/encoder.py`
- HG: `hetrep/hg/encoder.py` (stub)
- VC: `hetrep/vc/stub.py` (stub)

**Data Models**:

- `fluxmem/interfaces.py` (EpisodicUnit definition)
- `HyperMem/hypermem/structure.py` (Hypergraph schema, reused by hetrep/hg)

---

## Phase 2 Deliverables ✅

| Artifact                   | Status                  | Content                                |
| -------------------------- | ----------------------- | -------------------------------------- |
| Explore Report             | ✅ Complete             | VS/HG/VC output specs, files located   |
| ML-engineer Interpretation | ✅ Complete             | Semantic analysis, corrected VC design |
| Phase 2 Summary            | ✅ Complete (this file) | Gap analysis, design implications      |

---

## Ready for Phase 3

**Next**: Spawn data-scientist [Sonnet] + research-analyst [Haiku]

**Scope**: Design M ≤ 20 features for selector MLP

- Measure properties of VS/HG/VC outputs
- Option A: Design for intended (future) encoder behavior
- Modality-agnostic features (work for all three)
- Account for reward function: task_accuracy / memory_budget

**Constraint**: HG/VC features will be untestable (zero variance) until Phase 3 encoders land, but designed for correct semantics.

---

**Phase 2 Status**: ✅ COMPLETE  
**Ready for**: Phase 3 (Feature Engineering - CORE WORK)
