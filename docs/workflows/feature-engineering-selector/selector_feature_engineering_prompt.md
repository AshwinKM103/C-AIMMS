# Selector MLP Feature Engineering for C-AIMMS

## Objective

Design a novel feature set for a selector MLP that can choose between three heterogeneous encodings—**Vector Semantic (VS)**, **Hierarchical Graph (HG)**, and **Visual/Conceptual (VC)**—in the C-AIMMS system. This is a bottom-up, data-driven feature engineering exercise that extracts and derives features from the **raw outputs** of HyperMem, MemOCR, and VectorDB independently, without bias from prior architectural decisions or ADRs.

## Explicit Non-Bias Clause

**This prompt will read only:**
- Academic papers (FluxMem, C-AIMMS, HyperMem, MemOCR reference papers)
- Code implementations (selector, encoders, extractors)
- Data structures and outputs

**This prompt will NOT:**
- Reference past ADRs, design decisions, or architectural assumptions
- Assume constraints from prior analysis
- Inherit feature choices from existing work
- Use config flags or workarounds as justification

If a past decision conflicts with what the data reveals, the data wins.

---

## Phase 1: Understand the Baseline (FluxMem Selector)

### Task 1.1: Extract FluxMem's 12 Features
**Read:** `fluxmem/features.py` (lines 1–end)

List each of the 12 features with:
- Name and mathematical definition
- What modality/representation it measures (linear, graph, hierarchical)
- Input type (what does it consume?)
- Output range (scalar, vector, distribution?)
- Semantic meaning (why does it differentiate these modalities?)

### Task 1.2: Study FluxMem Selector Implementation
**Read:** `fluxmem/selector.py` (lines 1–end)

Document:
- MLP architecture (layers, activation, input/output shapes)
- How the 12 features are preprocessed/normalized before passing to MLP
- Training signal (what supervision drives the selector?)
- Inference: how does the selector output map to a routing decision?

### Task 1.3: Read FluxMem Paper Section on Feature Design
**Read:** The FluxMem paper (Yue et al., if available in `docs/` or references), Section on feature selection and the rationale for the 12 features.

Capture:
- Why these 12 features were chosen
- What assumptions they make about the input/output space
- Any limitations or caveats mentioned

---

## Phase 2: Map C-AIMMS Encoders and Their Outputs

### Task 2.1: Vector Semantic (VS) Encoder
**Read:** `hetrep/` (all Python files defining the VS encoder)

Document the **output structure** of VS:
- Input: raw text or episode description
- Processing: embedding model, pooling, normalization
- Output: shape, range, semantic properties (e.g., cosine-normalized, dense embedding)
- Metadata available: confidence, sparsity, dimensionality

### Task 2.2: Hierarchical Graph (HG) Encoder
**Read:** `HyperMem/hypermem/structure.py` and `HyperMem/hypermem/extractors/` (all extractor files)

Document the **output structure** of HG:
- Input: episode or sequence
- Processing: what nodes, edges, hierarchy are extracted?
- Output: graph representation (adjacency matrix, edge list, node features)
- Metadata available: graph density, node count, edge count, hierarchy depth, connectivity patterns

### Task 2.3: Visual/Conceptual (VC) Encoder
**Read:** `MemOCR/` (visual rendering and encoding pipeline)

Document the **output structure** of VC:
- Input: episode sequence
- Processing: image rendering, OCR/visual feature extraction
- Output: image features (CNN features, attention maps, bounding boxes) or conceptual embeddings
- Metadata available: rendering dimensions, feature maps per layer, spatial structure

### Task 2.4: Understand the "Final Outputs"
**Read:** 
- `fluxmem/fusion.py` (if exists) or relevant integration points
- `fluxmem/memocr_episodes.py` (interface between MemOCR and fluxmem)
- `tests/test_selector.py` (example inputs/outputs to the selector)

Capture:
- What does "final output of HyperMem/MemOCR/VectorDB" mean concretely?
- Are these intermediate representations or task-specific outputs?
- What shapes, types, and ranges do they have?

---

## Phase 3: Feature Engineering (Bottom-Up)

### Task 3.1: Analyze Representation Geometry
For each encoder (VS, HG, VC), compute:

**Intrinsic Properties:**
1. **Dimensionality**: size of the representation (embedding dim, graph nodes, image channels)
2. **Sparsity**: ratio of non-zero elements (for embeddings, relative magnitudes; for graphs, edge density; for images, active feature maps)
3. **Structure**: measure of internal organization (entropy for embeddings, clustering coefficient for graphs, spatial coherence for images)
4. **Scale**: typical magnitude of values (L2 norm for embeddings, edge weights for graphs, pixel intensities for images)

**Relative Properties:**
5. **Stability**: consistency across runs or perturbations (measure variance in repeated forward passes)
6. **Alignment**: how well does the representation align with a reference/ground truth? (cosine similarity to canonical vectors, if available)
7. **Separability**: can similar/dissimilar examples be distinguished? (compute silhouette score or Fisher's discriminant if labeled data exists)

### Task 3.2: Design Cross-Modality Comparison Features
These features measure **comparative properties** without assuming one modality is "correct":

**Redundancy:**
1. **Information overlap**: mutual information or cosine distance between pairs of encoders (VS vs HG, VS vs VC, HG vs VC)
2. **Complementarity**: how much does each encoder add? (compute explained variance when combined)

**Efficiency:**
3. **Compression ratio**: information per unit of representation (task accuracy per parameter, per dimension, per memory footprint)
4. **Latency**: time to produce the representation (wall-clock time or computational complexity proxy)

**Robustness:**
5. **Noise sensitivity**: how much do outputs change when inputs are perturbed? (add small noise, measure output delta)
6. **Uncertainty**: does the encoder produce confidence/entropy estimates? (aggregate and normalize)

### Task 3.3: Domain-Specific Features for Selection
These capture **task-relevant properties** that might drive selector decisions:

**For VS (Vector Semantics):**
- Embedding isotropy (are all directions equally used?)
- Nearest-neighbor retrieval precision (if a labeled dataset exists)
- Language model likelihood (if embeddings feed an LLM, does it assign high probability?)

**For HG (Hierarchical Graph):**
- Graph complexity (number of levels, branching factor, average path length)
- Node semantic coherence (do connected nodes represent related concepts?)
- Hierarchy quality (is the hierarchy meaningful or degenerate?)

**For VC (Visual/Conceptual):**
- Visual coverage (what fraction of the episode is visually rendered?)
- Concept alignment (do extracted concepts match domain expectations?)
- Rendering quality (sharpness, readability of OCR if text-based)

### Task 3.4: Synthesize Feature Set
Combine the above into a **unified feature vector** that:
1. Is **modality-agnostic**: uses only properties that can be computed for all three encoders
2. Is **normalized**: all features are scaled to [0, 1] or [−1, 1]
3. Is **interpretable**: each feature has a clear meaning and unit
4. Is **complete**: captures the key dimensions of variation (geometry, overlap, efficiency, robustness, domain fit)
5. Is **minimal**: prioritizes top-k features by importance (e.g., variance, mutual information with selector targets)

**Deliverable**: A concrete list of M features (where M ≤ 20), with:
- Feature name and definition
- Computation formula or pseudocode
- Expected range and units
- Why it matters for selecting among VS, HG, VC

---

## Phase 4: Validation and Refinement

### Task 4.1: Test Feature Extraction on Real Data
**Read:** `tests/test_selector.py`

Run the feature extraction on sample inputs and outputs from each encoder. Verify:
- All features compute without error
- Ranges are as expected
- Features are uncorrelated (if correlation exists, drop redundant ones)
- Features separate different encoder modalities (can you predict which encoder was used from features alone?)

### Task 4.2: Comparison to FluxMem Features
Create a **feature mapping table**:
- FluxMem feature → C-AIMMS counterpart (if applicable)
- Why the C-AIMMS feature differs (modality, output structure, use case)
- Is there a feature in C-AIMMS with no FluxMem equivalent? (new discovery)

### Task 4.3: Identify Gaps
List properties or distinctions that **cannot** be captured by the current feature set. For each gap:
- What would be needed to capture it?
- Is it worth adding complexity for?

---

## Phase 5: Output Specification

### Deliverables

1. **Feature Engineering Report** (with sections for each phase)
   - FluxMem baseline analysis
   - Encoder output structures (VS, HG, VC)
   - Derived feature set (M features with definitions, pseudocode, ranges)
   - Feature importance or prioritization
   - Validation results

2. **Feature Extraction Code Skeleton**
   - Python functions for each feature, parameterized by encoder type
   - Normalization/preprocessing logic
   - Pseudocode or actual code (candidate for `fluxmem/features_v2.py` or similar)

3. **Feature Mapping Reference**
   - How C-AIMMS features relate to FluxMem features
   - Documentation of why features are different or new

4. **Testing Plan**
   - How to validate that features are computed correctly
   - Expected failure modes and edge cases
   - Suggestions for ablation or sensitivity analysis

---

## Codebase Navigation (Relative Paths)

| Component | Path | Key Files |
|-----------|------|-----------|
| **FluxMem Selector** | `fluxmem/` | `selector.py`, `features.py`, `fusion.py` |
| **FluxMem Tests** | `tests/` | `test_selector.py` |
| **HyperMem (Graph)** | `HyperMem/` | `hypermem/structure.py`, `hypermem/extractors/*.py` |
| **MemOCR (Visual)** | `MemOCR/` | `scripts/eval.sh`, `memory_img*.py`, rendering pipeline |
| **HetRep (Semantic)** | `hetrep/` | VS encoder implementation |
| **ADRs (for context only, not bias)** | `docs/adr/` | ADR 0004, 0007, 0010, 0011, 0014 (reference, don't inherit constraints) |

---

## Success Criteria

- [ ] All FluxMem 12 features understood and documented
- [ ] VS, HG, VC encoder outputs characterized with concrete shapes/types/ranges
- [ ] At least 12–16 new C-AIMMS features designed, justified, and computable from encoder outputs
- [ ] Features are modality-agnostic and interpretable
- [ ] Features are validated on sample data (no crashes, expected ranges)
- [ ] Feature set is documented with pseudocode or runnable code
- [ ] Gaps identified and addressed (or acknowledged as future work)

---

## Implementation Next Steps (Post-Feature Engineering)

Once this feature engineering is complete:
1. Implement feature extraction as a reusable module (`fluxmem/features_caimms.py` or expand `features.py`)
2. Collect ground-truth selector targets (which encoder should the MLP prefer for each episode?)
3. Train a new selector MLP with the C-AIMMS feature set
4. Benchmark: compare selector accuracy and efficiency vs. the original FluxMem selector
5. Iterate: add/remove features based on feature importance analysis

---

## How to Use This Prompt with Optimization

1. **Run with `/prompt-engineer`** before handing to an agent:
   - Compress overlapping sections
   - Reorder phases for better context efficiency
   - Remove redundancy between "Understand the Baseline" and "Comparison"
   - Optimize for a specific agent (ml-engineer, data-scientist, or research-analyst)

2. **Run with agent**: Pick from the recommended list below.

3. **Iterate**: If results are incomplete, refine sections and re-optimize.

