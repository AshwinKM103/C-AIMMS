# 0005. HetRep as a sibling package and the `EpisodeEncoder` seam

## Status

Accepted — 2026-08-12. Gates `hetrep/` (Phase 0 tasks T-03, T-04 of the hetrepv2 plan) and the
addition of `EpisodeEncoder` to `fluxmem/interfaces.py`. No other `fluxmem/` module changes.
Supersedes revision 1 of `.claude/prompts/hetrepv2-3format-storage-implementation.md`, which
targeted `fluxmem/HetRep/` — that target is rejected in place, not merely deprioritized.

## Context

`fluxmem/` states, in three independent places, that HETREP (COLM's heterogeneous encoding
stage) is out of scope for the package:

- `fluxmem/__init__.py:5` — _"Does not implement HETREP, ITERRET, or WORKMEM."_
- `fluxmem/ltsm.py:51-53` — the encoding stage that would produce `(H_j, I_j, v_j, f_j)` per COLM
  Eq. 1 is explicitly out of scope; the module docstring adds _"This package builds no BM25
  fusion or vector retriever — retrieval (ITERRET) is an explicit non-goal."_
- `fluxmem/interfaces.py:19-24` — `MemoryFormat`'s three labels (`HG`, `VC`, `VS`) are documented
  as _"opaque HETREP representation labels ... the encoders themselves are an explicit non-goal."_

`EpisodicUnit.hyperedge_density` and `.visual_salience` (`interfaces.py:68-69`) are correspondingly
documented as _"opaque placeholders a real HETREP HG/VC encoder would populate,"_ defaulting to
`0.0`. `fluxmem/features.py:160-162` reads both fields straight through into the seven-feature
vector the selector consumes — the fields are load-bearing, not vestigial.

This non-goal is a scope boundary, not an implementation gap that can be closed by writing the
encoders wherever is convenient. `fluxmem` is downstream of HETREP in COLM's data flow (encoding
happens before storage), so wherever HetRep lives, it must be upstream of `fluxmem`, not nested
inside it. Revision 1 of the implementation plan proposed `fluxmem/HetRep/`. That inverts the
dependency direction the three docstrings above already commit to, and it does so silently —
nesting a package that produces the very fields `fluxmem` currently treats as opaque placeholders,
inside the package that declares producing them out of scope, contradicts the package's own
documentation without amending it. Revision 1 is rejected on this basis, not on a stylistic
preference for flat layouts.

### The existing boundary conflates two different jobs

Today the only boundary between "raw dialogue" and `fluxmem`'s STIM/MTEM pipeline is
`EpisodeProducer`:

```python
class EpisodeProducer(Protocol):
    """Boundary fake for HETREP (segmentation + encoding), a non-goal here."""
    def produce(self, count: int) -> list[EpisodicUnit]: ...
```

The docstring's own phrasing — "segmentation + encoding" in one Protocol — names the problem.
COLM treats these as two separate stages with separate section numbers: §1.3.1 (ADASTORE)
segments raw dialogue into episodic units; §1.2 (HETREP) encodes a segmented unit into the three
representations. `EpisodeProducer.produce(count)` is a _pull_ interface with no input — it cannot
express "take this already-segmented unit and enrich it," only "hand me `count` new units from
wherever you get them." Any real encoder wired in behind this Protocol has to also own
segmentation, whether or not it actually performs any.

ADR 0003 hit this directly. `MemOCREpisodeProducer` (`fluxmem/memocr_episodes.py:91`) is a real
`EpisodeProducer` for `MemoryFormat.VC`, and it sets `turns=[]` on every unit it returns, with the
comment _"MemOCR's memory is already a compacted summary, not raw dialogue turns, so there is
nothing accurate to backfill."_ That was the correct call given the Protocol available at the
time — MemOCR's `MemoryAgent` really does both segment (buffer dialogue into chunks) and encode
(draft + render) in one pass, so `EpisodeProducer` was not misapplied, it was the only seam that
existed.

The consequence is a protocol-level bug, not a MemOCR-specific one. `fluxmem/features.py:149-162`
computes five of its seven features — `temporal_ordering`, `topic_diversity`, `token_length`, and
the two entity-based features feeding into them — from `episode.turns`. When `turns=[]`, all five
evaluate to a constant (zero, or a degenerate ratio) for every VC episode, silently. This does not
raise; it reports a plausible-looking feature vector that happens to be wrong for one-third of the
label space. It is exactly the failure mode `.claude/rules/evidence-discipline.md` names: "it does
not crash, it reports a number that is wrong." The fix cannot live inside
`MemOCREpisodeProducer` — there is no accurate `turns` to backfill from a rendered image — it has
to live in the seam design: segmentation and encoding must be separable calls, so that whatever
does segmentation (EM-LLM) can hand a `turns`-populated unit to whatever does encoding (HetRep),
and the encoder enriches without discarding what segmentation already produced.

## Decision

### HetRep is `hetrep/`, a sibling package at repo root, upstream of `fluxmem/`

| Option                                 | Description                                          | Tradeoff                                                                                                                                                                                                                                    |
| -------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (a) `fluxmem/HetRep/` (revision 1)     | Nest HetRep as a subpackage of `fluxmem`             | Contradicts three existing docstrings without amending them; inverts the actual dependency (fluxmem would "contain" the thing it says it doesn't implement)                                                                                 |
| (b) `hetrep/` sibling package (chosen) | New top-level package, same level as `fluxmem/`      | Matches the real dependency direction (encode → store); requires a second `pyproject.toml` package entry and a CI guard against the reverse import                                                                                          |
| (c) Separate repository                | HetRep as its own git repo, imported as a dependency | Cleanest separation, but this is a single-team research monorepo (`CLAUDE.md`: "Multi-person team; workflow is code → review → merge") — splitting the repo adds release/version coordination overhead with no consumer outside C-AIMMS yet |

**Decision: (b).** `hetrep/` sits at repo root alongside `fluxmem/`. It may import
`fluxmem.interfaces` for the shared `EpisodicUnit`/`MemoryFormat` types; **`fluxmem` must never
import `hetrep`.** `pyproject.toml`'s `[tool.setuptools.packages.find]` gains
`include = ["fluxmem*", "hetrep*"]` and a CI guard (hetrepv2 plan T-09) checks
`grep -rn "import hetrep" fluxmem/` is empty on every PR.

### `EpisodeEncoder`: a second Protocol, not a change to `EpisodeProducer`

Added to `fluxmem/interfaces.py` alongside the existing `EpisodeProducer`:

```python
@runtime_checkable
class EpisodeEncoder(Protocol):
    """HETREP boundary (COLM §1.2, Alg. 1 line 4): enrich a segmented unit in place.

    Distinct from EpisodeProducer, which conflates segmentation with encoding.
    Segmentation is ADASTORE's (COLM §1.3.1, EM-LLM); this is encoding only.
    """
    def encode(self, unit: EpisodicUnit) -> EpisodicUnit: ...
```

`HetRepEncoder.encode` (the `hetrep/` implementation) populates `embedding` (from the VS arm),
`hyperedge_density` (from HG), and `visual_salience` (from VC). It does not touch `turns` or
`primary_format` — the former is segmentation's output and stays as handed in; the latter is
`fluxmem.selector.FormatSelector`'s decision, made downstream from the now-correct features.

Full data flow per COLM Algorithm 1:

```
raw dialogue
  → EM-LLM surprise segmentation (§1.3.1)     → EpisodicUnit(turns=[...])
  → HetRepEncoder.encode (§1.2)               → + embedding, hyperedge_density, visual_salience
  → FormatSelector.predict (§1.3.3)           → primary_format
  → select_merge_target / MTEM.add (§1.3)     → stored
  → ltsm.promote (§1.3.2)                     → consolidated
```

Because `turns` is now preserved from segmentation through to feature extraction regardless of
which format's encoder ran, all seven features in `fluxmem/features.py` compute correctly for
every unit — directly fixing the bug identified above. `EpisodeProducer` is not removed:
`StubEpisodeProducer` still exists for hermetic tests that need synthetic units with no upstream
segmentation stage at all, and `MemOCREpisodeProducer` remains valid wherever a caller genuinely
has no separate segmentation step to run first. `EpisodeEncoder` is additive.

## Consequences

### Positive

- **Dependency direction matches the paper.** `fluxmem` depends on `hetrep`-shaped data
  (via `EpisodicUnit` fields) without depending on the `hetrep` package itself — enforceable in CI,
  not just documented.
- **Segmentation and encoding are independently testable.** A test can hand
  `HetRepEncoder.encode` a hand-built `EpisodicUnit(turns=[...])` without any segmentation
  machinery, and a segmentation test can assert on `turns` without any encoder running.
  Both are ablatable in isolation, which the format-per-phase sequencing (VS → HG → VC, see the
  hetrepv2 plan §4) depends on.
- **Fixes a real, previously silent bug.** `turns=[]` for VC episodes is no longer possible once
  encoding is a distinct call over an already-segmented unit — the encoder has no reason to
  discard what it was handed, unlike a combined producer that has to reconstruct or omit it.
- **`EpisodeProducer` is not deprecated, only narrowed in role.** No existing caller of
  `StubEpisodeProducer` or `MemOCREpisodeProducer` breaks; `EpisodeEncoder` is a new, additive
  Protocol.

### Negative

- **One more top-level package to maintain.** `hetrep/` is new surface area at repo root,
  smaller than `fluxmem/` today but growing across the VS → HG → VC phases.
- **Shared-type coupling.** Both packages now depend on `EpisodicUnit`'s field set; adding a field
  to support one HETREP arm (e.g., a `summary` field for VS, noted as a Phase 1 open question in
  the hetrepv2 plan §5) requires coordinating the change across both packages rather than
  containing it in one.
- **Two Protocols to reason about at the seam** (`EpisodeProducer` and `EpisodeEncoder`) where
  there was one before. This is a deliberate trade of conceptual simplicity for correctness — the
  previous single-Protocol design produced a real bug — but it does mean new contributors need to
  understand the segmentation/encoding split explicitly rather than infer it.

### Risks

- **Circular-import temptation.** If a future test wants to construct a `hetrep`-encoded unit
  inside `fluxmem/tests/`, the path of least resistance is `import hetrep`, which breaks the
  stated direction. Mitigation: the CI guard in T-09; any such test should instead depend on a
  fake `EpisodeEncoder` implemented in `fluxmem/tests/` itself, the same pattern already used for
  `FakeEntityExtractor`.
- **`visual_salience` stays constant through Phases 1-2.** With VC stubbed
  (`hetrep/vc/stub.py`, ADR-adjacent decision in the hetrepv2 plan §4), `visual_salience = 0.0`
  for every unit until Phase 3 lands. This is not a regression introduced by this ADR — it is the
  same placeholder value the field already had — but it means the selector remains wired and
  tested, not trained, until Phase 3, consistent with the sequencing decision recorded in the
  hetrepv2 plan and the prior `memocr-integration-handoff` memory note.
- **`ltsm.promote` fails silently on a misconfigured encoder.** `fluxmem/ltsm.py:148-149` skips
  any episode where `embedding is None` with no error raised. If a `HetRepEncoder` implementation
  has a bug that leaves `embedding` unset, the symptom is an empty or undersized `LTSM` with no
  exception — the same "wrong number, not a crash" failure mode this ADR's own bug fix targets.
  Mitigation: the Phase 1 harness (hetrepv2 plan §5) asserts a non-empty `FaissVectorStore` rather
  than trusting the absence of an exception.

## Related

- **ADR 0003** established the injection-over-direct-import pattern this seam extends, and its
  `EpisodeProducer` choice and `visual_salience` proxy are partially superseded here — see that
  ADR's own text for the parts that still stand (the injection reasoning) versus what changes
  (the seam itself, and the salience question reopened for Phase 3).
- **ADR 0006** — the submodule conversion of `HyperMem`, `MemOCR`, `EM-LLM` that Phase 2/3 encoders
  will depend on.
- **ADR 0007** — how the HG arm of `HetRepEncoder` reuses HyperMem's data model without importing
  its pipeline.
- **ADR 0008** — the fidelity rule `HetRepEncoder`'s VS and HG arms follow when their reused
  components diverge from COLM's spec.
