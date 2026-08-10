# 0003. MemOCR/FluxMem integration: an `EpisodeProducer` adapter, not a retriever

## Status

Accepted — 2026-08-11. Adds `fluxmem/memocr_episodes.py`; no other `fluxmem/` module changes.

## Context

Step 4 of the MemOCR trimming plan (`.claude/prompts/memocr-codebase-trim-and-fluxmem-
integration.md`) called for "FluxMem integration," originally specced as a
`MemOCRVisualStore(MemoryStore)` adapter living at
`LightMem/lightmem/factory/retriever/memocr_visual_store.py`, with `add()`/`retrieve()`
methods and a factory registration.

Neither half of that target exists. `LightMem/` (the vendored fork) was removed in commit
`50d99be` ("chore: remove vendored lightmem directory") before this session. The FluxMem
implementation that does exist, `fluxmem/` at the repo root, has no retriever, no
`MemoryStore` interface, and no factory/registry pattern. Its own module docstring
(`fluxmem/ltsm.py`) states this explicitly: "This package builds no BM25 fusion --
retrieval (ITERRET) is an explicit non-goal -- so there are no fusion weights to record."
`fluxmem/__init__.py` repeats it at package level: "Does not implement HETREP, ITERRET, or
WORKMEM." The original design would have added a query/retrieve surface to a package that
has deliberately never had one -- not a stale detail, an architectural contradiction.

What `fluxmem/` does have, and does need, is `EpisodeProducer`
(`fluxmem/interfaces.py`) -- the Protocol `STIM -> MTEM` construction is built against:

```python
class EpisodeProducer(Protocol):
    """Boundary fake for HETREP (segmentation + encoding), a non-goal here."""
    def produce(self, count: int) -> list[EpisodicUnit]: ...
```

Today this Protocol is satisfied only by `StubEpisodeProducer`, a deterministic
test-only generator. MemOCR (`MemOCR/recurrent/impls/memory_img_final_only_triple.py`'s
`MemoryAgent`) already does real segmentation-and-encoding work for one specific
format: it drafts a persistent Markdown memory per chunk and renders it to an image
under a fixed pixel budget (COLM Sec 3.1-3.2, verified in
`docs/MemOCR-Paper-Alignment.md`). That is a real `EpisodeProducer` for
`MemoryFormat.VC`, not a fake -- the correct integration point given what both
codebases actually contain.

## Decision

**Adapter: `fluxmem/memocr_episodes.py:MemOCREpisodeProducer`, implementing
`EpisodeProducer`.** `produce(count)` fetches up to `count` pending Markdown snapshots
and renders each to an image, returning `EpisodicUnit`s with
`primary_format=MemoryFormat.VC`. No factory registration, because `fluxmem/` has none;
`MemOCREpisodeProducer` is constructed the same way `StubEpisodeProducer` is --
directly, by whatever assembles the STIM/MTEM pipeline.

### Dependency injection over direct MemOCR imports

| Option                                                    | Description                                                                                    | Tradeoff                                                                                                                                                                                                                                         |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| (a) Import `MemoryAgent` / `call_md_renderer` directly    | Adapter owns a live MemOCR agent and calls its render pipeline                                 | Pulls torch/verl/transformers and a Playwright-backed HTTP render service into every `fluxmem` test that touches this module; matches `.claude/rules/testing.md`'s "mock at boundaries" only if MemOCR itself is treated as an external boundary |
| (b) Inject `markdown_source` and `render_fn` as callables | Adapter is pure glue; caller supplies the real `MemoryAgent`/`batch_generate_images` or a fake | Adds one layer of indirection at the call site; keeps this module's own tests hermetic                                                                                                                                                           |

**Decision: (b).** Same reasoning `fluxmem/interfaces.py` already uses for
`EntityExtractor` (`FakeEntityExtractor` vs. `SpacyEntityExtractor`, injected rather than
imported) and the same reasoning ADR 0002 used for `JudgeReward`/`MemUtilReward`:
`fluxmem`'s test suite must stay hermetic (`.claude/rules/testing.md`), and MemOCR's
real render path is a network-calling, GPU-touching external system relative to this
package, not something its unit tests should construct. `MemOCR/recurrent/impls/
call_md_renderer.py:batch_generate_images` (`list[str] -> list[Optional[Image.Image]]`)
matches the injected `RenderFn` shape unchanged -- no wrapper needed at the call site
beyond passing it in.

### `visual_salience`: pixel-budget occupancy, not a paper-defined metric

`EpisodicUnit.visual_salience` (`fluxmem/interfaces.py`) is documented as "an opaque
placeholder a real HETREP HG/VC encoder would populate." Neither COLM nor FluxMem
defines it numerically -- COLM Sec 3.1 only says salience is "layout-based," steered
through the drafting prompt rather than measured after rendering.

| Option                                                                | Description                                                                                               | Tradeoff                                                                                                                                                                                 |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (a) Constant / unset (0.0)                                            | No signal computed                                                                                        | Cheapest, but throws away real information the render already carries (a failed or truncated render is meaningfully different from a full one)                                           |
| (b) Rendered pixel-area / configured patch budget, clipped to `[0,1]` | Reuses MemOCR's own budget accounting (`patch_count * 28**2`, `taskutils/memory_eval/utils/memocr_md.py`) | Is a proxy for "how much of the allocated visual budget this memory fills," not for the paper's own notion of layout salience; a large blank render would score as highly as a dense one |

**Decision: (b).** Grounding the proxy in MemOCR's existing pixel-budget accounting
(the same `28x28`-patch convention `docs/MemOCR-Paper-Alignment.md:87-89` verifies is
used for the paper's own token-budget sweeps) keeps it a documented approximation
rather than an arbitrary constant, per `.claude/rules/evidence-discipline.md`. **Known
gap, not a silent one:** this is occupancy, not layout quality -- it does not model
whether the filled pixels are actually salient content. If MTEM's utility score proves
sensitive to this term, replacing it with a real layout signal (e.g. from the render's
HTML/DOM structure) is a drop-in change inside `_visual_salience`; nothing calling
`MemOCREpisodeProducer` needs to change, because the seam is a private function, not a
public contract.

## Consequences

- `fluxmem/` gains a second, real `EpisodeProducer` alongside `StubEpisodeProducer`,
  with no change to `EpisodicUnit`, `MTEM`, `LTSM`, `selector`, or `fusion` -- all
  consume `EpisodicUnit`s the same way regardless of producer, confirming the boundary
  was the right integration point.
- No retrieval surface was added anywhere in `fluxmem/`. If ITERRET is ever implemented,
  it is a separate, later decision -- this ADR does not presuppose it or leave a stub
  for it (`.claude/rules/coding-style.md`: no dead code "for later").
- `fluxmem/memocr_episodes.py` has zero new runtime dependencies (no PIL, no torch, no
  verl) -- `RenderedImage` is a structural `Protocol` matching `PIL.Image.Image`'s
  `.size` attribute, not an import of PIL itself.
- Every episode from this producer carries `turns=[]`: MemOCR's memory is already a
  compacted summary at the point it is rendered, not raw dialogue turns, so there is
  nothing accurate to backfill into `EpisodicUnit.turns`.
- Superseded content: the "FluxMem Integration" sections of
  `.claude/prompts/memocr-codebase-trim-and-fluxmem-integration.md` and
  `.claude/prompts/STEP-5-NEXT-SESSION.md` were rewritten in this session to match this
  decision (`LightMem/factory/retriever/MemoryStore` design replaced with the
  `EpisodeProducer` design above) rather than left to describe a target that no longer
  exists.
