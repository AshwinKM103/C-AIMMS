"""fluxmem/memocr_episodes.py -- MemOCR-backed `EpisodeProducer` (COLM Sec 1.3.2).

`fluxmem.interfaces.EpisodeProducer` is documented there as a "Boundary fake
for HETREP (segmentation + encoding), a non-goal here," today satisfied only
by the test-only `StubEpisodeProducer`. MemOCR (`MemOCR/recurrent/impls/
memory_img_final_only_triple.py`'s `MemoryAgent`) already drafts a persistent
Markdown memory and renders it to an image under a pixel budget (COLM Sec
3.1-3.2) -- a real encoder for the visual-canvas format, not a fake. This
module wraps it as a second `EpisodeProducer` implementation.

This is an encoding-boundary adapter, not a retriever. `fluxmem/ltsm.py`
states retrieval (ITERRET) is out of scope for this package; nothing here
adds a query/retrieve surface, and downstream MTEM/LTSM/selector/fusion
consume the `EpisodicUnit`s this produces exactly as they do
`StubEpisodeProducer`'s.

Dependency-free by design, same reasoning as `fluxmem/interfaces.py`'s
`EntityExtractor` boundary: `MemOCREpisodeProducer` takes `markdown_source`
and `render_fn` as injected callables rather than importing MemOCR's
`MemoryAgent` (torch/verl/transformers) or `call_md_renderer`
(Playwright-backed HTTP calls) directly. That keeps this module importable
and its tests hermetic without those dependencies, and lets
`MemOCR/recurrent/impls/call_md_renderer.py:batch_generate_images` (or any
fake) be passed in as `render_fn` unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from fluxmem.interfaces import EpisodicUnit, MemoryFormat

MarkdownSourceFn = Callable[[int], Sequence[str]]
"""Returns up to `count` pending Markdown memory snapshots (decoded text)."""


class RenderedImage(Protocol):
    """Structural boundary for a rendered memory image.

    Matches `PIL.Image.Image`'s `.size` attribute (width, height) without
    importing PIL: `batch_generate_images` returns `list[Optional[Image.Image]]`,
    and `None` stands for a failed render (MemOCR's own convention).
    """

    size: tuple[int, int]


RenderFn = Callable[[Sequence[str]], Sequence["RenderedImage | None"]]
"""Renders each Markdown snapshot to an image, or None if rendering failed."""

DEFAULT_PATCH_PIXELS = 28 * 28
"""One vision-LLM patch, per `MemOCR/taskutils/memory_eval/utils/memocr_md.py`'s
`MIN/MAX_PIXELS_28_28 * 28 * 28` pixel-budget accounting."""

DEFAULT_BUDGET_PATCHES = 256
"""Mid-range of the paper's swept budgets (16/64/256/1024 patches,
`MemOCR/scripts/eval.sh:29`); used only to normalize `visual_salience`."""


class MemOCREpisodeProducer:
    """Adapter: MemOCR's rendered Markdown memory -> `EpisodicUnit` (`MemoryFormat.VC`).

    Concretizes `fluxmem.interfaces.EpisodeProducer` for the visual-canvas
    format. `produce(count)` fetches up to `count` pending Markdown snapshots
    and renders each to an image; MTEM/LTSM/selector/fusion downstream treat
    the result identically to any other `EpisodeProducer`'s output.
    """

    def __init__(
        self,
        markdown_source: MarkdownSourceFn,
        render_fn: RenderFn,
        budget_patches: int = DEFAULT_BUDGET_PATCHES,
    ) -> None:
        if budget_patches <= 0:
            raise ValueError(f"budget_patches must be > 0, got {budget_patches}")
        self._markdown_source = markdown_source
        self._render_fn = render_fn
        self._budget_pixels = budget_patches * DEFAULT_PATCH_PIXELS

    def produce(self, count: int) -> list[EpisodicUnit]:
        """Renders up to `count` pending Markdown memory snapshots into `EpisodicUnit`s."""
        snapshots = list(self._markdown_source(count))
        images = list(self._render_fn(snapshots))
        return [self._to_episode(index, image) for index, image in enumerate(images)]

    def _to_episode(self, index: int, image: RenderedImage | None) -> EpisodicUnit:
        return EpisodicUnit(
            episode_id=f"memocr-episode-{index}",
            turns=[],  # MemOCR's memory is already a compacted summary, not raw turns.
            primary_format=MemoryFormat.VC,
            visual_salience=_visual_salience(image, self._budget_pixels),
        )


def _visual_salience(image: RenderedImage | None, budget_pixels: int) -> float:
    """Rendered pixel-area as a fraction of the configured patch budget, clipped to [0,1].

    Neither COLM nor FluxMem defines `visual_salience` numerically -- COLM Sec
    3.1 only says salience is "layout-based," directed through the drafting
    prompt rather than measured after the fact (see
    `docs/MemOCR-Paper-Alignment.md`). This proxy is a modeling choice, not a
    cited fact: it reuses MemOCR's own pixel-budget accounting (patch_count *
    28**2, `memocr_md.py`) as the closest available signal for "how much of
    the allocated visual budget this memory actually occupies," and a failed
    render (`image is None`) scores 0.0 rather than raising, since a missing
    image carries no visual information for this format.
    """
    if image is None:
        return 0.0
    width, height = image.size
    return min(1.0, (width * height) / budget_pixels)
