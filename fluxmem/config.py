"""fluxmem/config.py -- pydantic v2 configuration models, validated at the boundary.

Assembled incrementally as each fluxmem module needs config (per
`.claude/rules/security.md`: validate external input -- here, YAML config --
at the boundary, not deep inside the modules that consume it). The full
assembly (`AdaStoreConfig` + `load_config`) lands in Step 8, once every
sub-config exists.

Do not add `tau_c`, `promotion_gate`, or any other flag that re-opens a
resolved paper ambiguity (see the plan's Reading rule) -- config is for
tuning, not for hedging.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class UtilityWeights(BaseModel):
    """w1, w2, w3 in COLM Eq. 5: `U(e_j) = w1*c(e_j) + w2*l(e_j) + w3*d(e_j)`.

    All three normalized-term outputs (access_frequency, interaction_intensity,
    recency_decay) are in `[0,1]` so these weights are directly comparable
    (see fluxmem/mtem.py) -- itself a modeling choice, since neither paper
    states term ranges.
    """

    w1: float = Field(ge=0.0, description="weight on access_frequency (c)")
    w2: float = Field(ge=0.0, description="weight on interaction_intensity (l)")
    w3: float = Field(ge=0.0, description="weight on recency_decay (d)")


class RewardConfig(BaseModel):
    """lambda_judge, lambda_mem in FluxMem Eq. 9:
    `r(s) = lambda_judge*r_judge(s) + lambda_mem*r_mem(s)`.

    Defaults are FluxMem's own tuned values (0.7/0.3) -- cited, not derived.
    `Judge(.)` and `MemUtil(.)` themselves are concretized in ADR 0002
    (`docs/adr/0002-format-selector-reward-design.md`), not by this config.
    """

    lambda_judge: float = Field(default=0.7, ge=0.0, le=1.0)
    lambda_mem: float = Field(default=0.3, ge=0.0, le=1.0)


class SelectorConfig(BaseModel):
    """Hyperparameters for `fluxmem.selector.FormatSelector` (FluxMem Alg. 3)."""

    hidden_dim: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=1e-2, gt=0.0)
    max_epochs: int = Field(default=200, ge=1)
    batch_size: int = Field(default=16, ge=1)
    patience: int = Field(default=10, ge=1)
    seed: int = Field(default=0)


class PromotionThresholds(BaseModel):
    """LTSM promotion gate (COLM Sec 1.3.2): `U(e) >= tau_u AND r(e) >= tau_r`.

    `tau_u` is on the composite `U`'s scale (`[0, w1+w2+w3]` given the
    `[0,1]`-normalized terms in fluxmem/mtem.py), not FluxMem's raw usage-
    count scale -- see fluxmem/ltsm.py's module docstring. `max_age` sets
    the linear horizon for the hard recency gate `r`.
    """

    tau_u: float = Field(gt=0.0)
    tau_r: float = Field(ge=0.0, le=1.0)
    max_age: float = Field(gt=0.0)


class FusionConfig(BaseModel):
    """BMM-gated fusion parameters (FluxMem Eqs. 17-26, Alg. 1).

    `m_min` (min-keep top-k fallback) appears nowhere in COLM but is
    load-bearing in FluxMem Alg. 1 -- see the plan's research finding.
    """

    tau: float = Field(default=0.7, gt=0.0, lt=1.0, description="posterior threshold")
    m_min: int = Field(default=1, ge=0, description="min-keep top-k fallback")
    em_iters: int = Field(default=50, ge=1, description="BMM EM iterations")
    eps: float = Field(default=1e-3, gt=0.0, lt=0.5, description="Eq. 17 min-max inset")


class StimConfig(BaseModel):
    """`fluxmem.stim.ShortTermInteractionMemory` capacity (COLM Sec 1.3.2: capacity 4)."""

    capacity: int = Field(default=4, ge=0)


class MtemConfig(BaseModel):
    """`fluxmem.mtem.MidTermEpisodicMemory` capacity and Eq. 5 term scales."""

    capacity: int = Field(default=2000, ge=1, description="FluxMem: up to 2000 units")
    reference_length: float = Field(default=50.0, gt=0.0, description="l(e_j) token-count scale")
    half_life: float = Field(default=100.0, gt=0.0, description="d(e_j) exponential half-life")


class AdaStoreConfig(BaseModel):
    """Full ADASTORE configuration: the assembly of every sub-config above.

    `utility_weights` and `promotion` have no defaults -- they are the two
    genuinely load-bearing modeling choices (Step 3 and Step 8's module
    docstrings), so a config file must state them explicitly rather than
    silently inherit a placeholder.
    """

    stim: StimConfig = Field(default_factory=StimConfig)
    mtem: MtemConfig = Field(default_factory=MtemConfig)
    utility_weights: UtilityWeights
    promotion: PromotionThresholds
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    selector: SelectorConfig = Field(default_factory=SelectorConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)


def load_config(path: str | Path) -> AdaStoreConfig:
    """Loads and validates `AdaStoreConfig` from a YAML file (e.g. `configs/default.yaml`).

    Validation happens at this boundary (`.claude/rules/security.md`): a
    malformed or incomplete config fails here, with pydantic's error, not
    deep inside whichever module first reads a missing/invalid field.
    """
    with Path(path).open("r") as f:
        raw = yaml.safe_load(f)
    return AdaStoreConfig(**raw)
