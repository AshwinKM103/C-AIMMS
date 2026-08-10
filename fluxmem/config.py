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
