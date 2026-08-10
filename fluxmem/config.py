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
