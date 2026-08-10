"""Shared fixtures for fluxmem tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Deterministic seeding for every test that touches torch (Step 6's MLP).

    `warn_only=True` avoids hard-failing on any CPU op lacking a deterministic
    kernel; the selector's ops (Linear, ReLU, Adam) are deterministic on CPU
    in practice, so this is a safety net, not a relied-upon escape hatch.
    """
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
