"""Shared fixtures for fluxmem tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Deterministic seeding for every test that touches torch (Step 6's MLP)."""
    torch.manual_seed(0)
