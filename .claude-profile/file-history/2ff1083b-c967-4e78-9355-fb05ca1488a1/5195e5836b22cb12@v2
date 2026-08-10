"""Guards the fluxmem/LightMem name collision (plan Step 0)."""

from __future__ import annotations

import pathlib

import fluxmem


def test_fluxmem_resolves_under_repo_root() -> None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    module_path = pathlib.Path(fluxmem.__file__).resolve()
    assert module_path.is_relative_to(repo_root), (
        f"fluxmem imported from {module_path}, not under repo root {repo_root} "
        "-- LightMem/src/fluxmem/ may be shadowing the root package"
    )
