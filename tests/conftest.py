"""Shared configuration for the whole test suite.

The suite is split by how much of the stack a test exercises, one directory per
tier, so the fast inner loop and the slow pipeline checks can be run apart:

* ``unit/`` -- one component in isolation, on hand-built inputs. No model is
  solved and nothing is fitted beyond a few-parameter network; the whole tier
  runs in seconds and is the one to keep green while editing.
* ``integration/`` -- the real ground-truth models (GROWTH and Smets-Wouters) and
  real fitted proxies, at sizes small enough to finish in seconds each. This is
  where the *contracts* live: reproducibility per seed, the proxy rollout
  interface, the environment's instrument and collapse rules.
* ``e2e/`` -- whole pipelines: a tiny training run and a tiny dataset written to
  disk and read back into a world.

Every test is marked with its tier from the directory it lives in, so
``pytest -m unit`` runs the inner loop and ``pytest -m "not e2e"`` everything
short of the pipelines. The tier is never written on a test by hand -- moving a
file between directories is all it takes to re-tier it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: The tier directories, each doubling as the marker its tests receive.
TIERS = ("unit", "integration", "e2e")

_ROOT = Path(__file__).parent


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark every collected test with the tier of the directory it lives in."""
    for item in items:
        tier = item.path.relative_to(_ROOT).parts[0]
        if tier in TIERS:
            item.add_marker(getattr(pytest.mark, tier))
