"""Fixtures for the integration tier: real models and fitted proxies, kept tiny.

Everything expensive is session-scoped and built once. A fixture handed out
here is shared, so a test that steps an environment or a proxy must ``reset`` it
first -- which every contract under test requires anyway.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from control.dsac.train import TrainConfig, TrainingResult, setup
from economic_models.ground_truth import GrowthExcitationConfig, GrowthRunGenerator
from economic_models.ground_truth.registry import MODELS
from economic_models.run import Run

#: The smallest training stack that still exercises every moving part: a
#: history long enough to fit a Kalman-encoded VARX on, a handful of futures in
#: each bank, and networks a few units wide.
TINY_STACK = dict(
    history_steps=120,
    horizon=8,
    n_train_futures=6,
    n_eval_futures=3,
    hidden=32,
    n_quantiles=8,
    excitation="default",
)


@pytest.fixture(scope="session")
def tiny_config() -> Callable[..., TrainConfig]:
    """A builder for a :data:`TINY_STACK` training config, with overrides."""

    def build(model: str = "growth", **overrides) -> TrainConfig:
        return TrainConfig(model=model, **(TINY_STACK | overrides))

    return build


@pytest.fixture(scope="session", params=list(MODELS))
def stack(request, tiny_config) -> TrainingResult:
    """A fitted training stack -- world, proxy, observer, envs, agent -- per model."""
    return setup(tiny_config(request.param), verbose=False)


@pytest.fixture(scope="session")
def growth_generator() -> GrowthRunGenerator:
    """An annual GROWTH generator under the calm default excitation."""
    return GrowthRunGenerator(GrowthExcitationConfig.default(), dt=1.0)


@pytest.fixture(scope="session")
def growth_runs(growth_generator) -> list[Run]:
    """Two independent excited GROWTH histories to fit proxies on."""
    return [growth_generator.generate(60, seed=seed) for seed in (0, 1)]


@pytest.fixture(scope="session")
def growth_holdout(growth_generator) -> Run:
    """A third, held-out GROWTH history to warm-start and drive rollouts from."""
    return growth_generator.generate(30, seed=9)
