"""The model drivers: how an episode is run against a proxy or the ground truth.

The ground-truth driver's whole claim is that a fresh model restored from a
branch state stands *exactly* where the history left off, so its first step
reproduces the period the history would have recorded next. That is checked
against a history one period longer drawn under the same seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from control.drivers import GroundTruthDriver, ProxyDriver
from control.world import WorldConfig, build_world
from economic_models.ground_truth.registry import MODELS

HISTORY = 40

#: The fresh-model restore drops pysolve's lag slots (``_Nfe__1`` and friends),
#: and on its first solve pysolve seeds the lags from the *current* context --
#: after this period's parameters were set. So equations reading a lagged
#: parameter (``ER = N(-1)/Nfe(-1)``, and the ``NPLk``/``Rln`` lags) see
#: period ``t`` instead of ``t - 1`` on the first step of every ground-truth
#: episode. Remove this marker once the restore carries those lags.
GROWTH_LAGGED_PARAMETERS = pytest.mark.xfail(
    strict=True,
    reason="GROWTH branch restore loses lagged parameters (Nfe, NPLk, Rln) on step one",
)


def _world_and_next_row(model: str):
    """A tiny world, and the history row its end would have been followed by."""
    spec = MODELS[model]
    dt = spec.fixed_dt or 1.0
    config = WorldConfig(
        model=model, dt=dt, history_steps=HISTORY, horizon=4,
        n_train_futures=1, n_eval_futures=1, excitation="default",
    )
    world = build_world(config, verbose=False)
    generator = spec.generator(config.excitation_config(), dt=dt, on_collapse="truncate")
    longer = generator.generate(HISTORY + 1, seed=config.seed)
    return world, longer


@pytest.mark.parametrize(
    "model",
    [pytest.param("growth", marks=GROWTH_LAGGED_PARAMETERS), "smets_wouters"],
)
def test_ground_truth_resumes_exactly_where_the_history_stopped(model):
    """One step from the branch state is the next period of the same history."""
    world, longer = _world_and_next_row(model)
    np.testing.assert_array_equal(longer.states[:HISTORY], world.history.states)

    interface = world.config.spec.interface
    driver = GroundTruthDriver(world)
    driver.reset(world.branch_point)
    state = driver.step(
        interface.parameters.from_row(longer.params[HISTORY]),
        interface.actions.from_row(longer.actions[HISTORY]),
        dict(zip(world.hidden_names, longer.hidden[HISTORY])),
        np.random.default_rng(0),
    )
    levels = np.array([getattr(state, n) for n in interface.state.names()], dtype=float)
    np.testing.assert_allclose(levels, longer.states[HISTORY], rtol=1e-6)


def test_ground_truth_driver_is_not_branchable():
    """The solver is deterministic and its state must not be copied."""
    world, _ = _world_and_next_row("smets_wouters")
    driver = GroundTruthDriver(world)
    assert not driver.branchable
    assert driver.required_window == 1


def test_ground_truth_driver_needs_a_reset():
    """Stepping before an episode is a lifecycle error with an actionable message."""
    world, longer = _world_and_next_row("smets_wouters")
    interface = world.config.spec.interface
    with pytest.raises(RuntimeError, match="reset"):
        GroundTruthDriver(world).step(
            interface.parameters.from_row(longer.params[0]),
            interface.actions.from_row(longer.actions[0]),
            None,
            np.random.default_rng(0),
        )


def test_proxy_driver_resets_to_the_same_position(stack):
    """The cached start is restored exactly: the first draw repeats under one seed."""
    driver = stack.env.driver
    assert isinstance(driver, ProxyDriver) and driver.branchable
    interface = stack.world.config.spec.interface
    episode = stack.world.eval_futures.all()[0]
    params = interface.parameters.from_row(episode.params[0])
    actions = interface.actions.from_row(stack.world.history.actions[-1])

    draws = []
    for _ in range(2):
        states, exog = driver.reset(episode.branch)
        np.testing.assert_array_equal(states[-1], stack.world.history.states[-1])
        draws.append(driver.step(params, actions, None, np.random.default_rng(1)))
    assert draws[0] == draws[1]
