"""End-to-end smoke tests: the ground truth simulates, proxies fit and roll out.

Kept deliberately tiny (short runs, cheap proxies) so the suite runs in seconds;
these guard the wiring, not the science.
"""

import numpy as np
import pytest

from economic_models import Actions, BaseEconomicModel, Parameters, State
from economic_models.ground_truth import (
    ExcitedRunGenerator,
    GrowthCalibration,
    GrowthModel,
    Run,
)
from economic_models.proxy import RandomWalkProxy, VARXProxy


def test_growth_model_self_seeds_and_steps() -> None:
    model = GrowthModel()
    model.run(3)
    state = model.state
    assert all(np.isfinite(v) for v in state.to_dict().values())
    assert state.Y > 0


def test_constructor_overrides_survive_seeding() -> None:
    assert GrowthModel(Rbbar=0.04).actions.Rbbar == 0.04


def test_shared_advance_drives_both_families() -> None:
    truth = GrowthModel()
    truth.run(2)
    parameters, actions = truth.parameters, truth.actions
    assert isinstance(truth, BaseEconomicModel)
    state = truth.advance(parameters, actions)
    assert isinstance(state, State)


def test_calibration_is_injectable() -> None:
    calibration = GrowthCalibration.baseline()
    model = GrowthModel(calibration=calibration)
    model.run(2)
    assert np.isfinite(model.state.Y)


@pytest.fixture(scope="module")
def short_run() -> Run:
    return ExcitedRunGenerator().generate(60, seed=0)


def test_proxy_fit_rollout_and_advance(short_run: Run) -> None:
    for proxy in (RandomWalkProxy(), VARXProxy()):
        proxy.fit([short_run])
        window = (
            short_run.states[40:50],
            short_run.params[40:50],
            short_run.actions[40:50],
        )
        levels = proxy.rollout(
            window, short_run.params[50:60], short_run.actions[50:60]
        )
        assert levels.shape == (10, len(State.names()))
        assert np.isfinite(levels).all()

        # The shared driving protocol works on proxies too.
        parameters = Parameters.from_dict(
            dict(zip(Parameters.names(), short_run.params[50]))
        )
        actions = Actions.from_dict(dict(zip(Actions.names(), short_run.actions[50])))
        state = proxy.advance(parameters, actions)
        assert isinstance(state, State)
