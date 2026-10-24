"""The GROWTH model as a :class:`~economic_models.BaseEconomicModel`.

Construction, the visible interface, and a handful of economic sanity checks
that a broken equation would fail long before a training run noticed.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from economic_models.ground_truth import GrowthCalibration, GrowthModel
from economic_models.ground_truth.models.growth.variables import (
    GrowthActions,
    GrowthParameters,
    GrowthState,
)


@pytest.fixture(scope="module")
def settled() -> GrowthModel:
    """A GROWTH model run twenty years from its calibration."""
    model = GrowthModel()
    model.run(20)
    return model


def test_fresh_model_exposes_the_calibration():
    """A constructed model is fully seeded: its levers are the book's baseline."""
    model = GrowthModel()
    baselines = GrowthCalibration.baseline().baselines()
    for name, value in model.actions.to_dict().items():
        assert value == pytest.approx(baselines[name])
    assert isinstance(model.parameters, GrowthParameters)


def test_constructor_overrides_win_over_the_calibration():
    """``GrowthModel(Rbbar=0.04)`` is applied after seeding, so it sticks."""
    assert GrowthModel(Rbbar=0.04).actions.Rbbar == 0.04


def test_only_visible_inputs_can_be_overridden():
    """A state variable is not an exogenous input."""
    with pytest.raises(ValueError, match="not visible parameters"):
        GrowthModel(Y=1.0)


def test_settled_economy_is_plausible(settled):
    """After burn-in the economy sits near full employment with positive levels."""
    state = settled.state
    levels = np.array(list(state.to_dict().values()))
    assert np.all(np.isfinite(levels))
    assert 0.95 < state.ER < 1.05
    assert -0.05 < state.PI < 0.1
    for name in ("Y", "Yk", "P", "Ck", "Ik", "Vk"):
        assert getattr(state, name) > 0.0, name


def test_advance_returns_the_new_state_and_records_it():
    """One call applies the inputs, solves one period and appends one solution."""
    model = GrowthModel()
    model.run(5)
    before = len(model.solutions)
    state = model.advance(model.parameters, model.actions)
    assert isinstance(state, GrowthState)
    assert len(model.solutions) == before + 1
    assert state == model.state


def test_bill_rate_follows_the_policy_rate():
    """The policy lever is the bill rate itself: ``Rb`` moves to ``Rbbar`` at once."""
    model = GrowthModel()
    model.run(5)
    actions = dataclasses.replace(model.actions, Rbbar=0.045)
    assert model.advance(model.parameters, actions).Rb == pytest.approx(0.045)


def test_identical_inputs_give_identical_paths():
    """The solver is deterministic: two fresh models agree period for period."""
    a, b = GrowthModel(), GrowthModel()
    for _ in range(10):
        sa = a.advance(a.parameters, a.actions)
        sb = b.advance(b.parameters, b.actions)
    assert sa == sb


def test_model_reports_its_value_spaces():
    """``STATE``/``PARAMETERS``/``ACTIONS`` are the GROWTH specializations."""
    assert (GrowthModel.STATE, GrowthModel.PARAMETERS, GrowthModel.ACTIONS) == (
        GrowthState,
        GrowthParameters,
        GrowthActions,
    )
