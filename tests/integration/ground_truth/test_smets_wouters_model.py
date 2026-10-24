"""The Smets-Wouters DSGE as a :class:`~economic_models.BaseEconomicModel`.

Determinacy is the first thing that breaks when an equation is mistranscribed,
so it is checked at the estimated calibration and across sampled ones; the sign
of monetary transmission is the second.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from economic_models.ground_truth.models.smets_wouters import SwCalibration, SwModel, SwState


@pytest.fixture(scope="module")
def model() -> SwModel:
    """The estimated economy, unstepped."""
    return SwModel()


def test_estimated_economy_is_determinate(model):
    """The calibration has a unique stable rational-expectations solution."""
    solution = model.solution
    assert solution.verdict.determinate
    # The levels integrate a balanced growth path, so the largest root sits at one.
    assert solution.spectral_radius <= 1.0 + 1e-8


@pytest.mark.parametrize("seed", range(5))
def test_sampled_calibrations_are_determinate(seed):
    """Every draw the run generator can make is a solvable economy."""
    calibration = SwCalibration.sample(np.random.default_rng(seed))
    assert SwModel(calibration).solution.verdict.determinate


def test_timestep_is_pinned_to_a_quarter():
    """Calvo rigidities cannot be rescaled; any other ``dt`` is refused."""
    with pytest.raises(ValueError, match="quarterly"):
        SwModel(dt=1.0)


def test_only_visible_inputs_can_be_overridden():
    """A name outside Parameters/Actions is rejected by name."""
    with pytest.raises(ValueError, match="bogus"):
        SwModel(bogus=1.0)


def test_resting_economy_is_steady(model):
    """At the calibration the bank does nothing: no rule deviation, on target."""
    assert model.actions.Rdev == 0.0
    state = SwModel().advance(model.parameters, model.actions)
    assert isinstance(state, SwState)
    assert state.ER == pytest.approx(0.95)
    assert np.all(np.isfinite(list(state.to_dict().values())))


def test_tightening_lowers_inflation_and_employment(model):
    """A positive rule deviation raises the rate and cools the economy."""
    calm, tight = SwModel(), SwModel()
    hawkish = dataclasses.replace(model.actions, Rdev=model.actions.Rdev + 0.01)
    for _ in range(8):
        s_calm = calm.advance(model.parameters, model.actions)
        s_tight = tight.advance(model.parameters, hawkish)
    assert s_tight.R > s_calm.R
    assert s_tight.PI < s_calm.PI
    assert s_tight.ER < s_calm.ER


def test_internal_state_restores_onto_a_fresh_model(model):
    """A recorded solution pins a fresh model exactly -- the branch contract."""
    source = SwModel()
    source.run(5)
    target = SwModel()
    target.set_values(dict(source.solutions[-1]))
    assert source.advance(model.parameters, model.actions) == target.advance(
        model.parameters, model.actions
    )
