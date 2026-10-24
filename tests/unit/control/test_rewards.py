"""The mandate: three scaled quadratic gaps and the weighted sum over them.

The reward is the one normative choice in the experiment, so its arithmetic is
pinned exactly: zero at the target, ``-1`` per leg at the reference deviation,
and growth annualised by ``dt`` while inflation (already annual) is not.
"""

from __future__ import annotations

import math

import pytest

from control.rewards import MandateReward, RewardContext, RewardFunction
from economic_models.ground_truth.models.growth.variables import (
    GrowthActions,
    GrowthParameters,
    GrowthState,
)


@pytest.fixture
def context(filled):
    """A builder for a one-period context around a GROWTH economy at target.

    By default real output grows exactly at potential (``GRpr``, with a constant
    labour force), employment is full and inflation is on the 2% target, so every
    leg is zero; keyword arguments move one thing away from that.
    """

    def build(*, er=1.0, pi=0.02, growth=0.02, grpr=0.02, labour=0.0, dt=1.0):
        prev_state = filled(GrowthState, Yk=100.0, ER=1.0, PI=0.02)
        state = filled(GrowthState, Yk=100.0 * math.exp(growth * dt), ER=er, PI=pi)
        prev_params = filled(GrowthParameters, Nfe=50.0, GRpr=grpr)
        params = filled(GrowthParameters, Nfe=50.0 * math.exp(labour * dt), GRpr=grpr)
        actions = filled(GrowthActions)
        return RewardContext(
            state=state,
            prev_state=prev_state,
            parameters=params,
            prev_parameters=prev_params,
            actions=actions,
            prev_actions=actions,
            dt=dt,
        )

    return build


def test_on_target_every_leg_is_zero(context):
    """Full employment, potential growth and target inflation cost nothing."""
    reward = MandateReward()
    ctx = context()
    assert reward.terms(ctx) == pytest.approx({"unemployment": 0.0, "growth": 0.0, "inflation": 0.0})
    assert reward(ctx) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs, leg",
    [
        ({"er": 0.98}, "unemployment"),
        ({"er": 1.02}, "unemployment"),
        ({"growth": 0.0}, "growth"),
        ({"growth": 0.04}, "growth"),
        ({"pi": 0.0}, "inflation"),
        ({"pi": 0.04}, "inflation"),
    ],
)
def test_reference_deviation_costs_one(context, kwargs, leg):
    """A 2pp miss on any leg, in either direction, contributes exactly ``-1``."""
    terms = MandateReward().terms(context(**kwargs))
    assert terms[leg] == pytest.approx(-1.0)
    assert sum(terms.values()) == pytest.approx(-1.0)


def test_potential_includes_labour_force_growth(context):
    """Growing at ``GRpr`` alone is a shortfall when the labour force is growing too."""
    terms = MandateReward().terms(context(growth=0.02, labour=0.02))
    assert terms["growth"] == pytest.approx(-1.0)
    on_potential = MandateReward().terms(context(growth=0.04, labour=0.02))
    assert on_potential["growth"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("dt", [1.0, 0.25, 1 / 12])
def test_growth_gap_is_annualised(context, dt):
    """The same annual growth path scores the same at every timestep."""
    terms = MandateReward().terms(context(growth=0.0, dt=dt))
    assert terms["growth"] == pytest.approx(-1.0)


def test_weights_scale_their_own_leg(context):
    """The scalar is the weighted sum of the named terms."""
    reward = MandateReward(w_unemployment=2.0, w_growth=0.5, w_inflation=3.0)
    ctx = context(er=0.98, growth=0.0, pi=0.04)
    assert reward(ctx) == pytest.approx(-(2.0 + 0.5 + 3.0))
    assert reward.term_names == ("unemployment", "growth", "inflation")


def test_employment_target_belongs_to_the_economy(context):
    """An economy resting below one is not charged for a gap no policy can close."""
    assert MandateReward(employment_target=0.95).terms(context(er=0.95))["unemployment"] == 0.0
    assert MandateReward().terms(context(er=0.95))["unemployment"] < 0.0


def test_base_combines_any_subclass_terms(context):
    """A new mandate supplies terms and weights; the base does the sum."""

    class Flat(RewardFunction):
        @property
        def weights(self):
            return {"a": 2.0, "b": -1.0}

        def _terms(self, ctx):
            return {"a": 1.5, "b": 4.0}

    assert Flat()(context()) == pytest.approx(2.0 * 1.5 - 4.0)
    assert Flat().term_names == ("a", "b")
