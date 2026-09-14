"""The excitation presets that turn the estimated model into an ensemble of economies.

Where GROWTH's bands were chosen by hand around a textbook calibration, these are
read off the estimated model. The disturbance sizes and persistences are Smets and
Wouters' own posterior (Table 1B), translated from quarterly to annual terms; the
corridors the drifting deep parameters wander in are their posterior 5th to 95th
percentiles (Table 1A). So the ensemble is not "plausible variation around a
guess" but the variation the US data actually support.

Three presets, matching GROWTH's so the two ground truths can be run under the
same names:

* :meth:`~SwExcitationConfig.default` -- the estimated economy and nothing else:
  no volatility regimes, no crises, no heterogeneity between runs.
* :meth:`~SwExcitationConfig.calm` -- the same, widened to record the columns
  ``realistic`` records, so a dataset drawn under either has the same shape.
* :meth:`~SwExcitationConfig.realistic` -- the world preset: stochastic
  volatility, crises that hit the risk premium and investment together, and a
  per-run climate that decides how stormy this particular economy is.
"""

from __future__ import annotations

from dataclasses import dataclass

from economic_models.ground_truth.excitation.base import ExcitationConfig
from economic_models.ground_truth.excitation.specs import (
    AR1Spec,
    ClimateSpec,
    CrisisSpec,
    ExcitationJitter,
    RandomWalkSpec,
    StochasticVolatilitySpec,
)
from economic_models.ground_truth.models.smets_wouters.excitation.specs import (
    SpendingSpec,
    around,
    from_quarterly,
)

import numpy as np


@dataclass(frozen=True)
class SwExcitationConfig(ExcitationConfig):
    """Excitation of the Smets-Wouters economy, with its two model-specific drivers."""

    gov_spending: SpendingSpec  #: the exogenous spending disturbance and its fiscal reflex
    labour_force: RandomWalkSpec  #: demographics
    energy: RandomWalkSpec  #: the observed cost-push driver

    def _perturb_model_specs(
        self, rng: np.random.Generator, jitter: ExcitationJitter
    ) -> "SwExcitationConfig":
        """A neighbouring economy: the model-specific specs perturbed alongside the rest."""
        return SwExcitationConfig(
            visible=self.visible,
            hidden=self.hidden,
            volatility=self.volatility,
            crisis=self.crisis,
            climate=self.climate,
            gov_spending=self.gov_spending.perturbed(rng, jitter),
            labour_force=self.labour_force,
            energy=self.energy,
        )

    # -- presets -----------------------------------------------------------

    @classmethod
    def default(cls) -> "SwExcitationConfig":
        """The estimated economy: the posterior's disturbances and nothing else."""
        return cls(
            visible=_visible(),
            hidden=_hidden(),
            gov_spending=_SPENDING,
            labour_force=RandomWalkSpec(sigma=0.004, max_logdev=0.06),
            energy=RandomWalkSpec(sigma=0.05, max_logdev=0.20),
        )

    @classmethod
    def calm(cls) -> "SwExcitationConfig":
        """``default`` widened to the columns ``realistic`` records.

        A dataset has to have the same shape whichever preset drew it, so the
        parameters that only drift in the stormy world still appear here -- barely
        moving, rather than absent.
        """
        return cls(
            visible=_visible(),
            hidden=_hidden(wide=False),
            gov_spending=_SPENDING,
            labour_force=RandomWalkSpec(sigma=0.004, max_logdev=0.06),
            energy=RandomWalkSpec(sigma=0.05, max_logdev=0.20),
        )

    @classmethod
    def realistic(cls) -> "SwExcitationConfig":
        """The world preset: volatility regimes, crises, and heterogeneous economies."""
        return cls(
            visible=_visible(wide=True),
            hidden=_hidden(wide=True),
            volatility=StochasticVolatilitySpec(rho=0.94, xi=0.22, max_logvol=1.2),
            crisis=CrisisSpec(
                prob=0.025,
                min_gap=8,
                decay_range=(0.68, 0.85),
                severity_range=(0.8, 1.2),
                # A crisis in this economy is a demand collapse: the wedge between
                # the policy rate and what households require blows out, capital
                # goods stop being worth building, and firms and unions both stop
                # cutting -- which is what makes it a problem monetary policy
                # cannot simply cut its way out of.
                impulses={"shock_b": 0.9, "shock_i": -0.7, "shock_p": 0.12},
                financial_prob=0.35,
                financial_impulses={"shock_b": 0.6, "xi_p": 0.03},
            ),
            climate=ClimateSpec(),
            gov_spending=SpendingSpec(
                gap_sigma=0.60, gap_clip=2.5, stabilizer=0.7, bounds=(-3.0, 4.0)
            ),
            labour_force=RandomWalkSpec(sigma=0.004, max_logdev=0.06),
            energy=RandomWalkSpec(sigma=0.07, max_logdev=0.30),
        )


# -- the bands -------------------------------------------------------------

#: Fiscal policy. The discretionary component is scaled to the paper's own
#: estimate of the spending shock, and the response lifts spending by about half a
#: percent of output for each point the employment rate falls short.
#:
#: Note how much smaller this is than GROWTH's equivalent, and why. There, fiscal
#: policy is the *only* fast stabiliser -- the policy rate's channel to inflation
#: runs about fifteen years, so a Taylor rule destabilises rather than anchors. In
#: this economy the Taylor rule is inside the model and does the anchoring itself,
#: so the fiscal reflex is a supporting actor. Made as strong as GROWTH's it does
#: not help but hurts: spending saturates its bounds, which is a permanent demand
#: shock, which the rule answers by tightening, which lowers employment, which
#: calls for more spending.
_SPENDING = SpendingSpec(gap_sigma=0.50, gap_clip=2.0, stabilizer=0.5, bounds=(-2.5, 3.0))


def _visible(*, wide: bool = False) -> dict[str, AR1Spec]:
    """Drift of the exogenous inputs the bank observes, including its own levers.

    The levers drift for the same reason they do in GROWTH: the recorded history
    has no agent in it, so unless something moved them a proxy fitted to that
    history has no way to learn what they do. The ranges are deliberately narrower
    than the environment's action box -- a historical central bank that had already
    explored the whole box would leave the agent nothing to discover.
    """
    span = 1.5 if wide else 1.0
    return {
        # observed exogenous conditions
        "theta": AR1Spec(sigma=0.004, lower=0.19, upper=0.26),
        "GRpr": AR1Spec(sigma=0.0010, lower=0.012, upper=0.024, phi=0.97),
        "ADDbl": AR1Spec(sigma=0.002, lower=0.0, upper=0.025 * span),
        # the bank's own levers, as a historical bank moved them
        "Rdev": AR1Spec(sigma=0.003, lower=-0.015 * span, upper=0.015 * span),
        "PIstar": AR1Spec(sigma=0.0005, lower=0.018, upper=0.045, phi=0.98),
        "QE": AR1Spec(sigma=0.002, lower=0.0, upper=0.015 * span),
    }


def _hidden(*, wide: bool = False) -> dict[str, AR1Spec]:
    """Drift of everything the bank cannot see.

    Two kinds, and the difference matters. The five disturbances are the model's
    own estimated shock processes, and they only *force* the economy. The five
    deep parameters underneath them change the economy's **solution** -- how
    sticky prices are, how much habit anchors consumption, how costly it is to
    move investment -- so as they drift the bank is chasing a structure that will
    not hold still. That is the sharpest form of the problem this whole project is
    about, and none of it is observable.
    """
    span = 1.25 if wide else 1.0
    disturbances = {
        "shock_a": from_quarterly(0.95, 0.45 * span),
        "shock_b": from_quarterly(0.18, 0.23 * span),
        "shock_i": from_quarterly(0.71, 0.45 * span),
        # The two mark-ups are ARMA(1,1); see ``from_quarterly`` for why their
        # moving-average terms cannot be left out.
        "shock_p": from_quarterly(0.90, 0.14 * span, ma=0.74),
        "shock_w": from_quarterly(0.97, 0.24 * span, ma=0.88),
    }
    # Corridors are the posterior 5th to 95th percentiles of Table 1A: the drift
    # stays inside what the data support, however long the run.
    reach = 0.35 if wide else 0.25
    structural = {
        "habit": around(0.64, 0.78, sigma=0.010 * reach / 0.25),
        "xi_p": around(0.56, 0.74, sigma=0.012 * reach / 0.25),
        "xi_w": around(0.60, 0.81, sigma=0.015 * reach / 0.25),
        "varphi": around(3.97, 7.42, sigma=0.20 * reach / 0.25),
        "sigma_c": around(1.16, 1.59, sigma=0.030 * reach / 0.25),
    }
    return {**disturbances, **structural}
