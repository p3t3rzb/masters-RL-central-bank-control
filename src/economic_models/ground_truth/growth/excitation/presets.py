"""The GROWTH :class:`GrowthExcitationConfig` and its calibrated presets.

A config bundles the model-agnostic per-input drift specs (visible and hidden)
plus the optional volatility/crisis/climate layers (all inherited from
:class:`~economic_models.ground_truth.excitation.base.ExcitationConfig`) and
GROWTH's own special inputs: the government-spending stabilizer and the ``Nfe``
random walk. :meth:`GrowthExcitationConfig.default` is the calm calibration the
training datasets were generated with; :meth:`GrowthExcitationConfig.realistic`
adds volatility clustering, crises and per-run climates.
"""

from __future__ import annotations

from dataclasses import dataclass

from economic_models.ground_truth.excitation.base import ExcitationConfig
from economic_models.ground_truth.excitation.specs import (
    AR1Spec,
    ClimateSpec,
    CrisisSpec,
    RandomWalkSpec,
    StochasticVolatilitySpec,
)
from economic_models.ground_truth.growth.excitation.specs import GovSpendingSpec


@dataclass(frozen=True)
class GrowthExcitationConfig(ExcitationConfig):
    """Everything tunable about how a GROWTH run's exogenous inputs are excited.

    Extends the generic :class:`ExcitationConfig` with GROWTH's special inputs:
    ``GRg`` and ``Nfe`` are handled by their own specs rather than a plain AR(1).
    The baselines the deviations are taken around come from the model calibration,
    not from here.
    """

    gov_spending: GovSpendingSpec  #: government spending growth ``GRg``
    nfe: RandomWalkSpec  #: full-employment level ``Nfe``

    @classmethod
    def default(cls) -> "GrowthExcitationConfig":
        """The calibrated excitation used to generate the training datasets.

        Sigmas and clips were tuned so the visible levers move enough to be
        identifiable while the economy stays inside its stable corridor around
        full employment under the worst combinations of hidden drift.
        """
        return cls(
            visible={
                # Actions
                "Rbbar": AR1Spec(0.0025, 0.015, 0.055),
                "NCAR": AR1Spec(0.002, 0.07, 0.14),
                "ro": AR1Spec(0.002, 0.03, 0.09),
                # Parameters
                "theta": AR1Spec(0.002, 0.21, 0.25),
                "GRpr": AR1Spec(0.001, 0.02, 0.04),
                "ADDbl": AR1Spec(0.001, 0.01, 0.03),
                "Rln": AR1Spec(0.0015, 0.055, 0.09),
                "NPLk": AR1Spec(0.00075, 0.01, 0.035),
            },
            # Small sigmas and tight clips (~a few % of baseline): structural
            # drift, not policy-scale shocks, and the corridor must survive their
            # worst combinations.
            hidden={
                "RA": AR1Spec(0.002, -0.008, 0.008),  # expected-sales disturbance (11.2)
                "alpha1": AR1Spec(0.004, 0.72, 0.78),  # consume out of income
                "alpha2": AR1Spec(0.0006, 0.058, 0.070),  # consume out of wealth
                "gamma0": AR1Spec(0.00015, 0.00022, 0.00222),  # investment animal spirits
                "eta0": AR1Spec(0.001, 0.068, 0.080),  # household new-credit demand
                "omega3": AR1Spec(0.008, 0.41, 0.50),  # wage adjustment speed
                "psid": AR1Spec(0.002, 0.14, 0.165),  # dividend payout ratio
            },
            gov_spending=GovSpendingSpec(
                gap_sigma=0.0015, gap_clip=0.0075, stabilizer=0.2, bounds=(0.005, 0.06)
            ),
            nfe=RandomWalkSpec(sigma=0.002, max_logdev=0.05),
        )

    @classmethod
    def realistic(cls) -> "GrowthExcitationConfig":
        """A crisis-capable excitation resembling a real economy's history.

        Extends :meth:`default` along three axes so a single ensemble spans the
        range a real economy walks through:

        * **Volatility clustering** -- a stochastic-volatility regime scales every
          innovation, so runs alternate between calm and turbulent stretches.
        * **Crises** -- rare joint demand/credit shocks that push the economy into
          a slump it usually recovers from; a minority (``financial_prob``) also
          collapse equity portfolio preference (``lambda40``), a financial crisis
          that crashes the equity price and destroys real wealth.
        * **Room to move** -- crisis-targeted inputs get wider clips so a shock is
          not immediately clipped away, and the government-spending stabilizer is
          strengthened so the recovery channel can pull a deep slump back.

        Drift baselines and non-targeted inputs are inherited from :meth:`default`.
        """
        base = cls.default()
        # Widen the crisis-targeted visible/hidden inputs so an impulse has room
        # to move the level before the hard safety clip bites.
        visible = dict(base.visible)
        visible["Rln"] = AR1Spec(0.0015, 0.055, 0.13)  # loan-spread blowout
        visible["NPLk"] = AR1Spec(0.00075, 0.01, 0.09)  # bad-loan spike
        hidden = dict(base.hidden)
        hidden["RA"] = AR1Spec(0.002, -0.05, 0.05)  # sales-expectation collapse
        hidden["gamma0"] = AR1Spec(0.00015, 0.00002, 0.00222)  # investment spirits
        hidden["eta0"] = AR1Spec(0.001, 0.045, 0.080)  # household credit demand
        hidden["alpha1"] = AR1Spec(0.004, 0.66, 0.78)  # consumption propensity
        # Equity portfolio preference (11.66): baseline 0.671, with room to crash
        # far below it so a financial-crisis impulse can collapse the equity price.
        hidden["lambda40"] = AR1Spec(0.003, 0.40, 0.75)
        return cls(
            visible=visible,
            hidden=hidden,
            # Higher gain and ceiling: the recovery channel must be strong enough
            # to climb out of a deep slump before the deflation spiral takes over.
            gov_spending=GovSpendingSpec(
                gap_sigma=0.0015, gap_clip=0.0075, stabilizer=0.35, bounds=(0.005, 0.10)
            ),
            nfe=base.nfe,
            volatility=StochasticVolatilitySpec(rho=0.94, xi=0.22, max_logvol=1.2),
            crisis=CrisisSpec(
                prob=0.03,
                min_gap=8,
                # Duration varies per onset: decay 0.68 clears a crisis in ~2-3
                # years, 0.88 drags it out past a decade. Severity scales the whole
                # bundle so crises differ in depth too. Kept inside the range where
                # the fiscal stabilizer still reliably recovers the corridor.
                decay_range=(0.68, 0.85),
                severity_range=(0.8, 1.2),
                impulses={
                    "NPLk": 0.030,  # non-performing loans jump
                    "Rln": 0.020,  # loan spread widens
                    "gamma0": -0.00080,  # investment animal spirits collapse
                    "RA": -0.028,  # sales expectations slump
                    "eta0": -0.014,  # household credit demand dries up
                    "alpha1": -0.030,  # precautionary rise in saving
                },
                # ~1 in 3 crises is also financial: households flee equities, the
                # equity price and Tobin's Q collapse, and real wealth falls.
                financial_prob=0.35,
                financial_impulses={
                    "lambda40": -0.16,  # equity portfolio preference collapses
                },
            ),
            # Draw each run's overall turbulence once: calm draws stay near the
            # steady path with few crises, stormy draws are crisis-prone.
            climate=ClimateSpec(),
        )
