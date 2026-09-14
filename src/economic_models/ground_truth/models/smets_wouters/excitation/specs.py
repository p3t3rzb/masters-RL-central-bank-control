"""Excitation specs specific to the Smets-Wouters economy.

Two things the model-agnostic machinery cannot supply on its own: a translation
from the paper's quarterly shock estimates into the annualised terms every spec
in this project is written in, and the countercyclical fiscal response that keeps
an economy driven this hard from wandering out of its own corridor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from economic_models.ground_truth.excitation.specs import AR1Spec, ExcitationJitter

#: The employment rate the economy rests at, which the fiscal response measures
#: shortfalls against.
_EMPLOYMENT_SS = 0.95

#: How many stationary standard deviations a disturbance is allowed to reach. The
#: clip is a corridor, not a shape: wide enough that it almost never binds on an
#: ordinary draw, tight enough that a crisis cannot push the model somewhere its
#: linearisation is meaningless.
_CLIP_SIGMAS = 5.0


def from_quarterly(
    rho: float, sigma: float, *, ma: float = 0.0, clip: float = _CLIP_SIGMAS
) -> AR1Spec:
    """An :class:`AR1Spec` matching a quarterly shock process of the paper's.

    Every spec in this project states persistence and innovation size *per year*,
    so that changing ``dt`` leaves the process it describes alone; Smets and
    Wouters estimate theirs per quarter. The translation matches the two moments
    that determine what a disturbance does to the economy -- its stationary
    variance and its first autocorrelation -- and then converts both to annual
    terms.

    ``ma`` matters far more than it looks. The two mark-up disturbances are
    ARMA(1,1), and their estimated moving-average terms very nearly cancel their
    autoregressive ones: the wage mark-up's ``rho`` of 0.97 with an ``ma`` of 0.88
    is not a process with a six-year half-life but one with a first
    autocorrelation of 0.2, and a standard deviation a quarter of what the ``rho``
    alone would imply. Driving it as a pure AR(1) makes the single most persistent
    driver of long-run output four times too large and far too persistent, and the
    economy walks away from its own steady state.
    """
    variance_ratio = (1.0 - 2.0 * rho * ma + ma**2) / max(1.0 - rho**2, 1e-12)
    stationary = sigma * math.sqrt(max(variance_ratio, 1e-12))
    autocorrelation = (
        (rho - ma) * (1.0 - rho * ma) / max(1.0 - 2.0 * rho * ma + ma**2, 1e-12)
        if ma
        else rho
    )
    phi = max(autocorrelation, 0.0) ** 4
    annual_sigma = stationary * math.sqrt(max(1.0 - phi**2, 1e-12))
    reach = clip * stationary
    return AR1Spec(sigma=annual_sigma, lower=-reach, upper=reach, phi=phi)


def around(low: float, high: float, *, sigma: float, phi: float = 0.9) -> AR1Spec:
    """An :class:`AR1Spec` drifting a deep parameter inside ``[low, high]``.

    The resting point is left to the calibration: a spec's anchor is its
    *baseline* plus its ``center`` offset, and the baseline here is already the
    value the run drew for that parameter. So a run whose posterior draw put a
    parameter near the edge of its band drifts around that edge rather than being
    quietly pulled back to the middle -- which is the difference between an
    ensemble of different economies and one economy with noise on top.
    """
    return AR1Spec(sigma=sigma, lower=low, upper=high, phi=phi)


@dataclass(frozen=True)
class SpendingSpec:
    """Exogenous spending: a drifting gap plus a countercyclical response.

    The same device GROWTH uses, and for the same reason. An economy driven only
    by unconditional noise has nothing leaning against a bad draw, and a long
    enough run will eventually find one it cannot come back from -- which makes
    the dataset a study of collapse rather than of policy. Real fiscal policy is
    the stabiliser of last resort, so it is modelled here as one: spending rises
    when employment falls, at a speed that is itself a property of the economy
    and so varies from run to run.
    """

    gap_sigma: float  #: innovation size of the discretionary component
    gap_clip: float  #: how far the discretionary component may drift
    stabilizer: float  #: response to a one-point shortfall in the employment rate
    bounds: tuple[float, float]  #: hard limits on the spending disturbance

    #: The visible parameter this response moves.
    target: ClassVar[str] = "EXg"

    def support(self, er_prev: float) -> float:
        """The countercyclical addition to spending at employment rate ``er_prev``.

        Measured in points of the employment rate away from its steady state,
        rather than from one as in GROWTH, because this economy's employment rate
        rests at 0.95 rather than at full employment by construction.
        """
        return self.stabilizer * (_EMPLOYMENT_SS - er_prev) * 100.0

    def gap_spec(self) -> AR1Spec:
        """The discretionary component, as a plain AR(1)."""
        return AR1Spec(sigma=self.gap_sigma, lower=-self.gap_clip, upper=self.gap_clip)

    def perturbed(self, rng: np.random.Generator, jitter: ExcitationJitter) -> "SpendingSpec":
        """A neighbouring fiscal authority: same limits, different reflexes.

        The bounds do not move -- they are the corridor. What moves is how
        vigorously the government leans against a downturn, which is a reaction
        function rather than an innovation size, and so is drawn under the
        jitter's ``feedback`` scale.
        """
        return SpendingSpec(
            gap_sigma=float(self.gap_sigma * np.exp(rng.normal(0.0, jitter.rate))),
            gap_clip=self.gap_clip,
            stabilizer=float(max(0.0, self.stabilizer * (1.0 + rng.normal(0.0, jitter.feedback)))),
            bounds=self.bounds,
        )
