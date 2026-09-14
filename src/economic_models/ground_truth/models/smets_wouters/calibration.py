"""Calibration of the Smets-Wouters model, and how a run draws its own economy.

:class:`SwCalibration` plays the part :class:`GrowthCalibration` plays for the
GROWTH model, and partitions a complete starting point the same way:

* :attr:`~SwCalibration.params` -- the deep parameters, which are structural and
  are not part of the visible interface;
* :attr:`~SwCalibration.shock_sigma` -- the size of each innovation;
* :attr:`~SwCalibration.exogenous_baselines` -- baseline values of the visible
  :class:`~...variables.SwParameters` and :class:`~...variables.SwActions`;
* :attr:`~SwCalibration.initial_levels` -- where the reconstructed observable
  levels start, since the model itself only speaks in deviations.

What is different here, and better, is where the numbers come from.
GROWTH's baseline is a single hand-calibrated vector out of a textbook, and its
excitation invents plausible bands around it. Smets and Wouters *estimated*
theirs, and Tables 1A and 1B report the whole posterior: mode, mean, and the 5th
and 95th percentiles of every structural parameter and every shock process. So
:meth:`SwCalibration.sample` can draw a genuinely different economy per run from
the distribution the data actually support, rather than from a corridor someone
chose.

Draws are rejected and redrawn when they leave the model without a unique stable
solution. That check is the Blanchard-Kahn condition, and it is this model's
counterpart to GROWTH's solver-failure resampling: some parameter vectors simply
do not describe an economy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from economic_models.ground_truth.models.smets_wouters.parameters import SwParams

#: Posterior mode of every deep parameter, Smets and Wouters (2007) Table 4 and
#: the mode column of Tables 1A and 1B. The five parameters they fix rather than
#: estimate -- depreciation, the spending share, the steady-state wage mark-up
#: and the two Kimball curvatures -- carry the values fixed in Section II.A.
_MODE: dict[str, float] = {
    # preferences
    "habit": 0.71,
    "sigma_c": 1.39,
    "sigma_l": 1.92,
    # technology and adjustment costs
    "varphi": 5.48,
    "psi": 0.54,
    "alpha": 0.19,
    "phi_p": 1.61,
    "delta": 0.025,  # fixed in estimation
    # nominal rigidities
    "xi_p": 0.65,
    "xi_w": 0.73,
    "iota_p": 0.22,
    "iota_w": 0.59,
    "epsilon_p": 10.0,  # fixed in estimation
    "epsilon_w": 10.0,  # fixed in estimation
    "lambda_w": 1.5,  # fixed in estimation
    # policy
    "rho_r": 0.81,
    "r_pi": 2.03,
    "r_y": 0.08,
    "r_dy": 0.22,
    # steady state
    "pi_bar": 0.81,
    "r_bar_discount": 0.16,
    "gamma_bar": 0.43,
    "l_bar": -0.10,
    "g_y": 0.18,  # fixed in estimation
    # shock persistence
    "rho_a": 0.95,
    "rho_b": 0.18,
    "rho_g": 0.97,
    "rho_i": 0.71,
    "rho_mp": 0.12,
    "rho_p": 0.90,
    "rho_w": 0.97,
    "mu_p": 0.74,
    "mu_w": 0.88,
    "rho_ga": 0.52,
    # additions this project makes; see the module docstring of
    # ``modules.policy`` for the credibility parameter and ``modules.exogenous``
    # for the two observed cost-push drivers.
    "credibility": 0.10,
    "xi_e": 0.60,
    "rho_qe": 0.85,
    "rho_tau": 0.95,
    "rho_pe": 0.90,
    "energy_share": 0.03,
}

#: Posterior 5th and 95th percentiles, Tables 1A and 1B. A parameter absent here
#: was fixed rather than estimated, or is one of this project's additions, and is
#: held at its mode across every run.
_INTERVAL: dict[str, tuple[float, float]] = {
    "varphi": (3.97, 7.42),
    "sigma_c": (1.16, 1.59),
    "habit": (0.64, 0.78),
    "xi_w": (0.60, 0.81),
    "sigma_l": (0.91, 2.78),
    "xi_p": (0.56, 0.74),
    "iota_w": (0.38, 0.78),
    "iota_p": (0.10, 0.38),
    "psi": (0.36, 0.72),
    "phi_p": (1.48, 1.73),
    "r_pi": (1.74, 2.33),
    "rho_r": (0.77, 0.85),
    "r_y": (0.05, 0.12),
    "r_dy": (0.18, 0.27),
    "pi_bar": (0.61, 0.96),
    "r_bar_discount": (0.07, 0.26),
    "gamma_bar": (0.40, 0.45),
    "alpha": (0.16, 0.21),
    "rho_a": (0.94, 0.97),
    "rho_b": (0.07, 0.36),
    "rho_g": (0.96, 0.99),
    "rho_i": (0.61, 0.80),
    "rho_mp": (0.04, 0.24),
    "rho_p": (0.80, 0.96),
    "rho_w": (0.94, 0.99),
    "mu_p": (0.54, 0.85),
    "mu_w": (0.75, 0.93),
    "rho_ga": (0.37, 0.66),
}

#: Standard error of each innovation, Table 1B, in the units of the variable the
#: disturbance enters. The three levers carry no innovation of their own here:
#: their sizes belong to the excitation that moves them, not to the estimated
#: model.
_SIGMA: dict[str, float] = {
    "z_a": 0.45,
    "z_b": 0.23,
    "z_g": 0.53,
    "z_i": 0.45,
    "z_p": 0.14,
    "z_w": 0.24,
    "z_r": 0.24,
    "z_pistar": 0.0,
    "z_qe": 0.0,
    "z_tau": 0.0,
    "z_pe": 0.0,
}

#: 5th and 95th percentiles of the innovation standard errors, Table 1B.
_SIGMA_INTERVAL: dict[str, tuple[float, float]] = {
    "z_a": (0.41, 0.50),
    "z_b": (0.19, 0.27),
    "z_g": (0.48, 0.58),
    "z_i": (0.37, 0.53),
    "z_p": (0.11, 0.16),
    "z_w": (0.20, 0.28),
    "z_r": (0.22, 0.27),
}

#: How wide a normal draw the reported 5-95 interval is treated as. The posteriors
#: are close to normal, and 90 percent of a normal lies within 1.645 standard
#: deviations of its mean.
_NORMAL_90 = 1.6448536269514722


@dataclass(frozen=True)
class SwCalibration:
    """A complete starting point for a Smets-Wouters economy."""

    params: SwParams  #: the deep parameters
    shock_sigma: Mapping[str, float]  #: standard error of each innovation
    exogenous_baselines: Mapping[str, float]  #: baselines of the visible exogenous inputs
    initial_levels: Mapping[str, float]  #: where the reconstructed observables start
    drift: tuple[str, ...] = field(default=())  #: deep parameters the excitation moves

    def sigma(self, innovation: str) -> float:
        """Standard error of ``innovation``; zero for the levers."""
        return float(self.shock_sigma.get(innovation, 0.0))

    def baselines(self) -> dict[str, float]:
        """Baseline value of everything the excitation can drift around.

        The deep parameters it is allowed to move, plus the visible exogenous
        inputs -- the same contract as
        :meth:`~...growth.calibration.GrowthCalibration.baselines`.
        """
        deep = {name: float(getattr(self.params, name)) for name in self.drift}
        return {**deep, **{k: float(v) for k, v in self.exogenous_baselines.items()}}

    @classmethod
    def baseline(cls) -> "SwCalibration":
        """The estimated economy: Smets and Wouters' posterior mode."""
        params = SwParams(**_MODE)
        return cls(
            params=params,
            shock_sigma=dict(_SIGMA),
            exogenous_baselines=_baselines(params),
            initial_levels=dict(_INITIAL_LEVELS),
            drift=_DRIFTING,
        )

    @classmethod
    def sample(cls, rng: np.random.Generator, *, strength: float = 1.0) -> "SwCalibration":
        """Draw one economy from the estimated posterior.

        Each parameter with a reported interval is drawn independently from the
        normal that interval describes, then clipped back into it so no draw is
        wilder than the data support. ``strength`` scales the spread: at zero
        every run is the estimated economy, at one they are as different as the
        posterior says they might be.

        The draw is *not* checked for determinacy here -- that needs the assembled
        system, so the run generator does it and redraws.
        """
        mode = dict(_MODE)
        for name, (low, high) in _INTERVAL.items():
            mode[name] = _draw(rng, mode[name], low, high, strength)
        sigma = dict(_SIGMA)
        for name, (low, high) in _SIGMA_INTERVAL.items():
            sigma[name] = _draw(rng, sigma[name], low, high, strength)
        params = SwParams(**mode)
        return cls(
            params=params,
            shock_sigma=sigma,
            exogenous_baselines=_baselines(params),
            initial_levels=dict(_INITIAL_LEVELS),
            drift=_DRIFTING,
        )


# -- the visible side ------------------------------------------------------

def _baselines(params: SwParams) -> dict[str, float]:
    """Baselines of the visible exogenous inputs, at the drawn parameters.

    The disturbances are deviations and so rest at zero; the levers rest where the
    estimated rule leaves them, which is what makes "do nothing" mean "follow
    Smets and Wouters' own estimated Federal Reserve reaction function". Two of
    them are tied to the economy that was drawn rather than fixed: the announced
    objective must start at the inflation rate the drawn steady state actually
    has, and trend productivity growth at the drawn trend, or the very first
    period would be a policy shock nobody chose.
    """
    return {
        "EXg": 0.0,
        "theta": 0.22,
        "GRpr": 4.0 * params.gamma_bar / 100.0,
        "Nfe": 100.0,
        "Penergy": 100.0,
        "ADDbl": 0.01,
        "Rdev": 0.0,
        "PIstar": 4.0 * params.pi_bar / 100.0,
        "QE": 0.0,
    }

#: Starting levels of the reconstructed observables. The scale is arbitrary --
#: everything the model and the mandate care about is a ratio or a growth rate --
#: so output starts at 100 and the rest at their steady-state shares of it.
_INITIAL_LEVELS: dict[str, float] = {
    "Yk": 100.0,
    "P": 100.0,
    "Nfe": 100.0,
    "employment_rate": 0.95,
    "debt_ratio": 0.60,
}

#: The deep parameters the excitation is allowed to move over a run. These are the
#: structural drift that makes the environment genuinely non-stationary -- the
#: counterpart of GROWTH's wandering ``alpha1``, ``gamma0``, ``eta0``, ``omega3``
#: and ``psid``. Every one of them changes the model's *solution*, not just its
#: forcing, so the economy the bank is learning is quietly moving underneath it.
_DRIFTING: tuple[str, ...] = ("habit", "xi_p", "xi_w", "varphi", "sigma_c")


def _draw(
    rng: np.random.Generator, mode: float, low: float, high: float, strength: float
) -> float:
    """One parameter draw: normal through the reported interval, clipped to it."""
    if strength <= 0.0:
        return mode
    centre = 0.5 * (low + high)
    scale = strength * (high - low) / (2.0 * _NORMAL_90)
    return float(np.clip(rng.normal(centre, scale), low, high))
