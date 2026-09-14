"""The Smets-Wouters deep parameters, and the steady state they imply.

A log-linearised model's coefficients are not its parameters: every equation's
coefficients are algebraic combinations of a handful of deep parameters and of
the steady-state ratios those parameters pin down. :class:`SwParams` holds the
deep parameters as fields and exposes the combinations as properties, so an
equation module can read ``p.c1`` or ``p.k_y`` and stay a transcription of the
paper rather than a page of algebra.

The steady state follows the model's own long-run relations: the rental rate from
the household's intertemporal condition, the real wage from cost minimisation,
and the great ratios from those two. Nothing here is calibrated separately, so a
drawn parameter vector is automatically internally consistent.

Parameter values, priors and posterior intervals are Smets & Wouters (2007),
Tables 1A and 1B. Everything is quarterly, and the three rate-like parameters
(``pi_bar``, ``r_bar_discount``, ``gamma_bar``) are in *percent per quarter*, as
the paper's measurement equation (15) has them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping


@dataclass(frozen=True)
class SwParams:
    """One economy: the deep parameters of the Smets-Wouters model.

    Field names follow the paper's symbols, spelled out. Instances are frozen;
    :meth:`with_values` returns a modified copy, which is how the drifting
    structural parameters are applied each period.
    """

    # -- preferences -------------------------------------------------------
    habit: float  #: lambda, external habit in consumption
    sigma_c: float  #: inverse intertemporal elasticity of substitution
    sigma_l: float  #: elasticity of labour supply to the real wage

    # -- technology and adjustment costs -----------------------------------
    varphi: float  #: elasticity of the investment adjustment cost
    psi: float  #: capital-utilisation adjustment elasticity, in (0, 1)
    alpha: float  #: capital share in production
    phi_p: float  #: one plus the share of fixed costs in production
    delta: float  #: depreciation rate

    # -- nominal rigidities ------------------------------------------------
    xi_p: float  #: Calvo probability a price is *not* reoptimised
    xi_w: float  #: Calvo probability a wage is *not* reoptimised
    iota_p: float  #: price indexation to past inflation
    iota_w: float  #: wage indexation to past inflation
    epsilon_p: float  #: curvature of the Kimball goods aggregator
    epsilon_w: float  #: curvature of the Kimball labour aggregator
    lambda_w: float  #: steady-state wage mark-up

    # -- policy ------------------------------------------------------------
    rho_r: float  #: interest-rate smoothing in the policy rule
    r_pi: float  #: long-run response to the inflation gap
    r_y: float  #: response to the output gap
    r_dy: float  #: response to the change in the output gap
    credibility: float  #: speed at which an announced target is believed
    xi_e: float  #: Calvo probability employment is *not* adjusted (Smets-Wouters 2003)

    # -- perceived shock persistence ---------------------------------------
    # The excitation supplies the realised path of each disturbance, but private
    # agents still have to forecast it; these are the law of motion they believe
    # in, and so they enter the solution rather than only the data generator.
    rho_a: float  #: persistence of total factor productivity
    rho_b: float  #: persistence of the risk premium
    rho_g: float  #: persistence of exogenous spending
    rho_i: float  #: persistence of investment-specific technology
    rho_mp: float  #: persistence of the monetary policy deviation
    rho_p: float  #: persistence of the price mark-up
    rho_w: float  #: persistence of the wage mark-up
    mu_p: float  #: moving-average term of the price mark-up
    mu_w: float  #: moving-average term of the wage mark-up
    rho_ga: float  #: loading of the productivity innovation on exogenous spending
    rho_qe: float  #: persistence of the credit-easing wedge
    rho_tau: float  #: persistence of the labour tax wedge
    rho_pe: float  #: persistence of energy and import prices
    energy_share: float  #: weight of energy and import prices in marginal cost

    # -- steady state ------------------------------------------------------
    pi_bar: float  #: steady-state inflation, percent per quarter
    r_bar_discount: float  #: 100*(1/beta - 1), percent per quarter
    gamma_bar: float  #: trend growth of the balanced path, percent per quarter
    l_bar: float  #: steady-state hours, a measurement constant
    g_y: float  #: steady-state share of exogenous spending in output

    # -- derived: the basic long-run objects -------------------------------

    @property
    def gamma(self) -> float:
        """Gross quarterly growth factor of the balanced path."""
        return 1.0 + self.gamma_bar / 100.0

    @property
    def beta(self) -> float:
        """The household's discount factor."""
        return 1.0 / (1.0 + self.r_bar_discount / 100.0)

    @property
    def beta_growth(self) -> float:
        """``beta * gamma**(1 - sigma_c)``: the discount factor as it appears in the
        Calvo and adjustment-cost coefficients, adjusted for trend growth."""
        return self.beta * self.gamma ** (1.0 - self.sigma_c)

    @property
    def rk_ss(self) -> float:
        """Steady-state rental rate of capital, from the intertemporal condition."""
        return self.gamma**self.sigma_c / self.beta - (1.0 - self.delta)

    @property
    def w_ss(self) -> float:
        """Steady-state real wage, from cost minimisation given ``rk_ss``."""
        a = self.alpha
        return (
            a**a * (1.0 - a) ** (1.0 - a) / (self.phi_p * self.rk_ss**a)
        ) ** (1.0 / (1.0 - a))

    @property
    def k_y(self) -> float:
        """Steady-state capital-services-to-output ratio."""
        labour_capital = ((1.0 - self.alpha) / self.alpha) * (self.rk_ss / self.w_ss)
        return self.phi_p * labour_capital ** (self.alpha - 1.0)

    @property
    def i_y(self) -> float:
        """Steady-state investment share of output."""
        return (1.0 - (1.0 - self.delta) / self.gamma) * self.gamma * self.k_y

    @property
    def c_y(self) -> float:
        """Steady-state consumption share of output, as the residual."""
        return 1.0 - self.g_y - self.i_y

    @property
    def z_y(self) -> float:
        """Steady-state utilisation-cost share of output, ``rk_ss * k_y``."""
        return self.rk_ss * self.k_y

    @property
    def wl_c(self) -> float:
        """Steady-state ratio of the wage bill to consumption, ``W*H L*/C*``."""
        return (
            (1.0 / self.lambda_w)
            * ((1.0 - self.alpha) / self.alpha)
            * self.rk_ss
            * self.k_y
            / self.c_y
        )

    @property
    def r_ss(self) -> float:
        """Steady-state nominal policy rate, percent per quarter.

        The Fisher relation on the balanced path; its negative is the effective
        lower bound expressed as a deviation, which is what the model works in.
        """
        return 100.0 * (
            self.beta ** (-1) * self.gamma**self.sigma_c * (1.0 + self.pi_bar / 100.0) - 1.0
        )

    # -- derived: equation coefficients ------------------------------------

    @property
    def c1(self) -> float:
        """Weight on lagged consumption in the Euler equation (2)."""
        hg = self.habit / self.gamma
        return hg / (1.0 + hg)

    @property
    def c2(self) -> float:
        """Weight on the expected change in hours in the Euler equation (2)."""
        hg = self.habit / self.gamma
        return ((self.sigma_c - 1.0) * self.wl_c) / (self.sigma_c * (1.0 + hg))

    @property
    def c3(self) -> float:
        """Weight on the ex ante real rate in the Euler equation (2)."""
        hg = self.habit / self.gamma
        return (1.0 - hg) / (self.sigma_c * (1.0 + hg))

    @property
    def i1(self) -> float:
        """Weight on lagged investment in the investment equation (3)."""
        return 1.0 / (1.0 + self.beta_growth)

    @property
    def i2(self) -> float:
        """Weight on the value of capital in the investment equation (3)."""
        return 1.0 / ((1.0 + self.beta_growth) * self.gamma**2 * self.varphi)

    @property
    def q1(self) -> float:
        """Weight on the expected future value of capital in the arbitrage equation (4)."""
        return (1.0 - self.delta) / (self.rk_ss + 1.0 - self.delta)

    @property
    def z1(self) -> float:
        """Elasticity of utilisation to the rental rate, equation (7)."""
        return (1.0 - self.psi) / self.psi

    @property
    def k1(self) -> float:
        """Weight on the existing capital stock in the accumulation equation (8)."""
        return (1.0 - self.delta) / self.gamma

    @property
    def k2(self) -> float:
        """Loading of the investment-specific shock in the accumulation equation (8)."""
        return (
            (1.0 - (1.0 - self.delta) / self.gamma)
            * (1.0 + self.beta_growth)
            * self.gamma**2
            * self.varphi
        )

    @property
    def pi1(self) -> float:
        """Weight on lagged inflation in the Phillips curve (10)."""
        return self.iota_p / (1.0 + self.beta_growth * self.iota_p)

    @property
    def pi2(self) -> float:
        """Weight on expected inflation in the Phillips curve (10)."""
        return self.beta_growth / (1.0 + self.beta_growth * self.iota_p)

    @property
    def pi3(self) -> float:
        """Slope of the Phillips curve on the price mark-up, equation (10)."""
        return (
            1.0
            / (1.0 + self.beta_growth * self.iota_p)
            * ((1.0 - self.beta_growth * self.xi_p) * (1.0 - self.xi_p))
            / (self.xi_p * ((self.phi_p - 1.0) * self.epsilon_p + 1.0))
        )

    @property
    def w1(self) -> float:
        """Weight on the lagged real wage in the wage equation (13)."""
        return 1.0 / (1.0 + self.beta_growth)

    @property
    def w2(self) -> float:
        """Weight on current inflation in the wage equation (13)."""
        return (1.0 + self.beta_growth * self.iota_w) / (1.0 + self.beta_growth)

    @property
    def w3(self) -> float:
        """Weight on lagged inflation in the wage equation (13)."""
        return self.iota_w / (1.0 + self.beta_growth)

    @property
    def e1(self) -> float:
        """Weight on lagged employment in the employment equation (Smets-Wouters 2003)."""
        return 1.0 / (1.0 + self.beta_growth)

    @property
    def e2(self) -> float:
        """Speed at which employment closes the gap to hours worked."""
        return (
            (1.0 - self.beta_growth * self.xi_e)
            * (1.0 - self.xi_e)
            / ((1.0 + self.beta_growth) * self.xi_e)
        )

    @property
    def w4(self) -> float:
        """Slope of the wage equation on the wage mark-up, equation (13)."""
        return (
            1.0
            / (1.0 + self.beta_growth)
            * ((1.0 - self.beta_growth * self.xi_w) * (1.0 - self.xi_w))
            / (self.xi_w * ((self.lambda_w - 1.0) * self.epsilon_w + 1.0))
        )

    # -- modification ------------------------------------------------------

    def with_values(self, values: Mapping[str, float]) -> "SwParams":
        """A copy with ``values`` applied -- how the drifting parameters are set.

        Rejects unknown names rather than ignoring them, so a renamed excitation
        target fails loudly instead of silently ceasing to drift.
        """
        unknown = set(values) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"not deep parameters: {sorted(unknown)}")
        return replace(self, **{k: float(v) for k, v in values.items()})
