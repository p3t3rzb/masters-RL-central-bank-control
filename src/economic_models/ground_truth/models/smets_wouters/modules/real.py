"""The real block: spending, capital, production and factor prices.

Smets-Wouters equations (1)-(9) and (11)-(12). The block is written once and used
*twice*, because the model carries a parallel flexible-price economy whose only
purpose is to define potential output -- and so the output gap the policy rule
responds to. The two differ in exactly three ways, all handled by the
``flexible`` flag:

* the relevant real rate is ``r_t - E_t pi_{t+1}`` under sticky prices and simply
  ``rf_t`` under flexible ones, since flexible-price inflation is not defined;
* the price mark-up is a variable under sticky prices and identically zero under
  flexible ones, which turns equation (9) from a definition into a restriction;
* the wage mark-up is likewise zero, so equation (12) collapses to "the real wage
  equals the marginal rate of substitution".

Writing it once is not only shorter: the two economies must share their
technology and preferences exactly, or the gap between them stops being a gap.

Every equation is stated in the residual convention -- everything moved to the
left, summing to zero.
"""

from __future__ import annotations

from economic_models.ground_truth.models.smets_wouters.parameters import SwParams
from economic_models.ground_truth.models.smets_wouters.system import LinearSystem


def add_real_block(system: LinearSystem, p: SwParams, *, flexible: bool) -> None:
    """Add equations (1)-(9), (11) and (12) for one of the two economies.

    ``flexible`` selects the flexible-price twin, whose variables carry an ``f``
    suffix and whose mark-ups are identically zero.
    """
    v = _Names(flexible)
    habit_growth = p.habit / p.gamma

    # The ex ante real rate, as coefficient contributions. Under sticky prices it
    # is the nominal rate less expected inflation; under flexible prices the real
    # rate is itself the variable.
    rate_cur = {v.r: 1.0}
    rate_lead: dict[str, float] = {} if flexible else {"pi": -1.0}

    # The risk-premium wedge between the rate the bank sets and the return
    # households require, less whatever of it the bank's credit-easing lever
    # offsets. Note the scaling: Smets and Wouters print this disturbance inside
    # the same bracket as the real rate in equation (2), so that it is multiplied
    # by c3 there and by one in equation (4) -- but the process they *estimate*,
    # and whose standard error Table 1B reports, is the term as it appears
    # additively in (2). Carrying the estimated units means entering (2) with a
    # coefficient of one and (4) with 1/c3. Getting this wrong is invisible in the
    # impulse responses' shape and shows up only in the variance decomposition,
    # where the risk premium would explain almost none of consumption instead of
    # a large share of it.
    wedge = {"eps_b": 1.0, "qe": -1.0}

    # 1 : the aggregate resource constraint -- output is absorbed by consumption,
    #     investment, the resource cost of varying capital utilisation, and the
    #     exogenous spending disturbance.
    system.add(
        f"{v.tag}resource",
        cur={v.y: 1.0, v.c: -p.c_y, v.inv: -p.i_y, v.z: -p.z_y, "eps_g": -1.0},
    )

    # 2 : the consumption Euler equation, with external habit (hence the lag) and
    #     non-separable leisure (hence the hours terms).
    system.add(
        f"{v.tag}euler",
        cur={v.c: 1.0, v.l: -p.c2, **_scaled(rate_cur, p.c3), **wedge},
        lag={v.c: -p.c1},
        lead={v.c: -(1.0 - p.c1), v.l: p.c2, **_scaled(rate_lead, p.c3)},
    )

    # 3 : investment, sluggish because the adjustment cost is on the *change* in
    #     investment rather than its level.
    system.add(
        f"{v.tag}investment",
        cur={v.inv: 1.0, v.q: -p.i2, "eps_i": -1.0},
        lag={v.inv: -p.i1},
        lead={v.inv: -(1.0 - p.i1)},
    )

    # 4 : the arbitrage equation for the value of installed capital.
    system.add(
        f"{v.tag}value_of_capital",
        cur={v.q: 1.0, **rate_cur, **_scaled(wedge, 1.0 / p.c3)},
        lead={v.q: -p.q1, v.rk: -(1.0 - p.q1), **rate_lead},
    )

    # 5 : the aggregate production function, with fixed costs (phi_p > 1).
    system.add(
        f"{v.tag}production",
        cur={
            v.y: 1.0,
            v.ks: -p.phi_p * p.alpha,
            v.l: -p.phi_p * (1.0 - p.alpha),
            "eps_a": -p.phi_p,
        },
    )

    # 6 : capital *services* are last period's installed capital, worked harder or
    #     less hard -- newly installed capital only becomes effective with a lag.
    system.add(f"{v.tag}capital_services", cur={v.ks: 1.0, v.z: -1.0}, lag={v.k: -1.0})

    # 7 : utilisation rises with the rental rate, at a speed set by how costly it
    #     is to vary (psi -> 1 makes utilisation effectively fixed).
    system.add(f"{v.tag}utilisation", cur={v.z: 1.0, v.rk: -p.z1})

    # 8 : capital accumulation.
    system.add(
        f"{v.tag}capital",
        cur={v.k: 1.0, v.inv: -(1.0 - p.k1), "eps_i": -p.k2},
        lag={v.k: -p.k1},
    )

    # 9 : the price mark-up is the gap between the marginal product of labour and
    #     the real wage. With flexible prices it is zero, and the equation instead
    #     pins the real wage to the marginal product.
    markup_terms = {v.ks: -p.alpha, v.l: p.alpha, "eps_a": -1.0, v.w: 1.0}
    if flexible:
        system.add(f"{v.tag}price_markup", cur=markup_terms)
    else:
        system.add(f"{v.tag}price_markup", cur={"mup": 1.0, **markup_terms})

    # 11 : cost minimisation ties the rental rate to the capital-labour ratio and
    #      the real wage.
    system.add(f"{v.tag}rental_rate", cur={v.rk: 1.0, v.ks: 1.0, v.l: -1.0, v.w: -1.0})

    # 12 : the wage mark-up is the gap between the real wage and the marginal rate
    #      of substitution. Flexible wages set it to zero.
    mrs_terms = {
        v.w: -1.0,
        v.l: p.sigma_l,
        v.c: 1.0 / (1.0 - habit_growth),
    }
    mrs_lag = {v.c: -habit_growth / (1.0 - habit_growth)}
    if flexible:
        system.add(f"{v.tag}wage_markup", cur=mrs_terms, lag=mrs_lag)
    else:
        system.add(f"{v.tag}wage_markup", cur={"muw": 1.0, **mrs_terms}, lag=mrs_lag)


# -- internals -------------------------------------------------------------


class _Names:
    """The variable names of one of the two economies, sticky or flexible."""

    def __init__(self, flexible: bool) -> None:
        s = "f" if flexible else ""
        self.tag = "flex_" if flexible else ""
        self.y, self.c, self.inv, self.q = f"y{s}", f"c{s}", f"inv{s}", f"q{s}"
        self.ks, self.k, self.z = f"ks{s}", f"k{s}", f"z{s}"
        self.rk, self.w, self.l, self.r = f"rk{s}", f"w{s}", f"l{s}", f"r{s}"


def _scaled(terms: dict[str, float], factor: float) -> dict[str, float]:
    """``terms`` with every coefficient multiplied by ``factor``."""
    return {name: value * factor for name, value in terms.items()}
