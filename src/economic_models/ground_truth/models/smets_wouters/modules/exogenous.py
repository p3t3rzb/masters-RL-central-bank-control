"""The disturbances, and the beliefs private agents hold about them.

Smets and Wouters drive their model with seven disturbances: five structural ones
the bank never sees (productivity, the risk premium, investment-specific
technology and the two mark-ups), exogenous spending, and the monetary policy
deviation. Here the last is the bank's own lever, and two more levers -- the
announced inflation objective and the credit-easing wedge -- join the block.

**These processes are beliefs, not the data generating process.** The realised
path of every disturbance is supplied from outside, by the excitation: clipped,
scaled by a stochastic-volatility regime, and occasionally hit by a crisis
impulse, none of which an AR(1) describes. What the equations below fix is the
law of motion private agents *forecast* with, which is the only thing the
solution depends on. The model then simply asks, each period, which innovation
would have produced the level the excitation actually chose -- so the realised
path is exactly the excitation's while expectations stay internally consistent
with the process agents believe in. That the two differ is the point: the
economy is genuinely harder than anyone's model of it, the central bank's
included.

The announced objective is a random walk, which is what makes a fully believed
target change neutral in the long run; the perceived objective chases it at the
speed ``credibility``, which is what makes an announcement cost something in the
short run.
"""

from __future__ import annotations

from economic_models.ground_truth.models.smets_wouters.parameters import SwParams
from economic_models.ground_truth.models.smets_wouters.system import LinearSystem

#: The disturbances whose *level* is supplied from outside each period, in the
#: order their innovations are backed out. One innovation each, so the mapping
#: between the two is square and invertible.
DRIVEN: tuple[str, ...] = (
    "eps_a",
    "eps_b",
    "eps_g",
    "eps_i",
    "eps_p",
    "eps_w",
    "eps_r",
    "pistar",
    "qe",
    "tau",
    "pe",
)

#: The equation whose replacement by a fixed rate implements the lower bound, and
#: the variable that bound applies to.
POLICY_EQUATION = "policy_rule"
POLICY_VARIABLE = "r"

#: The innovations, aligned with :data:`DRIVEN`.
INNOVATIONS: tuple[str, ...] = (
    "z_a",
    "z_b",
    "z_g",
    "z_i",
    "z_p",
    "z_w",
    "z_r",
    "z_pistar",
    "z_qe",
    "z_tau",
    "z_pe",
)


def add_exogenous_block(system: LinearSystem, p: SwParams) -> None:
    """Add the law of motion each disturbance is forecast with."""
    # Plain AR(1)s.
    system.exogenous("eps_a", rho=p.rho_a, shock="z_a")
    system.exogenous("eps_b", rho=p.rho_b, shock="z_b")
    system.exogenous("eps_i", rho=p.rho_i, shock="z_i")
    system.exogenous("eps_r", rho=p.rho_mp, shock="z_r")
    system.exogenous("qe", rho=p.rho_qe, shock="z_qe")

    # The two cost-push drivers the bank can actually measure: the labour tax
    # wedge, which shifts the wage households need to be paid, and energy and
    # import prices, which shift firms' marginal cost. Both are observed, which is
    # what makes them different in kind from the hidden mark-up disturbances that
    # push on the same two equations.
    system.exogenous("tau", rho=p.rho_tau, shock="z_tau")
    system.exogenous("pe", rho=p.rho_pe, shock="z_pe")

    # Exogenous spending is empirically driven partly by productivity: net exports
    # move with it. Agents know that, so the productivity innovation loads here.
    system.add(
        "eps_g",
        cur={"eps_g": 1.0},
        lag={"eps_g": -p.rho_g},
        shock={"z_g": -1.0, "z_a": -p.rho_ga},
    )

    # The two mark-ups follow ARMA(1,1): the moving-average term captures the
    # high-frequency fluctuations a pure AR(1) leaves in the residuals. Carrying
    # last period's innovation as a state is what makes the MA term expressible.
    system.exogenous("eps_p", rho=p.rho_p, shock="z_p", ma="eta_p_lag", ma_weight=p.mu_p)
    system.add("eta_p_lag", cur={"eta_p_lag": 1.0}, shock={"z_p": -1.0})
    system.exogenous("eps_w", rho=p.rho_w, shock="z_w", ma="eta_w_lag", ma_weight=p.mu_w)
    system.add("eta_w_lag", cur={"eta_w_lag": 1.0}, shock={"z_w": -1.0})

    # The announced objective is permanent -- anything less would make a target
    # change partly self-reversing, and the long-run neutrality the price and wage
    # equations are written for would not hold.
    system.exogenous("pistar", rho=1.0, shock="z_pistar")

    # The perceived objective chases the announced one. At credibility 1 the bank
    # is believed immediately and a target change is neutral on impact; at 0 it is
    # never believed and the lever does nothing but move the rule's intercept.
    system.add(
        "pitil",
        cur={"pitil": 1.0, "pistar": -p.credibility},
        lag={"pitil": -(1.0 - p.credibility)},
    )
