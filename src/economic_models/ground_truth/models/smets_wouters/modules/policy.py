"""The monetary policy rule, and the three levers that move it.

Smets-Wouters equation (14): an inertial Taylor rule reacting to the inflation
gap and to the level and change of the output gap, where potential output is the
flexible-price economy's.

**Why the rule stays in the model.** In a rational-expectations economy the
solution is computed *conditional on* the policy rule, and a pure interest-rate
peg leaves the price level indeterminate (Sargent and Wallace). The bank
therefore cannot simply be handed the rate. What it is handed instead is
``eps_r``, the discretionary deviation from the rule -- the object the monetary
policy shock literature has always used. Because the rule is inertial, a run of
deviations moves the whole path of the rate, so the lever is not as weak as its
name suggests; and because a deviation is naturally a *step*, it lines up with
the delta-action parametrisation the environment already uses.

The other two levers are ``pistar``, the announced inflation objective, and
``qe``, a credit-easing wedge that offsets part of the risk premium households
face. The rule targets the *announced* objective while private agents index to
the *perceived* one, so announcing a change buys the bank only as much as it is
believed -- which is what makes the lever a decision rather than a free lunch.
"""

from __future__ import annotations

from economic_models.ground_truth.models.smets_wouters.parameters import SwParams
from economic_models.ground_truth.models.smets_wouters.system import LinearSystem


def add_policy_block(system: LinearSystem, p: SwParams) -> None:
    """Add the policy rule (14), stated as a residual."""
    smoothing = 1.0 - p.rho_r
    system.add(
        "policy_rule",
        cur={
            "r": 1.0,
            "pi": -smoothing * p.r_pi,
            # The rule reacts to the gap against the announced objective, and
            # carries that objective as its own long-run intercept: with inflation
            # at target the nominal rate settles one-for-one on it (Fisher).
            "pistar": -smoothing * (1.0 - p.r_pi),
            "y": -(smoothing * p.r_y + p.r_dy),
            "yf": smoothing * p.r_y + p.r_dy,
            "eps_r": -1.0,
        },
        lag={"r": -p.rho_r, "y": p.r_dy, "yf": -p.r_dy},
    )
