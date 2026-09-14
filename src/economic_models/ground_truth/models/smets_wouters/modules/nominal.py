"""Price setting, wage setting and employment: the nominal side.

Smets-Wouters equations (10) and (13), plus the employment equation of Smets and
Wouters (2003). These exist only in the sticky economy -- the flexible-price twin
has no inflation and no mark-ups to speak of.

**Where this departs from the paper.** Smets and Wouters hold the inflation
objective fixed, so their indexation terms anchor inflation at a constant. Here
the objective is one of the central bank's levers, so both equations index to the
*perceived* target ``pitil`` instead. The coefficient on it is not free: it is
whatever makes the equation hold when inflation, its lag, its expectation and the
target are all equal, which is the statement that a fully believed change in the
target is neutral. Deriving it rather than calibrating it means the neutrality
survives any parameter draw.
"""

from __future__ import annotations

from economic_models.ground_truth.models.smets_wouters.parameters import SwParams
from economic_models.ground_truth.models.smets_wouters.system import LinearSystem


def add_nominal_block(system: LinearSystem, p: SwParams) -> None:
    """Add the Phillips curve, the wage equation and the employment equation."""
    # 10 : the New Keynesian Phillips curve. Inflation is dragged by its own lag
    #      (partial indexation), pulled by its expectation, and pushed down by a
    #      positive price mark-up -- firms whose price is above marginal cost cut.
    system.add(
        "phillips",
        cur={
            "pi": 1.0,
            "mup": p.pi3,
            "eps_p": -1.0,
            "pe": -p.energy_share,  # observed marginal-cost push
            "pitil": -(1.0 - p.pi1 - p.pi2),
        },
        lag={"pi": -p.pi1},
        lead={"pi": -p.pi2},
    )

    # 13 : the real wage adjusts slowly towards the wage households would demand.
    #      Its inflation terms must likewise sum to zero once the target is
    #      included, so a believed target change leaves real wages alone.
    inflation_weight = -(1.0 - p.w1) + p.w2 - p.w3
    system.add(
        "wage",
        cur={
            "w": 1.0,
            "pi": p.w2,
            "muw": p.w4,
            "eps_w": -1.0,
            "tau": -p.w4,  # a labour tax acts on the wage exactly as a mark-up does
            "pitil": -inflation_weight,
        },
        lag={"w": -p.w1, "pi": -p.w3},
        lead={"w": -(1.0 - p.w1), "pi": -(1.0 - p.w1)},
    )

    # Employment adjusts to hours worked only gradually, because hiring is itself
    # subject to a Calvo friction (Smets and Wouters 2003). Hours are what enters
    # production; employment is what the unemployment rate is computed from, and
    # so what the bank's mandate is written against.
    system.add(
        "employment",
        cur={"e": 1.0 + p.e2, "l": -p.e2},
        lag={"e": -p.e1},
        lead={"e": -(1.0 - p.e1)},
    )
