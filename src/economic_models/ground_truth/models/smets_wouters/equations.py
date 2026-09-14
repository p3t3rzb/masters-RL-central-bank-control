"""Assembling the whole Smets-Wouters model from its blocks.

Fixes the variable and shock vectors -- their order is the permanent internal
state order, so it is declared once, here -- and calls each equation module in
turn. The real block is called twice: once for the sticky-price economy the data
come from, and once for the flexible-price twin whose output is potential output.

The count is the first check that anything is right: 38 variables and 38
equations, and :class:`~...system.LinearSystem` refuses to assemble anything
else.
"""

from __future__ import annotations

from economic_models.ground_truth.models.smets_wouters.modules.exogenous import (
    INNOVATIONS,
    add_exogenous_block,
)
from economic_models.ground_truth.models.smets_wouters.modules.nominal import add_nominal_block
from economic_models.ground_truth.models.smets_wouters.modules.policy import add_policy_block
from economic_models.ground_truth.models.smets_wouters.modules.real import add_real_block
from economic_models.ground_truth.models.smets_wouters.parameters import SwParams
from economic_models.ground_truth.models.smets_wouters.system import LinearSystem

#: The sticky-price economy: what actually happens, and what is measured.
STICKY: tuple[str, ...] = (
    "y",  # output
    "c",  # consumption
    "inv",  # investment
    "q",  # value of installed capital (Tobin's Q)
    "ks",  # capital services
    "k",  # installed capital
    "z",  # capital utilisation
    "rk",  # rental rate of capital
    "mup",  # price mark-up
    "pi",  # inflation
    "muw",  # wage mark-up
    "w",  # real wage
    "l",  # hours worked
    "r",  # nominal policy rate
    "e",  # employment
)

#: The flexible-price twin. Never observed; its only job is to define the output
#: gap, which is why that gap is the model's most consequential hidden variable.
FLEXIBLE: tuple[str, ...] = (
    "yf",
    "cf",
    "invf",
    "qf",
    "ksf",
    "kf",
    "zf",
    "rkf",
    "wf",
    "lf",
    "rf",
)

#: The disturbances and levers, including the two mark-up moving-average states
#: and the perceived inflation objective.
EXOGENOUS: tuple[str, ...] = (
    "eps_a",
    "eps_b",
    "eps_g",
    "eps_i",
    "eps_r",
    "eps_p",
    "eps_w",
    "eta_p_lag",
    "eta_w_lag",
    "pistar",
    "qe",
    "tau",
    "pe",
    "pitil",
)

#: The full state vector, in its permanent order.
VARIABLES: tuple[str, ...] = (*STICKY, *FLEXIBLE, *EXOGENOUS)


def build_system(p: SwParams) -> LinearSystem:
    """Assemble the whole model at parameter vector ``p``."""
    system = LinearSystem(VARIABLES, INNOVATIONS)
    add_real_block(system, p, flexible=False)
    add_nominal_block(system, p)
    add_policy_block(system, p)
    add_real_block(system, p, flexible=True)
    add_exogenous_block(system, p)
    return system
