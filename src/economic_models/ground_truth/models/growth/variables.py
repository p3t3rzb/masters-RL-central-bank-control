"""The GROWTH model's visible interface: its :class:`State`/:class:`Parameters`/:class:`Actions`.

Concrete specializations of the abstract value spaces in
:mod:`economic_models.variables`, carrying the Godley-Lavoie chapter-11 model's
own observable aggregates and exogenous levers. A :class:`GrowthModel` exposes
these as its ``STATE`` / ``PARAMETERS`` / ``ACTIONS``.
"""

from __future__ import annotations

from dataclasses import dataclass

from economic_models.variables import Actions, Parameters, State, _observable


@dataclass(frozen=True)
class GrowthState(State):
    """The GROWTH model's central-bank-visible endogenous state (the observables)."""

    Y: float = _observable("Output at current prices (nominal GDP)")
    Yk: float = _observable("Real output")
    PI: float = _observable("Price inflation")
    ER: float = _observable("Employment rate")
    P: float = _observable("Price level")
    Ck: float = _observable("Real consumption")
    Ik: float = _observable("Gross investment in real terms")
    Rb: float = _observable("Interest rate on government bills")
    Rl: float = _observable("Interest rate on loans")
    Rm: float = _observable("Interest rate on deposits")
    Rbl: float = _observable("Interest rate on bonds")
    GD: float = _observable("Government debt")
    PSBR: float = _observable("Government deficit")
    Vk: float = _observable("Real wealth of households")
    Q: float = _observable("Tobin's Q")
    PE: float = _observable("Price earnings ratio")


@dataclass(frozen=True)
class GrowthParameters(Parameters):
    """The GROWTH model's exogenous levers the bank observes but does not control."""

    GRg: float = _observable("Growth of real government expenditures")
    theta: float = _observable("Income tax rate")
    Nfe: float = _observable("Full employment level")
    GRpr: float = _observable("Growth rate of productivity")
    ADDbl: float = _observable("Spread between long-term interest rate and rate on bills")
    Rln: float = _observable("Normal interest rate on loans")
    NPLk: float = _observable("Proportion of Non-Performing loans")


@dataclass(frozen=True)
class GrowthActions(Actions):
    """The three levers the GROWTH-model central bank directly controls."""

    Rbbar: float = _observable("Interest rate on bills, set exogenously (policy rate)")
    NCAR: float = _observable("Normal capital adequacy ratio of banks (capital requirement)")
    ro: float = _observable("Reserve requirement ratio on deposits")
