"""The Smets-Wouters model's visible interface: what the central bank sees and sets.

Concrete specialisations of the abstract value spaces in
:mod:`economic_models.variables`. The split they draw is the point of having this
model at all: Smets and Wouters *estimate* their model on seven macro series
while it carries some forty latent states, so the line between what a central
bank observes and what it does not is drawn by the literature rather than by us.

Several field names are shared with the GROWTH model on purpose. ``ER``, ``PI``
and ``Yk`` here, and ``GRpr`` and ``Nfe`` below, mean exactly what they mean
there, which lets the same mandate
(:class:`~control.rewards.mandate.MandateReward`) and the same plausibility
bounds serve both economies. Where the concepts genuinely differ -- the levers
above all -- the names differ too.

Levels are reconstructed from the model's log-deviations so that the observables
trend the way a real economy's data do, and so the proxy layer's existing
log-difference machinery applies unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from economic_models.variables import Actions, Parameters, State, _observable


@dataclass(frozen=True)
class SwState(State):
    """The Smets-Wouters model's central-bank-visible endogenous state.

    The first seven are Smets and Wouters' own measurement variables -- the
    series they estimate the model on. The rest are the additions this project
    needs: employment for the mandate, the long rate, and the fiscal position.
    """

    Y: float = _observable("Output at current prices (nominal GDP)")
    Yk: float = _observable("Real output")
    P: float = _observable("Price level")
    PI: float = _observable("Price inflation, annualised")
    Ck: float = _observable("Real consumption")
    Ik: float = _observable("Real investment")
    Wk: float = _observable("Real wage")
    L: float = _observable("Hours worked")
    ER: float = _observable("Employment rate")
    R: float = _observable("Nominal policy rate, annualised")
    Rbl: float = _observable("Long-term nominal interest rate, annualised")
    GD: float = _observable("Government debt")
    PSBR: float = _observable("Government deficit")


@dataclass(frozen=True)
class SwParameters(Parameters):
    """The exogenous drivers the bank reads off published data but does not set.

    Fiscal policy, the supply side and the two cost-push channels that are
    actually measurable. Everything a real bank would have to *infer* -- the
    productivity, risk-premium, investment and mark-up disturbances, and the
    output gap they imply -- is hidden instead.
    """

    EXg: float = _observable("Exogenous spending disturbance, percent of output")
    theta: float = _observable("Average tax rate on labour income")
    GRpr: float = _observable("Trend growth rate of productivity, annualised")
    Nfe: float = _observable("Labour force (full employment level)")
    Penergy: float = _observable("Energy and import price index")
    ADDbl: float = _observable("Term premium on long bonds over the policy rate")


@dataclass(frozen=True)
class SwActions(Actions):
    """The three levers the Smets-Wouters central bank controls.

    ``Rdev`` is the discretionary deviation from the estimated policy rule rather
    than the rate itself: the rule has to stay inside the model to pin down the
    price level, so what the bank is handed is the deviation the monetary shock
    literature has always used. Since the rule is inertial, a run of deviations
    moves the whole path of the rate.

    All three are annualised rates, so a value of ``0.01`` is one percentage
    point per year whichever lever it belongs to.
    """

    Rdev: float = _observable("Discretionary deviation from the policy rule, annualised")
    PIstar: float = _observable("Announced inflation objective, annualised")
    QE: float = _observable("Credit-easing wedge offsetting the risk premium, annualised")
