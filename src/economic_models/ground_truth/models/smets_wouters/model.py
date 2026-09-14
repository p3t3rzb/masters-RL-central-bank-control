"""The Smets-Wouters (2007) model as a ground-truth economy.

Wires the equation blocks to the linear rational-expectations machinery and, on
top of the deviations the model actually speaks in, reconstructs the *levels* a
central bank observes: a trending path for output, prices, consumption,
investment and wages, an employment rate, a policy rate and a long rate, and a
fiscal position. That reconstruction is what lets the existing proxy layer -- the
log-difference and ratio transforms, the encoders, the whole control stack -- work
on this model without knowing it is different in kind from GROWTH.

Three of the model's quantities are not in the paper and are worth naming:

* **employment.** Smets and Wouters (2007) measure hours; the mandate is written
  against an employment rate, so the employment equation of Smets and Wouters
  (2003) is carried alongside, with hiring subject to its own Calvo friction.
* **the long rate.** Priced off the solution by the expectations hypothesis: the
  average policy rate expected over the next ten years, plus the observed term
  premium. It costs nothing, because expected future rates are what the solution
  already is.
* **the fiscal position.** Debt accumulates at the policy rate against a primary
  balance, with a mild tax response that keeps it from wandering off. The model
  is Ricardian, so none of this feeds back into the real allocation -- it is an
  observable, not a mechanism, and is kept out of the rational-expectations
  system so it cannot add a spurious root to the determinacy count.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from economic_models.ground_truth.linear_re import LinearREEconomicModel
from economic_models.ground_truth.models.smets_wouters.calibration import SwCalibration
from economic_models.ground_truth.models.smets_wouters.equations import VARIABLES, build_system
from economic_models.ground_truth.models.smets_wouters.modules.exogenous import (
    DRIVEN,
    POLICY_EQUATION,
    POLICY_VARIABLE,
)
from economic_models.ground_truth.models.smets_wouters.variables import (
    SwActions,
    SwParameters,
    SwState,
)

#: Quarters of expected policy rate the long rate averages over: ten years.
_LONG_HORIZON = 40

#: The effective lower bound on the annualised nominal policy rate.
_ELB = 0.0

#: How strongly the tax share leans against the debt ratio each quarter. Small
#: enough to leave the observed debt path plenty of life, large enough that it
#: does not wander away over a long run. The model is Ricardian, so this is a rule
#: for the *observable*, not a channel: it never touches the real allocation.
_DEBT_FEEDBACK = 0.02

#: Share of output accruing to labour, which is what the observed tax rate is
#: levied on. Only its deviations matter, so an approximation is enough.
_LABOUR_SHARE = 0.65

#: The debt ratio the fiscal rule leans back towards.
_DEBT_TARGET = 0.60

#: The hidden disturbances, as the excitation names them. They are deliberately
#: *not* the model's own variable names: setting ``shock_a`` states the level the
#: disturbance is to take this period, while setting ``eps_a`` restores the state
#: vector, and branching needs both to be possible without ambiguity.
HIDDEN_SHOCKS: Mapping[str, str] = {
    "shock_a": "eps_a",
    "shock_b": "eps_b",
    "shock_i": "eps_i",
    "shock_p": "eps_p",
    "shock_w": "eps_w",
}

#: Levels carried between periods, in the order they are recorded.
#:
#: Note that last period's labour force is recorded as ``labour_force`` rather
#: than as ``Nfe``. The two are the same quantity, but ``Nfe`` is also the name of
#: a *visible parameter*, and a flat state mapping has one namespace: restoring a
#: branch would set the incoming parameter and silently leave the carried level at
#: whatever the fresh model started with, so every subsequent step would measure
#: labour-force growth from the wrong base. The symptom is a slow, plausible drift
#: in the level of output and nothing else -- which is exactly the kind of thing
#: bit-for-bit branching tests exist to catch.
_LEVELS: tuple[str, ...] = ("Yk", "P", "Ck", "Ik", "Wk", "labour_force", "GD")


class SwModel(LinearREEconomicModel):
    """The Smets-Wouters medium-scale New Keynesian model.

    The visible interface is :class:`~...variables.SwState` /
    ``SwParameters`` / ``SwActions``; everything else -- capital, utilisation,
    Tobin's Q, the two mark-ups and the entire flexible-price economy that
    defines potential output -- is hidden, exactly as it is to a real central
    bank.
    """

    STATE = SwState
    PARAMETERS = SwParameters
    ACTIONS = SwActions

    def __init__(
        self,
        calibration: SwCalibration | None = None,
        *,
        dt: float = 0.25,
        zero_lower_bound: bool = True,
        **param_overrides: float,
    ) -> None:
        """Build and seed the model at ``calibration`` (the estimated economy by default).

        ``dt`` must be a quarter: the Calvo probabilities, adjustment costs and
        shock persistences are all estimated at that frequency, and rescaling them
        is not a matter of arithmetic. ``param_overrides`` set visible exogenous
        inputs on top of the calibration.
        """
        if not math.isclose(dt, 0.25):
            raise ValueError(
                f"the Smets-Wouters model is estimated quarterly; dt={dt} would need its "
                "nominal rigidities and shock processes re-estimated, not rescaled"
            )
        self.calibration = calibration if calibration is not None else SwCalibration.baseline()
        self.params = self.calibration.params
        self.zero_lower_bound = zero_lower_bound

        super().__init__(dt=dt)
        self._policy_equation_row = build_system(self.params).row(POLICY_EQUATION)

        self._exogenous = dict(self.calibration.exogenous_baselines)
        self._shocks = {name: 0.0 for name in HIDDEN_SHOCKS}
        self._seed_levels()
        self._previous = dict(self._levels)
        self._y_previous = np.zeros(len(VARIABLES))
        self._state = self._observe()
        self._solutions.append(self._record())

        if param_overrides:
            unknown = set(param_overrides) - self._exogenous_names()
            if unknown:
                raise ValueError(f"not visible parameters: {sorted(unknown)}")
            self._exogenous.update({k: float(v) for k, v in param_overrides.items()})

    # -- the visible interface ---------------------------------------------

    @property
    def state(self) -> SwState:
        """Latest values of the visible endogenous state."""
        return self._state

    @property
    def parameters(self) -> SwParameters:
        """Current values of the visible exogenous parameters."""
        return SwParameters.from_dict(self._exogenous)

    @property
    def actions(self) -> SwActions:
        """Current values of the central bank's levers."""
        return SwActions.from_dict(self._exogenous)

    def advance(self, parameters: SwParameters, actions: SwActions) -> SwState:
        """Apply the exogenous inputs and advance one period."""
        self.set_parameters(parameters)
        self.set_actions(actions)
        self.step()
        return self.state

    def set_parameters(self, parameters: SwParameters) -> None:
        """Set the exogenous parameters the bank observes but does not control."""
        self._exogenous.update(parameters.to_dict())

    def set_actions(self, actions: SwActions) -> None:
        """Set the central bank's policy levers."""
        self._exogenous.update(actions.to_dict())

    def set_state(self, state: SwState) -> None:
        """Seed the visible levels, to pin the common initial condition of a scenario."""
        values = state.to_dict()
        for name in _LEVELS:
            if name in values:
                self._levels[name] = float(values[name])

    # -- the subclass contract ---------------------------------------------

    def _variable_names(self) -> Sequence[str]:
        """The model's variables, in their permanent order."""
        return VARIABLES

    def _driven_names(self) -> Sequence[str]:
        """The disturbances and levers whose level is supplied from outside."""
        return DRIVEN

    def _parameter_signature(self) -> tuple[float, ...]:
        """The deep parameters, so the solution is reused while they stand still."""
        return tuple(
            float(getattr(self.params, name)) for name in self.params.__dataclass_fields__
        )

    def _matrices(self) -> tuple[np.ndarray, ...]:
        """Assemble both forms of the system at the current parameters."""
        system = build_system(self.params)
        lead, canonical = system.lead_form(), system.gensys_form()
        return (
            lead.g0,
            lead.g1,
            lead.gf,
            lead.psi,
            lead.c,
            canonical.g0,
            canonical.g1,
            canonical.psi,
            canonical.pi,
        )

    def _targets(self) -> np.ndarray:
        """This period's level for each driven variable, in the model's own units.

        Rates arrive as annualised fractions, the way every other rate in this
        project is expressed, and are converted to the percent-per-quarter the
        log-linearised model works in. Two conversions are less mechanical than
        they look. The announced objective is an annualised *level*, while the
        model is written in deviations from its own steady-state inflation, so the
        steady state is subtracted -- announcing the rate the economy already has
        must be no announcement at all. And the credit-easing lever is scaled by
        ``c3``, because the risk-premium disturbance it offsets is measured as it
        enters the consumption equation rather than as a rate.
        """
        p, x = self.params, self._exogenous
        to_quarterly = 100.0 / 4.0
        base = self.calibration.exogenous_baselines
        return np.array(
            [
                self._shocks["shock_a"],
                self._shocks["shock_b"],
                float(x["EXg"]),
                self._shocks["shock_i"],
                self._shocks["shock_p"],
                self._shocks["shock_w"],
                float(x["Rdev"]) * to_quarterly,
                float(x["PIstar"]) * to_quarterly - p.pi_bar,
                float(x["QE"]) * to_quarterly * p.c3,
                (float(x["theta"]) - float(base["theta"])) * 100.0,
                100.0 * math.log(max(float(x["Penergy"]), 1e-9) / float(base["Penergy"])),
            ]
        )

    def _bound(self) -> tuple[int, float, np.ndarray] | None:
        """The effective lower bound, as the policy equation's replacement."""
        if not self.zero_lower_bound:
            return None
        replacement = np.zeros(len(VARIABLES))
        replacement[self._bound_variable] = 1.0
        level = _ELB * 100.0 / 4.0 - self.params.r_ss
        return self._policy_row, level, replacement

    @property
    def _bound_variable(self) -> int:
        """Index of the policy rate in the state vector."""
        return self._index[POLICY_VARIABLE]

    @property
    def _policy_row(self) -> int:
        """Row the policy rule occupies -- the equation the bound replaces."""
        return self._policy_equation_row

    def _prepare(self) -> None:
        """Fold the observed trend growth rate into the deep parameters.

        Trend productivity growth is something a central bank reads off the data,
        and also something the model's whole steady state is built on -- the great
        ratios, the rental rate, every discount factor in the coefficients. So a
        drift in it is not just forcing: it moves the economy the bank is trying
        to learn, and the solution has to be recomputed to match.
        """
        target = float(self._exogenous["GRpr"]) * 100.0 / 4.0
        if not math.isclose(target, self.params.gamma_bar, rel_tol=0.0, abs_tol=1e-12):
            self.params = self.params.with_values({"gamma_bar": target})

    # -- levels ------------------------------------------------------------

    def _seed_levels(self) -> None:
        """Start the reconstructed levels at their calibrated values."""
        initial = self.calibration.initial_levels
        self._levels = {
            "Yk": float(initial["Yk"]),
            "P": float(initial["P"]),
            "Ck": float(initial["Yk"]) * self.params.c_y,
            "Ik": float(initial["Yk"]) * self.params.i_y,
            "Wk": float(initial["Yk"]) * self.params.w_ss / 100.0,
            "labour_force": float(initial["Nfe"]),
            "GD": float(initial["Yk"]) * float(initial["debt_ratio"]),
        }
        self._employment_ss = float(initial["employment_rate"])
        self._psbr = 0.0

    def _update_levels(self) -> None:
        """Integrate the new deviations into the observable levels.

        Smets and Wouters' measurement equation (15) in reverse: each quantity
        grows at the balanced-path rate plus the change in its deviation. Labour
        force growth is added on top, since the model is per capita and the bank
        observes aggregates.
        """
        self._previous = dict(self._levels)
        y, y_prev = self._y, self._y_previous
        trend = self.params.gamma_bar / 100.0
        nfe = float(self._exogenous["Nfe"])
        head = math.log(max(nfe, 1e-9) / max(self._levels["labour_force"], 1e-9))

        for level, variable in (("Yk", "y"), ("Ck", "c"), ("Ik", "inv")):
            change = (y[self._index[variable]] - y_prev[self._index[variable]]) / 100.0
            self._levels[level] *= math.exp(trend + change + head)
        wage_change = (y[self._index["w"]] - y_prev[self._index["w"]]) / 100.0
        self._levels["Wk"] *= math.exp(trend + wage_change)
        self._levels["P"] *= math.exp(
            (self.params.pi_bar + y[self._index["pi"]]) / 100.0
        )
        self._levels["labour_force"] = nfe
        self._update_fiscal()
        self._y_previous = y.copy()
        self._state = self._observe()

    def _update_fiscal(self) -> None:
        """Accumulate debt at the policy rate against the primary balance.

        The baseline tax share is *derived*, not chosen: it is whatever holds the
        debt ratio still given the steady-state gap between the nominal interest
        rate and nominal growth. Choosing a plausible-looking tax rate instead
        would leave the debt path drifting to zero or to infinity over a long run,
        for reasons having nothing to do with the economy.
        """
        nominal = self._levels["Yk"] * self._levels["P"] / 100.0
        debt_ratio = self._levels["GD"] / max(nominal, 1e-9)
        base = self.calibration.exogenous_baselines
        tax = (
            self.params.g_y
            + self._debt_stabilising_balance()
            + (float(self._exogenous["theta"]) - float(base["theta"])) * _LABOUR_SHARE
            + _DEBT_FEEDBACK * (debt_ratio - _DEBT_TARGET)
        )
        spending = self.params.g_y * (1.0 + float(self._exogenous["EXg"]) / 100.0)
        interest = self._policy_rate() * self.dt * self._levels["GD"]
        self._psbr = (spending - tax) * nominal + interest
        self._levels["GD"] = max(self._levels["GD"] + self._psbr, 0.0)

    def _debt_stabilising_balance(self) -> float:
        """The primary surplus that holds the debt ratio at its target.

        The textbook condition, at this economy's own steady state: the surplus
        has to cover the excess of the nominal interest rate over nominal growth
        on the outstanding stock.
        """
        p = self.params
        growth = (1.0 + p.gamma_bar / 100.0) * (1.0 + p.pi_bar / 100.0) - 1.0
        return (p.r_ss / 100.0 - growth) / (1.0 + growth) * _DEBT_TARGET

    def _policy_rate(self) -> float:
        """The annualised nominal policy rate implied by the current deviation."""
        return 4.0 * (self.params.r_ss + self._y[self._index["r"]]) / 100.0

    def _long_rate(self) -> float:
        """The ten-year rate, as the average expected policy rate plus the term premium.

        Free from the solution: expected future variables are exactly what a
        state-space law of motion is.
        """
        expected = self.solution.forward(self._y, _LONG_HORIZON)[:, self._index["r"]]
        average = (self._y[self._index["r"]] + float(np.sum(expected[:-1]))) / _LONG_HORIZON
        return 4.0 * (self.params.r_ss + average) / 100.0 + float(self._exogenous["ADDbl"])

    def _observe(self) -> SwState:
        """The visible state implied by the current deviations and levels."""
        y, levels = self._y, self._levels
        real, price = levels["Yk"], levels["P"]
        inflation = (price / self._previous["P"]) ** (1.0 / self.dt) - 1.0
        return SwState(
            Y=real * price / 100.0,
            Yk=real,
            P=price,
            PI=inflation,
            Ck=levels["Ck"],
            Ik=levels["Ik"],
            Wk=levels["Wk"],
            L=levels["labour_force"]
            * math.exp((self.params.l_bar + y[self._index["l"]]) / 100.0),
            ER=self._employment_ss * math.exp(y[self._index["e"]] / 100.0),
            R=self._policy_rate(),
            Rbl=self._long_rate(),
            GD=levels["GD"],
            PSBR=self._psbr,
        )

    # -- saving and restoring ----------------------------------------------

    def _record(self) -> dict[str, float]:
        """The whole internal state, flat, as branching needs it."""
        record = {name: float(self._y[i]) for i, name in enumerate(VARIABLES)}
        record.update({f"prev_{name}": float(self._y_previous[i]) for i, name in enumerate(VARIABLES)})
        record.update({name: float(value) for name, value in self._levels.items()})
        record.update({f"prev_{name}": float(v) for name, v in self._previous.items()})
        record.update({name: float(v) for name, v in self._exogenous.items()})
        record.update({name: float(v) for name, v in self._shocks.items()})
        record.update(
            {name: float(getattr(self.params, name)) for name in self.params.__dataclass_fields__}
        )
        record.update(self._state.to_dict())  # the visible row the generator reads
        record["PSBR"] = self._psbr
        return record

    def _restore(self, values: Mapping[str, float]) -> None:
        """Apply ``values`` to whichever part of the internal state each name belongs to."""
        deep: dict[str, float] = {}
        for name, value in values.items():
            value = float(value)
            if name in self._shocks:
                self._shocks[name] = value
            elif name in self.params.__dataclass_fields__:
                deep[name] = value
            elif name in self._exogenous:
                self._exogenous[name] = value
            elif name in self._index:
                self._y[self._index[name]] = value
            elif name.startswith("prev_") and name[5:] in self._index:
                self._y_previous[self._index[name[5:]]] = value
            elif name in self._levels:
                self._levels[name] = value
            elif name.startswith("prev_") and name[5:] in self._levels:
                self._previous[name[5:]] = value
            elif name == "PSBR":
                self._psbr = value
        if deep:
            self.params = self.params.with_values(deep)
