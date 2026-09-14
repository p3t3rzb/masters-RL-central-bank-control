"""Driving the Smets-Wouters economy over a run.

Two classes, mirroring GROWTH's. :class:`SwExcitationProcess` draws each period's
exogenous inputs -- the disturbances, the drifting deep parameters and the levers
a historical central bank moved -- and fills in the two that are not plain AR(1)s:
exogenous spending, which leans against the employment shortfall, and the labour
force and energy prices, which are random walks in levels rather than deviations.

:class:`SwRunGenerator` is the model-specific end of the shared machinery in
:mod:`economic_models.ground_truth.excitation.base`: it builds the model, makes
the process, and says what feedback the process reads off the last period.

The one addition GROWTH does not need is the redraw at the top of
:meth:`SwRunGenerator._build_model`. A rational-expectations model has parameter
vectors that do not describe an economy at all -- ones where expectations of the
future have no stable resolution -- and the posterior does not exclude them. So a
run that draws its own economy checks the draw and takes another if it fails,
which is this model's counterpart to GROWTH resampling a run whose solver will
not converge.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from economic_models.ground_truth.excitation.base import ExcitationProcess, ExcitedRunGenerator
from economic_models.ground_truth.models.smets_wouters.calibration import SwCalibration
from economic_models.ground_truth.models.smets_wouters.excitation.presets import SwExcitationConfig
from economic_models.ground_truth.models.smets_wouters.model import SwModel
from economic_models.ground_truth.models.smets_wouters.variables import (
    SwActions,
    SwParameters,
    SwState,
)
from economic_models.ground_truth.solvers.state_space import IndeterminateSystem

#: How many parameter draws to try before giving up on finding a determinate one.
_MAX_DRAWS = 50


class SwExcitationProcess(ExcitationProcess):
    """The exogenous path of one Smets-Wouters run."""

    def _init_model_state(self) -> None:
        """Start the two level walks and the discretionary spending gap at rest."""
        self._spending_gap = 0.0
        self._log_labour = 0.0
        self._log_energy = 0.0
        self._reconfigure_model_state()

    def _reconfigure_model_state(self) -> None:
        """Rebuild the cached specs after a structural break at a branch."""
        self._gap_spec = self._config.gov_spending.gap_spec()

    def _model_inputs(self, values: dict[str, float], feedback: Any) -> None:
        """Fill in the inputs that are not plain AR(1) deviations.

        ``feedback`` is last period's employment rate, which is what the fiscal
        response leans against.
        """
        config: SwExcitationConfig = self._config
        employment = 0.95 if feedback is None else float(feedback)

        self._spending_gap, _ = self._gap_spec.advance(
            self._spending_gap, 0.0, self._rng, sigma_scale=self.vol_multiplier, dt=self._dt
        )
        # Spending leans against the employment shortfall, through the same method
        # the rollout-time stabiliser in :mod:`control.world` calls, so the two can
        # never drift apart.
        support = config.gov_spending.support(employment)
        values["EXg"] = float(np.clip(self._spending_gap + support, *config.gov_spending.bounds))

        self._log_labour = config.labour_force.advance(
            self._log_labour, self._rng, sigma_scale=self.vol_multiplier, dt=self._dt
        )
        values["Nfe"] = float(self._baselines["Nfe"]) * math.exp(self._log_labour)

        self._log_energy = config.energy.advance(
            self._log_energy, self._rng, sigma_scale=self.vol_multiplier, dt=self._dt
        )
        values["Penergy"] = float(self._baselines["Penergy"]) * math.exp(self._log_energy)


class SwRunGenerator(ExcitedRunGenerator):
    """Generates excited runs and scenario continuations of the Smets-Wouters economy."""

    STATE = SwState
    PARAMETERS = SwParameters
    ACTIONS = SwActions

    def __init__(
        self,
        config: SwExcitationConfig | None = None,
        *,
        calibration: SwCalibration | None = None,
        sample_economy: float = 0.0,
        seed: int | None = None,
        dt: float = 0.25,
        burn_in: float = 15.0,
        **kwargs: Any,
    ) -> None:
        """Set up the generator.

        ``sample_economy`` scales how far each run's own deep parameters are drawn
        from the estimated posterior: at zero every run is the estimated economy
        and only the disturbances differ, at one they are as different as the data
        allow. ``calibration`` pins the economy explicitly and overrides it.
        """
        self.sample_economy = sample_economy
        self._seed = seed
        self.calibration = (
            calibration
            if calibration is not None
            else self._draw_calibration(np.random.default_rng(seed))
        )
        super().__init__(config or SwExcitationConfig.default(), dt=dt, burn_in=burn_in, **kwargs)

    # -- the subclass contract ---------------------------------------------

    def _model_baselines(self) -> Mapping[str, float]:
        """Baseline value of everything the excitation drifts around."""
        baselines = dict(self.calibration.baselines())
        baselines.update({name: 0.0 for name in _SHOCKS})
        return baselines

    def _build_model(self) -> SwModel:
        """A fresh model at this generator's economy, dt and settings."""
        return SwModel(self.calibration, dt=self.dt)

    def _make_process(
        self, rng: np.random.Generator, climate: float | None
    ) -> SwExcitationProcess:
        """The exogenous process for one run."""
        return SwExcitationProcess(
            self.config, self._baselines, rng, climate=climate, dt=self.dt
        )

    def _feedback(self, model: SwModel) -> Any:
        """The employment rate the fiscal response leans against."""
        return float(model.solutions[-1]["ER"])

    def _scenario_feedback(self) -> Any:
        """Full employment, so a stateless scenario carries no fiscal response.

        The response is put back at rollout, against the employment rate that
        actually happens -- see the stabiliser in :mod:`control.world`.
        """
        return 0.95

    # -- drawing an economy ------------------------------------------------

    def _draw_calibration(self, rng: np.random.Generator) -> SwCalibration:
        """Draw deep parameters, rejecting any that leave the model indeterminate.

        Most of the posterior describes a perfectly good economy; a small tail of
        it describes one where nothing pins the price level down. Redrawing is the
        Blanchard-Kahn counterpart of GROWTH resampling a run its solver cannot
        advance -- in both cases the draw simply is not an economy.
        """
        if self.sample_economy <= 0.0:
            return SwCalibration.baseline()
        for _ in range(_MAX_DRAWS):
            candidate = SwCalibration.sample(rng, strength=self.sample_economy)
            try:
                SwModel(candidate).solution
            except IndeterminateSystem:
                continue
            return candidate
        raise RuntimeError(
            f"no determinate economy in {_MAX_DRAWS} draws at "
            f"sample_economy={self.sample_economy}"
        )


#: The hidden disturbances, as the excitation names them.
_SHOCKS = ("shock_a", "shock_b", "shock_i", "shock_p", "shock_w")
