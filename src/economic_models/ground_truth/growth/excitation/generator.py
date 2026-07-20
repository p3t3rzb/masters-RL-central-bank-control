"""The GROWTH-specific excitation process and run generator.

:class:`GrowthExcitationProcess` supplies GROWTH's special exogenous inputs on
top of the model-agnostic drift/volatility/crisis machinery: government spending
growth ``GRg`` (a bounded gap to productivity growth plus a countercyclical
response to the previous step's employment gap) and the full-employment level
``Nfe`` (a log random walk). :class:`GrowthRunGenerator` wires the base
:class:`~economic_models.ground_truth.excitation.base.ExcitedRunGenerator` to the
GROWTH model, calibration and interface.

The policy rate carries no feedback rule, only tightly clipped noise: in this
model its channel is so delayed (~15 years to inflation) that a Taylor-type
anchor is destabilizing. Stability comes instead from the one fast lever,
government spending growth, whose countercyclical response to the employment gap
keeps the economy inside its stable corridor around full employment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from economic_models.ground_truth.base import PysolveEconomicModel
from economic_models.ground_truth.excitation.base import (
    ExcitationProcess,
    ExcitedRunGenerator,
)
from economic_models.ground_truth.growth.calibration import GrowthCalibration
from economic_models.ground_truth.growth.excitation.presets import GrowthExcitationConfig
from economic_models.ground_truth.growth.model import GrowthModel
from economic_models.ground_truth.growth.variables import (
    GrowthActions,
    GrowthParameters,
    GrowthState,
)


class GrowthExcitationProcess(ExcitationProcess):
    """The GROWTH model's per-step exogenous-input draws.

    Adds GROWTH's two special inputs to the generic AR(1)/volatility/crisis
    drift: government spending growth ``GRg`` and the full-employment level
    ``Nfe``. The ``feedback`` passed to :meth:`step` is the previous step's
    employment rate ``ER``, which drives the countercyclical fiscal stabilizer.
    """

    def _init_model_state(self) -> None:
        """Seed the GRg-gap and Nfe log-deviation draw state for this run."""
        # The GRg gap to productivity growth is itself an AR(1) around 0.
        self._grg_gap_spec = self._config.gov_spending.gap_spec()
        self._grg_gap = 0.0
        self._log_nfe_dev = 0.0

    def _model_inputs(self, values: dict[str, float], feedback: Any) -> None:
        """Set ``GRg`` (gap + employment stabilizer) and ``Nfe`` (log random walk).

        ``feedback`` is the previous step's employment rate ``er_prev``; the
        countercyclical response moves spending against the employment gap
        ``1 - er_prev``.
        """
        er_prev = feedback
        gov = self._config.gov_spending
        # The gap is an AR(1) around 0 with the spec's own persistence, advanced
        # like every other input (annual persistence/variance invariant to dt).
        self._grg_gap, _ = self._grg_gap_spec.advance(
            self._grg_gap,
            0.0,
            self._rng,
            sigma_scale=self.vol_multiplier,
            dt=self._dt,
        )
        stabilizer = gov.stabilizer * (1.0 - er_prev)
        values["GRg"] = float(
            np.clip(values["GRpr"] + self._grg_gap + stabilizer, *gov.bounds)
        )

        self._log_nfe_dev = self._config.nfe.advance(
            self._log_nfe_dev, self._rng, sigma_scale=self.vol_multiplier, dt=self._dt
        )
        values["Nfe"] = float(self._baselines["Nfe"]) * float(np.exp(self._log_nfe_dev))


class GrowthRunGenerator(ExcitedRunGenerator):
    """Reproducible generator of excited GROWTH-model histories.

    Supplies the base :class:`ExcitedRunGenerator` with GROWTH's interface, its
    settled-model factory, its :class:`GrowthExcitationProcess` and the
    employment-rate feedback that drives the fiscal stabilizer.
    """

    STATE = GrowthState
    PARAMETERS = GrowthParameters
    ACTIONS = GrowthActions

    def __init__(
        self,
        config: GrowthExcitationConfig | None = None,
        *,
        calibration: GrowthCalibration | None = None,
        dt: float = 1.0,
        burn_in: int = 15,
        max_dampen_attempts: int = 8,
        solver_iterations: int = 1000,
        solver_threshold: float = 1e-6,
        on_collapse: str = "raise",
    ) -> None:
        """Configure how GROWTH runs are produced.

        ``config`` is the excitation spec (defaults to
        ``GrowthExcitationConfig.default()``) and ``calibration`` the model's
        baseline seeding (defaults to the book's baseline); the remaining knobs
        are the base generator's (``dt``, ``burn_in``, dampening budget, solver
        settings and ``on_collapse``).
        """
        self.calibration = (
            calibration if calibration is not None else GrowthCalibration.baseline()
        )
        super().__init__(
            config if config is not None else GrowthExcitationConfig.default(),
            dt=dt,
            burn_in=burn_in,
            max_dampen_attempts=max_dampen_attempts,
            solver_iterations=solver_iterations,
            solver_threshold=solver_threshold,
            on_collapse=on_collapse,
        )

    def _model_baselines(self):
        """Every structural and exogenous parameter of the calibration."""
        return self.calibration.baselines()

    def _build_model(self) -> GrowthModel:
        """A fresh GROWTH model seeded from the calibration at this ``dt``/solver."""
        return GrowthModel(
            self.calibration,
            dt=self.dt,
            iterations=self.solver_iterations,
            threshold=self.solver_threshold,
        )

    def _make_process(
        self, rng: np.random.Generator, climate: float | None
    ) -> GrowthExcitationProcess:
        """Build the GROWTH excitation process for one run."""
        return GrowthExcitationProcess(
            self.config, self._baselines, rng, climate=climate, dt=self.dt
        )

    def _feedback(self, model: PysolveEconomicModel) -> float:
        """The previous step's employment rate ``ER`` (drives the fiscal stabilizer)."""
        return float(model.solutions[-1]["ER"])

    def _scenario_feedback(self) -> float:
        """Full employment (``ER = 1``): the stabilizer contributes nothing."""
        return 1.0
