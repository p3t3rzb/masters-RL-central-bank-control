"""The excitation process and run generator for the GROWTH model.

:class:`_ExcitationProcess` draws each step's exogenous inputs from an
:class:`~economic_models.ground_truth.growth.excitation.presets.ExcitationConfig`;
:class:`ExcitedRunGenerator` drives a settled
:class:`~economic_models.ground_truth.growth.model.GrowthModel` with those
inputs and records the result as a
:class:`~economic_models.ground_truth.run.Run`.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from pysolve.model import CalculationError, SolutionNotFoundError

from economic_models.variables import Actions, Parameters, State
from economic_models.ground_truth.run import Run
from economic_models.ground_truth.growth.calibration import GrowthCalibration
from economic_models.ground_truth.growth.excitation.presets import ExcitationConfig
from economic_models.ground_truth.growth.model import GrowthModel


class _ExcitationProcess:
    """Stateful per-step draws of the exogenous inputs for one :class:`Run`.

    All inputs follow clipped AR(1) deviations from their baselines; government
    spending growth adds a symmetric countercyclical reaction to the previous
    step's employment gap on top of its own bounded gap to productivity growth.
    """

    def __init__(
        self,
        config: ExcitationConfig,
        baselines: Mapping[str, float],
        rng: np.random.Generator,
        climate: float | None = None,
        dt: float = 1.0,
    ) -> None:
        """Initialise the per-step draw state for one run.

        ``config`` is the excitation spec (AR(1)s, volatility, crises, climate);
        ``baselines`` are the input levels deviations are taken around; ``rng`` is
        the run's random generator; ``climate`` in ``[0, 1]`` biases volatility and
        crisis frequency for the whole run (``None`` for no bias); ``dt`` is the
        period length in years used to rescale annual rates to per-step ones.
        """
        self._config = config
        self._baselines = baselines
        self._rng = rng
        self._dt = dt
        self._ar1 = {**config.visible, **config.hidden}
        self._devs = {name: 0.0 for name in self._ar1}
        self._log_nfe_dev = 0.0
        # The GRg gap to productivity growth is itself an AR(1) around 0.
        self._grg_gap_spec = config.gov_spending.gap_spec()
        self._grg_gap = 0.0
        self._logvol = 0.0
        # Live crisis episodes as ``(deviation_by_input, per_step_decay)`` pairs,
        # summed each step into ``_crisis_total``; several may overlap, each fading
        # at its own drawn rate. ``_crisis_total`` is the level shock applied now.
        self._crisis_episodes: list[tuple[dict[str, float], float]] = []
        self._crisis_total: dict[str, float] = {}
        # ``min_gap`` is in years; convert to steps for the per-step onset guard.
        self._min_gap_steps = round(config.crisis.min_gap / dt) if config.crisis else 0
        self._since_crisis = self._min_gap_steps
        # This run's climate biases volatility and crisis frequency for its whole
        # life; a neutral 0.5-equivalent (no bias) applies when unset.
        if config.climate is not None and climate is not None:
            self._vol_offset = config.climate.vol_offset(climate)
            self._crisis_scale = config.climate.crisis_scale(climate)
        else:
            self._vol_offset = 0.0
            self._crisis_scale = 1.0
        #: diagnostics exposed after each :meth:`step` (not part of the interface)
        self.vol_multiplier = 1.0
        self.crisis_intensity = 0.0

    def _advance_volatility(self) -> float:
        """Advance the stochastic-volatility regime and return this step's multiplier."""
        vol = self._config.volatility
        if vol is None:
            return 1.0
        self._logvol = vol.advance(self._logvol, self._rng, dt=self._dt)
        # Climate shifts where in the band this run sits, but the clip still caps
        # the total so turbulence never leaves the solver-safe corridor.
        total = float(
            np.clip(self._logvol + self._vol_offset, -vol.max_logvol, vol.max_logvol)
        )
        return vol.multiplier(total)

    def _advance_crisis(self) -> None:
        """Fade the live crisis episodes, then maybe erupt a fresh one."""
        crisis = self._config.crisis
        if crisis is None:
            return
        # Fade every live episode by its own decay; drop the spent ones.
        faded = []
        for dev, decay in self._crisis_episodes:
            dev = {name: value * decay for name, value in dev.items()}
            if sum(abs(v) for v in dev.values()) > 1e-6:
                faded.append((dev, decay))
        self._crisis_episodes = faded

        self._since_crisis += 1
        # ``prob`` is the annual onset probability; the per-step hazard that yields
        # the same annual rate is ``1 - (1-prob)**dt``.
        onset_prob = (1.0 - (1.0 - crisis.prob) ** self._dt) * self._crisis_scale
        if self._since_crisis >= self._min_gap_steps and self._rng.random() < onset_prob:
            severity = self._rng.uniform(*crisis.severity_range)
            # ``decay_range`` is the annual decay; the per-step decay is its ``dt`` power.
            decay = self._rng.uniform(*crisis.decay_range) ** self._dt
            dev = {name: mag * severity for name, mag in crisis.impulses.items()}
            # A minority of crises are also financial: an equity-preference collapse
            # on top of the demand/credit bundle, crashing asset prices and wealth.
            if crisis.financial_impulses and self._rng.random() < crisis.financial_prob:
                for name, mag in crisis.financial_impulses.items():
                    dev[name] = dev.get(name, 0.0) + mag * severity
            self._crisis_episodes.append((dev, decay))
            self._since_crisis = 0

        total: dict[str, float] = {}
        for dev, _ in self._crisis_episodes:
            for name, value in dev.items():
                total[name] = total.get(name, 0.0) + value
        self._crisis_total = total
        self.crisis_intensity = sum(abs(v) for v in total.values())

    def step(self, er_prev: float) -> dict[str, float]:
        """Draw this step's exogenous input levels; ``er_prev`` is last step's employment rate.

        Advances the volatility and crisis state, then every AR(1)/random-walk
        input, and sets ``GRg`` from its productivity gap plus the countercyclical
        response to the employment gap ``1 - er_prev``.
        """
        cfg = self._config
        self.vol_multiplier = self._advance_volatility()
        self._advance_crisis()

        values: dict[str, float] = {}
        for name, spec in self._ar1.items():
            base = float(self._baselines[name])
            self._devs[name], values[name] = spec.advance(
                self._devs[name],
                base,
                self._rng,
                sigma_scale=self.vol_multiplier,
                extra=self._crisis_total.get(name, 0.0),
                dt=self._dt,
            )

        gov = cfg.gov_spending
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

        self._log_nfe_dev = cfg.nfe.advance(
            self._log_nfe_dev, self._rng, sigma_scale=self.vol_multiplier, dt=self._dt
        )
        values["Nfe"] = float(self._baselines["Nfe"]) * float(np.exp(self._log_nfe_dev))
        return values


class ExcitedRunGenerator:
    """Configurable, reproducible generator of excited GROWTH-model histories.

    A generator fixes *how* runs are produced -- the excitation
    :class:`ExcitationConfig`, the model calibration, the timestep, the burn-in
    length and the solver dampening budget -- while :meth:`generate` fixes
    *which* run, via its seed and length. The same generator can therefore
    stamp out an ensemble of independent runs (one per seed) that share
    identical excitation dynamics.
    """

    def __init__(
        self,
        config: ExcitationConfig | None = None,
        *,
        calibration: GrowthCalibration | None = None,
        dt: float = 1.0,
        burn_in: int = 15,
        max_dampen_attempts: int = 8,
        solver_iterations: int = 1000,
        solver_threshold: float = 1e-6,
        on_collapse: str = "raise",
    ) -> None:
        """Configure how runs are produced.

        ``config`` is the excitation spec (defaults to ``ExcitationConfig.default()``)
        and ``calibration`` the model's baseline seeding; ``dt`` is the period length
        in years; ``burn_in`` is the number of years run unrecorded to settle onto the
        growth path; ``max_dampen_attempts`` bounds the solver-dampening retries per
        step; ``solver_iterations`` and ``solver_threshold`` are the solver's iteration
        cap and tolerance; ``on_collapse`` is ``'raise'`` or ``'truncate'`` -- whether an
        unrecoverable solver failure raises or ends the history at the collapse.
        """
        if on_collapse not in ("raise", "truncate"):
            raise ValueError("on_collapse must be 'raise' or 'truncate'")
        self.config = config if config is not None else ExcitationConfig.default()
        self.calibration = (
            calibration if calibration is not None else GrowthCalibration.baseline()
        )
        self.dt = dt
        self.burn_in = burn_in
        self.max_dampen_attempts = max_dampen_attempts
        self.solver_iterations = solver_iterations
        self.solver_threshold = solver_threshold
        #: what to do when even fully dampened inputs will not converge -- for a
        #: crisis config ``"truncate"`` ends the history at the collapse (a slump
        #: the economy did not recover from) instead of raising.
        self.on_collapse = on_collapse
        # Baselines the drift deviations are taken around: every structural and
        # exogenous parameter of the calibration.
        self._baselines = self.calibration.baselines()

    def generate(
        self, n_steps: int, *, seed: int | None = None, excite: bool = True
    ) -> Run:
        """Simulate one run of ``n_steps`` recorded steps.

        The model is seeded with the calibration and run ``burn_in`` steps with
        baseline inputs to settle onto its steady growth path before the
        recorded, excited segment starts. With ``excite=False`` the inputs
        are held at baseline throughout, yielding the model's steady-growth
        reference path.
        """
        rng = np.random.default_rng(seed)
        model = self._settled_model()

        state_names = State.names()
        hidden_names = self.config.hidden_names
        exog_names = [*Parameters.names(), *Actions.names(), *hidden_names]
        # Draw this run's climate once, up front, so the whole history shares one
        # character (calm vs crisis-prone) instead of every run looking alike.
        climate = (
            self.config.climate.draw(rng)
            if excite and self.config.climate is not None
            else None
        )
        process = (
            _ExcitationProcess(self.config, self._baselines, rng, climate=climate,
                               dt=self.dt)
            if excite
            else None
        )

        # Exogenous values in effect during burn-in; dampening pulls back toward
        # the previously *applied* values, under which the solver last converged.
        applied = {name: float(self._baselines[name]) for name in exog_names}

        states, params, actions, hidden = [], [], [], []
        volatility, crisis_intensity = [], []
        dampened = 0
        collapsed = False
        for _ in range(n_steps):
            if process is not None:
                target = process.step(er_prev=float(model.solutions[-1]["ER"]))
            else:
                target = dict(applied)
            try:
                target, attempts = self._solve_step(
                    model, target, applied, len(states), seed
                )
            except RuntimeError:
                if self.on_collapse == "raise":
                    raise
                collapsed = True
                break
            dampened += attempts > 0
            applied = target

            solution = model.solutions[-1]
            states.append([solution[name] for name in state_names])
            params.append([target[name] for name in Parameters.names()])
            actions.append([target[name] for name in Actions.names()])
            hidden.append([target[name] for name in hidden_names])
            if process is not None:
                volatility.append(process.vol_multiplier)
                crisis_intensity.append(process.crisis_intensity)

        excited = process is not None

        # Keep the 2-D column shape even for an empty history (a run that
        # collapsed on its first step under ``on_collapse="truncate"``), so
        # ``run.states[:, i]`` and the transform see ``(0, n_col)``, not ``(0,)``.
        def _cols(rows: list, width: int) -> np.ndarray:
            return np.asarray(rows, dtype=float).reshape(len(rows), width)

        return Run(
            states=_cols(states, len(state_names)),
            params=_cols(params, len(Parameters.names())),
            actions=_cols(actions, len(Actions.names())),
            hidden=_cols(hidden, len(hidden_names)),
            dt=self.dt,
            seed=seed,
            dampened_steps=dampened,
            volatility=np.asarray(volatility, dtype=float) if excited else None,
            crisis_intensity=(
                np.asarray(crisis_intensity, dtype=float) if excited else None
            ),
            climate=climate,
            collapsed=collapsed,
        )

    def _settled_model(self) -> GrowthModel:
        """A model seeded with the calibration and settled onto its path.

        ``burn_in`` is a number of *years*; at a sub-annual ``dt`` it is run for the
        matching number of steps so the settling spans the same calendar time.
        """
        model = GrowthModel(
            self.calibration,
            dt=self.dt,
            iterations=self.solver_iterations,
            threshold=self.solver_threshold,
        )
        model.run(round(self.burn_in / self.dt))
        return model

    def _solve_step(
        self,
        model: GrowthModel,
        target: dict[str, float],
        applied: dict[str, float],
        step_index: int,
        seed: int | None,
    ) -> tuple[dict[str, float], int]:
        """Apply ``target`` and step the model, dampening toward ``applied`` on failure.

        Returns the exogenous inputs actually applied and how many dampening
        retries it took; raises if even fully dampened inputs will not converge.
        """
        for attempt in range(self.max_dampen_attempts):
            model.set_values(target)  # visible and hidden exogenous alike
            try:
                model.step()
                return target, attempt
            except (CalculationError, SolutionNotFoundError):
                target = {
                    name: 0.5 * (value + applied[name])
                    for name, value in target.items()
                }
        raise RuntimeError(
            f"solver failed even with dampened exogenous inputs "
            f"(seed={seed}, step={step_index})"
        )
