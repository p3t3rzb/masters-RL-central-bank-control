"""Model-agnostic excitation: the base config, per-step process and run generator.

A proxy can only learn the causal effect of the policy levers if the historic
run actually moved them. Every bank-visible exogenous input therefore drifts as a
clipped AR(1) around its baseline; the hidden structural parameters drift too,
reaching the dataset only as unexplained variance in the visible state. On top of
that drift sit optional stochastic-volatility, crisis and per-run-climate layers.

This module owns everything about excitation that is the same for every
ground-truth model:

* :class:`ExcitationConfig` -- the generic drift/volatility/crisis/climate bundle;
* :class:`ExcitationProcess` -- the per-step draw of exogenous inputs, with a
  model hook (:meth:`~ExcitationProcess._model_inputs`) for the model's own
  special inputs (e.g. a stabilizer or a level random walk);
* :class:`ExcitedRunGenerator` -- the reproducible driver that settles a model,
  steps it under the drawn inputs, dampens the solver on failure and records the
  result as a :class:`~economic_models.ground_truth.run.Run` (plus branching
  :class:`~economic_models.ground_truth.run.Scenario`\\ s).

A concrete model subclasses :class:`ExcitationProcess` and
:class:`ExcitedRunGenerator` (and usually :class:`ExcitationConfig`) to supply
its value spaces, its model factory, its per-step feedback signal and its
special inputs -- see :mod:`economic_models.ground_truth.growth.excitation`.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

import numpy as np
from pysolve.model import CalculationError, SolutionNotFoundError

from economic_models.ground_truth.base import PysolveEconomicModel
from economic_models.ground_truth.excitation.specs import (
    AR1Spec,
    ClimateSpec,
    CrisisSpec,
    StochasticVolatilitySpec,
)
from economic_models.ground_truth.run import Run, Scenario
from economic_models.variables import Actions, Parameters, State


@dataclass(frozen=True)
class ExcitationConfig:
    """The model-agnostic tunables of how a run's exogenous inputs are excited.

    ``visible`` and ``hidden`` map an input name to its :class:`AR1Spec`; the
    baselines the deviations are taken around come from the model calibration,
    not from here. A model's *special* inputs (a stabilizer, a level random walk)
    are handled by its :class:`ExcitationProcess` subclass, configured by fields
    a subclass of this config adds.
    """

    visible: Mapping[str, AR1Spec]  #: bank-visible Parameters/Actions drift
    hidden: Mapping[str, AR1Spec]  #: hidden structural parameter drift
    #: optional stochastic-volatility regime scaling every innovation; ``None``
    #: leaves innovation sizes constant (the calm default).
    volatility: StochasticVolatilitySpec | None = field(default=None, kw_only=True)
    #: optional rare recoverable crisis shocks; ``None`` disables them.
    crisis: CrisisSpec | None = field(default=None, kw_only=True)
    #: optional per-run turbulence draw mixing calm and crisis-prone runs;
    #: ``None`` gives every run the same character.
    climate: ClimateSpec | None = field(default=None, kw_only=True)

    @property
    def hidden_names(self) -> tuple[str, ...]:
        """Hidden parameter names, in the column order recorded on a :class:`Run`."""
        return tuple(self.hidden)


class ExcitationProcess(ABC):
    """Stateful per-step draws of the exogenous inputs for one :class:`Run`.

    Owns the model-agnostic machinery -- the stochastic-volatility regime, the
    crisis episodes and the clipped-AR(1) drift of every ``visible``/``hidden``
    input. Each :meth:`step` advances all of that, then calls the abstract
    :meth:`_model_inputs` hook so the concrete model can add or override its own
    special inputs (a stabilizer, a level random walk, ...) using the ``feedback``
    signal the generator passes in.
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

        ``config`` is the excitation spec; ``baselines`` are the input levels
        deviations are taken around; ``rng`` is the run's random generator;
        ``climate`` in ``[0, 1]`` biases volatility and crisis frequency for the
        whole run (``None`` for no bias); ``dt`` is the period length in years
        used to rescale annual rates to per-step ones.
        """
        self._config = config
        self._baselines = baselines
        self._rng = rng
        self._dt = dt
        self._ar1 = {**config.visible, **config.hidden}
        self._devs = {name: 0.0 for name in self._ar1}
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
        self._init_model_state()

    # -- subclass contract -------------------------------------------------

    def _init_model_state(self) -> None:
        """Initialise any model-specific per-run state (default: none).

        Called at the end of ``__init__``; a subclass that carries extra draw
        state for its special inputs (an AR(1) gap, a log-level deviation, ...)
        seeds it here.
        """

    @abstractmethod
    def _model_inputs(self, values: dict[str, float], feedback: Any) -> None:
        """Fill in the model's special exogenous inputs, mutating ``values``.

        Called by :meth:`step` after the generic AR(1)/volatility/crisis draws
        have populated ``values`` with every ``visible``/``hidden`` input.
        ``feedback`` is the model-specific signal the generator reads off the
        last solved state (for GROWTH, the previous employment rate). Reads the
        live ``self.vol_multiplier`` for volatility scaling.
        """

    # -- generic drift/volatility/crisis machinery -------------------------

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
            # A minority of crises are also financial: an extra impulse bundle on
            # top of the base one, drawn independently at onset.
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

    def step(self, feedback: Any) -> dict[str, float]:
        """Draw this step's exogenous input levels given the model ``feedback``.

        Advances the volatility and crisis state, then every AR(1)/random-walk
        input, then defers to :meth:`_model_inputs` for the model's special
        inputs. ``feedback`` is the model-specific signal read off the last
        solved state (passed straight through to :meth:`_model_inputs`).
        """
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

        self._model_inputs(values, feedback)
        return values


class ExcitedRunGenerator(ABC):
    """Configurable, reproducible generator of excited ground-truth histories.

    A generator fixes *how* runs are produced -- the excitation
    :class:`ExcitationConfig`, the timestep, the burn-in length and the solver
    dampening budget -- while :meth:`generate` fixes *which* run, via its seed and
    length. The same generator can therefore stamp out an ensemble of independent
    runs (one per seed) that share identical excitation dynamics.

    This base owns the generic driving machinery (settle, step, dampen, record,
    branch); a subclass supplies the model-specific pieces via
    :attr:`STATE`/:attr:`PARAMETERS`/:attr:`ACTIONS` and the abstract
    :meth:`_model_baselines`, :meth:`_build_model`, :meth:`_make_process`,
    :meth:`_feedback` and :meth:`_scenario_feedback`.
    """

    #: the driven model's value spaces (set by the concrete subclass)
    STATE: ClassVar[type[State]]
    PARAMETERS: ClassVar[type[Parameters]]
    ACTIONS: ClassVar[type[Actions]]

    def __init__(
        self,
        config: ExcitationConfig,
        *,
        dt: float = 1.0,
        burn_in: int = 15,
        max_dampen_attempts: int = 8,
        solver_iterations: int = 1000,
        solver_threshold: float = 1e-6,
        on_collapse: str = "raise",
    ) -> None:
        """Configure how runs are produced.

        ``config`` is the excitation spec; ``dt`` is the period length in years;
        ``burn_in`` is the number of years run unrecorded to settle onto the
        growth path; ``max_dampen_attempts`` bounds the solver-dampening retries
        per step; ``solver_iterations`` and ``solver_threshold`` are the solver's
        iteration cap and tolerance; ``on_collapse`` is ``'raise'`` or
        ``'truncate'`` -- whether an unrecoverable solver failure raises or ends
        the history at the collapse.
        """
        if on_collapse not in ("raise", "truncate"):
            raise ValueError("on_collapse must be 'raise' or 'truncate'")
        self.config = config
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
        # exogenous parameter of the model's calibration.
        self._baselines = self._model_baselines()

    # -- subclass contract -------------------------------------------------

    @abstractmethod
    def _model_baselines(self) -> Mapping[str, float]:
        """The baseline value of every model parameter drift is taken around."""

    @abstractmethod
    def _build_model(self) -> PysolveEconomicModel:
        """A fresh, fully-seeded model at this generator's ``dt`` and solver knobs.

        Not yet settled -- :meth:`_settled_model` runs the burn-in on top.
        """

    @abstractmethod
    def _make_process(
        self, rng: np.random.Generator, climate: float | None
    ) -> ExcitationProcess:
        """Build this model's :class:`ExcitationProcess` for one run."""

    @abstractmethod
    def _feedback(self, model: PysolveEconomicModel) -> Any:
        """The model-specific feedback signal read off the last solved state.

        Passed to :meth:`ExcitationProcess.step` for the model's stabilizer (for
        GROWTH, the previous employment rate).
        """

    @abstractmethod
    def _scenario_feedback(self) -> Any:
        """The neutral feedback used when drawing a stateless scenario.

        A scenario is generated without solving the model, so there is no realised
        state to read feedback from; the model supplies a neutral value (for
        GROWTH, full employment ``ER = 1``) that makes the stabilizer contribute
        nothing.
        """

    # -- public API --------------------------------------------------------

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
        process, climate = self._start_process(rng, excite)
        run, _ = self._drive(
            model, process, self._baseline_inputs(), n_steps, climate, seed
        )
        return run

    def generate_with_continuations(
        self,
        main_steps: int,
        continuation_steps: int,
        n_continuations: int,
        *,
        seed: int | None = None,
        continuation_seeds: list[int | None] | None = None,
        excite: bool = True,
    ) -> tuple[Run, list[Scenario], dict[str, float]]:
        """Generate one main run, its branch state, and ``n_continuations`` scenarios.

        The main run is ``main_steps`` recorded steps, exactly as :meth:`generate`
        would produce for ``seed`` -- a fully simulated :class:`Run`. From the
        economic state it ends in (the *branch state*, returned as a full internal
        model state dict), ``n_continuations`` alternative-future :class:`Scenario`\\ s
        branch off: each carries the excitation's drift/crisis state over from the
        branch and draws its own independent stream of future exogenous shocks
        (one per ``continuation_seeds`` entry), ``continuation_steps`` long.

        A scenario is only the exogenous forcing (``params`` + ``hidden``), so it is
        produced from the excitation process alone -- **the model is not solved for
        continuations**. The states a continuation would take depend on the actions
        an in-control agent chooses at rollout, so they are not fixed here; the
        branch state is the common starting point every scenario is rolled out from.

        ``continuation_seeds`` must have ``n_continuations`` entries when given;
        otherwise the scenarios draw from unseeded generators. Returns
        ``(main_run, scenarios, branch_state)``.
        """
        if continuation_seeds is not None and len(continuation_seeds) != n_continuations:
            raise ValueError(
                "continuation_seeds must have n_continuations entries "
                f"({len(continuation_seeds)} != {n_continuations})"
            )
        rng = np.random.default_rng(seed)
        model = self._settled_model()
        process, climate = self._start_process(rng, excite)
        main_run, _ = self._drive(
            model, process, self._baseline_inputs(), main_steps, climate, seed
        )

        # The full internal model state the main run ends in: the shared starting
        # point every scenario is rolled out from (the agent supplies the actions).
        branch_state = self._capture_state(model)
        scenarios: list[Scenario] = []
        for j in range(n_continuations):
            cont_seed = None if continuation_seeds is None else continuation_seeds[j]
            cont_process = self._fork_process(process, cont_seed)
            scenarios.append(
                self._drive_scenario(
                    cont_process, continuation_steps, climate, cont_seed
                )
            )
        return main_run, scenarios, branch_state

    # -- internals ---------------------------------------------------------

    def _baseline_inputs(self) -> dict[str, float]:
        """The exogenous inputs at their baselines, in the order a run records them.

        These are the values in effect during burn-in and the fallback dampening
        pulls back toward on the first recorded step.
        """
        exog_names = [
            *self.PARAMETERS.names(),
            *self.ACTIONS.names(),
            *self.config.hidden_names,
        ]
        return {name: float(self._baselines[name]) for name in exog_names}

    def _start_process(
        self, rng: np.random.Generator, excite: bool
    ) -> tuple[ExcitationProcess | None, float | None]:
        """Build the excitation process for a run and draw its one-off climate.

        Returns ``(process, climate)``; both are ``None`` when ``excite`` is
        ``False``. The climate is drawn up front (when the config has a climate
        spec) so the whole history shares one character rather than every step
        looking alike.
        """
        if not excite:
            return None, None
        climate = (
            self.config.climate.draw(rng) if self.config.climate is not None else None
        )
        return self._make_process(rng, climate), climate

    def _fork_process(
        self, process: ExcitationProcess | None, seed: int | None
    ) -> ExcitationProcess | None:
        """A copy of ``process`` at its current drift/crisis state with a fresh RNG.

        The deviation, volatility and live-crisis state are carried over so a
        continuation inherits *where the excitation is*, while reseeding the
        generator makes its future shocks an independent draw.
        """
        if process is None:
            return None
        forked = copy.deepcopy(process)
        forked._rng = np.random.default_rng(seed)
        return forked

    def _capture_state(self, model: PysolveEconomicModel) -> dict[str, float]:
        """The full end-of-run internal model state, as a plain name->value dict.

        Drops pysolve's private lag parameters (``_x__1``); the remaining visible
        and hidden variables fully pin the state, since every lag in the model is
        first order and reads back from the restored last solution.
        """
        return {
            name: value
            for name, value in model.solutions[-1].items()
            if not name.startswith("_")
        }

    def _drive_scenario(
        self,
        process: ExcitationProcess | None,
        n_steps: int,
        climate: float | None,
        seed: int | None,
    ) -> Scenario:
        """Draw ``n_steps`` of exogenous forcing from ``process``, without any model.

        Records only the visible :class:`Parameters` and the hidden structural
        parameters -- a :class:`Scenario`, the world an agent acts within. The
        model's stabilizer feedback is evaluated at its neutral value
        (:meth:`_scenario_feedback`): a scenario has no realised state, and the
        stabilizer response belongs to the rollout against the agent's own
        economy, not to the frozen forcing. A ``None`` ``process`` (unexcited)
        holds every input at its baseline.
        """
        params_names = self.PARAMETERS.names()
        hidden_names = self.config.hidden_names

        params, hidden = [], []
        volatility, crisis_intensity = [], []
        for _ in range(n_steps):
            if process is not None:
                values = process.step(self._scenario_feedback())
                volatility.append(process.vol_multiplier)
                crisis_intensity.append(process.crisis_intensity)
            else:
                values = self._baseline_inputs()
            params.append([values[name] for name in params_names])
            hidden.append([values[name] for name in hidden_names])

        excited = process is not None

        def _cols(rows: list, width: int) -> np.ndarray:
            return np.asarray(rows, dtype=float).reshape(len(rows), width)

        return Scenario(
            params=_cols(params, len(params_names)),
            hidden=_cols(hidden, len(hidden_names)),
            dt=self.dt,
            seed=seed,
            volatility=np.asarray(volatility, dtype=float) if excited else None,
            crisis_intensity=(
                np.asarray(crisis_intensity, dtype=float) if excited else None
            ),
            climate=climate,
        )

    def _drive(
        self,
        model: PysolveEconomicModel,
        process: ExcitationProcess | None,
        applied: dict[str, float],
        n_steps: int,
        climate: float | None,
        seed: int | None,
    ) -> tuple[Run, dict[str, float]]:
        """Step ``model`` forward ``n_steps`` times, recording the visible history.

        ``process`` supplies each step's exogenous inputs (or ``None`` for a
        baseline reference path); ``applied`` is the last-converged input set the
        dampening falls back toward. Returns the recorded :class:`Run` and the
        final applied inputs (so a continuation can pick up where this left off).
        """
        state_names = self.STATE.names()
        param_names = self.PARAMETERS.names()
        action_names = self.ACTIONS.names()
        hidden_names = self.config.hidden_names

        states, params, actions, hidden = [], [], [], []
        volatility, crisis_intensity = [], []
        dampened = 0
        collapsed = False
        for _ in range(n_steps):
            if process is not None:
                target = process.step(self._feedback(model))
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
            params.append([target[name] for name in param_names])
            actions.append([target[name] for name in action_names])
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

        run = Run(
            states=_cols(states, len(state_names)),
            params=_cols(params, len(param_names)),
            actions=_cols(actions, len(action_names)),
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
        return run, applied

    def _settled_model(self) -> PysolveEconomicModel:
        """A model seeded with the calibration and settled onto its path.

        ``burn_in`` is a number of *years*; at a sub-annual ``dt`` it is run for the
        matching number of steps so the settling spans the same calendar time.
        """
        model = self._build_model()
        model.run(round(self.burn_in / self.dt))
        return model

    def _solve_step(
        self,
        model: PysolveEconomicModel,
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
