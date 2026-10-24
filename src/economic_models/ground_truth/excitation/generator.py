"""The model-agnostic run generator: settle, step, dampen, record and branch.

:class:`ExcitedRunGenerator` is the reproducible driver that settles a model,
steps it under the inputs an
:class:`~economic_models.ground_truth.excitation.process.ExcitationProcess`
draws, dampens the solver on failure and records the result as a
:class:`~economic_models.run.Run` -- plus branching
:class:`~economic_models.run.Scenario`\\ s forked off rows of it, each resumed
from a :class:`BranchCapture` of where the excitation stood at that row.

It lives apart from the process because it is the only half of excitation that
touches a model: everything here is plumbing around a solver, while the process
is pure random draws. A concrete model subclasses it to supply its value
spaces, its model factory, its process and its per-step feedback signal -- see
:mod:`economic_models.ground_truth.models.growth.excitation`.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from pysolve.model import CalculationError, SolutionNotFoundError

from economic_models.base import ModelStepFailure
from economic_models.ground_truth.base import PysolveEconomicModel
from economic_models.ground_truth.excitation.process import (
    ExcitationConfig,
    ExcitationProcess,
)
from economic_models.ground_truth.linear_re import LinearREEconomicModel
from economic_models.run import Run, Scenario
from economic_models.variables import Actions, Parameters, State

#: The structural models a generator can drive: both offer ``set_values``,
#: ``step``, ``run`` and the ``solutions`` record the driver reads.
DrivenModel = PysolveEconomicModel | LinearREEconomicModel

#: Offset from a continuation's seed to the seed of its climate redraw: far
#: enough from any seed a caller hands out that the two streams never coincide.
_CLIMATE_SEED_OFFSET = 500_000_000


@dataclass(frozen=True)
class BranchCapture:
    """Everything needed to fork a future off one row of a finished run.

    A :class:`~economic_models.run.Scenario` is drawn from the excitation process
    alone, but *where* the excitation stands is not a property of the row a run
    records: the drift deviations, the volatility regime, the live crisis state
    and the ``Nfe`` log random walk all live inside the process. Forking a future
    off row ``index`` therefore means resuming that process from the copy it was
    at when the row was drawn -- ``state`` alone is not enough, and splicing a
    future drawn elsewhere onto this row would step the economy's exogenous
    inputs discontinuously.

    ``state`` is the full internal model state at the row (what a solver is
    restored onto), ``process`` the excitation as it stood having drawn it, and
    ``applied`` the last converged input set the dampening falls back toward.
    """

    index: int  #: row of the run this forks from
    state: dict[str, float]  #: full internal model state at that row
    process: ExcitationProcess | None  #: the excitation, as it stood there
    applied: dict[str, float]  #: last converged inputs at that row


@dataclass
class _RunRows:
    """The per-step rows a run accumulates while it is being driven."""

    states: list[list[float]] = field(default_factory=list)
    params: list[list[float]] = field(default_factory=list)
    actions: list[list[float]] = field(default_factory=list)
    hidden: list[list[float]] = field(default_factory=list)
    volatility: list[float] = field(default_factory=list)
    crisis_intensity: list[float] = field(default_factory=list)


def _cols(rows: list, width: int) -> np.ndarray:
    """Stack ``rows`` into a ``(len(rows), width)`` float array.

    Keeps the 2-D column shape even for an empty history (a run that
    collapsed on its first step under ``on_collapse="truncate"``), so
    ``run.states[:, i]`` and the transform see ``(0, n_col)``, not ``(0,)``.
    """
    return np.asarray(rows, dtype=float).reshape(len(rows), width)


def _trace(values: list[float], excited: bool) -> np.ndarray | None:
    """A per-step diagnostic trace as an array, or ``None`` for an unexcited run.

    An unexcited run has no volatility regime or crisis state to report, and
    ``None`` rather than a row of ones and zeros is what says so.
    """
    return np.asarray(values, dtype=float) if excited else None


def _entry[T](entries: Sequence[T | None] | None, j: int) -> T | None:
    """Entry ``j`` of an optional per-continuation argument, ``None`` when it is absent."""
    return None if entries is None else entries[j]


def _check_length(
    name: str, entries: Sequence[Any] | None, n_continuations: int
) -> None:
    """Check an optional per-continuation argument has ``n_continuations`` entries."""
    if entries is not None and len(entries) != n_continuations:
        raise ValueError(
            f"{name} must have n_continuations entries "
            f"({len(entries)} != {n_continuations})"
        )


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
    def _build_model(self) -> DrivenModel:
        """A fresh, fully-seeded model at this generator's ``dt`` and solver knobs.

        Not yet settled -- :meth:`_settled_model` runs the burn-in on top.
        """

    @abstractmethod
    def _make_process(
        self, rng: np.random.Generator, climate: float | None
    ) -> ExcitationProcess:
        """Build this model's :class:`ExcitationProcess` for one run."""

    @abstractmethod
    def _feedback(self, model: DrivenModel) -> Any:
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
        run, _, _ = self._drive(
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
        continuation_seeds: Sequence[int | None] | None = None,
        continuation_configs: Sequence[ExcitationConfig | None] | None = None,
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
        otherwise the scenarios draw from unseeded generators.

        ``continuation_configs`` optionally gives each continuation its own
        excitation spec, taking effect *at the branch* (:meth:`ExcitationProcess.reconfigure`):
        a ``None`` entry continues under the generator's own config, and anything
        else is a **structural break** at the moment the future forks -- the
        drift carries over, the weather changes. That is how a bank of futures
        harsher than the history the model was fitted on is drawn, which is the
        only setting in which an online correction has anything to correct. Every
        config must record the same hidden inputs, since a :class:`Scenario`'s
        hidden block is laid out by the generator's own column order.

        Returns ``(main_run, scenarios, branch_state)``. The single-branch case of
        :meth:`generate_branched`, which forks futures off several rows of the
        same run.
        """
        end = main_steps - 1
        run, scenarios, branches = self.generate_branched(
            main_steps,
            continuation_steps,
            n_continuations,
            seed=seed,
            continuation_seeds=continuation_seeds,
            continuation_configs=continuation_configs,
            excite=excite,
        )
        return run, scenarios, branches[end].state

    def generate_branched(
        self,
        main_steps: int,
        continuation_steps: int,
        n_continuations: int,
        *,
        branch_at: Sequence[int] | None = None,
        seed: int | None = None,
        continuation_seeds: Sequence[int | None] | None = None,
        continuation_configs: Sequence[ExcitationConfig | None] | None = None,
        excite: bool = True,
    ) -> tuple[Run, list[Scenario], dict[int, BranchCapture]]:
        """One main run and ``n_continuations`` futures forking off rows of it.

        The general form of :meth:`generate_with_continuations`: ``branch_at[j]``
        is the row of the main run continuation ``j`` forks from, defaulting to
        the last row for every one of them (which *is*
        :meth:`generate_with_continuations`). Every fork resumes the excitation
        from a :class:`BranchCapture` taken at its own row, so a future starts
        where its row's drift, volatility regime and level random walks actually
        stood -- the reason a future cannot simply be spliced onto a different
        row than the one it was drawn for.

        Forking off several rows is how an agent is trained from more than one
        initial condition. The end of a run is one draw from the economy's state
        distribution and, being one draw, it is typically an extreme of something:
        training every episode from it conditions the policy on that corner. The
        model is still solved exactly once -- for the main run -- because a
        scenario needs no solve, so a bank spread over many rows costs what a bank
        forked off one costs.

        Returns ``(main_run, scenarios, captures)``, ``captures`` keyed by the row
        index each was taken at. A run that collapsed before a requested row has
        no capture there and raises rather than silently forking elsewhere.
        """
        if branch_at is None:
            branch_at = [main_steps - 1] * n_continuations
        self._check_continuation_args(
            n_continuations, branch_at, continuation_seeds, continuation_configs
        )
        wanted = self._branch_rows(main_steps, branch_at)

        rng = np.random.default_rng(seed)
        model = self._settled_model()
        process, climate = self._start_process(rng, excite)
        main_run, _, captures = self._drive(
            model,
            process,
            self._baseline_inputs(),
            main_steps,
            climate,
            seed,
            branch_at=wanted,
        )
        self._require_captures(wanted, captures, len(main_run))

        scenarios = [
            self._continuation(
                captures[branch_at[j]].process,
                _entry(continuation_seeds, j),
                _entry(continuation_configs, j),
                climate,
                continuation_steps,
            )
            for j in range(n_continuations)
        ]
        return main_run, scenarios, captures

    def _branch_rows(self, main_steps: int, branch_at: Sequence[int]) -> list[int]:
        """The sorted rows to capture: every fork row, plus the run's last one.

        The last row is always captured, because it is the branch state
        :meth:`generate_with_continuations` returns. A row outside the run raises.
        """
        last = main_steps - 1
        wanted = sorted(set(branch_at) | {last})
        if wanted[0] < 0 or wanted[-1] > last:
            raise ValueError(
                f"branch_at must index rows of a {main_steps}-step run, got "
                f"[{wanted[0]}, {wanted[-1]}]"
            )
        return wanted

    @staticmethod
    def _require_captures(
        wanted: Sequence[int], captures: Mapping[int, BranchCapture], run_length: int
    ) -> None:
        """Raise if the run collapsed before a row a future was asked to fork from.

        Forking from the nearest row that *was* reached instead would silently
        hand the future a different starting point than the one requested.
        """
        missing = [t for t in wanted if t not in captures]
        if missing:
            raise RuntimeError(
                f"the run ended after {run_length} steps and never reached rows "
                f"{missing}, which futures were asked to fork from"
            )

    def _check_continuation_args(
        self,
        n_continuations: int,
        branch_at: Sequence[int],
        continuation_seeds: Sequence[int | None] | None,
        continuation_configs: Sequence[ExcitationConfig | None] | None,
    ) -> None:
        """Check every per-continuation argument has ``n_continuations`` entries.

        Also checks each continuation config records the generator's own hidden
        inputs, since a :class:`Scenario`'s hidden block is laid out by them.
        """
        _check_length("branch_at", branch_at, n_continuations)
        _check_length("continuation_seeds", continuation_seeds, n_continuations)
        _check_length("continuation_configs", continuation_configs, n_continuations)
        for cfg in continuation_configs or ():
            if cfg is not None and cfg.hidden_names != self.config.hidden_names:
                raise ValueError(
                    "a continuation config must excite the same hidden inputs "
                    f"as the generator's: {cfg.hidden_names} != "
                    f"{self.config.hidden_names}"
                )

    def _continuation(
        self,
        process: ExcitationProcess | None,
        seed: int | None,
        config: ExcitationConfig | None,
        climate: float | None,
        n_steps: int,
    ) -> Scenario:
        """Fork ``process`` under ``seed`` and draw an ``n_steps`` future from it.

        ``process`` is the excitation as captured at the branch row and
        ``climate`` the history's turbulence draw; a non-``None`` ``config`` is a
        structural break at the branch (:meth:`ExcitationProcess.reconfigure`).
        """
        cont_process = self._fork_process(process, seed)
        cont_climate = climate
        if cont_process is not None and config is not None:
            # A continuation only reaches here when it is a different economy
            # (a different preset, a perturbation, or both), so its
            # turbulence is redrawn under its *own* climate spec rather than
            # inherited: a bank whose members all share the history's weather
            # is a bank of one character. Drawn off a stream derived from --
            # but not equal to -- the continuation's own seed, so it stays
            # reproducible without shifting the shock stream that follows it.
            if config.climate is not None:
                cont_climate = config.climate.draw(
                    np.random.default_rng(
                        None if seed is None else seed + _CLIMATE_SEED_OFFSET
                    )
                )
            cont_process.reconfigure(config, climate=cont_climate)
        return self._drive_scenario(cont_process, n_steps, cont_climate, seed)

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

    def _capture_state(self, model: DrivenModel) -> dict[str, float]:
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
        return Scenario(
            params=_cols(params, len(params_names)),
            hidden=_cols(hidden, len(hidden_names)),
            dt=self.dt,
            seed=seed,
            volatility=_trace(volatility, excited),
            crisis_intensity=_trace(crisis_intensity, excited),
            climate=climate,
        )

    def _drive(
        self,
        model: DrivenModel,
        process: ExcitationProcess | None,
        applied: dict[str, float],
        n_steps: int,
        climate: float | None,
        seed: int | None,
        branch_at: Sequence[int] = (),
    ) -> tuple[Run, dict[str, float], dict[int, BranchCapture]]:
        """Step ``model`` forward ``n_steps`` times, recording the visible history.

        ``process`` supplies each step's exogenous inputs (or ``None`` for a
        baseline reference path); ``applied`` is the last-converged input set the
        dampening falls back toward. Returns the recorded :class:`Run`, the
        final applied inputs (so a continuation can pick up where this left off)
        and a :class:`BranchCapture` per row index in ``branch_at``.

        A capture is taken at the *end* of the step that produced that row, so it
        holds the model exactly as the row leaves it and the process exactly as it
        stands having drawn that row's inputs -- which is the position a
        continuation forking there has to start from. A row the run never reached
        (it collapsed first) simply has no capture.
        """
        wanted = frozenset(branch_at)
        captures: dict[int, BranchCapture] = {}
        rows = _RunRows()
        dampened = 0
        collapsed = False
        for _ in range(n_steps):
            target = self._next_target(model, process, applied)
            try:
                target, attempts = self._solve_step(
                    model, target, applied, len(rows.states), seed
                )
            except RuntimeError:
                if self.on_collapse == "raise":
                    raise
                collapsed = True
                break
            dampened += attempts > 0
            applied = target

            self._record_row(rows, model, target, process)
            row = len(rows.states) - 1
            if row in wanted:
                captures[row] = self._capture(row, model, process, applied)

        run = self._assemble_run(
            rows,
            seed,
            climate,
            dampened=dampened,
            collapsed=collapsed,
            excited=process is not None,
        )
        return run, applied, captures

    def _next_target(
        self,
        model: DrivenModel,
        process: ExcitationProcess | None,
        applied: dict[str, float],
    ) -> dict[str, float]:
        """This step's intended inputs: a fresh draw, or the last ones on a reference path."""
        if process is None:
            return dict(applied)
        return process.step(self._feedback(model))

    def _capture(
        self,
        row: int,
        model: DrivenModel,
        process: ExcitationProcess | None,
        applied: dict[str, float],
    ) -> BranchCapture:
        """Snapshot ``model`` and ``process`` as they stand having produced ``row``.

        The process is deep-copied because the run carries on drawing from it;
        a capture that aliased it would drift forward with the history.
        """
        return BranchCapture(
            index=row,
            state=self._capture_state(model),
            process=None if process is None else copy.deepcopy(process),
            applied=dict(applied),
        )

    def _assemble_run(
        self,
        rows: _RunRows,
        seed: int | None,
        climate: float | None,
        *,
        dampened: int,
        collapsed: bool,
        excited: bool,
    ) -> Run:
        """Stack the accumulated ``rows`` into the :class:`Run` they describe."""
        return Run(
            states=_cols(rows.states, len(self.STATE.names())),
            params=_cols(rows.params, len(self.PARAMETERS.names())),
            actions=_cols(rows.actions, len(self.ACTIONS.names())),
            hidden=_cols(rows.hidden, len(self.config.hidden_names)),
            dt=self.dt,
            seed=seed,
            dampened_steps=dampened,
            volatility=_trace(rows.volatility, excited),
            crisis_intensity=_trace(rows.crisis_intensity, excited),
            climate=climate,
            collapsed=collapsed,
        )

    def _record_row(
        self,
        rows: _RunRows,
        model: DrivenModel,
        target: dict[str, float],
        process: ExcitationProcess | None,
    ) -> None:
        """Append the row the model just solved, under inputs ``target``, to ``rows``."""
        solution = model.solutions[-1]
        rows.states.append([solution[name] for name in self.STATE.names()])
        rows.params.append([target[name] for name in self.PARAMETERS.names()])
        rows.actions.append([target[name] for name in self.ACTIONS.names()])
        rows.hidden.append([target[name] for name in self.config.hidden_names])
        if process is not None:
            rows.volatility.append(process.vol_multiplier)
            rows.crisis_intensity.append(process.crisis_intensity)

    def _settled_model(self) -> DrivenModel:
        """A model seeded with the calibration and settled onto its path.

        ``burn_in`` is a number of *years*; at a sub-annual ``dt`` it is run for the
        matching number of steps so the settling spans the same calendar time.
        """
        model = self._build_model()
        model.run(round(self.burn_in / self.dt))
        return model

    def _solve_step(
        self,
        model: DrivenModel,
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
            except (CalculationError, SolutionNotFoundError, ModelStepFailure):
                target = {
                    name: 0.5 * (value + applied[name])
                    for name, value in target.items()
                }
        raise RuntimeError(
            f"solver failed even with dampened exogenous inputs "
            f"(seed={seed}, step={step_index})"
        )
