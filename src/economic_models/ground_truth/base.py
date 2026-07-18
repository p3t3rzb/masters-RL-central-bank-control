"""The pysolve base for structural ground-truth models.

:class:`PysolveEconomicModel` wraps a :mod:`pysolve` ``Model``, keeps its own
solution history, and advances one period at a time by solving the system. It
surfaces the shared visible interface; a subclass registers and seeds the hidden
machinery (expectations, targets, sector stocks/flows, behavioural parameters) in
:meth:`~PysolveEconomicModel.build` and :meth:`~PysolveEconomicModel._seed`.
"""

from abc import abstractmethod

from pysolve.model import Model

from economic_models.base import BaseEconomicModel
from economic_models.variables import Actions, Parameters, State


class PysolveEconomicModel(BaseEconomicModel):
    """Base for structural models simulated one period at a time with pysolve.

    The subclass registers every variable, parameter and equation -- visible and
    hidden alike -- in :meth:`build`, and seeds their initial values in
    :meth:`_seed`; this base drives the solver and surfaces the shared visible
    interface on top of the resulting solution history.

    Solver knobs are constructor state: ``iterations`` and ``threshold`` are
    stored once and used by every :meth:`step` / :meth:`run` call.
    """

    def __init__(
        self,
        *,
        iterations: int = 200,
        threshold: float = 1e-6,
        **param_overrides: float,
    ) -> None:
        """Build, validate and seed the model, then apply constructor overrides.

        ``iterations`` and ``threshold`` are the solver's per-step iteration cap
        and convergence tolerance, stored for every :meth:`step`. Any
        ``param_overrides`` set visible exogenous parameters on top of the seeded
        baseline (so they always win over seeding).
        """
        self.iterations = iterations
        self.threshold = threshold
        self._model = Model()
        self._model.set_var_default(0)

        # The subclass registers every variable, parameter and equation,
        # including the hidden internals specific to it.
        self.build(self._model)
        self._validate_interface()

        # Seed initial values first, then apply the caller's overrides on top,
        # so a constructor override is never clobbered by the seeding.
        self._seed(self._model)
        if param_overrides:
            self._set_names(self._exogenous_names(), param_overrides)

    # -- subclass contract -------------------------------------------------

    @abstractmethod
    def build(self, model: Model) -> None:
        """Register *all* variables, parameters and equations onto ``model``.

        Visible and hidden quantities alike -- the visible names carried by
        :class:`State` / :class:`Parameters` / :class:`Actions` must be among
        those registered here; everything else is this model's hidden internals.
        """

    def _seed(self, model: Model) -> None:
        """Seed initial values onto ``model`` (default: no-op).

        Called once from the constructor, after :meth:`build` and interface
        validation but *before* the constructor's ``param_overrides`` are
        applied, so overrides always win over the seeded baseline.
        """

    def _validate_interface(self) -> None:
        """Check every visible state/exogenous name is registered on the model."""
        for name in self.STATE.names():
            if name not in self._model.variables:
                raise ValueError(f"visible state {name!r} is not a model variable")
        for name in self._exogenous_names():
            if name not in self._model.parameters:
                raise ValueError(f"visible parameter {name!r} is not a model parameter")

    # -- simulation --------------------------------------------------------

    def step(self) -> None:
        """Advance the simulation by one period."""
        self._model.solve(iterations=self.iterations, threshold=self.threshold)

    def run(self, n_steps: int) -> None:
        """Advance the simulation by ``n_steps`` periods."""
        for _ in range(n_steps):
            self.step()

    def advance(self, parameters: Parameters, actions: Actions) -> State:
        """Apply the exogenous inputs and advance one period, returning the new state."""
        self.set_parameters(parameters)
        self.set_actions(actions)
        self.step()
        return self.state

    # -- visible interface -------------------------------------------------

    @property
    def state(self) -> State:
        """Latest values of the visible state, as a whole :class:`State`."""
        latest = self._model.solutions[-1]
        return State.from_dict(latest)

    @property
    def parameters(self) -> Parameters:
        """Current values of the visible exogenous :class:`Parameters`."""
        return Parameters.from_dict(self._parameter_values())

    @property
    def actions(self) -> Actions:
        """Current values of the central bank's :class:`Actions`."""
        return Actions.from_dict(self._parameter_values())

    def set_state(self, state: State) -> None:
        """Seed the visible state variables' initial values.

        Sets only the observable aggregates; each model seeds its own hidden
        internals. Use to pin the common initial condition of a scenario.
        """
        self._model.set_values(state.to_dict())

    def set_parameters(self, parameters: Parameters) -> None:
        """Set the exogenous parameters the bank observes (fiscal, structural...)."""
        self._model.set_values(parameters.to_dict())

    def set_actions(self, actions: Actions) -> None:
        """Set the central bank's policy levers."""
        self._model.set_values(actions.to_dict())

    def _parameter_values(self) -> dict[str, float]:
        """The current value of every parameter registered on the model, by name."""
        return {name: p.value for name, p in self._model.parameters.items()}

    def _set_names(self, allowed: set[str], values: dict[str, float]) -> None:
        """Set ``values`` on the model, rejecting any name outside ``allowed``."""
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"not visible parameters: {sorted(unknown)}")
        self._model.set_values(values)

    # -- escape hatches / compatibility ------------------------------------

    def set_values(self, values: dict[str, float]) -> None:
        """Set any variables/parameters, including hidden ones."""
        self._model.set_values(values)

    @property
    def solutions(self) -> list:
        """The full solution history of the underlying model."""
        return self._model.solutions

    @property
    def model(self) -> Model:
        """The underlying pysolve model (escape hatch)."""
        return self._model
