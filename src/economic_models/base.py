"""Base class for economic models simulated with :mod:`pysolve`.

A model wraps a pysolve ``Model``, keeps its own solution history, and can be
advanced one simulation step at a time. The base fixes the *visible interface* --
the :class:`State` a central bank observes, the exogenous :class:`Parameters` it
sees but does not control, and the :class:`Actions` (policy levers) it does. This
interface is **shared by every model**: whatever economy sits behind it, the bank
observes the same macro aggregates and pulls the same levers. What differs
between models is the *hidden* machinery (expectations, targets, sector
stocks/flows, behavioural parameters), which each concrete subclass registers and
owns in :meth:`build`. Those hidden parameters remain fully modifiable through
:meth:`set_values`, they are simply not part of the exposed interface.
"""

from abc import ABC, abstractmethod

from pysolve.model import Model

from src.economic_models.variables import Actions, Parameters, State


class BaseEconomicModel(ABC):
    """A stateful economic model exposing the shared visible interface.

    The three value classes below double as the interface schema: their fields
    name the model quantities every concrete model must register. ``State`` fields
    are endogenous variables; ``Parameters`` and ``Actions`` fields are exogenous
    parameters.
    """

    STATE = State
    PARAMETERS = Parameters
    ACTIONS = Actions

    def __init__(self, **param_overrides: float) -> None:
        self._model = Model()
        self._model.set_var_default(0)

        # The subclass registers every variable, parameter and equation,
        # including the hidden internals specific to it.
        self.build(self._model)
        self._validate_interface()

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

    def _exogenous_names(self) -> set[str]:
        return {*self.PARAMETERS.names(), *self.ACTIONS.names()}

    def _validate_interface(self) -> None:
        for name in self.STATE.names():
            if name not in self._model.variables:
                raise ValueError(f"visible state {name!r} is not a model variable")
        for name in self._exogenous_names():
            if name not in self._model.parameters:
                raise ValueError(f"visible parameter {name!r} is not a model parameter")

    # -- simulation --------------------------------------------------------

    def step(self, iterations: int = 200, threshold: float = 1e-6) -> None:
        """Advance the simulation by one period."""
        self._model.solve(iterations=iterations, threshold=threshold)

    def run(self, n_steps: int, iterations: int = 200, threshold: float = 1e-6) -> None:
        """Advance the simulation by ``n_steps`` periods."""
        for _ in range(n_steps):
            self.step(iterations=iterations, threshold=threshold)

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
        return {name: p.value for name, p in self._model.parameters.items()}

    def _set_names(self, allowed: set[str], values: dict[str, float]) -> None:
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"not visible parameters: {sorted(unknown)}")
        self._model.set_values(values)

    # -- escape hatches / compatibility ------------------------------------

    def set_values(self, values) -> None:
        """Set any variables/parameters, including hidden ones."""
        self._model.set_values(values)

    def solve(self, iterations: int = 200, threshold: float = 1e-6) -> None:
        """Alias for :meth:`step` (pysolve-compatible)."""
        self.step(iterations=iterations, threshold=threshold)

    @property
    def solutions(self) -> list:
        """The full solution history of the underlying model."""
        return self._model.solutions

    @property
    def model(self) -> Model:
        """The underlying pysolve model (escape hatch)."""
        return self._model
