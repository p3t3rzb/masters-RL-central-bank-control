"""The shared economic-model interface every model family implements.

:class:`BaseEconomicModel` fixes the visible interface a model exposes,
regardless of what sits behind it: the three observation properties
(:class:`State`, :class:`Parameters`, :class:`Actions`) and the single driver
:meth:`~BaseEconomicModel.advance`. Both the structural ground-truth models and
the learned proxies implement it.
"""

from abc import ABC, abstractmethod

from economic_models.variables import Actions, Parameters, State


class BaseEconomicModel(ABC):
    """The central-bank-visible interface shared by every economic model.

    The three value classes below double as the interface schema: their fields
    name the model quantities every concrete model must expose. ``State`` fields
    are endogenous observables; ``Parameters`` and ``Actions`` fields are the
    exogenous inputs (the levers the bank sees but does not control, and the ones
    it does). A subclass -- structural or learned -- realises this interface by
    supplying the :attr:`state` / :attr:`parameters` / :attr:`actions`
    observations and by advancing one period at a time via :meth:`advance`;
    *how* it produces the next state is its own business.
    """

    STATE = State
    PARAMETERS = Parameters
    ACTIONS = Actions

    def _exogenous_names(self) -> set[str]:
        """The names of every exogenous input: ``Parameters`` and ``Actions`` combined."""
        return {*self.PARAMETERS.names(), *self.ACTIONS.names()}

    # -- visible interface -------------------------------------------------

    @property
    @abstractmethod
    def state(self) -> State:
        """Latest values of the visible endogenous state."""

    @property
    @abstractmethod
    def parameters(self) -> Parameters:
        """Current values of the visible exogenous :class:`Parameters`."""

    @property
    @abstractmethod
    def actions(self) -> Actions:
        """Current values of the central bank's :class:`Actions`."""

    @abstractmethod
    def advance(self, parameters: Parameters, actions: Actions) -> State:
        """Apply the exogenous inputs and advance one period, returning the new state."""
