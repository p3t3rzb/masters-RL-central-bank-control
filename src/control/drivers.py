"""Driving a model through an episode: the proxy and the ground truth alike.

The environment must be able to run the *same* policy against a learned proxy and
against the structural :class:`~economic_models.ground_truth.models.growth.model.GrowthModel`
without changing a line, because the whole point of the experiment is the gap
between those two returns. Both families already share
:meth:`~economic_models.BaseEconomicModel.advance`, but they start an episode
differently: a proxy warm-starts its encoder belief from a window of trailing
level rows, while the ground truth is a solver that has to be pinned to the
branch state the history ended in.

:class:`ModelDriver` owns that difference and nothing else. It is a template: the
base hands the environment the two trailing level rows every observation needs
(the same rows for both families, since the branch point *is* the history's end),
and the subclass supplies only how the model returns there and how it steps.

**Branching** is an optional third thing a driver may support: saving a rollout
position and stepping from it more than once. Only a stochastic model has
anything to gain from it (the solver would return the same period twice), so the
base declines by default and :class:`ProxyDriver` opts in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from economic_models.ground_truth import GrowthModel
from economic_models.proxy import BaseProxyModel, RolloutState
from economic_models.variables import Actions, Parameters, State

from control.world import TrainingWorld


class ModelDriver(ABC):
    """Runs one model through episodes that all start at the history's end."""

    def __init__(self, world: TrainingWorld) -> None:
        """Bind the driver to the ``world`` whose history it starts from."""
        self._world = world

    @property
    def world(self) -> TrainingWorld:
        """The world this driver starts every episode from."""
        return self._world

    # -- subclass contract ---------------------------------------------------

    @property
    @abstractmethod
    def required_window(self) -> int:
        """Trailing level rows this model needs to be returned to the branch."""

    @abstractmethod
    def _reset(self, window: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Return the model to the branch point, given the warm-start window."""

    @abstractmethod
    def step(
        self,
        parameters: Parameters,
        actions: Actions,
        hidden: dict[str, float] | None,
        rng: np.random.Generator,
    ) -> State:
        """Advance one period under the given exogenous inputs and actions.

        ``hidden`` are the episode's hidden structural parameters, which only a
        structural model consumes (a proxy never sees them); ``rng`` drives the
        stochastic transition of a proxy and is ignored by the deterministic
        solver.
        """

    # -- branching (opt-in) --------------------------------------------------

    @property
    def branchable(self) -> bool:
        """Whether this driver can step more than once from a saved position."""
        return False

    def snapshot(self) -> Any:
        """Capture the rollout position, so :meth:`step` can be redrawn from it."""
        raise NotImplementedError(f"{type(self).__name__} cannot branch")

    def restore(self, snapshot: Any) -> None:
        """Put the rollout back at a :meth:`snapshot`."""
        raise NotImplementedError(f"{type(self).__name__} cannot branch")

    # -- public API ----------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Return to the branch point; give the trailing state and exog levels.

        The returned blocks are the last two rows of the history's state levels
        and of its exogenous levels (parameters then actions) -- everything an
        observation needs before the first step of an episode.

        A stateful driver may ask for the whole history rather than its bare
        minimum (see :attr:`ProxyDriver.required_window`); two rows is the floor,
        since an observation needs a previous row to difference against.
        """
        window = self._world.window(max(self.required_window, 2))
        self._reset(window)
        states, params, actions = window
        return (
            states[-2:].astype(float),
            np.hstack([params[-2:], actions[-2:]]).astype(float),
        )


class ProxyDriver(ModelDriver):
    """Drives a fitted proxy, always sampling from its conditional law.

    The proxy carries the rollout state (encoder belief, previous features and
    levels), so one instance backs exactly one driver; :meth:`_reset` re-seeds it
    from the history tail before every episode.
    """

    def __init__(self, proxy: BaseProxyModel, world: TrainingWorld) -> None:
        """Bind the driver to a **fitted** ``proxy`` and its ``world``."""
        super().__init__(world)
        self.proxy = proxy
        #: the branch position, filtered once and restored thereafter.
        self._start: RolloutState | None = None

    @property
    def required_window(self) -> int:
        """The whole history, not the proxy's bare minimum.

        The proxy's belief conditions its every draw, and it is the same filter
        the observation carries, so a reset that starts it at the encoder's prior
        both mis-conditions the world model and desynchronises it from the
        policy's view of the same latent. Encoding the whole run is what the
        observer does (:meth:`~control.observation.Observer.fit`), and matching it
        is what makes the two beliefs agree; :meth:`_reset` pays for it once, so
        there is no shorter tail worth tuning.
        """
        return len(self._world.history)

    @property
    def branchable(self) -> bool:
        """Yes: the proxy can be put back and asked for another draw."""
        return True

    def snapshot(self) -> RolloutState:
        """The proxy's rollout position (belief, previous features and levels)."""
        return self.proxy.snapshot()

    def restore(self, snapshot: RolloutState) -> None:
        """Put the proxy back at ``snapshot``."""
        self.proxy.restore(snapshot)

    def _reset(self, window: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Warm-start the encoder belief and level history from the window.

        Filtering the whole history costs far more than the episode that follows
        it, and every episode branches from the same place -- so the window is
        filtered on the first reset only and the position restored afterwards.
        That is what :meth:`snapshot` and :meth:`restore` already do between
        branch draws; a reset is the same move over a longer gap.
        """
        if self._start is None:
            self.proxy.reset(*window)
            self._start = self.proxy.snapshot()
        else:
            self.proxy.restore(self._start)

    def step(
        self,
        parameters: Parameters,
        actions: Actions,
        hidden: dict[str, float] | None,
        rng: np.random.Generator,
    ) -> State:
        """One stochastic draw from the proxy's one-step conditional law.

        Always sampled, never the conditional mean: an agent trained against a
        deterministic surrogate learns to exploit that surrogate's quirks.
        """
        return self.proxy.step(parameters, actions, rng=rng)


class GroundTruthDriver(ModelDriver):
    """Drives the structural GROWTH model, pinned to the branch state.

    Evaluation only -- it solves the full system every step. Each episode builds a
    fresh model and restores the internal state the history ended in, so runs are
    independent and start exactly where the proxy's warm start does.

    Not branchable, and nothing is lost by that: the solver is deterministic, so
    two draws of the same period would agree exactly, and its position lives in a
    pysolve model that must not be copied.
    """

    def __init__(self, world: TrainingWorld, *, iterations: int = 1000,
                 threshold: float = 1e-6) -> None:
        """Configure the solver knobs used for every episode of this driver."""
        super().__init__(world)
        self.iterations = iterations
        self.threshold = threshold
        self.model_: GrowthModel | None = None

    @property
    def required_window(self) -> int:
        """The solver needs no window -- the branch state pins it."""
        return 1

    def _reset(self, window: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Build a fresh model and restore the branch state onto it."""
        self.model_ = GrowthModel(
            dt=self._world.dt,
            iterations=self.iterations,
            threshold=self.threshold,
        )
        self.model_.set_values(dict(self._world.branch))

    def step(
        self,
        parameters: Parameters,
        actions: Actions,
        hidden: dict[str, float] | None,
        rng: np.random.Generator,
    ) -> State:
        """Apply the exogenous inputs (visible and hidden) and solve one period."""
        if self.model_ is None:
            raise RuntimeError("driver has no episode; call reset() first")
        if hidden:
            self.model_.set_values(hidden)
        return self.model_.advance(parameters, actions)
