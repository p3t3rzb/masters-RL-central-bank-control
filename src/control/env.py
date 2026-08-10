"""The central-bank environment: one episode is one exogenous future.

An episode starts at the end of the history, runs for the horizon of one drawn
future, and asks the agent for ``(Rbbar, NCAR, ro)`` every period. The
environment owns four things the model itself does not: the action box, the
observation, the reward, and what counts as a collapse.

The API is gym-shaped (``reset`` / ``step`` returning
``(obs, reward, terminated, truncated, info)``) without depending on Gymnasium --
the training loop here is the only consumer, and the contract is three lines.

**The collapse penalty is not decoration.** Every reward is a penalty, so
``<= 0``; if terminating merely stopped the accumulation, the highest-return
policy available would be to crash the economy on the first step. A terminal
state must therefore be strictly worse than surviving badly -- and worse by an
amount that does not depend on *when* it happens, or collapsing late becomes a
bargain. :attr:`EnvConfig.collapse_penalty` is therefore charged **per remaining
step** of the episode: a collapse is the economy spending the rest of its horizon
at a floor far below anything a surviving policy can reach.

A step is therefore split in two: :meth:`CentralBankEnv._transition` *draws* one,
leaving the episode where it was, and :meth:`CentralBankEnv._commit` moves the
episode onto it. :meth:`CentralBankEnv.step` does both, and
:meth:`CentralBankEnv.branch_step` draws several from the one position and
commits one of them -- so branch draws and ordinary steps cannot end up scored by
two different copies of the reward and collapse rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from pysolve.model import CalculationError, SolutionNotFoundError

from economic_models.interface import ModelInterface
from economic_models.variables import Actions, Parameters, State

from control.drivers import ModelDriver
from control.observation import Observer
from control.rewards import RewardContext, RewardFunction
from control.world import Episode, EpisodeBank

#: The action box, defaulting to the AR(1) clip bands the excitation drew these
#: levers inside -- and therefore the region a proxy was fit in. Widening it puts
#: the agent off the surrogate's training support, which needs an OOD guard, not
#: just a bigger number.
ACTION_BOUNDS: Mapping[str, tuple[float, float]] = {
    "Rbbar": (0.015, 0.055),
    "NCAR": (0.07, 0.14),
    "ro": (0.03, 0.09),
}


@dataclass(frozen=True)
class EnvConfig:
    """The environment's non-model knobs.

    ``collapse_penalty`` is charged **per remaining step** when the economy
    collapses (see the module docstring): it must be well below the per-step
    reward any surviving policy earns, or ending the episode early becomes the
    best move available. ``er_bounds`` and ``pi_bounds`` are the plausibility
    corridor outside which the economy counts as collapsed; ``action_bounds`` is
    the box the agent's normalised action is mapped onto; ``stabilize``
    re-applies GROWTH's fiscal response to the realised employment rate.

    ``horizon`` caps an episode at that many steps of whatever future it drew,
    and ``None`` runs the whole thing. It exists because the length of a *future*
    and the length of an *episode* are separable questions: a deployment
    (:mod:`control.live`) runs one future for hundreds of periods where training
    cut the same bank into episodes of fifty, and both read the same world.

    ``action_rate`` caps how fast a lever may move, as a fraction of its box **per
    year**; ``None`` lets the agent swing it end to end between periods. It lives
    **here**, in the environment, rather than being applied to the agent's output
    by whoever happens to be running it. Real policy instruments have a slew rate
    and the mandate has no smoothing term, so without it the agent is free to do
    something no central bank does -- but a limit imposed outside the environment
    is one the agent never learns, never trains against and is never evaluated
    under, so the policy that gets optimised is not the policy that gets run. As
    a property of the *instrument* it is part of the world, for the same reason
    :class:`~control.world.FiscalStabilizer` is, and training, evaluation,
    synthetic rollouts and a live deployment all read it from one place.

    Annualized, like every other rate in this project (the excitation's AR(1)
    persistences and variances, the mandate's growth rates), and for the reason
    that convention exists: a *per-step* cap is a different economic statement at
    every ``dt``. At 0.1 per step a lever crosses 40% of its box a year at
    quarterly and 120% at monthly, so a number tuned at one frequency silently
    means something else at another -- the same trap
    :func:`~control.dsac.train.taylor_policy` documents for its own gain. Stated
    per year it means one thing everywhere, and the environment converts.

    It costs no extra episode state: the previous lever setting is already in the
    trailing exogenous block, which is also why a rate-limited environment stays
    Markov in its own observation (the action columns are part of it).
    """

    collapse_penalty: float = -25.0
    horizon: int | None = None
    action_rate: float | None = None
    er_bounds: tuple[float, float] = (0.5, 1.5)
    pi_bounds: tuple[float, float] = (-0.2, 0.5)
    action_bounds: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: dict(ACTION_BOUNDS)
    )
    stabilize: bool = True


@dataclass(frozen=True)
class Transition:
    """One drawn step, before the episode has been moved onto it.

    Carries both halves: what a gym-shaped caller is handed (:meth:`as_step`) and
    the trailing level blocks committing it would install. Several transitions can
    be drawn from one position, but only the committed one advances the episode.
    """

    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    states: np.ndarray  # (2, n_state) trailing levels this draw leaves behind
    exog: np.ndarray  # (2, n_exog) trailing exogenous levels
    #: the observer's encoder belief after this draw. It rides here for the same
    #: reason the level blocks do: the observation is a pure function of them, so
    #: a draw that is never committed must not leave its memory behind.
    belief: Any

    def as_step(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """The gym-shaped ``(obs, reward, terminated, truncated, info)`` tuple."""
        return self.obs, self.reward, self.terminated, self.truncated, self.info


class CentralBankEnv:
    """An episodic environment in which a policy sets the central bank's levers.

    Wraps a :class:`~control.drivers.ModelDriver` (a proxy for training, the
    ground truth for evaluation), a bank of exogenous futures, a mandate and an
    observer. The agent's action is a normalised ``[-1, 1]`` vector, mapped
    affinely onto :attr:`EnvConfig.action_bounds`.
    """

    def __init__(
        self,
        driver: ModelDriver,
        episodes: EpisodeBank,
        reward: RewardFunction,
        observer: Observer,
        interface: ModelInterface,
        config: EnvConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        """Wire the environment together.

        ``driver`` runs the model, ``episodes`` is the bank of exogenous futures
        to draw from, ``reward`` is the mandate, ``observer`` builds the policy's
        view, ``interface`` supplies the value spaces, and ``config`` holds the
        action box, the collapse rules and the stabilizer switch. ``seed`` fixes
        both the episode draws and the proxy's transition noise.
        """
        self.driver = driver
        self.episodes = episodes
        self.reward = reward
        self.observer = observer
        self.interface = interface
        self.config = config or EnvConfig()
        self._rng = np.random.default_rng(seed)
        self._world = driver.world

        self._param_names = interface.parameters.names()
        self._action_names = interface.actions.names()
        # The columns the stationarizing view log-differences: a non-positive
        # level there is not merely implausible, it is unrepresentable.
        state_names = interface.state.names()
        self._positive = np.array(
            [name in interface.transform_spec.log_diff for name in state_names]
        )
        bounds = np.array(
            [self.config.action_bounds[name] for name in self._action_names], dtype=float
        )
        self._low, self._high = bounds[:, 0], bounds[:, 1]

        # Episode state, populated by :meth:`reset`.
        self._episode: Episode | None = None
        self._t = 0
        self._states: np.ndarray | None = None  # (2, n_state) trailing levels
        self._exog: np.ndarray | None = None  # (2, n_exog) trailing levels
        self._obs: np.ndarray | None = None
        self._belief: Any = None  # the observer's encoder belief

    # -- shape ---------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        """Width of the observation vector."""
        return self.observer.dim

    @property
    def action_dim(self) -> int:
        """Number of policy levers."""
        return len(self._action_names)

    @property
    def horizon(self) -> int:
        """Steps in the current episode: its future's length, or the cap."""
        if self._episode is None:
            return 0
        if self.config.horizon is None:
            return len(self._episode)
        return min(len(self._episode), self.config.horizon)

    # -- where the episode currently stands ----------------------------------
    #
    # Read-only, and for one caller: something that wants to branch a *different*
    # model off the position this episode has reached (see
    # :mod:`control.live.deploy`, which rolls a corrected proxy forward from the
    # states a live ground-truth run visited). An observation is a pure function
    # of these three, so handing them out lets that caller reproduce this
    # environment's own bookkeeping instead of re-deriving it and drifting.

    @property
    def states(self) -> np.ndarray | None:
        """The trailing ``(2, n_state)`` block of state levels, or ``None``."""
        return self._states

    @property
    def exog(self) -> np.ndarray | None:
        """The trailing ``(2, n_exog)`` block of exogenous levels, or ``None``."""
        return self._exog

    @property
    def belief(self) -> Any:
        """The observer's encoder belief behind the latest observation."""
        return self._belief

    @property
    def step_index(self) -> int:
        """Steps taken in the current episode."""
        return self._t

    def to_actions(self, action: np.ndarray) -> Actions:
        """Map a normalised ``[-1, 1]`` vector onto the economic action box."""
        unit = (np.clip(np.asarray(action, dtype=float), -1.0, 1.0) + 1.0) / 2.0
        levels = self._low + unit * (self._high - self._low)
        return self.interface.actions.from_dict(dict(zip(self._action_names, levels)))

    def apply_action_rate(
        self, action: np.ndarray, prev_exog: np.ndarray
    ) -> np.ndarray:
        """The normalised action after the instrument's slew limit.

        ``prev_exog`` is the exogenous level row the move is measured from -- the
        trailing one of an episode, or a branch point's, so a caller simulating
        this environment's dynamics with another model limits by the *same* rule
        against the *same* reference rather than a re-stated copy of it.

        The two factors: the normalised box is two units wide, and the configured
        rate is per year while a step is ``dt`` of one.

        A no-op when :attr:`EnvConfig.action_rate` is ``None``, beyond the clip
        into the box that :meth:`to_actions` would apply anyway.
        """
        raw = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.config.action_rate is None:
            return raw
        previous = self.to_normalised(self._actions_of(prev_exog).to_dict())
        step = 2.0 * self.config.action_rate * self._world.dt
        return np.clip(raw, previous - step, previous + step)

    def to_normalised(self, levels: Mapping[str, float]) -> np.ndarray:
        """Map economic action levels back onto the normalised ``[-1, 1]`` vector.

        The inverse of :meth:`to_actions`, for policies stated in economic units
        (a constant baseline, a Taylor rule). Levels outside the box are clipped
        to it, which is how the action bounds bind on such a rule.
        """
        values = np.array([levels[name] for name in self._action_names], dtype=float)
        unit = (values - self._low) / (self._high - self._low)
        return np.clip(2.0 * unit - 1.0, -1.0, 1.0)

    # -- episode -------------------------------------------------------------

    def reset(self, *, seed: int | None = None, episode: Episode | None = None
              ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode at the branch point under a drawn future.

        ``episode`` forces a particular future (deterministic evaluation);
        otherwise one is drawn from the bank. ``seed`` re-seeds the environment's
        generator, which drives both the draw and the proxy's transition noise.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._episode = episode if episode is not None else self.episodes.draw(self._rng)
        self._t = 0

        states, exog = self.driver.reset()
        self._states, self._exog = states, exog
        # Every episode branches from the same point, so the memory it starts
        # from is the same object every time: the whole history filtered once at
        # fit time. It stops one period short of the branch, which this first
        # observation then folds in -- so reset and step each advance the belief
        # by exactly one period.
        self._belief = self.observer.branch_belief_
        self._obs, self._belief = self.observer.observe(states, exog, self._belief)
        return self._obs, {"episode": self._episode.index}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one period of policy and advance the economy."""
        transition = self._transition(action)
        self._commit(transition)
        return transition.as_step()

    def branch_step(self, action: np.ndarray, k: int = 1) -> list[Transition]:
        """Draw ``k`` transitions from this one position; continue along the last.

        Every draw applies the *same* action to the *same* state under the same
        exogenous row and from the same observer memory, and differs only in the
        model's transition noise -- so the ``k`` returned transitions are
        independent samples of one conditional law, rather than samples of ``k``
        different states as re-running an episode would give. (The memory has to
        be restored along with the model: an observer that advanced its belief in
        place would hand draw ``j + 1`` a memory containing draw ``j``, and the
        draws would stop being exchangeable. It cannot, because a belief is passed
        through :meth:`~control.observation.Observer.observe` rather than held.) That is the point: a replay buffer holding all ``k`` learns the
        spread of the kernel *at that state*, which single-sample targets can only
        recover by smoothing across neighbours -- thinnest exactly in the tails a
        risk-averse critic is fitted on.

        The last draw is the one committed, and the driver is left standing in it.
        Which draw continues is arbitrary (they are exchangeable), so taking the
        last costs no extra bookkeeping and keeps the trajectory an honest sample
        of the same law an unbranched rollout would follow.

        Requires a :attr:`~control.drivers.ModelDriver.branchable` driver unless
        ``k == 1``, which is exactly :meth:`step` and snapshots nothing.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if k == 1:
            transition = self._transition(action)
            self._commit(transition)
            return [transition]
        if not self.driver.branchable:
            raise RuntimeError(
                f"{type(self.driver).__name__} cannot branch; branch_step needs k=1"
            )

        start = self.driver.snapshot()
        draws = []
        for j in range(k):
            if j:
                self.driver.restore(start)
            draws.append(self._transition(action))
        self._commit(draws[-1])
        return draws

    # -- one step, in two halves ----------------------------------------------

    def _transition(self, action: np.ndarray) -> Transition:
        """Draw one step without moving the episode onto it.

        Advances the *model* (and consumes the environment's noise stream), but
        leaves the episode's clock and trailing blocks alone, so the caller may
        draw again from the same position after restoring the driver.
        """
        if self._episode is None or self._states is None or self._exog is None:
            raise RuntimeError("environment has no episode; call reset() first")

        prev_levels = self._states[-1]
        prev_exog = self._exog[-1]
        prev_state = self._state_of(prev_levels)
        prev_params = self._params_of(prev_exog)
        prev_actions = self._actions_of(prev_exog)

        params = self._exog_row(self._t, prev_state.ER)
        parameters = self.interface.parameters.from_dict(
            dict(zip(self._param_names, params))
        )
        # The slew limit is measured from the levers as they actually stand, which
        # the trailing exogenous row already records -- so every draw from this
        # position limits against the same reference, and nothing has to be
        # carried across a branch.
        applied = self.apply_action_rate(action, prev_exog)
        actions = self.to_actions(applied)
        hidden = self._hidden_row(self._t)

        try:
            state = self.driver.step(parameters, actions, hidden, self._rng)
        except (ValueError, CalculationError, SolutionNotFoundError, RuntimeError) as exc:
            return self._collapse(str(exc))

        levels = self.levels_of(state)
        if not self._plausible(state, levels):
            return self._collapse("state left the plausibility corridor")

        ctx = RewardContext(
            state=state,
            prev_state=prev_state,
            parameters=parameters,
            prev_parameters=prev_params,
            actions=actions,
            prev_actions=prev_actions,
            dt=self._world.dt,
        )
        terms = self.reward.terms(ctx)
        reward = float(sum(self.reward.weights[k] * v for k, v in terms.items()))

        exog_now = np.hstack([params, [getattr(actions, n) for n in self._action_names]])
        states = np.vstack([self._states[-1], levels])
        exog = np.vstack([self._exog[-1], exog_now])
        try:
            # Reads the episode's belief and returns a fresh one rather than
            # advancing it in place, so redrawing from this position starts from
            # the same memory each time.
            obs, belief = self.observer.observe(states, exog, self._belief)
        except ValueError as exc:
            return self._collapse(str(exc))

        return Transition(
            obs=obs,
            reward=reward,
            terminated=False,
            truncated=self._t + 1 >= self.horizon,
            info={
                "reward_terms": terms,
                "state": state,
                # The exogenous row *as applied*, stabilizer included -- which is
                # not the episode's frozen row, and is what a caller shadowing
                # this step with its own model has to be driven by.
                "parameters": parameters,
                "actions": actions,
                # The normalised action *as applied*, slew limit included, which
                # is not necessarily the one the caller asked for. A replay
                # buffer has to store this one: the reward and the next state are
                # what this action produced, and pairing them with a request the
                # environment declined teaches the critic a transition that never
                # happened.
                "action": applied,
                "episode": self._episode.index,
            },
            states=states,
            exog=exog,
            belief=belief,
        )

    def _commit(self, transition: Transition) -> None:
        """Move the episode onto a drawn ``transition``: its blocks, one tick on."""
        self._states = transition.states
        self._exog = transition.exog
        self._obs = transition.obs
        self._belief = transition.belief
        self._t += 1

    # -- the collapse rules, addressable from outside a step ------------------

    def plausible(self, state: State) -> bool:
        """Whether ``state`` is an economy this environment would keep running.

        The corridor :meth:`step` applies, exposed so a caller simulating this
        environment's dynamics with another model (a corrected proxy, say) marks
        a collapse by the *same* rule rather than a re-stated copy of it.
        """
        return self._plausible(state, self.levels_of(state))

    def levels_of(self, state: State) -> np.ndarray:
        """A row of state levels from a :class:`State`, in the interface's order."""
        values = state.to_dict()
        return np.array(
            [values[name] for name in self.interface.state.names()], dtype=float
        )

    # -- internals -----------------------------------------------------------

    def _exog_row(self, t: int, er_prev: float) -> np.ndarray:
        """The episode's parameter row for step ``t``, with the fiscal response.

        A frozen future records government spending drawn at full employment; the
        stabilizer puts the countercyclical response to the realised employment
        rate back (see :class:`~control.world.FiscalStabilizer`).
        """
        assert self._episode is not None
        row = self._episode.params[t]
        if not self.config.stabilize:
            return row.astype(float)
        return self._world.stabilizer.apply(row, er_prev)

    def _hidden_row(self, t: int) -> dict[str, float] | None:
        """The episode's hidden structural parameters for step ``t``, if recorded."""
        assert self._episode is not None
        if self._episode.hidden is None:
            return None
        return dict(zip(self._world.hidden_names, self._episode.hidden[t]))

    def _plausible(self, state: State, levels: np.ndarray) -> bool:
        """Whether the realised state is an economy the model could have produced."""
        if not np.all(np.isfinite(levels)):
            return False
        if not self.config.er_bounds[0] <= state.ER <= self.config.er_bounds[1]:
            return False
        if not self.config.pi_bounds[0] <= state.PI <= self.config.pi_bounds[1]:
            return False
        return bool(np.all(levels[self._positive] > 0.0))

    def _collapse(self, reason: str) -> Transition:
        """End the episode in collapse: the terminal penalty and a reason.

        The penalty covers every step the episode would still have run, so a
        collapse costs the same wherever it happens and is never a way to stop
        accumulating penalties early. A collapsed draw leaves the trailing blocks
        where they were -- there is no plausible state to move onto -- so its
        ``obs`` is the one the step started from, and its belief the memory behind
        that observation rather than one folding in a period that never happened.
        """
        assert self._obs is not None and self._episode is not None
        assert self._states is not None and self._exog is not None
        remaining = max(1, self.horizon - self._t)
        return Transition(
            obs=self._obs,
            reward=self.config.collapse_penalty * remaining,
            terminated=True,
            truncated=False,
            info={
                "reward_terms": {},
                "collapse": reason,
                "episode": self._episode.index,
            },
            states=self._states,
            exog=self._exog,
            belief=self._belief,
        )

    def _state_of(self, levels: np.ndarray) -> State:
        """A :class:`State` from a row of state levels."""
        return self.interface.state.from_dict(
            dict(zip(self.interface.state.names(), levels))
        )

    def _params_of(self, exog: np.ndarray) -> Parameters:
        """The :class:`Parameters` half of a row of exogenous levels."""
        return self.interface.parameters.from_dict(
            dict(zip(self._param_names, exog[: len(self._param_names)]))
        )

    def _actions_of(self, exog: np.ndarray) -> Actions:
        """The :class:`Actions` half of a row of exogenous levels."""
        return self.interface.actions.from_dict(
            dict(zip(self._action_names, exog[len(self._param_names) :]))
        )
