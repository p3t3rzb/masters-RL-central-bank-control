"""The buffer of transitions that actually happened, kept apart from the rest.

Two buffers, never merged. ``D_model`` is the ordinary
:class:`~control.dsac.replay.ReplayBuffer` holding synthetic transitions, rolling
and disposable. ``D_real`` is this: every transition the ground-truth economy
actually produced, and it carries two things a replay buffer does not need to.

* **A branch point.** MBPO's rollouts start from *real* states, so each row
  records where the world model has to be put back to in order to continue from
  the state that transition ended in: the shadow proxy's rollout handle, the
  observer's belief, and the trailing level blocks an observation is a function
  of. Without them a "rollout from a visited state" is a rollout from wherever
  the model happened to drift to.
* **A weight.** The historic rows are real ground-truth data too and there are
  more of them than a live run will ever produce, but they came from a different
  policy and, for the residual, from a proxy that had seen them. Trusting them
  less than live rows is a knob rather than an argument.

:func:`seed_from_run` is what fills it before the live run starts. The historic
run is worth spelling out as a free win: it is roughly 500 ground-truth
transitions against the ~200 a deployment will add, every one of its rewards is
recoverable (the mandate is a pure function of consecutive states, parameters and
actions), none of them collapsed, and -- most usefully for an off-policy critic --
its actions were drawn by the excitation's own AR(1) *inside the box the agent
acts in*, which makes it broad off-policy action coverage of exactly the kind a
single on-policy live run cannot provide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from economic_models.interface import ModelInterface
from economic_models.proxy import BaseProxyModel, RolloutState
from economic_models.run import Run

from control.dsac.replay import Batch
from control.env import CentralBankEnv
from control.observation import Observer
from control.rewards import RewardContext, RewardFunction


@dataclass(frozen=True)
class BranchPoint:
    """Everything needed to restart a world model from a state that happened.

    ``handle`` puts the model back (belief, previous features, previous levels);
    ``belief`` is the *observer's* separate memory, which the rollout threads
    through :meth:`~control.observation.Observer.observe` as it goes; ``states``
    and ``exog`` are the trailing ``(2, .)`` level blocks every observation, every
    reward and the fiscal stabilizer all read.
    """

    obs: np.ndarray
    belief: Any
    handle: RolloutState
    states: np.ndarray  # (2, n_state)
    exog: np.ndarray  # (2, n_exog)


class RealBuffer:
    """Ground-truth transitions, with a branch point and a weight on each row.

    Append-only and small by construction -- a history plus one live run is under
    a thousand rows -- so it preallocates once and refuses to overwrite. A real
    transition is the scarcest thing in the whole deployment; silently evicting
    one to make room would be the wrong trade every time.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        """Preallocate room for ``capacity`` real transitions."""
        self.capacity = capacity
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros((capacity, 1), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._done = np.zeros((capacity, 1), dtype=np.float32)
        self._weight = np.ones(capacity, dtype=np.float64)
        self._branches: list[BranchPoint | None] = []
        self._size = 0

    def __len__(self) -> int:
        """The number of real transitions stored."""
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        branch: BranchPoint | None = None,
        weight: float = 1.0,
    ) -> None:
        """Store one transition, and where a rollout could continue it from.

        ``branch`` describes the state this transition *ended* in, and is ``None``
        for a collapse -- there is no plausible state to branch from, and a
        rollout that started there would be imagining a future after the end of
        the world.
        """
        if self._size >= self.capacity:
            raise RuntimeError(
                f"the real buffer is full at {self.capacity} transitions; it is "
                "sized for the history plus the live run and never evicts"
            )
        i = self._size
        self._obs[i] = obs
        self._action[i] = action
        self._reward[i] = reward
        self._next_obs[i] = next_obs
        self._done[i] = float(done)
        self._weight[i] = weight
        self._branches.append(branch)
        self._size = i + 1

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """Draw ``batch_size`` transitions, in proportion to their weights."""
        if self._size == 0:
            raise RuntimeError("the real buffer is empty")
        w = self._weight[: self._size]
        idx = rng.choice(self._size, size=batch_size, p=w / w.sum())
        return Batch(
            obs=self._obs[idx],
            action=self._action[idx],
            reward=self._reward[idx],
            next_obs=self._next_obs[idx],
            done=self._done[idx],
        )

    def branches(
        self,
        n: int,
        rng: np.random.Generator,
        *,
        decay: float = 0.99,
        half: int | None = None,
    ) -> list[BranchPoint]:
        """Draw ``n`` branch points, weighted toward recently visited states.

        Geometric in age at rate ``decay``, which is the compromise the setting
        asks for: the correction is most trustworthy near where it was just
        estimated, so rollouts should mostly start in the current regime -- but
        the history is the only broad coverage there is, and a policy improved
        only on the last few quarters is a policy that has forgotten the rest of
        the economy.

        ``half`` restricts the draw to the even (``0``) or odd (``1``) stored
        positions, which is the one held-out split a live run can afford. There
        is no second economy to evaluate a candidate in, so a policy improved on
        rollouts from *these* states and then scored on rollouts from the same
        states is being tested where it was fitted. Splitting the states by
        arrival parity gives the two an interleaved and therefore
        distributionally identical pair of pools -- a contiguous split would hand
        the test set a different stretch of the run and confound overfitting with
        drift. ``None`` draws from everything, which is the behaviour when the
        split is off.
        """
        usable = [i for i, b in enumerate(self._branches[: self._size]) if b is not None]
        if half is not None:
            usable = [i for i in usable if i % 2 == half]
        if not usable:
            raise RuntimeError(
                "the real buffer holds no branchable states"
                + ("" if half is None else f" in half {half}")
            )
        age = np.array([self._size - 1 - i for i in usable], dtype=float)
        w = decay**age * self._weight[usable]
        idx = rng.choice(len(usable), size=n, p=w / w.sum())
        branches = self._branches
        return [branches[usable[j]] for j in idx]  # type: ignore[misc]

    @property
    def rewards(self) -> np.ndarray:
        """Every stored reward, in arrival order (diagnostics)."""
        return self._reward[: self._size, 0].copy()


def seed_from_run(
    buffer: RealBuffer,
    run: Run,
    env: CentralBankEnv,
    observer: Observer,
    reward: RewardFunction,
    shadow: BaseProxyModel,
    *,
    interface: ModelInterface,
    dt: float,
    weight: float = 1.0,
) -> None:
    """Replay a historic ground-truth run into the buffer, transition by transition.

    Walks the run forward once, carrying the two memories a live step carries --
    the observer's belief and the shadow proxy's rollout state -- and records the
    same five-tuple the environment would have produced, plus a branch point at
    every state. Nothing here re-solves anything: the run already happened.

    The walk starts as soon as both memories are warm, which costs the first few
    rows and no more. It **leaves the shadow standing at the run's last row**,
    which is the branch point the live deployment starts from -- so the same
    object that filtered the history goes on to shadow the live run, and there is
    no seam where a second filter would have to catch up.

    ``weight`` is the trust placed in these rows relative to live ones, and is
    carried into every minibatch drawn from the buffer.

    A historic lever outside the agent's action box would be stored at the box's
    edge (:meth:`~control.env.CentralBankEnv.to_normalised` clips), which is a
    row whose recorded action did not produce its recorded reward. The excitation
    draws these levers inside the box, so this should not happen; it is checked
    rather than assumed, because the failure is silent and teaches the critic a
    wrong pairing.
    """
    states, params, actions = run.states, run.params, run.actions
    exog = np.hstack([params, actions])
    action_names = interface.actions.names()
    normalised = np.array(
        [env.to_normalised(dict(zip(action_names, row))) for row in actions]
    )
    # Tested on the levels rather than on the normalised rows, because a lever
    # sitting exactly *on* a bound is legitimate and normalises to the same +-1 a
    # clipped one does. Only strictly outside is a mismatch.
    box = np.array([env.config.action_bounds[name] for name in action_names])
    outside = int(
        np.sum(np.any((actions < box[:, 0]) | (actions > box[:, 1]), axis=1))
    )
    if outside:
        raise ValueError(
            f"{outside} of {len(actions)} historic rows have a lever outside the "
            "action box; they would be stored with an action that did not produce "
            "their reward. Widen EnvConfig.action_bounds or drop the rows."
        )
    # What the buffer records is the action in the *policy's* space: the step
    # the position took, in units of the period's maximum move
    # (:attr:`~control.env.EnvConfig.delta_rate`). A historic step faster than
    # that maximum stores beyond [-1, 1], and faithfully so: the linear relation
    # between step and position holds off the box too, so the pairing of action
    # and reward stays exact, and these rows are only ever *read* by the critic,
    # never emitted by the actor. Row ``j - 1`` belongs to the transition into
    # period ``j``.
    stored = np.diff(normalised, axis=0) / env.delta_step

    start = max(observer.required_window, shadow.required_window) - 1
    if len(run) < start + 2:
        raise ValueError(
            f"a {len(run)}-step run is too short to warm both memories and step"
        )
    shadow.reset(states[: start + 1], params[: start + 1], actions[: start + 1])
    belief = observer.init_belief(
        (states[: start + 1], params[: start + 1], actions[: start + 1])
    )
    obs, belief = observer.observe(
        states[start - 1 : start + 1], exog[start - 1 : start + 1], belief
    )

    for i in range(start, len(run) - 1):
        j = i + 1
        state, prev_state = (
            interface.state.from_row(states[j]),
            interface.state.from_row(states[i]),
        )
        pars, prev_pars = (
            interface.parameters.from_row(params[j]),
            interface.parameters.from_row(params[i]),
        )
        acts, prev_acts = (
            interface.actions.from_row(actions[j]),
            interface.actions.from_row(actions[i]),
        )

        r = reward(
            RewardContext(
                state=state,
                prev_state=prev_state,
                parameters=pars,
                prev_parameters=prev_pars,
                actions=acts,
                prev_actions=prev_acts,
                dt=dt,
            )
        )
        shadow.absorb(state, pars, acts)
        next_obs, next_belief = observer.observe(
            states[i : j + 1], exog[i : j + 1], belief
        )
        buffer.add(
            obs,
            stored[j - 1],
            r,
            next_obs,
            False,  # the history is required not to have collapsed
            branch=BranchPoint(
                obs=next_obs,
                belief=next_belief,
                handle=shadow.snapshot(),
                states=states[i : j + 1].astype(float),
                exog=exog[i : j + 1].astype(float),
            ),
            weight=weight,
        )
        obs, belief = next_obs, next_belief
