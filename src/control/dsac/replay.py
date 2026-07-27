"""The off-policy replay buffer: a preallocated ring of transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Batch:
    """One sampled minibatch of transitions, aligned row for row."""

    obs: np.ndarray  # (n, obs_dim)
    action: np.ndarray  # (n, action_dim) normalised to [-1, 1]
    reward: np.ndarray  # (n, 1)
    next_obs: np.ndarray  # (n, obs_dim)
    done: np.ndarray  # (n, 1) 1.0 only on a *terminal* state, never on truncation


class ReplayBuffer:
    """A fixed-capacity ring buffer of transitions, sampled uniformly.

    ``done`` marks genuine termination (a collapsed economy) and never mere
    truncation at the horizon: a truncated episode's value must still bootstrap,
    or the agent learns that the world ends every horizon steps.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        """Preallocate storage for ``capacity`` transitions."""
        self.capacity = capacity
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros((capacity, 1), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._done = np.zeros((capacity, 1), dtype=np.float32)
        self._cursor = 0
        self._size = 0

    def __len__(self) -> int:
        """The number of transitions currently stored."""
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition, overwriting the oldest when full."""
        i = self._cursor
        self._obs[i] = obs
        self._action[i] = action
        self._reward[i] = reward
        self._next_obs[i] = next_obs
        self._done[i] = float(done)
        self._cursor = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """Draw ``batch_size`` transitions uniformly with replacement."""
        idx = rng.integers(self._size, size=batch_size)
        return Batch(
            obs=self._obs[idx],
            action=self._action[idx],
            reward=self._reward[idx],
            next_obs=self._next_obs[idx],
            done=self._done[idx],
        )
