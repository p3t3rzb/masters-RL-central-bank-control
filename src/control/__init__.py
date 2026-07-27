"""Reinforcement-learning control of the central bank inside a proxy economy.

The models package gives two ways to advance an economy one period -- the
structural ground truth and a learned proxy -- behind one interface. This package
puts a *policy* on the action side of that interface and trains it:

* :mod:`~control.world` -- the setting: one simulated history (the only data the
  proxy is fit on) and the bank of exogenous futures branching off its end, plus
  the fiscal stabilizer a frozen future cannot carry.
* :mod:`~control.observation` -- what the policy sees: the proxy's own stationary
  feature view, standardised with statistics frozen on the history.
* :mod:`~control.drivers` -- how an episode is run against a proxy or against the
  ground truth, behind one contract.
* :mod:`~control.env` -- :class:`~control.env.CentralBankEnv`, the episodic
  environment: the action box, the reward, and what counts as a collapse.
* :mod:`~control.rewards` -- the mandate, as a swappable weighted sum of named
  terms.
* :mod:`~control.dsac` -- the distributional soft actor-critic and its training
  loop.

Train from the command line with ``uv run python -m control``.
"""

from control.drivers import GroundTruthDriver, ModelDriver, ProxyDriver
from control.env import ACTION_BOUNDS, CentralBankEnv, EnvConfig, Transition
from control.observation import Observer
from control.rewards import MandateReward, RewardContext, RewardFunction
from control.world import (
    Episode,
    EpisodeBank,
    FiscalStabilizer,
    NullStabilizer,
    TrainingWorld,
    WorldConfig,
    build_world,
)

__all__ = [
    "WorldConfig",
    "TrainingWorld",
    "build_world",
    "Episode",
    "EpisodeBank",
    "FiscalStabilizer",
    "NullStabilizer",
    "Observer",
    "ModelDriver",
    "ProxyDriver",
    "GroundTruthDriver",
    "CentralBankEnv",
    "EnvConfig",
    "Transition",
    "ACTION_BOUNDS",
    "RewardFunction",
    "RewardContext",
    "MandateReward",
]
