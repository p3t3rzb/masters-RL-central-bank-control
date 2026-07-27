"""Reward functions: what the central bank in the environment is trying to do.

The objective is the one normative choice in the experiment, so it is a swappable
component rather than a constant in the environment:

* :class:`~control.rewards.base.RewardFunction` -- the base mandate, a weighted
  sum over named, individually-scaled components, with the breakdown exposed for
  logging;
* :class:`~control.rewards.base.RewardContext` -- everything one period's reward
  may condition on (realised and previous state, exogenous parameters, the
  actions applied and the ones they moved from, and ``dt``);
* :class:`~control.rewards.mandate.MandateReward` -- the first concrete mandate:
  minimise unemployment, hold growth at potential, hit the inflation target.

A new mandate is a subclass supplying ``_terms`` and ``weights``; the context
already carries what financial-stability or action-smoothing terms would need.
"""

from control.rewards.base import RewardContext, RewardFunction
from control.rewards.mandate import MandateReward

__all__ = [
    "RewardContext",
    "RewardFunction",
    "MandateReward",
]
