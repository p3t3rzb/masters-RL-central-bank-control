"""Distributional soft actor-critic, and the loop that trains it in a proxy.

* :class:`~control.dsac.agent.DSACAgent` -- the maximum-entropy actor over twin
  **quantile** critics: the critic learns the return *distribution* rather than
  its mean, which curbs value overestimation under heavy-tailed transitions and
  lets the actor optimise a risk functional of that distribution.
* :mod:`~control.dsac.risk` -- those functionals:
  :class:`~control.dsac.risk.MeanRisk` (risk-neutral, the SAC-equivalent) and
  :class:`~control.dsac.risk.CVaRRisk` (a crisis-averse central bank).
* :mod:`~control.dsac.networks` / :mod:`~control.dsac.replay` -- the squashed
  Gaussian policy, the quantile critic, and the transition buffer.
* :mod:`~control.dsac.train` -- :func:`~control.dsac.train.train`, which builds
  the world, fits the proxy, and runs the loop, plus the evaluation helpers and
  the reference policies a learned one must beat.
"""

from control.dsac.agent import DSACAgent
from control.dsac.networks import QuantileCritic, SquashedGaussianPolicy
from control.dsac.replay import Batch, ReplayBuffer
from control.dsac.risk import CVaRRisk, MeanRisk, RiskMeasure
from control.dsac.train import (
    PROXIES,
    TrainConfig,
    TrainingResult,
    build_env,
    build_proxy,
    calibration_actions,
    constant_policy,
    evaluate,
    random_policy,
    taylor_policy,
    train,
)

__all__ = [
    "DSACAgent",
    "SquashedGaussianPolicy",
    "QuantileCritic",
    "ReplayBuffer",
    "Batch",
    "RiskMeasure",
    "MeanRisk",
    "CVaRRisk",
    "TrainConfig",
    "TrainingResult",
    "train",
    "evaluate",
    "build_proxy",
    "build_env",
    "random_policy",
    "constant_policy",
    "taylor_policy",
    "calibration_actions",
    "PROXIES",
]
