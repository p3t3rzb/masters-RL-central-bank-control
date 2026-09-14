"""Deploying the trained agent in the ground truth, for one unrepeatable run.

The offline phase ends with an agent that is optimal with respect to a *proxy's*
conditional law. This package is what happens next: the agent is placed in the
structural economy, gets one pass through it with no reset and no branching, and
spends that pass learning how the proxy is wrong rather than trying to relearn a
policy from two hundred samples.

* :mod:`~control.live.residual` -- the correction itself: a recursive Bayesian
  linear map plus a fast bias state, seeded on **cross-fitted** historic
  residuals, with a calibrated predictive spread. :class:`~control.live.residual.NullResidual`
  is the null object that recovers the uncorrected proxy exactly.
* :mod:`~control.live.corrected` -- :class:`~control.live.corrected.CorrectedProxy`,
  which is proxy-plus-correction behind the proxy interface, so every existing
  driver, environment and rollout path takes it unchanged.
* :mod:`~control.live.forcing` -- the exogenous forecast a synthetic rollout has
  to be driven by, since the future forcing has not happened yet.
* :mod:`~control.live.buffer` -- the buffer of transitions that really happened,
  kept apart from the synthetic one and carrying a branch point on every row.
* :mod:`~control.live.monitor` -- the out-of-distribution tripwire that freezes
  learning when the economy leaves the region the whole apparatus was calibrated
  in.
* :mod:`~control.live.oracle` -- the optimal run: a genetic search over the
  bank's own lever paths with the ground truth in hand, the lab-only bound the
  figures can score every policy against.
* :mod:`~control.live.deploy` -- the loop itself, and :func:`~control.live.deploy.rehearse`,
  which replays it on held-out futures to turn one-shot deployment into a
  distribution that can be tuned against.

The design and its justification are written up in ``docs/live_deployment.md``.
Run a rehearsal from the command line with
``uv run python scripts/deploy_agent.py``.
"""

from control.live.buffer import BranchPoint, RealBuffer, seed_from_run
from control.live.corrected import CorrectedProxy, CorrectedRolloutState
from control.live.deploy import (
    TAYLOR_GAINS,
    FORCINGS,
    ONLINE_POLICIES,
    LiveConfig,
    RehearsalResult,
    RunRecord,
    SyntheticRollouts,
    build_forcing,
    build_residual,
    deploy,
    rehearse,
    residual_seed,
    run_reference,
)
from control.live.forcing import (
    BlockBootstrapForcing,
    ForcingModel,
    OracleForcing,
    VARForcing,
)
from control.live.monitor import MonitorReading, OODMonitor
from control.live.oracle import OracleConfig, optimal_run
from control.live.residual import (
    BlockBootstrapResidualNoise,
    GaussianResidualNoise,
    NullResidual,
    Residual,
    ResidualModel,
    ResidualNoise,
    action_columns,
    cross_fitted_residuals,
    design,
    in_sample_residuals,
)

__all__ = [
    "TAYLOR_GAINS",
    "Residual",
    "ResidualModel",
    "NullResidual",
    "ResidualNoise",
    "GaussianResidualNoise",
    "BlockBootstrapResidualNoise",
    "cross_fitted_residuals",
    "in_sample_residuals",
    "design",
    "CorrectedProxy",
    "CorrectedRolloutState",
    "ForcingModel",
    "VARForcing",
    "BlockBootstrapForcing",
    "OracleForcing",
    "RealBuffer",
    "BranchPoint",
    "seed_from_run",
    "OODMonitor",
    "MonitorReading",
    "OracleConfig",
    "optimal_run",
    "LiveConfig",
    "RunRecord",
    "RehearsalResult",
    "SyntheticRollouts",
    "deploy",
    "rehearse",
    "run_reference",
    "residual_seed",
    "action_columns",
    "build_residual",
    "build_forcing",
    "FORCINGS",
    "ONLINE_POLICIES",
]
