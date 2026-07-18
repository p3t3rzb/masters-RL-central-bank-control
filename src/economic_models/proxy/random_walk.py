"""A random-walk proxy of the economy -- the naive baseline every proxy must beat.

Models the feature dynamics as a driftless random walk ``f_{t+1} = f_t + e_t``,
so the one-step prediction is just the current feature row. Fitting estimates
only the covariance of the one-step changes ``e_t``, pooled over the training
runs, so a stochastic rollout can perturb the walk with correlated noise of the
right size:

* the **conditional mean** ``f_t`` drives the deterministic rollout (pure
  persistence in feature space);
* a **stochastic draw** adds ``L z`` with ``L`` the Cholesky factor of the change
  covariance, preserving the contemporaneous correlation across state dimensions.

Because it needs no summary of the past, this is the one **encoder-free** proxy:
the constructor hard-wires the zero-width
:class:`~economic_models.encoders.base.NullEncoder`. Since trending aggregates
enter as log-differences, persistence there is a random walk *with drift* in
level space -- the appropriate naive benchmark for trending series.
"""

from __future__ import annotations

import numpy as np

from economic_models.encoders import NullEncoder
from economic_models.proxy.base import BaseProxyModel, FitData, StepContext
from economic_models.proxy.transform import StationarizingTransform


class RandomWalkProxy(BaseProxyModel):
    """Driftless random walk in a transform's feature space (naive baseline)."""

    def __init__(self, *, transform: StationarizingTransform | None = None) -> None:
        """Configure the baseline.

        Takes only ``transform`` (the shared feature view); the encoder is
        hard-wired to the zero-width :class:`NullEncoder`, since a persistence
        forecast conditions on no summary of the past.
        """
        # No ``encoder`` parameter by design: a persistence forecast needs no
        # summary of the past, so the null (zero-width) encoder is hard-wired.
        super().__init__(encoder=NullEncoder(), transform=transform)
        self.change_cov_: np.ndarray | None = None  # cov of one-step feature changes
        self._chol: np.ndarray | None = None

    # -- estimation ----------------------------------------------------------

    def _fit(self, data: FitData) -> None:
        """Estimate the one-step feature-change covariance and its Cholesky factor for stochastic draws."""
        # ``y - f_prev`` are exactly the within-run one-step feature changes.
        changes = data.y - data.f_prev
        self.change_cov_ = np.cov(changes.T)
        jitter = 1e-12 * np.trace(self.change_cov_) / len(self.change_cov_)
        self._chol = np.linalg.cholesky(
            self.change_cov_ + jitter * np.eye(len(self.change_cov_))
        )

    # -- inference -----------------------------------------------------------

    def _predict_batch(
        self, z: np.ndarray, u_next: np.ndarray, f_prev: np.ndarray
    ) -> np.ndarray:
        """Persistence: predict the current feature row unchanged."""
        return f_prev  # persistence: predict the current feature row

    def _sample_step(
        self, ctx: StepContext, rng: np.random.Generator
    ) -> np.ndarray:
        """One draw: the current row plus correlated feature-change noise."""
        return ctx.f_prev + self._chol @ rng.standard_normal(len(ctx.f_prev))
