"""A linear (VARX) proxy of the economy, fit on excited historic runs.

A memoryless linear map on the encoder latent: the past is carried by the
encoder, so the estimator only learns how the next state-feature row depends on
that latent and the contemporaneous exogenous inputs,

    f_{t+1} = c + W_z z_t + W_u u_{t+1} + e_t,

with ``z_t`` the encoder latent and ``u_{t+1}`` the transformed ``Parameters`` +
``Actions`` of the target period (contemporaneous, as in the real model's
within-period solve). Estimation is per-equation ridge-OLS on standardized
regressors; the residual covariance is kept so rollouts can be stochastic.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from economic_models.encoders import StateEncoder
from economic_models.interface import ModelInterface
from economic_models.proxy.base import BaseProxyModel, FitData, StepContext
from economic_models.proxy.transform import StationarizingTransform


class VARXProxy(BaseProxyModel):
    """Linear ridge map from ``(latent, next exog)`` to the next feature row."""

    def __init__(
        self,
        interface: ModelInterface,
        ridge: float = 1e-6,
        *,
        encoder: StateEncoder | None = None,
        transform: StationarizingTransform | None = None,
    ) -> None:
        """Configure the ridge map for the model ``interface`` it mimics.

        ``interface`` is the ground-truth model's
        :class:`~economic_models.interface.ModelInterface`. ``ridge`` is the
        per-``n`` L2 penalty strength on the standardized regressors
        (size-invariant shrinkage). ``encoder`` / ``transform`` are the shared
        conditioning latent and feature view.
        """
        super().__init__(interface, encoder=encoder, transform=transform)
        self.ridge = ridge
        self.reg_: Pipeline | None = None  # standardize -> multi-output ridge
        self.residual_cov_: np.ndarray | None = None
        self._chol: np.ndarray | None = None

    # -- estimation ----------------------------------------------------------

    def _fit(self, data: FitData) -> None:
        """Fit the standardized ridge regression; cache the residual covariance and its Cholesky factor for stochastic draws."""
        X = self._design(data.z, data.u_next)
        Y = data.y

        # Standardize the regressors so the L2 penalty is scale-invariant, then
        # ridge-regress each state feature. The penalty scales with ``n`` so the
        # effective shrinkage is invariant to the training-set size.
        self.reg_ = make_pipeline(
            StandardScaler(), Ridge(alpha=self.ridge * len(X))
        ).fit(X, Y)

        residuals = Y - self.reg_.predict(X)
        self.residual_cov_ = np.cov(residuals.T)
        jitter = 1e-12 * np.trace(self.residual_cov_) / len(self.residual_cov_)
        self._chol = np.linalg.cholesky(
            self.residual_cov_ + jitter * np.eye(len(self.residual_cov_))
        )

    # -- inference -----------------------------------------------------------

    def _predict_batch(
        self, z: np.ndarray, u_next: np.ndarray, f_prev: np.ndarray
    ) -> np.ndarray:
        """Ridge conditional-mean predictions for the design rows."""
        return self.reg_.predict(self._design(z, u_next))

    def _sample_step(
        self, ctx: StepContext, rng: np.random.Generator
    ) -> np.ndarray:
        """One draw: the ridge mean plus correlated residual noise."""
        mean = self.reg_.predict(self._design(ctx.z, ctx.u_next)[None, :])[0]
        return mean + self._chol @ rng.standard_normal(len(mean))
