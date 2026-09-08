"""A linear (VARX) proxy of the economy, fit on excited historic runs.

A memoryless linear map on the encoder latent: the past is carried by the
encoder, so the estimator only learns how the next state-feature row depends on
that latent and the contemporaneous exogenous inputs,

    f_{t+1} = c + W_z z_t + W_u u_{t+1} + e_t,

with ``z_t`` the encoder latent and ``u_{t+1}`` the transformed ``Parameters`` +
``Actions`` of the target period (contemporaneous, as in the real model's
within-period solve). Estimation is per-equation ridge-OLS on standardized
regressors; the residual covariance is kept so rollouts can be stochastic.

**Two different uncertainties, and only one of them is free.** The residual
covariance is *aleatoric*: the irreducible noise of the law that was fitted. It is
one constant matrix, so a draw taken where the history has no data at all is
exactly as tight as one taken in the middle of it -- which is why a stochastic
rollout, on its own, tells an agent nothing about having left the data behind.
What grows off-support is the *epistemic* part, the uncertainty about the
coefficients themselves, and for a ridge it is closed-form rather than something
an ensemble has to approximate: the Bayesian linear model's predictive variance
is ``sigma^2 (1 + h)`` with

    h = x' (X'X + alpha I)^-1 x

the query's leverage against the design it was fitted on. ``h`` is small inside
the data and grows quadratically as a regressor leaves it -- on a GROWTH history
it runs from about 0.06 at the centre of the lever range to 2.8 at the edge of the
action box, so the draw fans out by up to a factor of two exactly where a policy
has gone hunting for a region the proxy cannot vouch for. :attr:`epistemic`
switches it on; see :meth:`VARXProxy._sample_step`.
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
        epistemic: bool = False,
    ) -> None:
        """Configure the ridge map for the model ``interface`` it mimics.

        ``interface`` is the ground-truth model's
        :class:`~economic_models.interface.ModelInterface`. ``ridge`` is the
        per-``n`` L2 penalty strength on the standardized regressors
        (size-invariant shrinkage). ``encoder`` / ``transform`` are the shared
        conditioning latent and feature view.

        ``epistemic`` widens a draw by the ridge's own parameter uncertainty (see
        the module docstring and :meth:`_sample_step`). Off by default: it changes
        the conditional law every rollout samples from, so it is a property of the
        experiment rather than a free improvement, and leaving it off reproduces
        every result taken before it existed.
        """
        super().__init__(interface, encoder=encoder, transform=transform)
        self.ridge = ridge
        self.epistemic = epistemic
        self.reg_: Pipeline | None = None  # standardize -> multi-output ridge
        self.residual_cov_: np.ndarray | None = None
        self._chol: np.ndarray | None = None
        #: ``(X'X + alpha I)^-1`` over the standardized, centred design, and the
        #: centre it was taken about: everything :meth:`_leverage` needs.
        self.precision_: np.ndarray | None = None
        self._centre: np.ndarray | None = None

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

        # The design as the penalized normal equations actually saw it: scaled by
        # the pipeline's scaler and centred, since ``Ridge`` fits its intercept by
        # centring rather than by penalizing one. Inverting a (p, p) matrix once at
        # fit time is what makes the leverage a quadratic form per step.
        design = self._standardized(X)
        self._centre = design.mean(axis=0)
        centred = design - self._centre
        self.precision_ = np.linalg.inv(
            centred.T @ centred + self.reg_[-1].alpha * np.eye(centred.shape[1])
        )

    def _standardized(self, X: np.ndarray) -> np.ndarray:
        """The design rows as the fitted ridge sees them, scaler applied."""
        return self.reg_[0].transform(np.atleast_2d(X))

    def _leverage(self, x: np.ndarray) -> float:
        """``h = x' (X'X + alpha I)^-1 x``: how far ``x`` is off the design.

        Zero-ish inside the data the coefficients were estimated from, growing
        quadratically as a regressor leaves it. Uncertainty about the *map*, which
        is the part a constant residual covariance cannot express and the part
        that matters when a policy steers somewhere the history never went.
        """
        centred = self._standardized(x)[0] - self._centre
        return float(centred @ self.precision_ @ centred)

    # -- inference -----------------------------------------------------------

    def _predict_batch(
        self, z: np.ndarray, u_next: np.ndarray, f_prev: np.ndarray
    ) -> np.ndarray:
        """Ridge conditional-mean predictions for the design rows."""
        return self.reg_.predict(self._design(z, u_next))

    def _sample_step(
        self, ctx: StepContext, rng: np.random.Generator
    ) -> np.ndarray:
        """One draw: the ridge mean plus correlated residual noise.

        Under :attr:`epistemic` the noise is scaled by ``sqrt(1 + h)``, the
        Bayesian linear model's predictive standard deviation relative to its
        residual one -- so the law widens smoothly as the query leaves the design,
        and a rollout that walks off-support is drawn from something visibly
        less certain rather than from the same tight Gaussian as everywhere else.
        The scaling is on the whole residual covariance, which is exact here: one
        design serves every output equation, so every output's predictive variance
        is inflated by the same factor.

        It costs one quadratic form in ``p`` per draw and no extra fit. What it
        does **not** capture is misspecification -- a linear map can be confidently
        wrong off-support in a way its own parameter posterior never sees -- so
        read it as a lower bound on how little the model knows out there, not as
        the whole of it.
        """
        x = self._design(ctx.z, ctx.u_next)
        mean = self.reg_.predict(x[None, :])[0]
        noise = self._chol @ rng.standard_normal(len(mean))
        if not self.epistemic:
            return mean + noise
        return mean + np.sqrt(1.0 + self._leverage(x)) * noise
