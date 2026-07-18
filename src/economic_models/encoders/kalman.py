"""A linear-Gaussian state-space encoder fit by EM, filtered by Kalman.

Learns a latent linear-Gaussian system and hands each proxy its filtered state
as the conditioning latent. In column-standardized feature space the model is

    z_t = A z_{t-1} + B u_t + w_t,   w_t ~ N(0, Q)      (latent transition)
    f_t = C z_t + v_t,               v_t ~ N(0, R)      (observation)

with ``z_1 ~ N(mu0, V0)`` and ``R`` diagonal. ``f`` is the transformed ``State``
row and ``u`` the transformed ``Parameters`` + ``Actions`` row; the exogenous
inputs drive the latent transition, so ``z`` integrates their history and the
bank's drifting hidden parameters are absorbed as slow latent states. Estimation
is Expectation-Maximisation (Shumway & Stoffer): the E-step is a Kalman filter +
RTS smoother, the M-step the closed-form Gaussian updates, warm-started from a
PCA factor init.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.preprocessing import StandardScaler

from economic_models.encoders.base import GenerativeEncoder

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class _Belief:
    """A filtered latent belief: posterior mean and covariance of ``z_t``."""

    mean: np.ndarray  # (m,)
    cov: np.ndarray  # (m, m)


class KalmanEncoder(GenerativeEncoder):
    """A linear-Gaussian state-space encoder (Kalman filter, EM-fit)."""

    def __init__(
        self,
        latent_dim: int = 10,
        max_iter: int = 75,
        tol: float = 1e-4,
        warmup: int = 8,
        jitter: float = 1e-8,
        max_radius: float = 1.0,
        seed: int | None = 0,
    ) -> None:
        """Configure the EM-fit linear-Gaussian encoder.

        ``latent_dim`` is the latent state dimension ``m``; ``max_iter`` and
        ``tol`` bound the EM loop (iterations and relative log-likelihood
        convergence); ``warmup`` is the warm-start window length; ``jitter`` is
        the diagonal regularisation added to covariances; ``max_radius`` caps the
        spectral radius of the fitted transition ``A`` for rollout stability;
        ``seed`` seeds the (currently deterministic) fit.
        """
        self._m = latent_dim
        self.max_iter = max_iter
        self.tol = tol
        self.warmup = warmup
        self.jitter = jitter
        # Cap on the spectral radius of the fitted transition A. EM on limited or
        # near-integrated data can overshoot to radius > 1 -- fine one-step, but it
        # compounds into an explosive multi-step rollout; clipping keeps rollouts
        # bounded. 1.0 forbids only the explosive region (near-unit-root is allowed,
        # as befits a trending economy in stationarized space).
        self.max_radius = max_radius
        self.seed = seed

        # Fitted system matrices, in standardized (f, u) space; set by :meth:`fit`
        # (trailing underscore: estimated-from-data attributes, like ``coef_``).
        self.A_: np.ndarray | None = None  # (m, m)
        self.B_: np.ndarray | None = None  # (m, ku)
        self.C_: np.ndarray | None = None  # (dy, m)
        self.Q_: np.ndarray | None = None  # (m, m)
        self.R_: np.ndarray | None = None  # (dy, dy) diagonal
        self.mu0_: np.ndarray | None = None  # (m,)
        self.V0_: np.ndarray | None = None  # (m, m)
        # Column standardisers for the observations (f) and exog inputs (u),
        # fit on the pooled features by :meth:`fit`.
        self._y_scaler: StandardScaler | None = None
        self._u_scaler: StandardScaler | None = None
        self.loglik_: list[float] = []  # per-EM-iteration observation log-likelihood

    # -- interface ----------------------------------------------------------

    @property
    def latent_dim(self) -> int:
        """Latent state dimension ``m``."""
        return self._m

    @property
    def min_window(self) -> int:
        """Feature rows needed to warm-start a belief (one)."""
        return 1

    # -- standardisation ----------------------------------------------------

    def _sy(self, F: np.ndarray) -> np.ndarray:
        """Column-standardise observation rows ``F``."""
        return self._y_scaler.transform(F)

    def _su(self, U: np.ndarray) -> np.ndarray:
        """Column-standardise exog input rows ``U``."""
        return self._u_scaler.transform(U)

    def _inv_y(self, Y: np.ndarray) -> np.ndarray:
        """Map standardised observations ``Y`` back to original feature units."""
        return self._y_scaler.inverse_transform(Y)

    # -- Kalman filter / smoother (single run, standardized) ----------------

    def _filter(
        self, Y: np.ndarray, Uc: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Forward pass. Returns filtered/predicted means, covs, gains and loglik.

        ``mf, Pf`` are the filtered ``z_t|t``; ``mp, Pp`` the one-step-ahead
        predicted ``z_t|t-1`` (needed by the smoother); ``Kf`` the filter gains
        (needed for the lag-one smoother init); ``ll`` the run log-likelihood.
        """
        N, m, dy = len(Y), self._m, self.C_.shape[0]
        A, B, C, Q, R = self.A_, self.B_, self.C_, self.Q_, self.R_
        I = np.eye(m)
        mf, Pf = np.zeros((N, m)), np.zeros((N, m, m))
        mp, Pp = np.zeros((N, m)), np.zeros((N, m, m))
        Kf = np.zeros((N, m, dy))
        ll = 0.0
        for t in range(N):
            if t == 0:
                mp[t], Pp[t] = self.mu0_, self.V0_
            else:
                mp[t] = A @ mf[t - 1] + B @ Uc[t]
                Pp[t] = A @ Pf[t - 1] @ A.T + Q
            S = C @ Pp[t] @ C.T + R
            S = 0.5 * (S + S.T) + self.jitter * np.eye(dy)
            # One Cholesky factor serves the gain, the quadratic form and the
            # log-det (2 sum log diag L) -- replacing a separate matrix inverse
            # and slogdet, each its own factorisation, in this innermost EM loop.
            cf = cho_factor(S, lower=True)
            K = Pp[t] @ cho_solve(cf, C).T  # Pp C' S^-1, since S symmetric
            Kf[t] = K
            innov = Y[t] - C @ mp[t]
            mf[t] = mp[t] + K @ innov
            ImKC = I - K @ C
            Pf[t] = ImKC @ Pp[t] @ ImKC.T + K @ R @ K.T  # Joseph form
            logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
            ll += -0.5 * (dy * _LOG_2PI + logdet + innov @ cho_solve(cf, innov))
        return mf, Pf, mp, Pp, Kf, ll

    def _smooth(
        self,
        mf: np.ndarray,
        Pf: np.ndarray,
        mp: np.ndarray,
        Pp: np.ndarray,
        Kf: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RTS backward pass. Returns smoothed means, covs and lag-one covs.

        ``Plag[t]`` is ``Cov(z_t, z_{t-1} | f_{1:N})`` for ``t >= 1`` (``Plag[0]``
        is unused), the cross-covariance the transition M-step needs.
        """
        N, m = mf.shape
        A, C = self.A_, self.C_
        ms, Ps = mf.copy(), Pf.copy()
        J = np.zeros((N, m, m))
        for t in range(N - 2, -1, -1):
            # J = Pf A' Pp^-1 as a solve (Pp symmetric): J' = Pp^-1 (Pf A')'.
            J[t] = np.linalg.solve(Pp[t + 1], A @ Pf[t].T).T
            ms[t] = mf[t] + J[t] @ (ms[t + 1] - mp[t + 1])
            Ps[t] = Pf[t] + J[t] @ (Ps[t + 1] - Pp[t + 1]) @ J[t].T
        Plag = np.zeros((N, m, m))
        if N >= 2:
            Plag[N - 1] = (np.eye(m) - Kf[N - 1] @ C) @ A @ Pf[N - 2]
            for t in range(N - 2, 0, -1):
                Plag[t] = (
                    Pf[t] @ J[t - 1].T
                    + J[t] @ (Plag[t + 1] - A @ Pf[t]) @ J[t - 1].T
                )
        return ms, Ps, Plag

    # -- stability ----------------------------------------------------------

    def _stabilize(self, A: np.ndarray) -> np.ndarray:
        """Project the transition onto spectral radius ``<= max_radius``.

        A radius > 1 is fine one-step but compounds into an explosive multi-step
        rollout. This clips each eigenvalue's modulus to ``max_radius`` (scaling
        only the offending modes and preserving conjugate pairs, so the result is
        real), falling back to a uniform contraction if the eigendecomposition is
        ill-conditioned.
        """
        r = self.max_radius
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        if rho <= r:
            return A
        try:
            w, V = np.linalg.eig(A)
            mag = np.abs(w)
            w_clipped = np.where(mag > r, w * (r / mag), w)
            A_new = (V @ np.diag(w_clipped) @ np.linalg.inv(V)).real
            if np.all(np.isfinite(A_new)) and (
                np.max(np.abs(np.linalg.eigvals(A_new))) <= r + 1e-6
            ):
                return A_new
        except np.linalg.LinAlgError:
            pass
        return A * (r / rho)  # robust fallback: uniform contraction

    # -- EM -----------------------------------------------------------------

    def _pca_init(self, runs: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Warm-start the system matrices from a PCA factor model."""
        m = self._m
        Yall = np.vstack([Y for Y, _ in runs])
        # Top-m principal directions as the observation loading C.
        _, _, Vt = np.linalg.svd(Yall - Yall.mean(axis=0), full_matrices=False)
        C = Vt[:m].T  # (dy, m), orthonormal columns
        # Latent scores per run, then regress z_t on [z_{t-1}, u_t] for A, B.
        Z = [Y @ C for Y, _ in runs]
        zt = np.vstack([z[1:] for z in Z])
        prev = np.vstack(
            [np.hstack([z[:-1], U[1:]]) for z, (_, U) in zip(Z, runs)]
        )
        AB, *_ = np.linalg.lstsq(prev, zt, rcond=None)
        A = AB[:m].T
        B = AB[m:].T
        resid = zt - prev @ AB
        self.A_, self.B_, self.C_ = self._stabilize(A), B, C
        self.Q_ = np.cov(resid.T) + self.jitter * np.eye(m)
        obs_resid = Yall - (Yall @ C) @ C.T
        self.R_ = np.diag(obs_resid.var(axis=0) + self.jitter)
        z0 = np.array([z[0] for z in Z])
        self.mu0_ = z0.mean(axis=0)
        self.V0_ = (np.cov(z0.T) if len(z0) > 1 else np.eye(m)) + np.eye(m) * self.jitter

    def fit(self, feature_runs: list[tuple[np.ndarray, np.ndarray]]) -> Self:
        """Fit the state-space model by EM and cache the standardisers."""
        # Fit standardisers on the pooled features, then work standardized.
        Fall = np.vstack([F for F, _ in feature_runs])
        Uall = np.vstack([U for _, U in feature_runs])
        self._y_scaler = StandardScaler().fit(Fall)
        self._u_scaler = StandardScaler().fit(Uall)
        runs = [(self._sy(F), self._su(U)) for F, U in feature_runs]

        self._pca_init(runs)
        m, dy, ku = self._m, self.C_.shape[0], self.B_.shape[1]
        self.loglik_ = []
        prev_ll = -np.inf
        for _ in range(self.max_iter):
            # -- E-step: filter + smooth every run, accumulate sufficient stats.
            Syy = np.zeros((dy, dy))
            Syx = np.zeros((dy, m))
            Sxx = np.zeros((m, m))  # E[z_t z_t'] over all t
            P11 = np.zeros((m, m))  # E[z_t z_t']    for t >= 1
            P10 = np.zeros((m, m))  # E[z_t z_{t-1}']
            P1u = np.zeros((m, ku))
            P00 = np.zeros((m, m))  # E[z_{t-1} z_{t-1}']
            P0u = np.zeros((m, ku))
            Puu = np.zeros((ku, ku))
            n_all = n_trans = 0
            m0_acc = np.zeros(m)
            V0_acc = np.zeros((m, m))
            total_ll = 0.0
            for Y, Uc in runs:
                mf, Pf, mp, Pp, Kf, ll = self._filter(Y, Uc)
                ms, Ps, Plag = self._smooth(mf, Pf, mp, Pp, Kf)
                total_ll += ll
                N = len(Y)
                Ezz = Ps + ms[:, :, None] * ms[:, None, :]  # (N,m,m) E[z_t z_t']
                Syy += Y.T @ Y
                Syx += Y.T @ ms
                Sxx += Ezz.sum(axis=0)
                n_all += N
                if N >= 2:
                    Ezz1 = Plag[1:] + ms[1:, :, None] * ms[:-1, None, :]
                    P11 += Ezz[1:].sum(axis=0)
                    P10 += Ezz1.sum(axis=0)
                    P1u += ms[1:].T @ Uc[1:]
                    P00 += Ezz[:-1].sum(axis=0)
                    P0u += ms[:-1].T @ Uc[1:]
                    Puu += Uc[1:].T @ Uc[1:]
                    n_trans += N - 1
                m0_acc += ms[0]
                V0_acc += Ps[0] + np.outer(ms[0], ms[0])
            self.loglik_.append(total_ll)

            # -- M-step: closed-form Gaussian updates.
            self.C_ = np.linalg.solve(Sxx.T, Syx.T).T  # Syx @ inv(Sxx)
            R = (Syy - self.C_ @ Syx.T) / n_all
            self.R_ = np.diag(np.maximum(np.diag(R), self.jitter))

            Num = np.hstack([P10, P1u])  # (m, m+ku)
            Den = np.block([[P00, P0u], [P0u.T, Puu]])
            Den += self.jitter * np.eye(m + ku)
            AB = np.linalg.solve(Den.T, Num.T).T  # Num @ inv(Den)
            self.A_, self.B_ = self._stabilize(AB[:, :m]), AB[:, m:]
            Q = (P11 - AB @ Num.T) / max(n_trans, 1)
            self.Q_ = 0.5 * (Q + Q.T) + self.jitter * np.eye(m)

            n_runs = len(runs)
            self.mu0_ = m0_acc / n_runs
            V0 = V0_acc / n_runs - np.outer(self.mu0_, self.mu0_)
            self.V0_ = 0.5 * (V0 + V0.T) + self.jitter * np.eye(m)

            if total_ll - prev_ll < self.tol * (abs(prev_ll) + 1.0):
                break
            prev_ll = total_ll
        self._fitted = True
        return self

    # -- batch encode -------------------------------------------------------

    def encode_run(self, F: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Causal filtered latent means for one run, ``(len(F), m)``."""
        self._require_fitted()
        mf, *_ = self._filter(self._sy(F), self._su(U))
        return mf

    # -- online -------------------------------------------------------------

    def init_belief(self, F_win: np.ndarray, U_win: np.ndarray) -> _Belief:
        """Warm-start a belief by filtering the window; keeps the last posterior."""
        self._require_fitted()
        mf, Pf, *_ = self._filter(self._sy(F_win), self._su(U_win))
        return _Belief(mean=mf[-1].copy(), cov=Pf[-1].copy())

    def latent(self, belief: _Belief) -> np.ndarray:
        """Return the belief's posterior mean as the conditioning latent."""
        return belief.mean.copy()

    def _predict(self, belief: _Belief, u_next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One-step latent prior ``(mean, cov)`` given standardized next exog."""
        mp = self.A_ @ belief.mean + self.B_ @ u_next
        Pp = self.A_ @ belief.cov @ self.A_.T + self.Q_
        return mp, Pp

    def advance(self, belief: _Belief, f_next: np.ndarray, u_next: np.ndarray) -> _Belief:
        """Fold one realised ``(f_next, u_next)`` in via a Kalman predict+update."""
        y = self._sy(f_next[None, :])[0]
        u = self._su(u_next[None, :])[0]
        mp, Pp = self._predict(belief, u)
        dy = self.C_.shape[0]
        S = self.C_ @ Pp @ self.C_.T + self.R_
        S = 0.5 * (S + S.T) + self.jitter * np.eye(dy)
        K = Pp @ self.C_.T @ np.linalg.inv(S)
        ImKC = np.eye(self._m) - K @ self.C_
        mean = mp + K @ (y - self.C_ @ mp)
        cov = ImKC @ Pp @ ImKC.T + K @ self.R_ @ K.T
        return _Belief(mean=mean, cov=cov)

    # -- generative forecast ------------------------------------------------

    def forecast(
        self,
        belief: _Belief,
        u_next: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """One-step feature forecast ``C(A z + B u)``; a draw if ``rng`` is given."""
        u = self._su(u_next[None, :])[0]
        mp, Pp = self._predict(belief, u)
        y = self.C_ @ mp
        if rng is not None:
            dy = self.C_.shape[0]
            S = self.C_ @ Pp @ self.C_.T + self.R_
            S = 0.5 * (S + S.T) + self.jitter * np.eye(dy)
            y = y + np.linalg.cholesky(S) @ rng.standard_normal(dy)
        return self._inv_y(y[None, :])[0]

    def forecast_means(self, Z: np.ndarray, U_next: np.ndarray) -> np.ndarray:
        """Batch predictive means: next feature row from filtered latents + exog.

        ``Z`` are filtered latent means (as returned by :meth:`encode_run`) and
        ``U_next`` the raw exog features of each target period; returns the
        deterministic ``C (A z + B u)`` forecast in original feature units.
        """
        Uc = self._su(U_next)
        Ystd = (Z @ self.A_.T + Uc @ self.B_.T) @ self.C_.T
        return self._inv_y(Ystd)
