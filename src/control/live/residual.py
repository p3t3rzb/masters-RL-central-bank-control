"""The online correction to the proxy: what it gets wrong, learned as it happens.

The proxy predicts a state-feature row ``mu(z_t, u_{t+1}, f_t)``; the ground truth
produces ``f*_{t+1}``. The gap between them,

    eps_{t+1} = f*_{t+1} - mu(z_t, u_{t+1}, f_t),

is the only thing a live run can teach cheaply, and it is worth far more than the
same number of samples spent on the policy: hundreds of transitions cannot retrain
an actor, but they can say *which way the world model is wrong right now*.

**The residual is not white noise, and that is what makes this work.** GROWTH is
deterministic given its hidden structural path (``RA``, ``gamma0``, ``eta0``,
``alpha1``, ``lambda40``), and the excitation drives those as slow AR(1)
processes. What looks like transition noise from the proxy's side is the
projection of persistent unobserved structure, so a regime the proxy
mis-predicts is a regime it will go on mis-predicting for a while. The correction
is therefore split in two:

    c_t(x) = g_psi(x)  +  b_t
             \\_ slow _/    \\_ fast _/

* ``g_psi`` is a fitted map over the same context the proxy conditions on. It is
  estimated by **recursive Bayesian linear regression with a forgetting factor**
  -- a Kalman filter on the coefficients, with a zero mean prior that says *the
  proxy is unbiased until the data say otherwise* and shrinks hard toward that.
  With hundreds of rows against a design in the tens, anything less opinionated
  overfits.
* ``b_t`` is a recursive estimate of the leftover bias *here, now*: an
  exponentially weighted average of what ``g_psi`` still misses, with no refit and
  a memory of about a year at quarterly steps. It is what adapts within a handful
  of observations, and it decays over the steps of a synthetic rollout because a
  bias read off the present is evidence about the present.

The spread matters as much as the mean, because the critic is distributional and
under CVaR the actor optimises the *lower tail* of the return: a corrected model
whose predictive spread is wrong corrupts the objective rather than merely the
value estimate. :class:`ResidualNoise` is that half -- calibrated on the realised
leftover residuals, either as a Gaussian with an EWMA covariance or, where the
tail shape matters, as a block bootstrap of recent ones.

:class:`NullResidual` is the null object: correction identically zero and the
proxy's own noise passed straight through, so "the uncorrected proxy" is a
configuration of this machinery rather than a separate code path -- which is what
makes the ablation of §8.2 a one-line change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np

from economic_models.proxy import BaseProxyModel
from economic_models.run import Run


def design(z: np.ndarray, u_next: np.ndarray, f_prev: np.ndarray) -> np.ndarray:
    """The context the residual is a function of: ``[z_t, u_{t+1}, f_t]``.

    Everything the proxy conditioned on, in one row (or one batch of rows, if the
    inputs are 2-D): the encoder latent at the current period, the exogenous
    features of the period being predicted, and the current state-feature row.
    The intercept is added later, by the standardisation, so that a caller
    building a design does not have to know about it.
    """
    return np.hstack([z, u_next, f_prev])


def action_columns(proxy: BaseProxyModel, width: int) -> np.ndarray:
    """The columns of a :func:`design` row that carry the policy's own levers.

    A design row is ``[z, u_next, f_prev]`` and ``u_next`` is the parameters
    followed by the actions, so the action block ends exactly where ``f_prev``
    begins -- which is fixed by the width of a state-feature row regardless of
    how wide the latent is.
    """
    n_feat = len(proxy.transform.state_feature_names)
    n_act = len(proxy.ACTIONS.names())
    return np.arange(width - n_feat - n_act, width - n_feat)


# -- the noise half ---------------------------------------------------------


class ResidualNoise(ABC):
    """The corrected model's one-step predictive spread, calibrated on live data.

    Fed the *leftover* residual ``eps - c(x)`` after every real step
    (:meth:`observe`) and asked for a draw of the same law inside every synthetic
    one (:meth:`draw`). Those leftovers already carry the world's own noise, so
    this is the whole spread and not a component of one; ``inflation`` adds the
    correction's **epistemic** variance on top (:meth:`Residual.variance`), which
    is the one thing they cannot contain. Where the correction is uncertain the
    corrected model should be less confident -- not equally confident about a
    different number, and not twice as uncertain as its own errors have been.
    """

    @abstractmethod
    def observe(self, leftover: np.ndarray) -> None:
        """Fold one realised leftover residual row into the calibration."""

    @abstractmethod
    def draw(self, rng: np.random.Generator, inflation: np.ndarray) -> np.ndarray:
        """One noise row, with ``inflation`` extra variance per dimension."""

    def restart(self) -> None:
        """Begin a fresh rollout; a memoryless noise model has nothing to do."""


class GaussianResidualNoise(ResidualNoise):
    """A Gaussian with an exponentially weighted covariance of the leftovers.

    Tracks the whole covariance rather than per-dimension variances because the
    proxy's errors are strongly correlated across the feature block -- output,
    consumption and investment growth miss together -- and a diagonal draw would
    hand the critic economies that cannot happen.
    """

    def __init__(self, n_outputs: int, *, kappa: float = 0.05, floor: float = 1e-12
                 ) -> None:
        """Track ``n_outputs`` dimensions with EWMA rate ``kappa``.

        ``kappa`` is deliberately slower than the bias state's: a mean can be
        re-estimated from a few observations, a covariance cannot. ``floor``
        keeps the factorisation defined before any data has arrived.
        """
        self.kappa = kappa
        self.floor = floor
        self.cov_ = np.eye(n_outputs) * floor
        self._chol = np.linalg.cholesky(self.cov_)

    def observe(self, leftover: np.ndarray) -> None:
        """Update the covariance with one leftover row and refactor it."""
        outer = np.outer(leftover, leftover)
        self.cov_ = (1.0 - self.kappa) * self.cov_ + self.kappa * outer
        jitter = self.floor + 1e-10 * np.trace(self.cov_) / len(self.cov_)
        self._chol = np.linalg.cholesky(self.cov_ + jitter * np.eye(len(self.cov_)))

    def draw(self, rng: np.random.Generator, inflation: np.ndarray) -> np.ndarray:
        """A correlated draw plus an independent one carrying ``inflation``.

        The sum of two independent Gaussians is Gaussian with the summed
        covariance, so this is exact and avoids refactorising a matrix that
        changes on every draw.
        """
        n = len(self.cov_)
        correlated = self._chol @ rng.standard_normal(n)
        return correlated + np.sqrt(np.maximum(inflation, 0.0)) * rng.standard_normal(n)


class BlockBootstrapResidualNoise(ResidualNoise):
    """Resampled blocks of recent leftover residuals: shape as well as scale.

    A Gaussian gets the covariance right and the *tails* wrong, and the tails are
    what a CVaR actor is fitted on. Several proxy families are worse than
    Gaussian here rather than better: the ones that resample observed rows (DRF,
    kNN) cannot emit a crisis larger than the largest in the history at all.
    Drawing the noise from recent realised leftovers instead keeps whatever shape
    the truth actually had, and walking a *contiguous* block through a synthetic
    rollout keeps its autocorrelation too -- which matters because these errors
    persist, which is the premise of this whole module.
    """

    def __init__(self, n_outputs: int, *, memory: int = 200, block: int = 4) -> None:
        """Keep the last ``memory`` leftovers and walk them ``block`` at a time."""
        self.memory = memory
        self.block = block
        self.rows_: list[np.ndarray] = []
        self.n_outputs = n_outputs
        self._cursor: int | None = None
        self._taken = 0

    def observe(self, leftover: np.ndarray) -> None:
        """Record one leftover row, evicting the oldest past ``memory``."""
        self.rows_.append(np.asarray(leftover, dtype=float).copy())
        del self.rows_[: max(0, len(self.rows_) - self.memory)]

    def restart(self) -> None:
        """Drop the current block, so the next draw starts a new one."""
        self._cursor = None
        self._taken = 0

    def draw(self, rng: np.random.Generator, inflation: np.ndarray) -> np.ndarray:
        """The next row of the current block, plus the ``inflation`` term."""
        if not self.rows_:
            return np.sqrt(np.maximum(inflation, 0.0)) * rng.standard_normal(
                self.n_outputs
            )
        if self._cursor is None or self._taken >= self.block:
            self._cursor = int(rng.integers(len(self.rows_)))
            self._taken = 0
        row = self.rows_[self._cursor % len(self.rows_)]
        self._cursor += 1
        self._taken += 1
        return row + np.sqrt(np.maximum(inflation, 0.0)) * rng.standard_normal(
            self.n_outputs
        )


# -- the correction ---------------------------------------------------------


class Residual(ABC):
    """A correction to a proxy's one-step conditional law.

    Two responsibilities, mirroring how a deployment uses one: **learn** from the
    residual of a period that actually happened (:meth:`update`), and **apply**
    the correction and its noise inside a synthetic rollout
    (:meth:`correction`, :meth:`noise`).
    """

    @abstractmethod
    def correction(self, x: np.ndarray, depth: int = 0) -> np.ndarray:
        """The mean correction at context ``x``, ``depth`` steps into a rollout.

        ``depth`` is how far the rollout has travelled from the real state it
        branched off: the fast bias state is evidence about *now*, so it is
        decayed away over a rollout rather than carried undiminished into a
        future the correction was never fit on.
        """

    @abstractmethod
    def variance(self, x: np.ndarray) -> np.ndarray:
        """Per-dimension **epistemic** variance of the correction at ``x``.

        How uncertain the *coefficients* are here -- not how noisy the world is.
        The distinction is load-bearing because of who reads this. It is added on
        top of a spread that is already calibrated on realised leftovers
        (:class:`ResidualNoise`), so folding the observation noise in here would
        count it twice and hand a CVaR actor a model roughly twice as uncertain
        as the data says; and the out-of-distribution monitor wants exactly the
        epistemic part, since a spike in it is the earliest warning that the live
        economy has walked out of the region the correction was estimated in,
        with no prediction yet seen to be wrong.
        """

    @abstractmethod
    def update(self, x: np.ndarray, eps: np.ndarray, weight: float = 1.0) -> None:
        """Fold in one realised residual ``eps`` observed at context ``x``."""

    @abstractmethod
    def noise(
        self, innovation: np.ndarray, x: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """The corrected model's noise, given the draw the raw proxy would make."""

    def restart(self) -> None:
        """Tell the correction a fresh rollout is beginning."""

    @property
    def bias(self) -> np.ndarray:
        """The fast bias state; zero for a correction that has none."""
        return np.zeros(0)


class NullResidual(Residual):
    """The null object correction: the raw proxy, reached through this interface.

    Correction identically zero, no uncertainty, and the proxy's own noise passed
    through untouched -- so a :class:`~control.live.corrected.CorrectedProxy`
    wrapping this *is* the proxy it wraps, draw for draw under the same stream.
    That exactness is the point: it makes "MBPO on the uncorrected proxy" an
    ablation that changes one object rather than one that re-runs a different
    program.
    """

    def __init__(self, n_outputs: int) -> None:
        """Answer with zeros of the same width a fitted correction would."""
        self.zero_ = np.zeros(n_outputs)

    def correction(self, x: np.ndarray, depth: int = 0) -> np.ndarray:
        """Zero, wherever and however deep it is asked for."""
        return self.zero_

    def variance(self, x: np.ndarray) -> np.ndarray:
        """Zero: nothing is estimated, so nothing is uncertain."""
        return self.zero_

    def update(self, x: np.ndarray, eps: np.ndarray, weight: float = 1.0) -> None:
        """Ignore the observation; there is nothing to learn."""

    def noise(
        self, innovation: np.ndarray, x: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """The proxy's own draw, unchanged."""
        return innovation

    @property
    def bias(self) -> np.ndarray:
        """Zero: this correction has no fast state to carry."""
        return self.zero_


class ResidualModel(Residual):
    """Recursive Bayesian linear correction plus a fast recursive bias state.

    Fit once on cross-fitted historic residuals (:meth:`seed`) and updated by one
    rank-one correction per real step thereafter (:meth:`update`) -- the same
    recursion either way, so there is no seam between what the history taught it
    and what the live run does.
    """

    def __init__(
        self,
        n_outputs: int,
        *,
        tau: float = 1.0,
        forgetting: float = 0.999,
        kappa: float = 0.2,
        bias_decay: float | None = None,
        noise: ResidualNoise | None = None,
        ignore: Sequence[int] | None = None,
    ) -> None:
        """Configure the two halves of the correction.

        ``n_outputs`` is the width of a state-feature row. ``tau`` is the prior
        scale on the coefficients (small means "the proxy is right"), and
        ``forgetting`` the factor ``lambda <= 1`` that discounts old rows, which
        is what lets the correction track a drifting truth rather than average
        over all of history. ``kappa`` is the bias state's EWMA rate -- 0.2 is a
        memory of roughly a year at quarterly steps -- and ``bias_decay`` how
        much of that bias survives one step of a synthetic rollout, defaulting to
        the ``1 - kappa`` that makes the pair an AR(1). ``noise`` is the
        predictive-spread model (Gaussian by default).

        ``ignore`` names design columns the correction is not allowed to read,
        and exists for one of them: the **action block**. A correction fitted on
        a live run sees levers that a policy moved in response to the state, so
        those columns are a near-exact function of the rest of the design (a
        variance inflation upwards of a thousand) and their coefficients are
        decided by the prior rather than by the data. That costs nothing in
        forecast -- they carry under a thirtieth of the error reduction -- and
        everything in control, because a policy is determined by exactly the
        derivative those coefficients rewrite. Ignoring them leaves the proxy's
        own response to the instrument in place, which was identified honestly:
        the history excites the levers as independent AR(1) processes precisely
        so that it would be.
        """
        self.n_outputs = n_outputs
        self.tau = tau
        self.forgetting = forgetting
        self.kappa = kappa
        self.bias_decay = (1.0 - kappa) if bias_decay is None else bias_decay
        self.noise_ = noise or GaussianResidualNoise(n_outputs)
        # ``ignore or ()`` would truth-test a numpy array; the caller's block
        # of column indices is usually exactly that.
        self.ignore = tuple(sorted({int(c) for c in (() if ignore is None else ignore)}))
        self.keep_: np.ndarray | None = None  # design columns actually read

        self.mean_: np.ndarray | None = None  # design standardisation, frozen
        self.std_: np.ndarray | None = None
        self.psi_: np.ndarray | None = None  # (d, n_outputs) coefficients
        self.P_: np.ndarray | None = None  # (d, d) coefficient covariance
        self.sigma_: np.ndarray | None = None  # (n_outputs,) per-dim noise variance
        self.bias_ = np.zeros(n_outputs)
        self.n_seen_ = 0

    # -- fitting -------------------------------------------------------------

    def seed(
        self, X: np.ndarray, E: np.ndarray, *, weight: float = 1.0
    ) -> ResidualModel:
        """Prime the correction from a block of already-observed residuals.

        ``X`` are :func:`design` rows and ``E`` the residuals observed at them --
        in practice the *cross-fitted* historic residuals of
        :func:`cross_fitted_residuals`, never the in-sample ones. The
        standardisation of the design is fit here and then frozen, for the reason
        the observation's is: the correction must be the same function of the
        same context before and after the live run starts.

        The rows are then pushed through the ordinary recursive :meth:`update`,
        in run order and at ``weight`` each. That is not a shortcut -- it is the
        same estimator, and taking the history through it in order is what makes
        the forgetting factor treat the history as *old* rather than as
        contemporaneous with the live rows that follow.

        ``weight`` below one is the second half of the answer to the in-sample
        problem: cross-fitting removes the bias, and down-weighting says that a
        residual from a proxy fit on a *neighbouring* stretch of the same run is
        still not quite a residual from the proxy that is actually deployed.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        E = np.atleast_2d(np.asarray(E, dtype=float))
        if len(X) != len(E):
            raise ValueError(f"{len(X)} design rows against {len(E)} residual rows")
        if E.shape[1] != self.n_outputs:
            raise ValueError(
                f"residuals are {E.shape[1]}-wide, model was built for {self.n_outputs}"
            )

        bad = [c for c in self.ignore if not 0 <= c < X.shape[1]]
        if bad:
            raise ValueError(
                f"ignore names column(s) {bad} outside a {X.shape[1]}-wide design"
            )
        self.keep_ = np.array(
            [c for c in range(X.shape[1]) if c not in self.ignore], dtype=int
        )
        kept = X[:, self.keep_]
        std = kept.std(axis=0)
        self.mean_ = kept.mean(axis=0)
        self.std_ = np.where(std > 1e-12, std, 1.0)

        d = len(self.keep_) + 1  # + intercept
        self.psi_ = np.zeros((d, self.n_outputs))
        self.P_ = self.tau**2 * np.eye(d)
        # Start the per-dimension noise scale at the residuals' own variance: the
        # prior is that the correction explains none of it.
        self.sigma_ = E.var(axis=0) + 1e-12
        for x, eps in zip(X, E):
            self.update(x, eps, weight=weight)
        return self

    # -- the recursion --------------------------------------------------------

    def update(self, x: np.ndarray, eps: np.ndarray, weight: float = 1.0) -> None:
        """One rank-one recursive-least-squares step, plus the bias EWMA.

        The Kalman update on the coefficients: gain, correction, covariance
        deflation, all divided through by the forgetting factor so that an old
        row's influence decays geometrically. ``weight`` scales one row's
        influence by inflating its effective observation noise, which is how the
        historic block is trusted less than the live one.
        """
        self._require_seeded()
        assert self.psi_ is not None and self.P_ is not None and self.sigma_ is not None
        row = self._row(x)
        eps = np.asarray(eps, dtype=float)

        # Everything the calibration reads is computed against the *prior*
        # coefficients and the *prior* bias, so it measures a genuine one-step-
        # ahead error rather than a fit that has already seen this row.
        error = eps - self.psi_.T @ row
        corrected_error = error - self.bias_

        denom = self.forgetting / max(weight, 1e-12) + row @ self.P_ @ row
        gain = (self.P_ @ row) / denom
        self.psi_ = self.psi_ + np.outer(gain, error)
        self.P_ = (self.P_ - np.outer(gain, row @ self.P_)) / self.forgetting
        # Keep it symmetric: the deflation above is symmetric in exact arithmetic
        # and drifts out of it in floating point over hundreds of updates.
        self.P_ = 0.5 * (self.P_ + self.P_.T)

        self.sigma_ = (1.0 - self.kappa) * self.sigma_ + self.kappa * error**2
        self.bias_ = (1.0 - self.kappa) * self.bias_ + self.kappa * (
            eps - self.psi_.T @ row
        )
        self.noise_.observe(corrected_error)
        self.n_seen_ += 1

    # -- reading it -----------------------------------------------------------

    def correction(self, x: np.ndarray, depth: int = 0) -> np.ndarray:
        """``g_psi(x)`` plus the bias state, decayed for a rollout's ``depth``."""
        self._require_seeded()
        assert self.psi_ is not None
        return self.psi_.T @ self._row(x) + self.bias_decay**depth * self.bias_

    def mean(self, x: np.ndarray) -> np.ndarray:
        """The fitted map alone, without the fast bias state."""
        self._require_seeded()
        assert self.psi_ is not None
        return self.psi_.T @ self._row(x)

    def variance(self, x: np.ndarray) -> np.ndarray:
        """``sigma_j^2 (x' P x)``: the epistemic variance per dimension.

        The full Bayesian predictive variance is ``sigma^2 (1 + x' P x)``, and
        the ``1`` is the observation noise. It is dropped here because
        :meth:`noise` is not a predictive distribution built from scratch -- it
        draws from a spread already calibrated on the realised leftovers, whose
        variance *is* that ``sigma^2``. What is missing from that calibration is
        only the parameter uncertainty, which is the term kept.
        """
        self._require_seeded()
        assert self.P_ is not None and self.sigma_ is not None
        row = self._row(x)
        return self.sigma_ * float(row @ self.P_ @ row)

    def noise(
        self, innovation: np.ndarray, x: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """A draw from the *corrected* model's spread, ignoring the proxy's own.

        The realised leftover residual is, by construction, the corrected model's
        own one-step error against the truth -- measured directly, every step. So
        the calibrated spread of those leftovers is a better description of what
        the corrected model does not know than the raw proxy's fitted noise,
        which was estimated on the history against a different mean. The proxy's
        ``innovation`` is what this replaces, and is passed in only so that
        :class:`NullResidual` can hand it back untouched.
        """
        return self.noise_.draw(rng, self.variance(x))

    def restart(self) -> None:
        """Begin a fresh synthetic rollout."""
        self.noise_.restart()

    @property
    def bias(self) -> np.ndarray:
        """The fast bias state: where the correction is missing *right now*."""
        return self.bias_

    # -- internals ------------------------------------------------------------

    def _row(self, x: np.ndarray) -> np.ndarray:
        """Drop the ignored columns, standardise, and append the intercept.

        Callers keep handing over the *whole* design row -- the correction's
        blind spots are its own business, not theirs -- so the selection happens
        here and nowhere else.
        """
        assert self.mean_ is not None and self.std_ is not None
        assert self.keep_ is not None
        x = np.asarray(x, dtype=float)[self.keep_]
        return np.append((x - self.mean_) / self.std_, 1.0)

    def _require_seeded(self) -> None:
        """Raise a clear error before :meth:`seed` has fixed the design."""
        if self.psi_ is None:
            raise RuntimeError("residual model must be seed()ed before use")


# -- cross-fitting the historic residuals -----------------------------------


def cross_fitted_residuals(
    build: Callable[[], BaseProxyModel],
    history: Run,
    *,
    folds: int = 5,
    initial: float = 0.5,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Historic residuals from proxies that never saw the periods they score.

    The historic run is real ground-truth data and there is four times as much of
    it as a live run will ever produce, so it should be used. But **the proxy was
    fit on it**, and its residuals there are in-sample: for an OLS-type family
    they are orthogonal to the design by construction, so a linear correction fit
    on them returns approximately zero, contributes nothing, and dilutes the
    handful of live rows that would have said something. Using them naively is
    worse than not using them.

    The fix is the standard one from debiased machine learning: score each block
    with a model that never saw it. **Rolling-origin** rather than random k-fold,
    because the encoder latent is a causal filter of the past and a random split
    would let a proxy be fit on the future of the rows it is scored on. The first
    ``initial`` fraction of the run is the opening training window; the rest is
    cut into ``folds`` blocks, and block ``j`` is scored by a proxy fit on
    everything before it.

    ``build`` makes a *fresh, unfitted* proxy of the deployed family each time,
    carrying a copy of the **deployed encoder**. Only the estimator is refit per
    fold, deliberately: a latent is identified only up to a change of basis, so
    an encoder refit per fold would give each block's design a different set of
    columns and the pooled regression would be fitting the disagreement between
    filters rather than the proxy's error. What that leaves is a second-order
    leak -- the encoder is a filter of the whole history -- which is the price of
    having a design that means one thing.

    Returns the ``(design, residual)`` rows of every block stacked in run order,
    ready for :meth:`ResidualModel.seed`.
    """
    if not 0.0 < initial < 1.0:
        raise ValueError(f"initial must be a fraction in (0, 1), got {initial}")
    if folds < 1:
        raise ValueError(f"need at least one fold, got {folds}")

    T = len(history)
    edges = np.linspace(int(initial * T), T, folds + 1).astype(int)
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for j in range(folds):
        lo, hi = int(edges[j]), int(edges[j + 1])
        if hi - lo < 1:
            continue
        proxy = build()
        proxy.fit([_prefix(history, lo)], refit_encoder=False)
        X, E = _teacher_forced_residuals(proxy, history)
        # Feature row i is built from level rows (i, i + 1), so design row i
        # predicts the period at level row i + 2: that is the offset the block
        # boundaries, which are in level rows, have to be shifted by.
        keep = slice(max(lo - 2, 0), max(hi - 2, 0))
        blocks.append((X[keep], E[keep]))
        if verbose:
            rmse = float(np.sqrt(np.mean(E[keep] ** 2)))
            print(f"  fold {j + 1}/{folds}: fit on [0, {lo}), scored [{lo}, {hi}) "
                  f"-- {hi - lo} rows, RMSE {rmse:.4g}")
    if not blocks:
        raise ValueError(f"a {T}-step history leaves no held-out rows at {folds} folds")
    return np.vstack([b[0] for b in blocks]), np.vstack([b[1] for b in blocks])


def in_sample_residuals(
    proxy: BaseProxyModel, history: Run
) -> tuple[np.ndarray, np.ndarray]:
    """The same residuals without cross-fitting -- the ablation, not the method.

    What §8.2 compares :func:`cross_fitted_residuals` against: the deployed
    proxy's residuals on the very run it was fit on. Cheap, wrong, and worth
    measuring, because "does the in-sample bias matter in practice" is an
    empirical question about this economy rather than a theorem.
    """
    return _teacher_forced_residuals(proxy, history)


def _teacher_forced_residuals(
    proxy: BaseProxyModel, run: Run
) -> tuple[np.ndarray, np.ndarray]:
    """One-step residuals of ``proxy`` along ``run``, each from the realised past.

    Teacher-forced: every prediction conditions on what actually happened up to
    that period, never on the proxy's own earlier predictions, which is the
    definition the residual model is written against and the only one that
    isolates *one-step* error from accumulated drift.
    """
    F, U = proxy.transform.transform_run(run)
    z = proxy.encoder.encode_run(F, U)
    X = design(z[:-1], U[1:], F[:-1])
    return X, F[1:] - proxy.predict_means(z[:-1], U[1:], F[:-1])


def _prefix(run: Run, n: int) -> Run:
    """The first ``n`` steps of a run, as a run in its own right."""
    return Run(
        states=run.states[:n],
        params=run.params[:n],
        actions=run.actions[:n],
        dt=run.dt,
    )
