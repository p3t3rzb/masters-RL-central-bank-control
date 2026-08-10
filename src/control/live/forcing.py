"""Forecasting the exogenous environment, because a synthetic rollout needs one.

A rollout of ``k`` steps through the corrected model needs ``k`` future rows of
exogenous forcing -- and in a live run the future forcing has not happened yet.
At training time this was free: the episode bank *is* a set of frozen forcing
paths, drawn from the excitation process without solving anything. Live, they
have to be forecast.

**The forecast must respect observability.** The bank sees ``GRg``, ``theta``,
``Nfe``, ``GRpr``, ``ADDbl``, ``Rln`` and ``NPLk``; it does not see the process
generating them, and it certainly does not see the hidden structural parameters
riding alongside. So the honest object is a small time-series model fit on the
observed history and extended with the live rows -- which is, not incidentally,
a thing central banks actually do. :class:`VARForcing` is that model, and the
default.

The two alternatives exist to bound it. :class:`BlockBootstrapForcing` makes no
functional-form assumption at all and is the robustness check; :class:`OracleForcing`
replays the forcing that *will* actually arrive, which is lab-only information and
belongs in the results as an upper bound on what knowing the exogenous law would
buy -- never as the headline.

Everything here works in the proxy's **exogenous feature space** (levels, except
``Nfe``, which enters as a log-difference) and converts back to levels on the way
out, for the same reason a proxy is fit there: ``Nfe`` trends, and a model of its
level is a model of a trend rather than of a process. The agent's own three action
columns are never forecast -- they come from the policy being evaluated, which is
the point of the rollout.

**A forecast is re-based onto whatever it is about to drive.** Undoing a
log-difference needs a level to cumulate from, and the level that matters is the
one at the *branch point* the rollout starts from -- not the latest observed one.
A rollout branched off a state fifty periods back and driven by a path cumulated
from today's ``Nfe`` makes its first step's ``dlog(Nfe)`` the accumulated drift
between the two, which is not a one-period change and is not what anything
downstream was fit on. Hence the ``previous`` argument on :meth:`ForcingModel.sample`.

**What a forecast is allowed to know.** Government spending is the one column
that is not really exogenous: the generator sets it as a rule on the previous
period's employment rate, and a realised run therefore records ``GRg`` with that
response already in it while a frozen future does not. A deployment that intends
to re-apply the response at rollout (:class:`~control.world.FiscalStabilizer`)
must therefore remove it here first, or count it twice -- and doing so is an
assumption that the bank knows the fiscal reaction function. Passing
``stabilizer=None`` is the position that it does not: ``GRg`` is then forecast as
an opaque time series, the historic response survives only as unconditional level
and variance, and nothing re-applies it. See
:attr:`~control.live.deploy.LiveConfig.rollout_stabilizer`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from economic_models.interface import ModelInterface
from economic_models.run import Run


class Stabilizer(Protocol):
    """The half of :class:`~control.world.FiscalStabilizer` a forecast needs."""

    def strip(self, params: np.ndarray, er_prev: float) -> np.ndarray:
        """Remove the spending response ``er_prev`` provoked from a realised row."""


class ForcingModel(ABC):
    """A forecast of the exogenous parameter path a rollout will be driven by.

    Fit on the observed history, extended with :meth:`absorb` as live rows
    arrive, and asked for a path of ``k`` future **level** rows by :meth:`sample`.
    The columns are the model's :class:`~economic_models.variables.Parameters`
    only; the actions the policy supplies.
    """

    def __init__(
        self, interface: ModelInterface, *, stabilizer: Stabilizer | None = None
    ) -> None:
        """Bind to the value spaces whose parameter columns this forecasts.

        ``stabilizer``, when given, is used to strip the fiscal response out of
        every observed row on the way in -- which is required of, and only of, a
        deployment that re-applies it at rollout. ``None`` forecasts the rows as
        they were observed and assumes nothing about the rule.
        """
        self.interface = interface
        self.names = interface.parameters.names()
        self.stabilizer = stabilizer
        spec = interface.transform_spec
        #: which parameter columns are modelled in log-differences rather than
        #: in levels -- the trending ones.
        self.log_diff = np.array([name in spec.exog_log_diff for name in self.names])
        self._er = interface.state.names().index("ER")
        self.rows_: list[np.ndarray] = []  # observed level rows, oldest first

    # -- data ---------------------------------------------------------------

    def fit(self, history: Run) -> ForcingModel:
        """Learn from the parameter path of an observed run.

        The employment rate is read off the run's own states to undo the fiscal
        response, when there is one to undo. Row ``i``'s response was provoked by
        the employment rate at ``i - 1``; the very first row has no predecessor
        and is stripped against its own, which is one approximate row in several
        hundred.
        """
        rows = [row.astype(float) for row in history.params]
        if self.stabilizer is not None:
            er = history.states[:, self._er].astype(float)
            er_prev = np.concatenate([er[:1], er[:-1]])
            rows = [self.stabilizer.strip(r, e) for r, e in zip(rows, er_prev)]
        self.rows_ = rows
        self._fit()
        return self

    def absorb(self, row: np.ndarray, er_prev: float | None = None) -> None:
        """Record one newly observed parameter row.

        ``er_prev`` is the employment rate that provoked this row's fiscal
        response, and is required exactly when this model was built with a
        ``stabilizer`` to strip it back out.

        Cheap by design and called every live step: what the estimate costs is
        paid in :meth:`refit`, which a deployment runs on the policy's clock
        rather than the economy's.
        """
        row = np.asarray(row, dtype=float).copy()
        if self.stabilizer is not None:
            if er_prev is None:
                raise ValueError(
                    "this forcing model strips the fiscal response, so absorb() "
                    "needs the employment rate that provoked it"
                )
            row = self.stabilizer.strip(row, er_prev)
        self.rows_.append(row)

    def refit(self) -> None:
        """Re-estimate from everything observed so far, history and live alike."""
        self._fit()

    @property
    def last(self) -> np.ndarray:
        """The most recently observed parameter level row."""
        if not self.rows_:
            raise RuntimeError("forcing model has seen no rows; call fit() first")
        return self.rows_[-1]

    # -- subclass contract ---------------------------------------------------

    @abstractmethod
    def _fit(self) -> None:
        """Estimate whatever this family needs from :attr:`rows_`."""

    @abstractmethod
    def sample(
        self, k: int, rng: np.random.Generator, *, previous: np.ndarray | None = None
    ) -> np.ndarray:
        """``(k, n_params)`` future parameter **levels**, continuing the history.

        ``previous`` is the level row the path is cumulated from, and must be the
        one of whatever position the path is about to drive -- a branch point's,
        not necessarily the latest observed. Defaults to :attr:`last`, which is
        right only when the caller is standing at the end of the history.
        """

    # -- the feature space ---------------------------------------------------

    def features(self) -> np.ndarray:
        """The observed rows as exogenous features: levels, ``Nfe`` differenced."""
        rows = np.array(self.rows_, dtype=float)
        feats = rows[1:].copy()
        feats[:, self.log_diff] = np.diff(np.log(rows[:, self.log_diff]), axis=0)
        return feats

    def to_levels(self, feats: np.ndarray, previous: np.ndarray) -> np.ndarray:
        """Turn a feature path back into levels, cumulating from ``previous``."""
        levels = np.array(feats, dtype=float).copy()
        prev = np.asarray(previous, dtype=float).copy()
        for row in levels:
            row[self.log_diff] = prev[self.log_diff] * np.exp(row[self.log_diff])
            prev = row
        return levels


class VARForcing(ForcingModel):
    """A first-order vector autoregression on the observed exogenous features.

    ``y_{t+1} = c + A y_t + e``, estimated by ridge least squares with the
    residual covariance kept so the path can be sampled rather than merely
    projected. First order and heavily shrunk on purpose: the object of interest
    is a three-to-five-step-ahead path, the sample is a few hundred rows wide
    enough to be badly conditioned, and a richer lag structure buys variance
    rather than accuracy at that horizon.

    Draws are clipped to a widened box around what the history contained. Not
    cosmetic: a VAR with an estimated ``A`` whose spectral radius creeps above
    one produces explosive paths, and an explosive forcing path fed through the
    corrected model into the replay buffer is a lesson the critic should never be
    taught. The bank knows what it has seen, and that is the bound used.
    """

    def __init__(
        self,
        interface: ModelInterface,
        *,
        ridge: float = 1e-4,
        slack: float = 0.5,
        stabilizer: Stabilizer | None = None,
    ) -> None:
        """Shrink the coefficients by ``ridge``; clip ``slack`` outside the range."""
        super().__init__(interface, stabilizer=stabilizer)
        self.ridge = ridge
        self.slack = slack
        self.coef_: np.ndarray | None = None  # (n + 1, n): intercept then A'
        self._chol: np.ndarray | None = None
        self._low: np.ndarray | None = None
        self._high: np.ndarray | None = None

    def _fit(self) -> None:
        """Ridge-regress each feature on the previous row; keep the noise."""
        Y = self.features()
        if len(Y) < 3:
            raise ValueError(f"a VAR(1) needs at least 3 feature rows, got {len(Y)}")
        X = np.hstack([np.ones((len(Y) - 1, 1)), Y[:-1]])
        target = Y[1:]
        gram = X.T @ X + self.ridge * len(X) * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(gram, X.T @ target)

        residuals = target - X @ self.coef_
        cov = np.atleast_2d(np.cov(residuals.T))
        jitter = 1e-12 + 1e-10 * np.trace(cov) / len(cov)
        self._chol = np.linalg.cholesky(cov + jitter * np.eye(len(cov)))

        span = Y.max(axis=0) - Y.min(axis=0)
        self._low = Y.min(axis=0) - self.slack * span
        self._high = Y.max(axis=0) + self.slack * span

    def sample(
        self, k: int, rng: np.random.Generator, *, previous: np.ndarray | None = None
    ) -> np.ndarray:
        """Simulate ``k`` steps forward, cumulated onto ``previous``.

        The *process* is continued from the latest observed feature row -- "what
        does the exogenous environment look like now" is the forecast the bank
        actually has, whatever state a rollout happens to branch from. Only the
        reconstruction of levels is re-based, which is the part that has to agree
        with the position being driven.
        """
        assert self.coef_ is not None and self._chol is not None
        assert self._low is not None and self._high is not None
        y = self.features()[-1]
        path = np.empty((k, len(y)))
        for j in range(k):
            mean = self.coef_[0] + self.coef_[1:].T @ y
            y = np.clip(
                mean + self._chol @ rng.standard_normal(len(y)), self._low, self._high
            )
            path[j] = y
        return self.to_levels(path, self.last if previous is None else previous)


class BlockBootstrapForcing(ForcingModel):
    """Contiguous blocks of the observed feature path, replayed forward.

    Assumes nothing about the functional form and reproduces the history's
    autocorrelation, cross-correlation and tail shape exactly, because it *is*
    the history -- resampled. What it cannot do is produce a configuration the
    history never contained, which is precisely the way a bootstrap and a fitted
    VAR fail differently, and the reason to report both.
    """

    def __init__(
        self,
        interface: ModelInterface,
        *,
        block: int = 8,
        stabilizer: Stabilizer | None = None,
    ) -> None:
        """Replay the observed path in contiguous blocks of ``block`` steps."""
        super().__init__(interface, stabilizer=stabilizer)
        self.block = block
        self._feats: np.ndarray | None = None

    def _fit(self) -> None:
        """Cache the feature path; there is nothing to estimate."""
        self._feats = self.features()
        if len(self._feats) < 2:
            raise ValueError("a bootstrap needs at least two feature rows")

    def sample(
        self, k: int, rng: np.random.Generator, *, previous: np.ndarray | None = None
    ) -> np.ndarray:
        """Draw blocks until ``k`` steps are covered, cumulated onto ``previous``."""
        assert self._feats is not None
        drawn = []
        while len(drawn) < k:
            start = int(rng.integers(max(1, len(self._feats) - self.block)))
            drawn.extend(self._feats[start : start + self.block])
        return self.to_levels(
            np.array(drawn[:k]), self.last if previous is None else previous
        )


class OracleForcing(ForcingModel):
    """The forcing that is actually going to arrive -- a bound, not a method.

    Reads the future straight off the :class:`~control.world.Episode` the live
    run is being driven by, which is information the bank does not have and could
    not have. It answers exactly one question: how much of the deployment's
    remaining gap is *not knowing the exogenous environment*, as against the
    world model being wrong about the economy's response to it. Reported as an
    upper bound and labelled as one.

    The rows it reads are the episode's frozen, pre-stabilizer ones, the same the
    environment reads before applying the fiscal response -- so a caller
    stabilizes them exactly as it stabilizes forecast ones, and the two are
    comparable. (Which is also why the ``stabilizer`` it may be given is never
    applied to :attr:`future`: those rows have nothing in them to strip. It
    reaches only the observed history, and only so that :attr:`last` agrees with
    the other families.)

    What it knows is the exogenous *path*, and re-basing keeps it to that: the
    realised changes are replayed **onto the branch point**, rather than the
    realised levels being pasted next to a state from another period. Otherwise
    the one family that is meant to be the clean upper bound would be the one
    carrying the largest re-basing error.
    """

    def __init__(
        self,
        interface: ModelInterface,
        future: np.ndarray,
        *,
        stabilizer: Stabilizer | None = None,
    ) -> None:
        """Read the coming parameter rows off ``future``, oldest first."""
        super().__init__(interface, stabilizer=stabilizer)
        self.future = np.asarray(future, dtype=float)
        self.cursor = 0

    def _fit(self) -> None:
        """Nothing to estimate: the answer was handed over at construction."""

    def advance(self, steps: int = 1) -> None:
        """Move the read head on, as the live run consumes the real rows."""
        self.cursor += steps

    def sample(
        self, k: int, rng: np.random.Generator, *, previous: np.ndarray | None = None
    ) -> np.ndarray:
        """The next ``k`` rows that will arrive, re-based onto ``previous``."""
        window = self.future[self.cursor : self.cursor + k]
        if len(window) < k:
            tail = self.future[-1] if len(self.future) else self.last
            window = np.vstack([window, np.tile(tail, (k - len(window), 1))])
        if previous is None:
            return window
        # Differencing against the row *before* the window turns the realised
        # levels into the realised changes, which is the part that transfers.
        anchor = self.future[self.cursor - 1] if self.cursor else self.last
        feats = window.copy()
        feats[:, self.log_diff] = np.diff(
            np.log(np.vstack([anchor, window])[:, self.log_diff]), axis=0
        )
        return self.to_levels(feats, previous)
