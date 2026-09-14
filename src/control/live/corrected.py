"""The corrected world model: a fitted proxy read through an online residual.

``CorrectedProxy`` is a :class:`~economic_models.proxy.base.BaseProxyModel` whose
one-step law is another proxy's plus a correction:

    f~_{t+1} = mu(z_t, u_{t+1}, f_t) + c_t(x_{t+1}) + eta~_{t+1}

It is a proxy rather than a wrapper around one deliberately. Everything
downstream of the world model in this project -- :class:`~control.drivers.ProxyDriver`,
:class:`~control.env.CentralBankEnv`, the observation, the mandate, the whole
synthetic-rollout path -- is written against that one interface, and the corrected
model is only useful if it can be dropped into all of it without a line changing.
In particular it is **branchable**, which is what lets short on-policy rollouts be
restarted over and over from the real states a live run visited.

The estimator is shared with the proxy being corrected; the *rollout state* is
not. That split is the whole design: the inner proxy stays wherever its owner put
it (a live deployment keeps it teacher-forced against the realised economy, so
that snapshots taken from it are real branch points), while this object carries
its own belief and level history through whatever synthetic future it is asked
to imagine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from economic_models.proxy import BaseProxyModel, FitData, RolloutState, StepContext
from economic_models.variables import State

from control.live.residual import NullResidual, Residual, design


@dataclass(frozen=True)
class CorrectedRolloutState(RolloutState):
    """A rollout position that also records how far it is from a real state.

    The fast bias state is evidence about *now*, so a correction decays it over
    the steps of a synthetic rollout (see
    :meth:`~control.live.residual.Residual.correction`). That decay needs a step
    counter, and the counter has to travel with the snapshot: restoring a branch
    point must put the rollout back at depth zero, or the second rollout from a
    branch point would start already discounted by the first.
    """

    depth: int = 0


class CorrectedProxy(BaseProxyModel):
    """A fitted proxy plus a residual correction, behind the proxy interface."""

    def __init__(self, proxy: BaseProxyModel, residual: Residual | None = None) -> None:
        """Correct a **fitted** ``proxy`` with ``residual``.

        ``residual`` defaults to :class:`~control.live.residual.NullResidual`, in
        which case this model is exactly the proxy it wraps -- same mean, same
        draw off the same random stream. The encoder and the feature transform
        are the proxy's own instances rather than copies: they are fitted,
        stateless-after-fitting components, and two copies of a filter would be
        two filters that could disagree.
        """
        super().__init__(proxy.interface, encoder=proxy.encoder, transform=proxy.transform)
        self.proxy = proxy
        self.residual = residual or NullResidual(len(self.transform.state_feature_names))
        self._depth = 0
        #: predictive variance of the step just taken, per state feature.
        #:
        #: Recorded rather than returned because the proxy interface has one
        #: return value and every caller downstream wants the state. A caller
        #: that wants to be *pessimistic* about an imagined step -- charging a
        #: synthetic rollout for what the model does not know, which is the one
        #: thing that makes a large model-based policy step safe -- reads it
        #: here, immediately after :meth:`step`. Zero until the first step.
        self.last_variance_ = np.zeros(len(self.transform.state_feature_names))
        # The wrapped proxy is already fitted; this object adds no parameters of
        # its own, so it is usable immediately.
        self._fitted = True

    # -- the estimator, corrected --------------------------------------------

    def _predict_step(
        self, ctx: StepContext, rng: np.random.Generator | None
    ) -> np.ndarray:
        """The proxy's one-step law, shifted by the correction and re-spread.

        Overrides the base template rather than :meth:`_sample_step`, because the
        proxy families differ in which of the two they implement (the
        encoder-native one forecasts from the belief and overrides this method
        itself) and this is the single point all of them pass through.

        The mean is taken separately from the draw so that the proxy's own
        innovation can be isolated and handed to the correction, which decides
        whether to keep it (the null correction does, exactly) or replace it with
        a spread calibrated on the live leftovers.
        """
        x = design(ctx.z, ctx.u_next, ctx.f_prev)
        base = self.proxy.predict_mean(ctx)
        mean = base + self.residual.correction(x, self._depth)
        self.last_variance_ = self.residual.variance(x)
        if rng is None:
            return mean
        return mean + self.residual.noise(self.proxy.sample(ctx, rng) - base, x, rng)

    def _predict_batch(
        self, z: np.ndarray, u_next: np.ndarray, f_prev: np.ndarray
    ) -> np.ndarray:
        """Corrected conditional means for aligned context arrays (diagnostics).

        The batch twin of the mean above, and the one that answers "how much
        better is the corrected model than the raw one, one step ahead" over a
        whole run at once.
        """
        base = self.proxy.predict_means(z, u_next, f_prev)
        rows = design(z, u_next, f_prev)
        return base + np.vstack([self.residual.correction(row) for row in rows])

    def _sample_step(
        self, ctx: StepContext, rng: np.random.Generator
    ) -> np.ndarray:
        """One corrected draw; reached only through :meth:`_predict_step`."""
        return self._predict_step(ctx, rng)

    def _fit(self, data: FitData) -> None:
        """Refuse: this model corrects a proxy that is already fitted.

        The correction is not estimated from pooled historic pairs -- it is
        estimated from *residuals*, recursively, by whoever owns it (see
        :mod:`control.live.deploy`). Failing loudly here keeps the inherited
        :meth:`~economic_models.proxy.base.BaseProxyModel.fit` from looking like
        a way to train one.
        """
        raise RuntimeError(
            "a CorrectedProxy wraps an already-fitted proxy; fit that one, and "
            "update the correction with ResidualModel.seed()/update()"
        )

    # -- rollout position, with depth ----------------------------------------

    def reset(
        self, states: np.ndarray, params: np.ndarray, actions: np.ndarray
    ) -> None:
        """Warm-start from a window, at depth zero and a fresh noise block."""
        super().reset(states, params, actions)
        self._depth = 0
        self.residual.restart()

    def snapshot(self) -> CorrectedRolloutState:
        """The rollout position, carrying its distance from the last real state."""
        base = super().snapshot()
        return CorrectedRolloutState(
            belief=base.belief,
            feat_prev=base.feat_prev,
            levels_prev=base.levels_prev,
            exog_prev=base.exog_prev,
            n_solutions=base.n_solutions,
            depth=self._depth,
        )

    def restore(self, snapshot: RolloutState) -> None:
        """Put the rollout back, and the bias decay back with it."""
        super().restore(snapshot)
        self._depth = getattr(snapshot, "depth", 0)
        self.residual.restart()

    def _advance_rollout(
        self,
        features: np.ndarray,
        levels: np.ndarray,
        u_next: np.ndarray,
        exog_now: np.ndarray,
    ) -> State:
        """Advance one period and record that the rollout got one step further."""
        self._depth += 1
        return super()._advance_rollout(features, levels, u_next, exog_now)
