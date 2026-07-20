"""Base class for proxy (learned surrogate) economic models.

Owns everything a proxy shares regardless of its estimator, three pieces of
plumbing:

* **a stationary feature view** -- a proxy is fit on the
  :class:`~economic_models.proxy.transform.StationarizingTransform` view of a
  run, not on raw trending levels; this base owns every level<->feature
  conversion.
* **a swappable state encoder** -- the conditioning latent summarising state and
  past (any :class:`~economic_models.encoders.base.StateEncoder`, Kalman by
  default). This base fits the encoder, batch-encodes training runs, and carries
  an online belief through a rollout.
* **open-loop simulation** -- :meth:`reset` warm-starts belief and level history
  from a trailing window, :meth:`step` advances one period under given exogenous
  inputs, :meth:`rollout` chains the two over a whole exogenous path.

A concrete proxy therefore reduces to a memoryless map from ``(latent, next
exog)`` to the next state-feature row: it implements :meth:`~BaseProxyModel._fit`,
:meth:`~BaseProxyModel._predict_batch` and :meth:`~BaseProxyModel._sample_step`
in feature space. The deterministic step routes through
:meth:`~BaseProxyModel._predict_batch`, so the batch and rollout paths cannot
drift apart.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from economic_models.base import BaseEconomicModel
from economic_models.encoders import KalmanEncoder, StateEncoder
from economic_models.interface import ModelInterface
from economic_models.proxy.transform import StationarizingTransform
from economic_models.variables import Actions, Parameters, State

if TYPE_CHECKING:
    from economic_models.ground_truth import Run


@dataclass
class FitData:
    """Pooled one-step training pairs, aligned row-for-row across all runs.

    Every row is one supervised example: predict the next state-feature row ``y``
    from the encoder latent ``z`` at the current period and the exogenous features
    ``u_next`` applied in the target period. ``f_prev`` is the current state-
    feature row (the persistence baseline the random walk uses). Under a
    :class:`~economic_models.encoders.base.NullEncoder`, ``z`` is an ``(n, 0)``
    array, so stacking it into a design matrix is a no-op.
    """

    z: np.ndarray  # (n, latent_dim) latent at the current period
    u_next: np.ndarray  # (n, n_exog_feat) exog features of the target period
    f_prev: np.ndarray  # (n, n_state_feat) current state-feature row
    y: np.ndarray  # (n, n_state_feat) target next state-feature row


@dataclass(frozen=True)
class StepContext:
    """Everything one online prediction step conditions on.

    ``belief`` is the opaque encoder belief (for proxies that read the encoder's
    own forecast); ``z`` the conditioning latent extracted from it; ``u_next``
    the exog features of the period being predicted; ``f_prev`` the current
    state-feature row.
    """

    belief: Any
    z: np.ndarray
    u_next: np.ndarray
    f_prev: np.ndarray


class BaseProxyModel(BaseEconomicModel):
    """A learned surrogate fit on ground-truth runs, in a transform's feature space.

    Subclasses supply the memoryless feature-space estimator; this base owns the
    transform, the state encoder, the rollout belief, and the shared visible
    interface.
    """

    def __init__(
        self,
        interface: ModelInterface,
        encoder: StateEncoder | None = None,
        transform: StationarizingTransform | None = None,
    ) -> None:
        """Wire up the shared proxy plumbing for the model ``interface`` it mimics.

        ``interface`` is the ground-truth model's
        :class:`~economic_models.interface.ModelInterface` -- its value spaces and
        stationarization spec; it has no default, so the caller always says which
        model this proxy stands in for. ``encoder`` is the state encoder to
        condition on (defaults to a :class:`KalmanEncoder`); ``transform`` is the
        stationary feature view to fit in (defaults to one built from
        ``interface.transform_spec``).
        """
        self._interface = interface
        # The value spaces this proxy observes through, as instance attrs so
        # ``self.STATE`` resolves the same way it does on a ground-truth model.
        self.STATE = interface.state
        self.PARAMETERS = interface.parameters
        self.ACTIONS = interface.actions
        self._transform = transform or StationarizingTransform(interface.transform_spec)
        # The filtered latent of a linear-Gaussian state-space model by default;
        # an encoder-free proxy passes a NullEncoder instead.
        self._encoder = encoder or KalmanEncoder()
        self._fitted = False

        # Rollout buffers, populated by :meth:`reset` and advanced by :meth:`step`.
        self._belief: Any = None  # opaque encoder belief
        self._feat_prev: np.ndarray | None = None  # last state-feature row
        self._levels_prev: np.ndarray | None = None  # previous state levels
        self._exog_prev: np.ndarray | None = None  # previous exog levels
        self._solutions: list[dict[str, float]] = []  # recorded level history

    # -- components (read-only) ----------------------------------------------

    @property
    def encoder(self) -> StateEncoder:
        """The state encoder this proxy conditions on (set at construction)."""
        return self._encoder

    @property
    def transform(self) -> StationarizingTransform:
        """The stationary feature view this proxy is fit in (set at construction)."""
        return self._transform

    # -- concrete-estimator contract ---------------------------------------

    @staticmethod
    def _design(z: np.ndarray, u_next: np.ndarray) -> np.ndarray:
        """The shared design-matrix convention: latent stacked with next exog.

        Works row-wise (1-D inputs) and batch-wise (2-D inputs) alike; a
        zero-width latent stacks away to just the exog block.
        """
        return np.hstack([z, u_next])

    @abstractmethod
    def _fit(self, data: FitData) -> None:
        """Estimate the memoryless one-step map from pooled training pairs."""

    @abstractmethod
    def _predict_batch(
        self, z: np.ndarray, u_next: np.ndarray, f_prev: np.ndarray
    ) -> np.ndarray:
        """Batch conditional-mean predictions for aligned context arrays.

        The deterministic estimator core, over the arrays
        :meth:`_context_arrays` builds; returns one prediction per row. The
        online deterministic step routes through it too (via
        :meth:`_predict_step`), so batch diagnostics and rollouts share one
        code path.
        """

    @abstractmethod
    def _sample_step(
        self, ctx: StepContext, rng: np.random.Generator
    ) -> np.ndarray:
        """One stochastic one-step draw from the fitted noise model."""

    def _predict_step(
        self, ctx: StepContext, rng: np.random.Generator | None
    ) -> np.ndarray:
        """One-step-ahead state-feature row for the online rollout.

        A template: the deterministic path is the single-row case of
        :meth:`_predict_batch`; the stochastic path is :meth:`_sample_step`.
        Subclasses normally override neither (the encoder-native proxy is the
        exception -- it forecasts from the belief itself).
        """
        if rng is None:
            return self._predict_batch(
                ctx.z[None], ctx.u_next[None], ctx.f_prev[None]
            )[0]
        return self._sample_step(ctx, rng)

    # -- training ----------------------------------------------------------

    def _context_arrays(
        self, F: np.ndarray, U: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One run's aligned ``(z, u_next, f_prev, y)`` one-step pairs."""
        z = self._encoder.encode_run(F, U)
        return z[:-1], U[1:], F[:-1], F[1:]

    def fit(self, runs: list[Run]) -> Self:
        """Fit the proxy on ground-truth runs, in the chosen feature space."""
        feature_runs = [self._transform.transform_run(run) for run in runs]
        self._encoder.fit(feature_runs)

        blocks = [self._context_arrays(F, U) for F, U in feature_runs]
        data = FitData(
            z=np.vstack([b[0] for b in blocks]),
            u_next=np.vstack([b[1] for b in blocks]),
            f_prev=np.vstack([b[2] for b in blocks]),
            y=np.vstack([b[3] for b in blocks]),
        )
        self._fit(data)
        self._fitted = True
        return self

    def predict_features(self, F: np.ndarray, U: np.ndarray) -> np.ndarray:
        """One-step-ahead conditional-mean predictions for a run (diagnostics)."""
        if not self._fitted:
            raise RuntimeError("proxy must be fit before it can predict")
        z_ctx, u_next, f_prev, _ = self._context_arrays(F, U)
        return self._predict_batch(z_ctx, u_next, f_prev)

    # -- simulation --------------------------------------------------------

    @property
    def required_window(self) -> int:
        """Level rows a warm-start window must supply for :meth:`reset`."""
        return self._encoder.min_window + 1

    def reset(
        self, states: np.ndarray, params: np.ndarray, actions: np.ndarray
    ) -> None:
        """Warm-start the belief and level history from a trailing window.

        Needs at least :attr:`required_window` level rows so the encoder can seed
        its belief from enough feature rows.
        """
        if not self._fitted:
            raise RuntimeError("proxy must be fit before it can be rolled out")
        if len(states) < self.required_window:
            raise ValueError(
                f"need at least {self.required_window} rows, got {len(states)}"
            )
        feats = self._transform.transform_states(states)
        exog_feats = self._transform.transform_exog(params, actions)
        self._belief = self._encoder.init_belief(feats, exog_feats)
        self._feat_prev = feats[-1]
        self._levels_prev = states[-1].astype(float)
        self._exog_prev = np.hstack([params[-1], actions[-1]]).astype(float)
        self._solutions = []

    def advance(self, parameters: Parameters, actions: Actions) -> State:
        """Deterministically advance one period (the family-shared driver).

        The proxy's implementation of the abstract
        :meth:`~economic_models.BaseEconomicModel.advance`: exactly the
        conditional-mean :meth:`step`. :meth:`step` remains the proxy-specific
        richer API (its ``rng`` enables stochastic rollouts).
        """
        return self.step(parameters, actions)

    def step(
        self,
        parameters: Parameters,
        actions: Actions,
        rng: np.random.Generator | None = None,
    ) -> State:
        """Advance one period under the given exogenous inputs.

        Deterministic (conditional mean) unless ``rng`` is given, in which case a
        residual draw from the fitted noise model is added. Requires a warm start:
        call :meth:`reset` (or :meth:`rollout`, which resets for you) first.
        """
        if self._feat_prev is None:
            raise RuntimeError(
                "proxy has no warm start; call reset() (or rollout(), which resets "
                "for you) before step()/advance()"
            )
        exog_now = np.array(
            [*parameters.to_dict().values(), *actions.to_dict().values()]
        )
        u_next = self._transform.transform_exog_row(exog_now, self._exog_prev)
        ctx = StepContext(
            belief=self._belief,
            z=self._encoder.latent(self._belief),
            u_next=u_next,
            f_prev=self._feat_prev,
        )

        features = self._predict_step(ctx, rng)
        levels = self._transform.invert_states(features, self._levels_prev)

        self._belief = self._encoder.advance(self._belief, features, u_next)
        self._feat_prev = features
        self._levels_prev = levels
        self._exog_prev = exog_now

        state = self.STATE.from_dict(dict(zip(self._transform.state_names, levels)))
        self._solutions.append(state.to_dict())
        return state

    def rollout(
        self,
        window: tuple[np.ndarray, np.ndarray, np.ndarray],
        params: np.ndarray,
        actions: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate ``len(params)`` periods from a warm-start window; returns levels."""
        self.reset(*window)
        param_names, action_names = self.PARAMETERS.names(), self.ACTIONS.names()
        levels = [
            self.step(
                self.PARAMETERS.from_dict(dict(zip(param_names, params[t]))),
                self.ACTIONS.from_dict(dict(zip(action_names, actions[t]))),
                rng=rng,
            ).to_dict()
            for t in range(len(params))
        ]
        return np.array(
            [[row[name] for name in self._transform.state_names] for row in levels]
        )

    # -- visible interface -------------------------------------------------

    @property
    def state(self) -> State:
        """Latest simulated :class:`State` (available after the first :meth:`step`)."""
        if not self._solutions:
            raise RuntimeError("no state yet; call step()/rollout() first")
        return self.STATE.from_dict(self._solutions[-1])

    @property
    def parameters(self) -> Parameters:
        """Exogenous :class:`Parameters` applied in the latest :meth:`step`."""
        return self.PARAMETERS.from_dict(self._exog_dict())

    @property
    def actions(self) -> Actions:
        """Central-bank :class:`Actions` applied in the latest :meth:`step`."""
        return self.ACTIONS.from_dict(self._exog_dict())

    @property
    def solutions(self) -> list[dict[str, float]]:
        """Recorded per-step level history since the last :meth:`reset`."""
        return self._solutions

    def _exog_dict(self) -> dict[str, float]:
        """Latest exogenous levels keyed by name (raises before any step)."""
        if self._exog_prev is None:
            raise RuntimeError("no exogenous inputs yet; call reset()/step() first")
        return dict(zip(self._transform.exog_names, self._exog_prev))
