"""The :class:`StateEncoder` interface a proxy conditions on.

A proxy predicts the next state-feature row from a summary of the current state
and the relevant past. The GROWTH observables are a partially observed
projection of a larger structural state, so that summary is a filtered state
estimate rather than the raw current row; abstracting it as a
:class:`StateEncoder` lets the summary be swapped (Kalman by default, an LSTM,
or the zero-width :class:`NullEncoder`).

An encoder has two responsibilities, mirroring how a proxy uses it: **batch
encode** (:meth:`~StateEncoder.encode_run`) turns a run's feature arrays into one
latent row per period for the training design; **online filter**
(:meth:`~StateEncoder.init_belief` / :meth:`~StateEncoder.advance` /
:meth:`~StateEncoder.latent`) carries a recursively updated, proxy-opaque belief
through a rollout one step at a time. A :class:`GenerativeEncoder` additionally
carries its own one-step feature forecast, which
:class:`~economic_models.proxy.encoder_native.EncoderNativeProxy` reads off.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np


class StateEncoder(ABC):
    """Map a run's history to a fixed-dimensional conditioning latent.

    Subclasses own whatever they need to fit; this interface fixes only how a
    proxy drives them: batch-encode training runs, and carry an online belief
    through a rollout.
    """

    #: whether :meth:`fit` has run; guards the data-dependent methods.
    _fitted: bool = False

    def _require_fitted(self) -> None:
        """Raise a clear error when a data-dependent method is called pre-fit."""
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fit before this method is called"
            )

    # -- fitting -------------------------------------------------------------

    @abstractmethod
    def fit(self, feature_runs: list[tuple[np.ndarray, np.ndarray]]) -> Self:
        """Learn the encoder from per-run ``(state_features, exog_features)`` pairs.

        Each element is one run's ``(F, U)``, both ``(T-1, .)`` aligned arrays as
        produced by :meth:`StationarizingTransform.transform_run`. A stateless
        encoder (the :class:`NullEncoder`) may treat this as a no-op. Returns
        ``self`` so fitting chains.
        """

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimension of the latent vector this encoder emits.

        Available *before* :meth:`fit`: every encoder answers it from its
        constructor arguments alone, so a proxy can size designs up front.
        """

    @property
    def min_window(self) -> int:
        """Feature rows needed to warm-start a belief in :meth:`init_belief`."""
        return 1

    # -- batch ---------------------------------------------------------------

    @abstractmethod
    def encode_run(self, F: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Latent row per period for one run, ``(len(F), latent_dim)``.

        Row ``i`` is the *causal* summary of history through period ``i`` -- it may
        use ``F[:i+1]`` / ``U[:i+1]`` but never a future row, so it matches what a
        rollout has available online. Requires a fitted encoder.
        """

    # -- online --------------------------------------------------------------

    @abstractmethod
    def init_belief(self, F_win: np.ndarray, U_win: np.ndarray) -> Any:
        """Warm-start an opaque belief from a trailing window of feature rows.

        Requires a fitted encoder.
        """

    @abstractmethod
    def latent(self, belief: Any) -> np.ndarray:
        """Extract the conditioning latent ``(latent_dim,)`` from a belief."""

    @abstractmethod
    def advance(self, belief: Any, f_next: np.ndarray, u_next: np.ndarray) -> Any:
        """Fold one newly realised step ``(f_next, u_next)`` into the belief.

        ``f_next`` is the feature row just produced for the new period and
        ``u_next`` its exogenous inputs; the returned belief is the one to
        condition the *following* step on.

        **Purity contract:** ``advance`` must *not* mutate the input belief; it
        returns a fresh belief. A caller may therefore hold on to an old belief
        (e.g. to branch several rollouts off one warm start) without defensive
        copies.
        """


class GenerativeEncoder(StateEncoder):
    """A state encoder with its own generative one-step feature forecast.

    Beyond summarising the past, a generative encoder can *predict* the next
    feature row from its belief -- it is a full sequence model, not only a
    filter. :class:`~economic_models.proxy.encoder_native.EncoderNativeProxy`
    requires one; other proxies work with any :class:`StateEncoder`.
    """

    @abstractmethod
    def forecast(
        self,
        belief: Any,
        u_next: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generative one-step feature forecast from the belief and next exog.

        Deterministic (predictive mean) unless ``rng`` is given, in which case a
        draw from the encoder's own predictive noise model is returned.
        """

    @abstractmethod
    def forecast_means(self, Z: np.ndarray, U_next: np.ndarray) -> np.ndarray:
        """Batch predictive means: next feature row from latents ``Z`` and next exog.

        The vectorised, deterministic twin of :meth:`forecast`, over the
        latents :meth:`~StateEncoder.encode_run` returns and the target
        periods' exog features; returns one predicted feature row per input row.
        """


class NullEncoder(StateEncoder):
    """The null object encoder: a zero-width latent for encoder-free proxies.

    A proxy that needs no summary of the past (the random walk) still speaks
    the same :class:`StateEncoder` protocol through this class instead of
    special-casing ``None`` everywhere: the latent has dimension 0, so stacking
    it into a design matrix is a no-op (``np.hstack([z, u]) == u``), fitting
    records nothing, and the belief is trivial.
    """

    _fitted = True  # nothing to fit; usable immediately

    @property
    def latent_dim(self) -> int:
        """Zero -- this encoder emits an empty latent."""
        return 0

    def fit(self, feature_runs: list[tuple[np.ndarray, np.ndarray]]) -> Self:
        """No-op fit: there is nothing to learn; returns ``self``."""
        return self  # nothing to learn

    def encode_run(self, F: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Return one zero-width latent row per period, ``(len(F), 0)``."""
        return np.zeros((len(F), 0))

    def init_belief(self, F_win: np.ndarray, U_win: np.ndarray) -> Any:
        """Return the trivial (``None``) belief; nothing to warm-start."""
        return None  # the trivial belief

    def latent(self, belief: Any) -> np.ndarray:
        """Return the empty latent ``(0,)``."""
        return np.zeros(0)

    def advance(self, belief: Any, f_next: np.ndarray, u_next: np.ndarray) -> Any:
        """Return the belief unchanged: stateless, so nothing to fold in."""
        # Stateless: the trivial belief never changes, so returning it unchanged
        # trivially satisfies the purity contract (there is nothing to mutate).
        return belief
