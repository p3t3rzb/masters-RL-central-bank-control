"""An encoder-native proxy: forecast straight from the encoder's own dynamics.

The degenerate proxy that learns nothing on top of the encoder: it reads the
one-step forecast off the encoder's own generative model, whatever the chosen
:class:`~economic_models.encoders.base.GenerativeEncoder` predicts natively. With
the :class:`~economic_models.encoders.kalman.KalmanEncoder` that is the Kalman
prediction ``f_{t+1} = C (A z_t + B u_{t+1}) + noise``; with the
:class:`~economic_models.encoders.lstm.LSTMEncoder` it is the trained readout head
``f_{t+1} = Head([h_t, u_{t+1}]) + noise``. Either way it is the pure
"encoder alone" baseline -- what the learned state predicts by itself, before any
proxy adds its own estimator.

Because all the fitting lives in the encoder, :meth:`~EncoderNativeProxy._fit` is
a no-op; a :class:`~economic_models.encoders.base.GenerativeEncoder` is required,
checked at construction.
"""

from __future__ import annotations

import numpy as np

from economic_models.encoders import GenerativeEncoder
from economic_models.proxy.base import BaseProxyModel, FitData, StepContext
from economic_models.proxy.transform import StationarizingTransform


class EncoderNativeProxy(BaseProxyModel):
    """Forecast from the encoder's own generative one-step dynamics."""

    def __init__(
        self,
        *,
        encoder: GenerativeEncoder | None = None,
        transform: StationarizingTransform | None = None,
    ) -> None:
        """Configure the encoder-native proxy.

        ``encoder`` must be a :class:`GenerativeEncoder` (one with its own
        ``forecast()``), validated here; ``transform`` is the shared feature
        view. There are no estimator hyperparameters -- all learning lives in the
        encoder.
        """
        super().__init__(encoder=encoder, transform=transform)
        if not isinstance(self.encoder, GenerativeEncoder):
            raise TypeError(
                f"{type(self).__name__} needs a GenerativeEncoder (one with its "
                f"own forecast()); {type(self.encoder).__name__} is not one -- "
                "pass e.g. KalmanEncoder() or LSTMEncoder()"
            )

    def _fit(self, data: FitData) -> None:
        """No-op: all fitting lives in the encoder; this proxy only reads its forecast."""
        # All learning is the encoder's; this proxy only reads its forecast.
        pass

    def _predict_batch(
        self, z: np.ndarray, u_next: np.ndarray, f_prev: np.ndarray
    ) -> np.ndarray:
        """The encoder's own one-step forecast means from the latent design rows."""
        return self.encoder.forecast_means(z, u_next)

    def _predict_step(
        self, ctx: StepContext, rng: np.random.Generator | None
    ) -> np.ndarray:
        """Forecast the next row directly from the encoder belief (deterministic or stochastic)."""
        # Overrides the base template deliberately: the native step forecasts
        # from the *belief itself* (``encoder.forecast``) for both the
        # deterministic and stochastic paths, rather than routing the
        # deterministic case through the latent-based ``forecast_means``. The
        # two are mathematically identical when ``latent(belief)`` is the mean
        # the forecast reads, but the belief-based call is this proxy's defining
        # semantics.
        return self.encoder.forecast(ctx.belief, ctx.u_next, rng)

    def _sample_step(
        self, ctx: StepContext, rng: np.random.Generator
    ) -> np.ndarray:
        """One draw from the encoder's predictive distribution about its belief."""
        return self.encoder.forecast(ctx.belief, ctx.u_next, rng)
