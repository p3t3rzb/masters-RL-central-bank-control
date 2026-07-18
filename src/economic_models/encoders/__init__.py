"""State encoders: how a proxy summarizes state plus past into one latent vector.

Every proxy forecasts from a fixed-dimensional conditioning latent produced by a
:class:`StateEncoder`:

* :class:`KalmanEncoder` (the default) -- the Kalman-filtered latent of a learned
  linear-Gaussian state-space model;
* :class:`LSTMEncoder` -- its nonlinear sibling, a gated recurrent network;
* :class:`NullEncoder` -- a zero-width latent for the encoder-free random-walk
  baseline.

Both learned encoders are :class:`GenerativeEncoder`\\ s: they carry their own
one-step feature forecast, which the encoder-native proxy reads off directly.
"""

from economic_models.encoders.base import (
    GenerativeEncoder,
    NullEncoder,
    StateEncoder,
)
from economic_models.encoders.kalman import KalmanEncoder
from economic_models.encoders.lstm import LSTMEncoder

__all__ = [
    "StateEncoder",
    "GenerativeEncoder",
    "NullEncoder",
    "KalmanEncoder",
    "LSTMEncoder",
]
