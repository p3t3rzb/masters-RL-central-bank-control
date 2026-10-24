"""The state encoders behind every proxy and the policy's observation.

Two properties are what make an encoder usable online: its batch latents are
**causal** (row ``i`` never reads a future row, so it is what a rollout can
compute), and :meth:`~economic_models.encoders.base.StateEncoder.advance` is
**pure** (a held belief can be branched from without copying). Both are checked
against the batch path the encoder is fit through.
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest

from economic_models.encoders import KalmanEncoder, LSTMEncoder, NullEncoder, StateEncoder
from economic_models.ground_truth import GROWTH_INTERFACE
from economic_models.proxy import StationarizingTransform

LATENT = 4

FACTORIES: dict[str, Callable[[], StateEncoder]] = {
    "kalman": lambda: KalmanEncoder(latent_dim=LATENT),
    "lstm": lambda: LSTMEncoder(latent_dim=LATENT, epochs=20),
    "null": lambda: NullEncoder(),
}


@pytest.fixture(scope="module")
def features(growth_runs):
    """The stationary ``(F, U)`` view of the shared GROWTH runs."""
    transform = StationarizingTransform(GROWTH_INTERFACE.transform_spec)
    return [transform.transform_run(run) for run in growth_runs]


@pytest.fixture(scope="module", params=list(FACTORIES))
def encoder(request, features) -> StateEncoder:
    """One encoder family, fit on the shared features."""
    return FACTORIES[request.param]().fit(features)


def test_latent_width_is_known_before_fitting():
    """A proxy sizes its design from the constructor alone."""
    assert KalmanEncoder(latent_dim=7).latent_dim == 7
    assert LSTMEncoder(latent_dim=5).latent_dim == 5
    assert NullEncoder().latent_dim == 0


@pytest.mark.parametrize("name", ["kalman", "lstm"])
def test_unfitted_encoder_refuses_data(name, features):
    """A learned encoder must be fit before it encodes anything."""
    F, U = features[0]
    with pytest.raises(RuntimeError, match="fit"):
        FACTORIES[name]().encode_run(F, U)


def test_encode_run_has_one_latent_per_period(encoder, features):
    """``(len(F), latent_dim)``, finite."""
    F, U = features[0]
    Z = encoder.encode_run(F, U)
    assert Z.shape == (len(F), encoder.latent_dim)
    assert np.all(np.isfinite(Z))


def test_batch_latents_are_causal(encoder, features):
    """Truncating the future leaves every earlier latent unchanged."""
    F, U = features[0]
    full = encoder.encode_run(F, U)
    prefix = encoder.encode_run(F[:20], U[:20])
    np.testing.assert_allclose(prefix, full[:20], atol=1e-8)


def test_online_belief_tracks_the_batch_latent(encoder, features):
    """Warm start then ``advance`` reproduces ``encode_run`` row for row."""
    F, U = features[0]
    full = encoder.encode_run(F, U)
    start = max(encoder.min_window, 10)
    belief = encoder.init_belief(F[:start], U[:start])
    np.testing.assert_allclose(encoder.latent(belief), full[start - 1], atol=1e-6)
    for t in range(start, start + 5):
        belief = encoder.advance(belief, F[t], U[t])
        np.testing.assert_allclose(encoder.latent(belief), full[t], atol=1e-6)


def test_advance_does_not_mutate_its_input(encoder, features):
    """The purity contract: an old belief is still the old belief."""
    F, U = features[0]
    belief = encoder.init_belief(F[:10], U[:10])
    held = copy.deepcopy(belief)
    encoder.advance(belief, F[10], U[10])
    np.testing.assert_array_equal(encoder.latent(belief), encoder.latent(held))


def test_kalman_transition_is_not_explosive(features):
    """The fitted ``A`` is capped at ``max_radius`` so rollouts stay bounded."""
    encoder = KalmanEncoder(latent_dim=LATENT, max_radius=0.95).fit(features)
    radius = np.max(np.abs(np.linalg.eigvals(encoder.A_)))
    assert radius <= 0.95 + 1e-8
    assert encoder.loglik_ and np.all(np.isfinite(encoder.loglik_))
