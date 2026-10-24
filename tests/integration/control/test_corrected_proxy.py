"""The live deployment's corrected proxy, under the null correction.

The deployment compares an adapting agent against a frozen one, and the
comparison is only fair if wrapping a proxy in a
:class:`~control.live.corrected.CorrectedProxy` with nothing to correct changes
*nothing*: the same means and the same draws off the same random stream.
"""

from __future__ import annotations

import numpy as np
import pytest

from control.live.corrected import CorrectedProxy
from economic_models.encoders import KalmanEncoder
from economic_models.ground_truth import GROWTH_INTERFACE
from economic_models.proxy import VARXProxy


@pytest.fixture(scope="module")
def proxy(growth_runs) -> VARXProxy:
    """A small Kalman-encoded VARX fit on the shared runs."""
    return VARXProxy(GROWTH_INTERFACE, encoder=KalmanEncoder(latent_dim=3)).fit(growth_runs)


@pytest.fixture(scope="module")
def drive(proxy, growth_holdout):
    """A warm-start window and the next ten periods of inputs."""
    w = proxy.required_window + 4
    h = growth_holdout
    return (h.states[:w], h.params[:w], h.actions[:w]), h.params[w : w + 10], h.actions[w : w + 10]


def test_null_correction_leaves_the_mean_alone(proxy, drive):
    """With nothing to correct the corrected rollout is the raw one."""
    window, params, actions = drive
    corrected = CorrectedProxy(proxy)
    np.testing.assert_allclose(
        corrected.rollout(window, params, actions), proxy.rollout(window, params, actions), rtol=1e-12
    )


def test_null_correction_leaves_the_draws_alone(proxy, drive):
    """Same stream, same draws -- the paired comparison rests on this."""
    window, params, actions = drive
    corrected = CorrectedProxy(proxy)
    np.testing.assert_allclose(
        corrected.rollout(window, params, actions, rng=np.random.default_rng(3)),
        proxy.rollout(window, params, actions, rng=np.random.default_rng(3)),
        rtol=1e-12,
    )


def test_corrected_proxy_refuses_to_be_fit(proxy, growth_runs):
    """The correction is estimated from live residuals, never by ``fit``."""
    with pytest.raises(RuntimeError, match="already-fitted"):
        CorrectedProxy(proxy).fit(growth_runs)


def test_corrected_proxy_shares_the_fitted_components(proxy):
    """One filter and one transform, never two copies that could disagree."""
    corrected = CorrectedProxy(proxy)
    assert corrected.encoder is proxy.encoder
    assert corrected.transform is proxy.transform
