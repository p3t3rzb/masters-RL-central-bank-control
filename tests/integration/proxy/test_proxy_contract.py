"""The :class:`~economic_models.proxy.base.BaseProxyModel` contract, for every family.

The control stack treats proxies as interchangeable -- it reads only ``reset``,
``step``, ``snapshot``/``restore``, ``absorb`` and the estimator accessors -- so
every family is held to the same behaviour here, on a pair of short GROWTH runs.
A new estimator is covered by adding one line to :data:`FACTORIES`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from economic_models.encoders import KalmanEncoder
from economic_models.ground_truth import GROWTH_INTERFACE
from economic_models.proxy import (
    BaseProxyModel,
    DRFProxy,
    EncoderNativeProxy,
    KNNProxy,
    MDNProxy,
    RandomWalkProxy,
    VARXProxy,
)

#: Every proxy family, at the smallest size that still fits.
FACTORIES: dict[str, Callable[[], BaseProxyModel]] = {
    "varx": lambda: VARXProxy(GROWTH_INTERFACE, encoder=KalmanEncoder(latent_dim=3)),
    "drf": lambda: DRFProxy(
        GROWTH_INTERFACE, n_estimators=20, encoder=KalmanEncoder(latent_dim=3)
    ),
    "knn": lambda: KNNProxy(GROWTH_INTERFACE, k=5, encoder=KalmanEncoder(latent_dim=3)),
    "mdn": lambda: MDNProxy(
        GROWTH_INTERFACE, epochs=50, hidden=16, encoder=KalmanEncoder(latent_dim=3)
    ),
    "random-walk": lambda: RandomWalkProxy(GROWTH_INTERFACE),
    "encoder-native": lambda: EncoderNativeProxy(
        GROWTH_INTERFACE, encoder=KalmanEncoder(latent_dim=3)
    ),
}

HORIZON = 10
STATE_NAMES = GROWTH_INTERFACE.state.names()


@pytest.fixture(scope="module", params=list(FACTORIES))
def family(request) -> str:
    """The name of one proxy family."""
    return request.param


@pytest.fixture(scope="module")
def proxy(family, growth_runs) -> BaseProxyModel:
    """That family, fit on the shared GROWTH runs."""
    return FACTORIES[family]().fit(growth_runs)


@pytest.fixture(scope="module")
def drive(proxy, growth_holdout):
    """A warm-start window off the held-out run and the inputs that follow it."""
    w = proxy.required_window + 4
    h = growth_holdout
    window = (h.states[:w], h.params[:w], h.actions[:w])
    return window, h.params[w : w + HORIZON], h.actions[w : w + HORIZON], h.states[w]


def _inputs(params_row, actions_row):
    """Typed exogenous inputs from one row of each."""
    return (
        GROWTH_INTERFACE.parameters.from_row(params_row),
        GROWTH_INTERFACE.actions.from_row(actions_row),
    )


def _levels(state) -> np.ndarray:
    """A state's levels in the interface's column order."""
    return np.array([getattr(state, name) for name in STATE_NAMES])


# -- lifecycle ---------------------------------------------------------------


def test_unfitted_proxy_refuses_to_roll_out(family, growth_holdout):
    """Rolling out before fitting is a lifecycle error, not garbage."""
    fresh = FACTORIES[family]()
    h = growth_holdout
    with pytest.raises(RuntimeError, match="fit"):
        fresh.reset(h.states[:5], h.params[:5], h.actions[:5])


def test_step_needs_a_warm_start(family):
    """``step`` before ``reset`` says what to call."""
    with pytest.raises(RuntimeError, match="reset"):
        FACTORIES[family]().step(*_inputs(np.ones(7), np.ones(3)))


def test_short_window_is_rejected(proxy, growth_holdout):
    """A window shorter than :attr:`required_window` cannot seed the encoder."""
    h = growth_holdout
    n = proxy.required_window - 1
    with pytest.raises(ValueError, match="rows"):
        proxy.reset(h.states[:n], h.params[:n], h.actions[:n])


# -- rollouts ----------------------------------------------------------------


def test_rollout_returns_plausible_levels(proxy, drive):
    """One finite row per input row; trending levels stay positive."""
    window, params, actions, _ = drive
    levels = proxy.rollout(window, params, actions)
    assert levels.shape == (HORIZON, len(STATE_NAMES))
    assert np.all(np.isfinite(levels))
    positive = [STATE_NAMES.index(n) for n in GROWTH_INTERFACE.transform_spec.log_diff]
    assert np.all(levels[:, positive] > 0.0)


def test_mean_rollout_is_repeatable(proxy, drive):
    """Without an ``rng`` a rollout is the conditional mean, every time."""
    window, params, actions, _ = drive
    np.testing.assert_allclose(
        proxy.rollout(window, params, actions),
        proxy.rollout(window, params, actions),
        rtol=1e-12,
    )


def test_seeded_rollout_is_reproducible_and_noisy(proxy, drive):
    """The same seed redraws the same path, and that path is not the mean."""
    window, params, actions, _ = drive
    draw = proxy.rollout(window, params, actions, rng=np.random.default_rng(0))
    again = proxy.rollout(window, params, actions, rng=np.random.default_rng(0))
    np.testing.assert_allclose(draw, again, rtol=1e-12)
    assert not np.allclose(draw, proxy.rollout(window, params, actions))


def test_advance_is_the_mean_step(proxy, drive):
    """The family-shared driver is exactly the deterministic ``step``."""
    window, params, actions, _ = drive
    inputs = _inputs(params[0], actions[0])
    proxy.reset(*window)
    advanced = proxy.advance(*inputs)
    proxy.reset(*window)
    np.testing.assert_allclose(
        _levels(advanced), _levels(proxy.step(*inputs)), rtol=1e-12
    )


def test_predict_mean_is_what_step_would_take(proxy, drive):
    """The estimator, read out of a rollout, agrees with the step that moves it."""
    window, params, actions, _ = drive
    inputs = _inputs(params[0], actions[0])
    proxy.reset(*window)
    features = proxy.predict_mean(proxy.context(*inputs))
    levels = proxy.transform.invert_states(features, window[0][-1])
    np.testing.assert_allclose(levels, _levels(proxy.step(*inputs)), rtol=1e-12)


def test_snapshot_restore_redraws_the_same_period(proxy, drive):
    """Restoring and redrawing under the same stream lands on the same state."""
    window, params, actions, _ = drive
    inputs = _inputs(params[0], actions[0])
    proxy.reset(*window)
    snapshot = proxy.snapshot()
    first = proxy.step(*inputs, rng=np.random.default_rng(5))
    proxy.step(*_inputs(params[1], actions[1]))
    proxy.restore(snapshot)
    assert len(proxy.solutions) == 0
    assert proxy.step(*inputs, rng=np.random.default_rng(5)) == first


def test_absorb_stands_where_the_truth_does(proxy, drive):
    """Teacher forcing lands the rollout exactly on the realised period."""
    window, params, actions, realised = drive
    inputs = _inputs(params[0], actions[0])
    proxy.reset(*window)
    state = GROWTH_INTERFACE.state.from_row(realised)
    proxy.absorb(state, *inputs)
    assert proxy.state == state
    assert proxy.parameters == inputs[0] and proxy.actions == inputs[1]


def test_batch_predictions_cover_every_transition(proxy, growth_holdout):
    """One-step diagnostics return one prediction per consecutive feature pair."""
    F, U = proxy.transform.transform_run(growth_holdout)
    predicted = proxy.predict_features(F, U)
    assert predicted.shape == (len(F) - 1, F.shape[1])
    assert np.all(np.isfinite(predicted))
