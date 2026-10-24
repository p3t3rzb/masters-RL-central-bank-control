"""The policy's observation: the proxy's stationary view plus an optional memory.

The observation has to be computable on the ground-truth side as well, so it is a
pure function of the trailing level rows and a belief that is passed in and
handed back -- never held. That purity is what makes branched draws
exchangeable, and it is checked directly here.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from control.observation import Observer
from economic_models.encoders import KalmanEncoder
from economic_models.ground_truth import GROWTH_INTERFACE


@pytest.fixture(scope="module")
def observer(growth_runs) -> Observer:
    """A Kalman-memory observer fit on one GROWTH history, with two start rows."""
    return Observer(GROWTH_INTERFACE, encoder=KalmanEncoder(latent_dim=3)).fit(
        growth_runs[0], branches=[30, 45]
    )


def _blocks(run, t):
    """The trailing two level rows ending at ``t``, as the environment keeps them."""
    return run.states[t - 1 : t + 1], np.hstack([run.params, run.actions])[
        t - 1 : t + 1
    ]


def test_latent_block_comes_last(observer):
    """Economic features keep their columns whichever encoder is in use."""
    names = observer.names
    assert len(names) == observer.dim
    assert names[-3:] == ("z0", "z1", "z2")
    assert observer.index("dlog(Y)") == 0


def test_memoryless_by_default():
    """With no encoder the observation is just the stationary features."""
    plain = Observer(GROWTH_INTERFACE)
    assert plain.encoder.latent_dim == 0
    assert not any(name.startswith("z") and name[1:].isdigit() for name in plain.names)


def test_start_beliefs_cover_the_declared_rows(observer, growth_runs):
    """A belief exists for each declared start and for the history's end -- no others."""
    end = len(growth_runs[0]) - 1
    assert sorted(observer.branch_beliefs_) == [30, 45, end]
    assert observer.branch_belief_ is observer.belief_at(end)
    with pytest.raises(KeyError, match="not fit with row"):
        observer.belief_at(10)


def test_observation_matches_the_batch_features(observer, growth_runs):
    """The online observation standardises the same row ``run_features`` builds."""
    run = growth_runs[0]
    row = 45
    obs, _ = observer.observe(*_blocks(run, row), observer.belief_at(row))
    raw = observer.run_features(run)[row - 1]
    expected = np.clip(
        (raw - observer.mean_) / observer.std_, -observer.clip, observer.clip
    )
    np.testing.assert_allclose(obs, expected, atol=1e-6)
    np.testing.assert_allclose(observer.unstandardise(obs), raw, atol=1e-6)


def test_observe_is_pure(observer, growth_runs):
    """The same inputs give the same observation, and the input belief is untouched."""
    run = growth_runs[0]
    belief = observer.belief_at(30)
    held = copy.deepcopy(belief)
    first, _ = observer.observe(*_blocks(run, 30), belief)
    again, _ = observer.observe(*_blocks(run, 30), belief)
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(
        observer.encoder.latent(belief), observer.encoder.latent(held)
    )


def test_unfitted_observer_cannot_standardise(growth_runs):
    """Standardisation statistics are frozen at fit time, so they must exist."""
    with pytest.raises(RuntimeError, match="fit"):
        Observer(GROWTH_INTERFACE).observe(*_blocks(growth_runs[0], 5), None)
