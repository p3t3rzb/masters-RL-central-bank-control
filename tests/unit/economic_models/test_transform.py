"""The stationarizing transform every proxy and the observation are fit in.

A proxy predicts *features* and rolls out in *levels*, so the transform has to be
exactly invertible one row at a time; anything less and an open-loop rollout
drifts for a reason that has nothing to do with the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from economic_models.ground_truth.models.growth.interface import GROWTH_INTERFACE
from economic_models.ground_truth.models.smets_wouters.interface import SW_INTERFACE
from economic_models.proxy.transform import StationarizingTransform
from economic_models.run import Run

INTERFACES = [GROWTH_INTERFACE, SW_INTERFACE]
IDS = ["growth", "smets_wouters"]


def _levels(interface, steps: int = 12, seed: int = 0) -> Run:
    """A run of strictly positive, gently trending synthetic levels."""
    rng = np.random.default_rng(seed)
    ts = interface.transform_spec
    trend = np.exp(np.cumsum(rng.normal(0.01, 0.02, size=(steps, 1)), axis=0))
    states = trend * rng.uniform(0.5, 2.0, size=(steps, len(ts.state_names)))
    exog = rng.uniform(0.01, 2.0, size=(steps, len(ts.exog_names)))
    n_params = len(interface.parameters.names())
    return Run(states=states, params=exog[:, :n_params], actions=exog[:, n_params:])


@pytest.mark.parametrize("interface", INTERFACES, ids=IDS)
def test_state_features_follow_the_rules(interface):
    """Log-diff, ratio and pass-through columns each get their own treatment."""
    transform = StationarizingTransform(interface.transform_spec)
    run = _levels(interface)
    F = transform.transform_states(run.states)
    names = transform.state_names
    y = names.index(interface.transform_spec.denominator)

    assert F.shape == (len(run) - 1, len(names))
    for j, name in enumerate(names):
        if name in interface.transform_spec.log_diff:
            expected = np.log(run.states[1:, j] / run.states[:-1, j])
        elif name in interface.transform_spec.ratio_to:
            expected = run.states[1:, j] / run.states[1:, y]
        else:
            expected = run.states[1:, j]
        np.testing.assert_allclose(F[:, j], expected, rtol=1e-12, err_msg=name)


@pytest.mark.parametrize("interface", INTERFACES, ids=IDS)
def test_state_features_invert_row_by_row(interface):
    """Each feature row and the previous levels recover the next level row exactly."""
    transform = StationarizingTransform(interface.transform_spec)
    run = _levels(interface)
    F = transform.transform_states(run.states)
    recovered = np.array(
        [transform.invert_states(F[i], run.states[i]) for i in range(len(F))]
    )
    np.testing.assert_allclose(recovered, run.states[1:], rtol=1e-12)


@pytest.mark.parametrize("interface", INTERFACES, ids=IDS)
def test_exog_row_form_matches_batch_form(interface):
    """The online one-row path and the batch path cannot drift apart."""
    transform = StationarizingTransform(interface.transform_spec)
    run = _levels(interface)
    U = transform.transform_exog(run.params, run.actions)
    exog = np.hstack([run.params, run.actions])
    rows = np.array(
        [transform.transform_exog_row(exog[t], exog[t - 1]) for t in range(1, len(run))]
    )
    np.testing.assert_allclose(rows, U, rtol=1e-12)


@pytest.mark.parametrize("interface", INTERFACES, ids=IDS)
def test_transform_run_is_both_halves(interface):
    """``transform_run`` is the state and exogenous transforms, row-aligned."""
    transform = StationarizingTransform(interface.transform_spec)
    run = _levels(interface)
    F, U = transform.transform_run(run)
    np.testing.assert_array_equal(F, transform.transform_states(run.states))
    np.testing.assert_array_equal(U, transform.transform_exog(run.params, run.actions))
    assert len(F) == len(U) == len(run) - 1


def test_feature_names_describe_the_treatment():
    """Names read as the feature they label, latent-free and in column order."""
    transform = StationarizingTransform(GROWTH_INTERFACE.transform_spec)
    names = transform.state_feature_names
    assert names[transform.state_names.index("Y")] == "dlog(Y)"
    assert names[transform.state_names.index("GD")] == "GD/Y"
    assert names[transform.state_names.index("PI")] == "PI"
    assert "dlog(Nfe)" in transform.exog_feature_names
    assert "Rbbar" in transform.exog_feature_names


def test_non_positive_level_is_named_not_nan():
    """A slump that drives a trending level to zero fails loudly, naming the series."""
    transform = StationarizingTransform(GROWTH_INTERFACE.transform_spec)
    run = _levels(GROWTH_INTERFACE)
    states = run.states.copy()
    states[5, transform.state_names.index("Ik")] = 0.0
    with pytest.raises(ValueError, match="Ik"):
        transform.transform_states(states)
