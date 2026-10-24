"""The ground-truth registry and each model's :class:`ModelInterface`.

A :class:`~economic_models.ground_truth.registry.GroundTruthSpec` is a lookup
table the control and data-generation stacks trust without checking, so its
internal consistency -- action boxes naming the model's levers, the resting
levers inside the box, transform rules naming real columns -- is checked here.
"""

from __future__ import annotations

import pytest

from control.env import ACTION_BOUNDS
from economic_models.ground_truth.registry import (
    DEFAULT_MODEL,
    GROWTH_ACTION_BOUNDS,
    MODELS,
    ground_truth,
)

SPECS = list(MODELS.values())
IDS = list(MODELS)


def test_default_model_is_registered():
    """Everything that names no model falls back on one that exists."""
    assert DEFAULT_MODEL in MODELS


def test_unknown_model_is_rejected_by_name():
    """A typo in a config fails loudly and lists what was available."""
    with pytest.raises(ValueError, match="unknown ground-truth model"):
        ground_truth("godley")


def test_env_action_bounds_match_the_registry():
    """``control.env`` keeps its own copy of GROWTH's box; the two must not drift."""
    assert dict(ACTION_BOUNDS) == dict(GROWTH_ACTION_BOUNDS)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_spec_is_registered_under_its_own_name(spec):
    """The key a config selects by is the name the spec reports."""
    assert ground_truth(spec.name) is spec


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_action_box_covers_exactly_the_levers(spec):
    """One non-empty interval per action, and none for anything else."""
    assert set(spec.action_bounds) == set(spec.interface.actions.names())
    for low, high in spec.action_bounds.values():
        assert low < high


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_resting_levers_sit_inside_the_box(spec):
    """Doing nothing must be an action the environment can express."""
    baseline = spec.baseline_actions()
    assert set(baseline) == set(spec.interface.actions.names())
    for name, value in baseline.items():
        low, high = spec.action_bounds[name]
        assert low <= value <= high, name


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_resting_economy_is_inside_its_corridor(spec):
    """The employment target cannot itself count as a collapse."""
    low, high = spec.er_bounds
    assert low < spec.employment_target <= high
    assert spec.pi_bounds[0] < 0.0 < spec.pi_bounds[1]


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_reference_lever_is_an_action(spec):
    """The Taylor references move a lever the model actually has."""
    assert spec.reference_lever in spec.interface.actions.names()


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_every_named_preset_builds(spec):
    """Each advertised excitation preset is a classmethod of the family."""
    for preset in spec.excitations:
        assert isinstance(spec.excitation_config(preset), spec.excitation)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_unknown_preset_is_rejected(spec):
    """A preset the family does not offer is an error, not an attribute lookup."""
    with pytest.raises(ValueError, match="unknown excitation"):
        spec.excitation_config("apocalyptic")


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_transform_spec_is_aligned_with_the_interface(spec):
    """The stationarizing rules name real columns, in the run's column order."""
    interface = spec.interface
    ts = interface.transform_spec
    state, exog = (
        interface.state.names(),
        (
            *interface.parameters.names(),
            *interface.actions.names(),
        ),
    )
    assert ts.state_names == state
    assert ts.exog_names == exog
    assert set(ts.log_diff) <= set(state)
    assert set(ts.ratio_to) <= set(state)
    assert ts.denominator in state
    assert set(ts.exog_log_diff) <= set(exog)
    assert not set(ts.log_diff) & set(ts.ratio_to)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_mandate_reads_columns_the_model_has(spec):
    """``Yk`` is log-differenced into the growth leg, so it must be a trending column."""
    assert "Yk" in spec.interface.transform_spec.log_diff
    assert {"ER", "PI"} <= set(spec.interface.state.names())
