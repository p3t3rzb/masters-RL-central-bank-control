"""The tuning search space: named dimensions a policy's knobs are drawn from."""

from __future__ import annotations

import optuna
import pytest

from control.tuning.space import Categorical, Integer, Real, SearchSpace


@pytest.fixture
def space() -> SearchSpace:
    """A three-knob space, one of each kind."""
    return SearchSpace(
        [
            Real("phi_y", 0.0, 1.0),
            Integer("window", 1, 8),
            Categorical("encoder", ("kalman", "lstm")),
        ]
    )


def test_space_keeps_declaration_order(space):
    """Names are the keys a policy is built from, in the order declared."""
    assert space.names == ("phi_y", "window", "encoder")
    assert len(space) == 3


def test_space_rejects_repeated_names():
    """Two dimensions sharing a name would silently drop a knob."""
    with pytest.raises(ValueError, match="duplicate"):
        SearchSpace([Real("a", 0.0, 1.0), Integer("a", 0, 3)])


def test_suggest_draws_one_value_per_dimension(space):
    """A trial yields a full candidate keyed by dimension name."""
    trial = optuna.trial.FixedTrial({"phi_y": 0.25, "window": 3, "encoder": "lstm"})
    assert space.suggest(trial) == {"phi_y": 0.25, "window": 3, "encoder": "lstm"}


def test_describe_is_a_json_ready_report(space):
    """The run report records each dimension's kind and range."""
    assert space.describe() == {
        "phi_y": {"kind": "real", "low": 0.0, "high": 1.0, "log": False},
        "window": {"kind": "integer", "low": 1, "high": 8, "log": False},
        "encoder": {"kind": "categorical", "choices": ["kalman", "lstm"]},
    }
