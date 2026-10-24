"""The value spaces: the typed currency every model and proxy speaks in.

Column order is load-bearing -- a :class:`~economic_models.run.Run` stores levels
as plain arrays ordered by :meth:`~economic_models.variables.ValueSpace.names`,
and every consumer rebuilds typed values from those rows -- so the round trips
between the dict, row and instance forms are pinned here for every concrete space.
"""

from __future__ import annotations

import dataclasses

import pytest

from economic_models.ground_truth.models.growth.variables import (
    GrowthActions,
    GrowthParameters,
    GrowthState,
)
from economic_models.ground_truth.models.smets_wouters.variables import (
    SwActions,
    SwParameters,
    SwState,
)
from economic_models.variables import Actions, Parameters, State

SPACES = [GrowthState, GrowthParameters, GrowthActions, SwState, SwParameters, SwActions]


@pytest.mark.parametrize("space", [State, Parameters, Actions])
def test_abstract_spaces_carry_no_fields(space):
    """The abstract bases fix the split only; every field belongs to a model."""
    assert space.names() == ()


@pytest.mark.parametrize("space", SPACES)
def test_every_field_is_described(space):
    """Each observable carries a non-empty human-readable description."""
    descriptions = space.describe()
    assert tuple(descriptions) == space.names()
    assert all(descriptions.values())


@pytest.mark.parametrize("space", SPACES)
def test_row_round_trip_preserves_order(space):
    """``from_row`` reads positionally in :meth:`names` order and ``to_dict`` gives it back."""
    row = [float(i) + 0.5 for i in range(len(space.names()))]
    value = space.from_row(row)
    assert list(value.to_dict().values()) == row
    assert list(value.to_dict()) == list(space.names())


@pytest.mark.parametrize("space", SPACES)
def test_from_dict_ignores_foreign_names(space):
    """A solver solution carries every model quantity; only this space's are picked."""
    values = {name: 2.0 for name in space.names()} | {"not_a_field": 99.0}
    assert space.from_dict(values).to_dict() == {name: 2.0 for name in space.names()}


def test_from_dict_requires_every_field():
    """A missing quantity is an error, never a silent default."""
    values = {name: 1.0 for name in GrowthActions.names() if name != "ro"}
    with pytest.raises(KeyError):
        GrowthActions.from_dict(values)


def test_values_are_frozen():
    """Observations are value objects: mutating one in place is refused."""
    actions = GrowthActions(Rbbar=0.035, NCAR=0.1, ro=0.05)
    with pytest.raises(dataclasses.FrozenInstanceError):
        actions.Rbbar = 0.04  # type: ignore[misc]


def test_shared_names_mean_the_same_across_models():
    """The mandate reads ``ER``, ``PI`` and ``Yk`` off either economy by name."""
    for name in ("Yk", "PI", "ER", "Y", "GD", "PSBR"):
        assert name in GrowthState.names()
        assert name in SwState.names()
    for name in ("GRpr", "Nfe"):
        assert name in GrowthParameters.names()
        assert name in SwParameters.names()
