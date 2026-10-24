"""Fixtures for the unit tier: hand-built inputs, nothing solved or fitted."""

from __future__ import annotations

from typing import Callable

import pytest

from economic_models.variables import ValueSpace


@pytest.fixture
def filled() -> Callable[..., ValueSpace]:
    """A builder for a value-space instance with every field at ``1.0``.

    The value spaces have no defaults, so a test that cares about two fields of
    sixteen would otherwise spell out all sixteen. ``filled(GrowthState, ER=0.9)``
    sets the fields named and leaves the rest at a harmless positive level.
    """

    def build(space: type[ValueSpace], **values: float) -> ValueSpace:
        unknown = set(values) - set(space.names())
        if unknown:
            raise ValueError(f"not fields of {space.__name__}: {sorted(unknown)}")
        return space.from_dict({name: values.get(name, 1.0) for name in space.names()})

    return build
