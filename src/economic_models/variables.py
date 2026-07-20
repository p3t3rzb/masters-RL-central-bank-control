"""The central-bank-visible interface, as three abstract whole-state value spaces.

* :class:`State`      -- the endogenous macro observables (GDP, inflation, ...).
* :class:`Parameters` -- the exogenous levers the bank observes but does **not**
  control (fiscal stance, productivity growth, loan spreads, ...).
* :class:`Actions`    -- the levers the central bank controls (policy rate,
  capital requirement, reserve requirement, ...).

These three are *abstract* here: field-less base classes that fix the tripartite
split (endogenous / exogenous-observed / controlled) every model shares, without
committing to any particular economy's quantities. Each ground-truth model
specializes them with its own fields -- GROWTH, for instance, defines
:class:`~economic_models.ground_truth.growth.variables.GrowthState` and friends.

Splitting ``Parameters`` from ``Actions`` lets a scenario be written as an
initial state plus the exogenous parameters over time -- without actions -- and
replayed against any model, while a policy supplies the actions at run time.
These objects only carry values for already-registered model quantities; a
model's hidden internals are owned by the concrete model itself.
"""

from __future__ import annotations

from dataclasses import Field, dataclass, field, fields
from typing import Any, ClassVar, Mapping, TypeVar

T = TypeVar("T", bound="ValueSpace")


def _observable(desc: str) -> float:
    """A required ``float`` field carrying a human-readable description."""
    return field(metadata={"desc": desc})  # type: ignore[return-value]


class ValueSpace:
    """Mixin for a fixed, fully-specified group of named model quantities.

    Subclasses are frozen dataclasses whose fields are the visible quantities.
    This mixin adds conversion to/from the flat ``{name: value}`` dicts that
    :mod:`pysolve` speaks, plus introspection over the field names.
    """

    # Every concrete subclass is a ``@dataclass``; declaring the attribute the
    # decorator injects lets the type checker treat ``cls``/``self`` as dataclasses
    # in the :func:`fields` calls below.
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """The model-quantity names carried by this space, in declaration order."""
        return tuple(f.name for f in fields(cls))

    @classmethod
    def describe(cls) -> dict[str, str]:
        """Map each field name to its human-readable description."""
        return {f.name: f.metadata.get("desc", "") for f in fields(cls)}

    @classmethod
    def from_dict(cls: type[T], values: Mapping[str, float]) -> T:
        """Build an instance by picking this space's names out of ``values``."""
        return cls(**{name: values[name] for name in cls.names()})

    def to_dict(self) -> dict[str, float]:
        """Flatten to a ``{name: value}`` dict the solver can consume."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class State(ValueSpace):
    """The central-bank-visible endogenous state: the macro observables.

    Abstract: a field-less base a concrete model subclasses with its own
    observable aggregates (see
    :class:`~economic_models.ground_truth.growth.variables.GrowthState`).
    """


@dataclass(frozen=True)
class Parameters(ValueSpace):
    """Exogenous levers the bank observes but does not control.

    Fiscal stance, structural/productivity assumptions and lending conditions --
    the parts of a scenario that vary over time independently of monetary policy.
    Abstract: each model supplies its own fields.
    """


@dataclass(frozen=True)
class Actions(ValueSpace):
    """The levers the central bank directly controls.

    Abstract: each model supplies its own fields.
    """
