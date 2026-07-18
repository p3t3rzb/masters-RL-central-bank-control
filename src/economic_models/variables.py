"""The central-bank-visible interface, as three frozen whole-state value classes.

* :class:`State`      -- the endogenous macro observables (GDP, inflation, ...).
* :class:`Parameters` -- the exogenous levers the bank observes but does **not**
  control (fiscal stance, productivity growth, loan spreads, ...).
* :class:`Actions`    -- the three levers the central bank controls (policy
  rate, capital requirement, reserve requirement).

Splitting ``Parameters`` from ``Actions`` lets a scenario be written as an
initial ``State`` plus the exogenous ``Parameters`` over time -- without actions
-- and replayed against any model, while a policy supplies the ``Actions`` at
run time. These objects only carry values for already-registered model
quantities; a model's hidden internals are owned by the concrete model itself.
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

    Shared by every model -- whatever economy sits behind it, the bank observes
    these same aggregates.
    """

    Y: float = _observable("Output at current prices (nominal GDP)")
    Yk: float = _observable("Real output")
    PI: float = _observable("Price inflation")
    ER: float = _observable("Employment rate")
    P: float = _observable("Price level")
    Ck: float = _observable("Real consumption")
    Ik: float = _observable("Gross investment in real terms")
    Rb: float = _observable("Interest rate on government bills")
    Rl: float = _observable("Interest rate on loans")
    Rm: float = _observable("Interest rate on deposits")
    Rbl: float = _observable("Interest rate on bonds")
    GD: float = _observable("Government debt")
    PSBR: float = _observable("Government deficit")
    Vk: float = _observable("Real wealth of households")
    Q: float = _observable("Tobin's Q")
    PE: float = _observable("Price earnings ratio")


@dataclass(frozen=True)
class Parameters(ValueSpace):
    """Exogenous levers the bank observes but does not control.

    Fiscal stance, structural/productivity assumptions and lending conditions --
    the parts of a scenario that vary over time independently of monetary policy.
    """

    GRg: float = _observable("Growth of real government expenditures")
    theta: float = _observable("Income tax rate")
    Nfe: float = _observable("Full employment level")
    GRpr: float = _observable("Growth rate of productivity")
    ADDbl: float = _observable("Spread between long-term interest rate and rate on bills")
    Rln: float = _observable("Normal interest rate on loans")
    NPLk: float = _observable("Proportion of Non-Performing loans")


@dataclass(frozen=True)
class Actions(ValueSpace):
    """The three levers the central bank directly controls."""

    Rbbar: float = _observable("Interest rate on bills, set exogenously (policy rate)")
    NCAR: float = _observable("Normal capital adequacy ratio of banks (capital requirement)")
    ro: float = _observable("Reserve requirement ratio on deposits")
