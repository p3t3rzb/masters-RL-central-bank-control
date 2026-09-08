"""The search space: what a policy's hyperparameters are, before any optimiser.

A hyperparameter search has two halves that want to move independently -- *what*
may be varied, which belongs to the policy, and *how* the next candidate is
chosen, which belongs to the optimiser. This module is the first half, written as
plain value objects: a :class:`Dimension` per knob and a :class:`SearchSpace`
holding them in order.

The only place the two halves meet is :meth:`Dimension.suggest`, which asks an
Optuna trial for a value of the right kind. Everything else -- a policy declaring
its knobs, a report describing them -- goes through this module without importing
an optimiser at all, so swapping the backend touches one method per dimension
rather than every policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import optuna


# -- dimensions ------------------------------------------------------------


@dataclass(frozen=True)
class Dimension(ABC):
    """One named hyperparameter and the values it may take."""

    name: str

    @abstractmethod
    def suggest(self, trial: optuna.Trial) -> Any:
        """Ask ``trial`` for this dimension's value in the candidate it is building."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """A JSON-serialisable description, for the run report."""


@dataclass(frozen=True)
class Real(Dimension):
    """A continuous dimension over ``[low, high]``, optionally on a log scale.

    ``log`` is for a knob whose interesting variation is multiplicative -- a
    learning rate, a regularisation weight -- and is wrong for one that may
    legitimately be zero, such as a feedback gain.
    """

    low: float
    high: float
    log: bool = False

    def suggest(self, trial: optuna.Trial) -> float:
        """A float drawn from the trial's sampler over this range."""
        return trial.suggest_float(self.name, self.low, self.high, log=self.log)

    def describe(self) -> dict[str, Any]:
        """This dimension as ``{kind, low, high, log}``."""
        return {"kind": "real", "low": self.low, "high": self.high, "log": self.log}


@dataclass(frozen=True)
class Integer(Dimension):
    """A discrete dimension over the integers in ``[low, high]``."""

    low: int
    high: int
    log: bool = False

    def suggest(self, trial: optuna.Trial) -> int:
        """An integer drawn from the trial's sampler over this range."""
        return trial.suggest_int(self.name, self.low, self.high, log=self.log)

    def describe(self) -> dict[str, Any]:
        """This dimension as ``{kind, low, high, log}``."""
        return {"kind": "integer", "low": self.low, "high": self.high, "log": self.log}


@dataclass(frozen=True)
class Categorical(Dimension):
    """An unordered dimension over a fixed set of choices."""

    choices: tuple[Any, ...]

    def suggest(self, trial: optuna.Trial) -> Any:
        """One of the choices, drawn from the trial's sampler."""
        return trial.suggest_categorical(self.name, list(self.choices))

    def describe(self) -> dict[str, Any]:
        """This dimension as ``{kind, choices}``."""
        return {"kind": "categorical", "choices": list(self.choices)}


# -- the space -------------------------------------------------------------


class SearchSpace:
    """An ordered collection of :class:`Dimension`\\ s: one policy's knobs."""

    def __init__(self, dimensions: Sequence[Dimension]) -> None:
        """Hold ``dimensions``, rejecting a repeated name.

        Names are the keys of the parameter dict a policy is later built from, so
        two dimensions sharing one would silently drop a knob.
        """
        names = [d.name for d in dimensions]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate dimension names in {names}")
        self._dimensions = tuple(dimensions)

    @property
    def names(self) -> tuple[str, ...]:
        """The dimension names, in declaration order."""
        return tuple(d.name for d in self._dimensions)

    def __len__(self) -> int:
        """How many knobs this space has."""
        return len(self._dimensions)

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """One candidate: a value per dimension, drawn from ``trial``."""
        return {d.name: d.suggest(trial) for d in self._dimensions}

    def describe(self) -> dict[str, dict[str, Any]]:
        """The whole space as a JSON-serialisable mapping, for the run report."""
        return {d.name: d.describe() for d in self._dimensions}
