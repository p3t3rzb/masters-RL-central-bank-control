"""The reward interface: a context and a weighted sum of named terms.

What a central bank is *for* is the one genuinely normative choice in this
project, so it is kept swappable and inspectable rather than inlined into the
environment. A :class:`RewardFunction` is a template: the subclass supplies named
components through :meth:`~RewardFunction._terms`, and this base combines them
with the mandate's weights. The environment logs the named breakdown alongside
the scalar, so a training curve can be read per mandate leg without re-running
anything.

Each component is written as a *contribution to reward* (a penalty is therefore
negative) and is expected to be scaled to O(1) for a typical business-cycle
deviation, so no single leg dominates the critic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

from economic_models.variables import Actions, Parameters, State


@dataclass(frozen=True)
class RewardContext:
    """Everything one period's reward may condition on.

    The realised ``state`` and the ``prev_state`` it came from, the exogenous
    ``parameters`` (and ``prev_parameters``) in force, the ``actions`` the bank
    applied and the ``prev_actions`` it moved from, plus ``dt``, the length of
    the period in years -- needed to annualise any per-step growth rate.
    """

    state: State
    prev_state: State
    parameters: Parameters
    prev_parameters: Parameters
    actions: Actions
    prev_actions: Actions
    dt: float


class RewardFunction(ABC):
    """A mandate: named, scaled reward components and the weights over them.

    Subclasses implement :meth:`_terms` and :attr:`weights`; the scalar reward is
    always their weighted sum, so a mandate can be inspected leg by leg.
    """

    def __call__(self, ctx: RewardContext) -> float:
        """The scalar reward: the weighted sum of :meth:`terms`."""
        terms = self.terms(ctx)
        return float(sum(self.weights[name] * value for name, value in terms.items()))

    def terms(self, ctx: RewardContext) -> dict[str, float]:
        """The named, already-scaled reward contributions for this period."""
        return self._terms(ctx)

    @property
    @abstractmethod
    def weights(self) -> Mapping[str, float]:
        """Weight per term name; the keys must match :meth:`_terms`."""

    @abstractmethod
    def _terms(self, ctx: RewardContext) -> dict[str, float]:
        """Compute this mandate's named contributions (penalties are negative)."""

    @property
    def term_names(self) -> tuple[str, ...]:
        """The term names this mandate reports, in weight order."""
        return tuple(self.weights)
