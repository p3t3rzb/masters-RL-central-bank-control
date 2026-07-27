"""Risk measures: how the actor scores a whole return distribution.

A point critic leaves the actor one choice -- maximise the expected return. A
distributional critic hands it the entire law of the return, so the objective can
be any functional of that law. That is where a *risk-averse* central bank lives:
one that weights the crisis tail more heavily than the average outcome, which is
much closer to what a real mandate means than expected loss.

:class:`MeanRisk` is the null object, recovering risk-neutral DSAC (the
apples-to-apples comparison with SAC); :class:`CVaRRisk` optimises the conditional
value at risk, the mean of the worst ``alpha`` fraction of outcomes. Swapping one
for the other is the only change needed to switch the mandate's risk attitude.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class RiskMeasure(ABC):
    """Reduces a batch of return quantiles to the scalar the actor maximises."""

    @abstractmethod
    def aggregate(self, quantiles: torch.Tensor) -> torch.Tensor:
        """Score each row of a ``(batch, n_quantiles)`` tensor, keeping the batch."""

    @property
    def name(self) -> str:
        """Short label for logging."""
        return type(self).__name__


class MeanRisk(RiskMeasure):
    """The null object risk measure: the plain expectation (risk-neutral)."""

    def aggregate(self, quantiles: torch.Tensor) -> torch.Tensor:
        """The mean over quantiles -- identical to a point critic's ``Q``."""
        return quantiles.mean(dim=-1, keepdim=True)


class CVaRRisk(RiskMeasure):
    """Conditional value at risk: the mean of the worst ``alpha`` of outcomes."""

    def __init__(self, alpha: float = 0.1) -> None:
        """Optimise the lowest ``alpha`` fraction of the return distribution."""
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha

    def aggregate(self, quantiles: torch.Tensor) -> torch.Tensor:
        """The mean of the lowest ``alpha`` fraction of the sorted quantiles."""
        n = max(1, int(round(self.alpha * quantiles.shape[-1])))
        worst, _ = torch.sort(quantiles, dim=-1)
        return worst[..., :n].mean(dim=-1, keepdim=True)

    @property
    def name(self) -> str:
        """Short label for logging, carrying the tail fraction."""
        return f"CVaR@{self.alpha:g}"
