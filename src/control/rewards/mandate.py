"""The dual-mandate-plus-growth reward: employment, growth and inflation gaps.

Three quadratic gaps, one per stated objective:

* **unemployment** -- the employment gap ``1 - ER`` (``ER = 1`` is full
  employment). Squared, so it is symmetric: an overheating boom is penalised too,
  which keeps the term bounded and matches the standard quadratic loss. Whenever
  ``ER <= 1`` this is exactly "minimise unemployment".
* **growth** -- the gap between realised real growth and *potential* growth.
  Rewarding growth linearly would be unbounded above and is precisely the lever a
  model-based agent uses to exploit a surrogate's extrapolation; the gap to
  potential is bounded, and potential is observable: productivity growth ``GRpr``
  plus growth in the full-employment labour level ``Nfe``.
* **inflation** -- the gap to the target. ``PI`` is already an annualised rate in
  the GROWTH model (11.x: ``PI = dt*((P/P(-1))**(1/dt) - 1) + (1 - dt)*PI(-1)``),
  so it needs no conversion -- unlike growth, which is a per-step log-difference
  and must be divided by ``dt``.

Every gap is divided by a reference deviation before squaring, so each term is
O(1) for a typical business-cycle excursion and no leg dominates the critic. The
weights then express the mandate's priorities on a common scale.
"""

from __future__ import annotations

import math
from typing import Mapping

from control.rewards.base import RewardContext, RewardFunction


class MandateReward(RewardFunction):
    """Minimise unemployment, hold growth at potential, hit the inflation target."""

    def __init__(
        self,
        pi_target: float = 0.02,
        *,
        w_unemployment: float = 1.0,
        w_growth: float = 1.0,
        w_inflation: float = 1.0,
        u_scale: float = 0.02,
        g_scale: float = 0.02,
        pi_scale: float = 0.02,
    ) -> None:
        """Configure the mandate's target, weights and reference scales.

        ``pi_target`` is the annual inflation target. ``w_unemployment``,
        ``w_growth`` and ``w_inflation`` weight the three legs against each other.
        ``u_scale``, ``g_scale`` and ``pi_scale`` are the reference deviations each
        gap is divided by before squaring -- the excursion at which a leg
        contributes ``-1`` -- so the defaults treat a 2pp employment gap, a 2pp
        shortfall from potential growth and a 2pp inflation miss as equally bad.
        """
        self.pi_target = pi_target
        self.w_unemployment = w_unemployment
        self.w_growth = w_growth
        self.w_inflation = w_inflation
        self.u_scale = u_scale
        self.g_scale = g_scale
        self.pi_scale = pi_scale

    @property
    def weights(self) -> Mapping[str, float]:
        """The weight of each mandate leg."""
        return {
            "unemployment": self.w_unemployment,
            "growth": self.w_growth,
            "inflation": self.w_inflation,
        }

    def _terms(self, ctx: RewardContext) -> dict[str, float]:
        """The three squared gaps, scaled and negated."""
        return {
            "unemployment": -((1.0 - ctx.state.ER) / self.u_scale) ** 2,
            "growth": -(self._growth_gap(ctx) / self.g_scale) ** 2,
            "inflation": -((ctx.state.PI - self.pi_target) / self.pi_scale) ** 2,
        }

    def _growth_gap(self, ctx: RewardContext) -> float:
        """Annualised real growth minus annualised potential growth.

        Realised growth is the per-step log-difference of real output divided by
        ``dt``; potential is productivity growth plus growth in the
        full-employment labour level, both observed parameters.
        """
        realised = math.log(ctx.state.Yk / ctx.prev_state.Yk) / ctx.dt
        labour = math.log(ctx.parameters.Nfe / ctx.prev_parameters.Nfe) / ctx.dt
        return realised - (ctx.parameters.GRpr + labour)
