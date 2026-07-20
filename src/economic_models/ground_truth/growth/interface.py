"""The GROWTH model's :class:`ModelInterface` -- what a proxy mimics GROWTH through.

Bundles GROWTH's value spaces with the stationarization rules its learned proxies
are fit on. GROWTH levels trend, so a regression on them is ill-posed; the proxy
feature space instead enters trending aggregates (``Y``, ``Yk``, ``P``, ``Ck``,
``Ik``, ``Vk``) as one-step log-differences, the government stock/flow (``GD``,
``PSBR``) as ratios to nominal GDP (``Y``), and the exogenous ``Nfe`` level as a
log-difference; everything already stationary passes through as its level.
"""

from __future__ import annotations

from economic_models.interface import ModelInterface, TransformSpec
from economic_models.ground_truth.growth.variables import (
    GrowthActions,
    GrowthParameters,
    GrowthState,
)

GROWTH_INTERFACE = ModelInterface(
    state=GrowthState,
    parameters=GrowthParameters,
    actions=GrowthActions,
    transform_spec=TransformSpec(
        state_names=GrowthState.names(),
        exog_names=(*GrowthParameters.names(), *GrowthActions.names()),
        log_diff=("Y", "Yk", "P", "Ck", "Ik", "Vk"),
        ratio_to=("GD", "PSBR"),
        denominator="Y",
        exog_log_diff=("Nfe",),
    ),
)
