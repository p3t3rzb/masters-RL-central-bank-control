"""Model-agnostic excitation scaffolding shared by every ground-truth model.

The generic process specs (:class:`AR1Spec`, :class:`RandomWalkSpec`,
:class:`CrisisSpec`, ...) and the base :class:`ExcitationConfig`,
:class:`ExcitationProcess` and :class:`ExcitedRunGenerator` a concrete model
subclasses to produce its excited :class:`~economic_models.ground_truth.run.Run`
datasets. GROWTH's own subclasses live in
:mod:`economic_models.ground_truth.growth.excitation`.
"""

from economic_models.ground_truth.excitation.base import (
    ExcitationConfig,
    ExcitationProcess,
    ExcitedRunGenerator,
)
from economic_models.ground_truth.excitation.specs import (
    AR1Spec,
    ClimateSpec,
    CrisisSpec,
    RandomWalkSpec,
    StochasticVolatilitySpec,
    discretize_ar1,
)

__all__ = [
    "AR1Spec",
    "RandomWalkSpec",
    "StochasticVolatilitySpec",
    "CrisisSpec",
    "ClimateSpec",
    "discretize_ar1",
    "ExcitationConfig",
    "ExcitationProcess",
    "ExcitedRunGenerator",
]
