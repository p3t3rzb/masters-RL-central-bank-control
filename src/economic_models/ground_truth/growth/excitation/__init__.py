"""Generate "historic" runs of the GROWTH model with excited exogenous inputs.

GROWTH's concrete excitation: :class:`GrowthExcitationConfig` (the drift +
volatility/crisis/climate bundle plus the government-spending stabilizer and the
``Nfe`` random walk), :class:`GrowthExcitationProcess` (the per-step draws) and
:class:`GrowthRunGenerator` (the reproducible driver that turns a config into
:class:`~economic_models.ground_truth.run.Run` datasets). The model-agnostic base
classes and the generic process specs live in
:mod:`economic_models.ground_truth.excitation`; the public names are re-exported
here for convenience. Hidden parameter paths are recorded for diagnostics only --
a proxy must never train on them.
"""

from economic_models.ground_truth.excitation.specs import (
    AR1Spec,
    ClimateSpec,
    CrisisSpec,
    RandomWalkSpec,
    StochasticVolatilitySpec,
    discretize_ar1,
)
from economic_models.ground_truth.growth.excitation.generator import (
    GrowthExcitationProcess,
    GrowthRunGenerator,
)
from economic_models.ground_truth.growth.excitation.presets import GrowthExcitationConfig
from economic_models.ground_truth.growth.excitation.specs import GovSpendingSpec
from economic_models.ground_truth.run import Run

__all__ = [
    "AR1Spec",
    "RandomWalkSpec",
    "StochasticVolatilitySpec",
    "CrisisSpec",
    "ClimateSpec",
    "GovSpendingSpec",
    "GrowthExcitationConfig",
    "GrowthExcitationProcess",
    "GrowthRunGenerator",
    "Run",
    "discretize_ar1",
]
