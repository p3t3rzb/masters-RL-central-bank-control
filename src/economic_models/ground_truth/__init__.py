"""Ground-truth economic models: full structural simulations of the economy.

Each ground-truth model is a self-contained subpackage built on
:class:`PysolveEconomicModel`. The excited-run machinery is split into a
model-agnostic base (:class:`ExcitationConfig`, :class:`ExcitationProcess`,
:class:`ExcitedRunGenerator`) that owns the generic drift/solve/record loop, and
per-model subclasses that specialise it. GROWTH is the one implemented so far
(:class:`GrowthModel`, :class:`GrowthExcitationConfig`,
:class:`GrowthRunGenerator`); its excited runs are the :class:`Run` datasets a
:mod:`~economic_models.proxy` is fit on.
"""

from economic_models.ground_truth.base import PysolveEconomicModel
from economic_models.ground_truth.excitation import (
    ExcitationConfig,
    ExcitationProcess,
    ExcitedRunGenerator,
)
from economic_models.ground_truth.growth import (
    GROWTH_INTERFACE,
    GrowthCalibration,
    GrowthExcitationConfig,
    GrowthModel,
    GrowthRunGenerator,
)
from economic_models.ground_truth.run import Run, Scenario

__all__ = [
    "PysolveEconomicModel",
    "ExcitationConfig",
    "ExcitationProcess",
    "ExcitedRunGenerator",
    "GrowthModel",
    "GrowthCalibration",
    "GrowthExcitationConfig",
    "GrowthRunGenerator",
    "GROWTH_INTERFACE",
    "Run",
    "Scenario",
]
