"""Ground-truth economic models: full structural simulations of the economy.

Each ground-truth model is a self-contained subpackage built on
:class:`PysolveEconomicModel`. GROWTH is the one implemented so far; it also
defines the excited-run machinery (:class:`ExcitationConfig`,
:class:`ExcitedRunGenerator`) that generates the :class:`Run` datasets a
:mod:`~economic_models.proxy` is fit on. The individual excitation spec classes
stay importable from :mod:`economic_models.ground_truth.growth.excitation`.
"""

from economic_models.ground_truth.base import PysolveEconomicModel
from economic_models.ground_truth.growth import (
    ExcitationConfig,
    ExcitedRunGenerator,
    GrowthCalibration,
    GrowthModel,
)
from economic_models.ground_truth.run import Run

__all__ = [
    "PysolveEconomicModel",
    "GrowthModel",
    "GrowthCalibration",
    "Run",
    "ExcitationConfig",
    "ExcitedRunGenerator",
]
