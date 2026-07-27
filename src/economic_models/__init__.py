"""Economic models split into two families behind one shared interface.

* :mod:`~economic_models.ground_truth` -- full structural simulations of the
  economy (e.g. GROWTH), the "real" model an agent is judged against.
* :mod:`~economic_models.proxy` -- cheap learned surrogates fit on runs of a
  ground-truth model, swappable behind the same interface.

Both families realise the same tripartite interface -- the abstract
:class:`State`, :class:`Parameters`, :class:`Actions` value spaces, which each
ground-truth model specializes -- and share one driver,
:meth:`BaseEconomicModel.advance`, which applies the exogenous inputs and
advances one period. A proxy is pointed at a model's
:class:`~economic_models.interface.ModelInterface` (its value spaces plus its
stationarization spec) to stand in for that particular model.

The :class:`Run` and :class:`Scenario` dataset containers live here too, at the
root rather than under either family: a ground-truth model produces them, a
proxy trains on them, and :mod:`data_generation` and :mod:`control` read them, so
they are the currency all three packages speak in.
"""

from economic_models.base import BaseEconomicModel
from economic_models.interface import ModelInterface, TransformSpec
from economic_models.run import Run, Scenario
from economic_models.variables import Actions, Parameters, State

__all__ = [
    "BaseEconomicModel",
    "State",
    "Parameters",
    "Actions",
    "ModelInterface",
    "TransformSpec",
    "Run",
    "Scenario",
]
