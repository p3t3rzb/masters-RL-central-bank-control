"""Economic models split into two families behind one shared interface.

* :mod:`~economic_models.ground_truth` -- full structural simulations of the
  economy (e.g. GROWTH), the "real" model an agent is judged against.
* :mod:`~economic_models.proxy` -- cheap learned surrogates fit on runs of a
  ground-truth model, swappable behind the same interface.

Both families expose the same central-bank-visible interface --
:class:`State`, :class:`Parameters`, :class:`Actions` -- and share one driver,
:meth:`BaseEconomicModel.advance`, which applies the exogenous inputs and
advances one period. A scenario written in the three value classes is therefore
model-agnostic data, replayable against either family.
"""

from economic_models.base import BaseEconomicModel
from economic_models.variables import Actions, Parameters, State

__all__ = [
    "BaseEconomicModel",
    "State",
    "Parameters",
    "Actions",
]
