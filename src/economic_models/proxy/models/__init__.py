"""The concrete proxy estimators, one module each.

Every model here subclasses :class:`~economic_models.proxy.base.BaseProxyModel`
and fits on the shared
:class:`~economic_models.proxy.transform.StationarizingTransform`; all they
supply is how the one-step conditional is estimated and sampled. The contract
they implement lives one level up, in :mod:`economic_models.proxy.base`. A VARX,
a distributional random forest, a mixture density network, a kNN analog
forecaster, an encoder-native forecast reader and a random-walk baseline are
implemented so far.
"""

from economic_models.proxy.models.drf import DRFProxy
from economic_models.proxy.models.encoder_native import EncoderNativeProxy
from economic_models.proxy.models.knn import KNNProxy
from economic_models.proxy.models.mdn import MDNProxy
from economic_models.proxy.models.random_walk import RandomWalkProxy
from economic_models.proxy.models.varx import VARXProxy

__all__ = [
    "VARXProxy",
    "DRFProxy",
    "MDNProxy",
    "KNNProxy",
    "RandomWalkProxy",
    "EncoderNativeProxy",
]
