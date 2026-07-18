"""Proxy economic models: cheap learned surrogates of a ground-truth model.

A proxy is fit on excited historic runs of a ground-truth model and then exposes
the same :class:`~economic_models.State` / :class:`~economic_models.Parameters`
/ :class:`~economic_models.Actions` currency via ``reset``/``step``/``rollout``
and the family-shared deterministic ``advance`` -- observed and driven in the
same terms as the real economy.

Several interchangeable estimators are provided -- a VARX, a distributional
random forest, a mixture density network, a kNN analog forecaster, an
encoder-native forecast reader, and a random-walk baseline -- all behind the
:class:`BaseProxyModel` contract, all fitting on the shared
:class:`~economic_models.proxy.transform.StationarizingTransform`. The
conditioning encoders live in :mod:`economic_models.encoders`; the excitation
machinery a proxy is fit on lives in :mod:`~economic_models.ground_truth`.
"""

from economic_models.proxy.base import BaseProxyModel, FitData, StepContext
from economic_models.proxy.drf import DRFProxy
from economic_models.proxy.encoder_native import EncoderNativeProxy
from economic_models.proxy.knn import KNNProxy
from economic_models.proxy.mdn import MDNProxy
from economic_models.proxy.random_walk import RandomWalkProxy
from economic_models.proxy.transform import StationarizingTransform
from economic_models.proxy.varx import VARXProxy

__all__ = [
    "StationarizingTransform",
    "BaseProxyModel",
    "FitData",
    "StepContext",
    "VARXProxy",
    "DRFProxy",
    "MDNProxy",
    "KNNProxy",
    "RandomWalkProxy",
    "EncoderNativeProxy",
]
