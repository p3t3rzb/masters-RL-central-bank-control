"""A k-nearest-neighbour analog-forecasting proxy of the economy.

The fixed-metric sibling of the DRF proxy: both map the encoder latent ``z_t``
stacked with the contemporaneous exogenous features ``u_{t+1}`` onto the next
state-feature row by resampling observed training rows, and both are
distributional. This one uses a plain **Euclidean** metric on standardized
regressors and gives the ``k`` nearest rows **uniform** weight -- the classic
analog forecast -- so pairing it with the DRF isolates what the forest's learned
metric buys over off-the-shelf neighbours. From the ``k`` neighbours:

* the **conditional mean** -- the average of their next-feature rows -- drives the
  deterministic rollout;
* a **stochastic draw** picks one whole neighbour target row uniformly,
  preserving the cross-equation correlation of the state features.

Regressors are standardized column-wise before distances are taken, so the metric
is not dominated by the widest-scaled feature.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from economic_models.encoders import StateEncoder
from economic_models.proxy.base import BaseProxyModel, FitData, StepContext
from economic_models.proxy.transform import StationarizingTransform


class KNNProxy(BaseProxyModel):
    """k-nearest-neighbour analog forecasting over the latent one-step dynamics."""

    def __init__(
        self,
        k: int = 10,
        *,
        encoder: StateEncoder | None = None,
        transform: StationarizingTransform | None = None,
    ) -> None:
        """Configure the analog forecaster.

        ``k`` is the number of nearest neighbours averaged (or sampled among).
        ``encoder`` / ``transform`` are the shared conditioning latent and
        feature view.
        """
        super().__init__(encoder=encoder, transform=transform)
        self.k = k

        self.knn_: NearestNeighbors | None = None
        self._targets: np.ndarray | None = None  # (n_train, n_state) training f_{t+1}
        self._scaler: StandardScaler | None = None

    # -- design --------------------------------------------------------------

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """Column-wise standardize the design so the Euclidean metric is scale-fair."""
        return self._scaler.transform(X)

    # -- estimation ----------------------------------------------------------

    def _fit(self, data: FitData) -> None:
        """Fit the standardizer and index the standardized training designs for neighbour lookup."""
        X = self._design(data.z, data.u_next)
        Y = data.y

        self._scaler = StandardScaler().fit(X)

        # A neighbourhood can be no larger than the training set.
        n_neighbors = min(self.k, len(X))
        self.knn_ = NearestNeighbors(n_neighbors=n_neighbors).fit(self._standardize(X))
        self._targets = Y

    # -- inference -----------------------------------------------------------

    def _predict_batch(
        self, z: np.ndarray, u_next: np.ndarray, f_prev: np.ndarray
    ) -> np.ndarray:
        """Conditional mean: the average target row of the ``k`` nearest neighbours."""
        _, idx = self.knn_.kneighbors(self._standardize(self._design(z, u_next)))
        return self._targets[idx].mean(axis=1)

    def _sample_step(
        self, ctx: StepContext, rng: np.random.Generator
    ) -> np.ndarray:
        """One draw: a whole target row of one uniformly chosen neighbour."""
        x = self._design(ctx.z[None, :], ctx.u_next[None, :])
        _, idx = self.knn_.kneighbors(self._standardize(x))
        return self._targets[rng.choice(idx[0])]
