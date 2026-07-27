"""A Distributional Random Forest (DRF) proxy of the economy.

A nonparametric, fully *distributional* surrogate: a multi-output forest maps the
encoder latent ``z_t`` stacked with the contemporaneous exogenous features
``u_{t+1}`` onto the next state-feature row, and the one-step conditional
distribution is read off the forest.

Following Ćevid, Michel, Näf, Meinshausen & Bühlmann (2022), the forest is used
as an adaptive nearest-neighbour weighting scheme rather than a point predictor.
For a query ``x`` each tree assigns it to a leaf, and the training rows sharing
that leaf define weights

    w_i(x) = (1 / n_trees) * sum_t  1{leaf_t(x) = leaf_t(x_i)} / |leaf_t(x)|

approximating the conditional law as ``sum_i w_i(x) * delta_{f_i}``. From those
weights:

* the **conditional mean** ``sum_i w_i(x) f_i`` drives the deterministic rollout;
* a **stochastic draw** resamples one whole observed target row ``f_i`` with
  probability ``w_i(x)``, preserving the cross-equation correlation of the state
  features rather than adding independent per-dimension noise.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from economic_models.encoders import StateEncoder
from economic_models.interface import ModelInterface
from economic_models.proxy.base import BaseProxyModel, FitData, StepContext
from economic_models.proxy.transform import StationarizingTransform


class DRFProxy(BaseProxyModel):
    """Distributional random forest over the latent one-step dynamics."""

    def __init__(
        self,
        interface: ModelInterface,
        n_estimators: int = 300,
        min_samples_leaf: int = 5,
        max_features: float | str = 0.5,
        seed: int | None = 0,
        *,
        encoder: StateEncoder | None = None,
        transform: StationarizingTransform | None = None,
    ) -> None:
        """Configure the distributional forest for the model ``interface`` it mimics.

        ``interface`` is the ground-truth model's
        :class:`~economic_models.interface.ModelInterface`. ``n_estimators`` trees;
        ``min_samples_leaf`` the minimum leaf size (the neighbourhood
        granularity); ``max_features`` the fraction/rule of features tried per
        split; ``seed`` the forest RNG. ``encoder`` / ``transform`` are the shared
        conditioning latent and feature view.
        """
        super().__init__(interface, encoder=encoder, transform=transform)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed

        self.forest_: RandomForestRegressor | None = None
        self._targets: np.ndarray | None = None  # (n_train, n_state) training f_{t+1}
        # per-tree {leaf_id -> training-row indices} membership, for DRF weights
        self._leaf_members: list[dict[int, np.ndarray]] | None = None

    # -- estimation ----------------------------------------------------------

    def _fit(self, data: FitData) -> None:
        """Grow the multi-output forest and cache per-tree leaf-membership maps for fast DRF weighting."""
        X = self._design(data.z, data.u_next)
        Y = data.y

        self.forest_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=-1,
            random_state=self.seed,
        ).fit(X, Y)
        self._targets = Y

        # Cache each tree's leaf -> member-rows map so DRF weights are a few dict
        # lookups per step rather than a scan over the whole training set.
        train_leaves = self.forest_.apply(X)  # (n_train, n_trees)
        self._leaf_members = []
        for t in range(train_leaves.shape[1]):
            leaves = train_leaves[:, t]
            order = np.argsort(leaves, kind="stable")
            uniq, starts = np.unique(leaves[order], return_index=True)
            groups = np.split(order, starts[1:])  # member rows, grouped by leaf
            self._leaf_members.append(
                {int(u): g for u, g in zip(uniq, groups)}
            )

    def _predict_batch(
        self, z: np.ndarray, u_next: np.ndarray, f_prev: np.ndarray
    ) -> np.ndarray:
        """Forest conditional-mean predictions for the design rows."""
        return self.forest_.predict(self._design(z, u_next))

    # -- DRF weights ---------------------------------------------------------

    def _weights(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nonzero DRF weights ``(indices, weights)`` for one query row ``x``."""
        query_leaves = self.forest_.apply(x[None, :])[0]  # (n_trees,)
        acc: dict[int, float] = {}
        for members, leaf in zip(self._leaf_members, query_leaves):
            rows = members[int(leaf)]
            w = 1.0 / (len(query_leaves) * len(rows))
            for i in rows:
                acc[int(i)] = acc.get(int(i), 0.0) + w
        idx = np.fromiter(acc.keys(), dtype=int, count=len(acc))
        wts = np.fromiter(acc.values(), dtype=float, count=len(acc))
        return idx, wts

    # -- inference -----------------------------------------------------------

    def _sample_step(
        self, ctx: StepContext, rng: np.random.Generator
    ) -> np.ndarray:
        """One draw: resample a whole observed target row with its DRF weight."""
        x = self._design(ctx.z, ctx.u_next)
        idx, wts = self._weights(x)
        draw = idx[rng.choice(len(idx), p=wts)]
        return self._targets[draw]
