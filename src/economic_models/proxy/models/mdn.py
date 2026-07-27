"""A Mixture Density Network (MDN) proxy of the economy.

A small neural network whose output *is* a probability distribution over the next
state-feature row. The encoder latent ``z_t`` stacked with the contemporaneous
exogenous features ``u_{t+1}`` is fed through a ``tanh`` MLP with three heads
(Bishop, 1994) -- means ``mu_k(x)``, input-dependent (heteroscedastic) log-scales
``log s_k(x)``, and mixing logits ``-> pi_k(x)`` -- describing the conditional law
as a mixture of ``K`` diagonal Gaussians,

    p(f_{t+1} | x) = sum_k pi_k(x) * N(f_{t+1} ; mu_k(x), diag(s_k(x)^2)).

From it:

* the **conditional mean** ``sum_k pi_k mu_k`` drives the deterministic rollout;
* a **stochastic draw** picks a component ``k ~ pi(x)`` then samples its diagonal
  Gaussian. The shared component identity couples the state dimensions, so the
  mixture reproduces regime-switching co-movement -- multimodality a single
  Gaussian (VARX) cannot express.

Trained by minimising the mixture negative log-likelihood with Adam on
column-standardised features, restoring the weights from the lowest-validation-NLL
epoch (early-stopping-style model selection).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from economic_models._torch import resolve_device
from economic_models.encoders import StateEncoder
from economic_models.interface import ModelInterface
from economic_models.proxy.base import BaseProxyModel, FitData, StepContext
from economic_models.proxy.transform import StationarizingTransform

_LOG_SIG_MIN, _LOG_SIG_MAX = -7.0, 3.0


class _MDNNet(nn.Module):
    """A small MLP with mean / log-scale / mixing-logit heads for a diagonal MoG."""

    def __init__(
        self, n_in: int, n_out: int, n_components: int, hidden: int, n_layers: int
    ) -> None:
        """Build the ``tanh`` MLP body and the three mixture heads.

        ``n_in`` design width, ``n_out`` the ``D`` target dimensions,
        ``n_components`` the ``K`` mixture components, ``hidden`` the width and
        ``n_layers`` the depth of the shared body.
        """
        super().__init__()
        self.K, self.D = n_components, n_out
        layers: list[nn.Module] = []
        prev = n_in
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.Tanh()]
            prev = hidden
        self.body = nn.Sequential(*layers)
        self.head_mu = nn.Linear(prev, n_components * n_out)
        self.head_ls = nn.Linear(prev, n_components * n_out)
        self.head_pi = nn.Linear(prev, n_components)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logits, mu, log_sig)`` with the mixture axis broken out."""
        h = self.body(x)
        n = x.shape[0]
        mu = self.head_mu(h).view(n, self.K, self.D)
        log_sig = self.head_ls(h).view(n, self.K, self.D).clamp(_LOG_SIG_MIN, _LOG_SIG_MAX)
        return self.head_pi(h), mu, log_sig


def _mixture_nll(
    y: torch.Tensor, logits: torch.Tensor, mu: torch.Tensor, log_sig: torch.Tensor
) -> torch.Tensor:
    """Mean negative log-likelihood of ``y`` under the diagonal Gaussian mixture."""
    mixture = MixtureSameFamily(
        Categorical(logits=logits),  # mixing weights pi_k(x)
        Independent(Normal(mu, log_sig.exp()), 1),  # diagonal Gaussian per component
    )
    return -mixture.log_prob(y).mean()


class MDNProxy(BaseProxyModel):
    """Mixture density network over the latent one-step dynamics."""

    def __init__(
        self,
        interface: ModelInterface,
        n_components: int = 3,
        hidden: int = 64,
        n_layers: int = 2,
        lr: float = 3e-3,
        epochs: int = 2000,
        weight_decay: float = 1e-4,
        val_fraction: float = 0.2,
        patience: int = 300,
        seed: int | None = 0,
        device: str | None = None,
        *,
        encoder: StateEncoder | None = None,
        transform: StationarizingTransform | None = None,
    ) -> None:
        """Configure the mixture density network and its training loop.

        ``interface`` is the ground-truth model's
        :class:`~economic_models.interface.ModelInterface` the proxy mimics.
        Network shape: ``n_components`` mixture components over a body of
        ``n_layers`` ``tanh`` layers of width ``hidden``. Optimisation: Adam with
        learning rate ``lr`` and L2 ``weight_decay`` for up to ``epochs``, holding
        out ``val_fraction`` of rows to pick the best epoch and stopping after
        ``patience`` epochs without improvement. ``seed`` seeds torch/NumPy;
        ``device`` selects the compute device (``None``/``"cpu"`` -> CPU,
        ``"auto"`` -> CUDA/MPS/CPU). ``encoder`` / ``transform`` are the shared
        conditioning latent and feature view.
        """
        super().__init__(interface, encoder=encoder, transform=transform)
        self.n_components = n_components
        self.hidden = hidden
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.patience = patience
        self.seed = seed
        # ``None``/``"cpu"`` -> CPU (reproducible default); ``"auto"`` picks
        # CUDA, then Apple MPS, then CPU; any explicit device passes through.
        self.device = resolve_device(device)

        self.net_: _MDNNet | None = None
        # Column-wise standardisers, fit on the training split.
        self._x_scaler: StandardScaler | None = None
        self._y_scaler: StandardScaler | None = None
        self.best_epoch_ = self.best_val_ = None
        self.history_: list[float] = []  # per-epoch validation NLL

    # -- estimation ----------------------------------------------------------

    def _fit(self, data: FitData) -> None:
        """Train the MDN by Adam on the mixture NLL, restoring the best-validation-epoch weights."""
        X = self._design(data.z, data.u_next)
        Y = data.y
        torch.manual_seed(self.seed if self.seed is not None else 0)

        # Shuffled train/validation split for best-epoch model selection.
        rng = np.random.default_rng(self.seed)
        n = len(X)
        n_val = max(1, int(round(self.val_fraction * n))) if n > 4 else 0
        perm = rng.permutation(n)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        self._x_scaler = StandardScaler().fit(X[tr_idx])
        self._y_scaler = StandardScaler().fit(Y[tr_idx])
        Xs = self._t(self._x_scaler.transform(X))
        Ys = self._t(self._y_scaler.transform(Y))
        Xtr, Ytr = Xs[tr_idx], Ys[tr_idx]
        Xval, Yval = (Xs[val_idx], Ys[val_idx]) if n_val else (Xtr, Ytr)

        self.net_ = _MDNNet(
            X.shape[1], Y.shape[1], self.n_components, self.hidden, self.n_layers
        ).to(self.device)
        opt = torch.optim.Adam(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val, best_state, best_epoch, stale = float("inf"), None, 0, 0
        self.history_ = []
        for epoch in range(1, self.epochs + 1):
            self.net_.train()
            opt.zero_grad()
            _mixture_nll(Ytr, *self.net_(Xtr)).backward()
            opt.step()

            self.net_.eval()
            with torch.no_grad():
                val = float(_mixture_nll(Yval, *self.net_(Xval)))
            self.history_.append(val)
            if val < best_val - 1e-6:
                best_val, best_epoch, stale = val, epoch, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                stale += 1
                if self.patience and stale >= self.patience:
                    break

        # ``best_state`` stays ``None`` only if no epoch ever improved on the
        # initial ``inf`` -- i.e. the validation NLL was ``nan``/``inf`` throughout
        # (a degenerate or ill-scaled batch). Fall back to the final weights rather
        # than passing ``None`` to ``load_state_dict``.
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.best_epoch_, self.best_val_ = best_epoch, best_val

    # -- inference -----------------------------------------------------------

    def _t(self, A: np.ndarray) -> torch.Tensor:
        """Convert an array to a float32 tensor on this proxy's device."""
        return torch.as_tensor(A, dtype=torch.float32, device=self.device)

    def _mixture_batch(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardised-space mixture params ``(pi, mu, sigma)`` for query rows.

        One forward pass over the whole ``(n, n_in)`` design; returns ``pi``
        ``(n, K)``, ``mu`` ``(n, K, D)`` and ``sigma`` ``(n, K, D)``. Batching the
        forward pass is the only change from a per-row call -- each row's outputs
        are computed independently, so the results are identical.
        """
        xs = self._t(self._x_scaler.transform(X))
        with torch.no_grad():
            logits, mu, log_sig = self.net_(xs)
            pi = torch.softmax(logits, dim=1).cpu().numpy()
        return pi, mu.cpu().numpy(), np.exp(log_sig.cpu().numpy())

    def _sample_feature(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One draw from the predicted mixture, in level-feature space."""
        pi, mu, sig = (a[0] for a in self._mixture_batch(x[None, :]))
        k = rng.choice(len(pi), p=pi)
        draw = mu[k] + sig[k] * rng.standard_normal(mu.shape[1])
        return self._y_scaler.inverse_transform(draw[None, :])[0]

    def _predict_batch(
        self, z: np.ndarray, u_next: np.ndarray, f_prev: np.ndarray
    ) -> np.ndarray:
        """Conditional mean ``sum_k pi_k mu_k`` of the predicted mixture, in level-feature space."""
        pi, mu, _ = self._mixture_batch(self._design(z, u_next))
        mean = np.einsum("nk,nkd->nd", pi, mu)  # conditional mean sum_k pi_k mu_k
        return self._y_scaler.inverse_transform(mean)

    def _sample_step(
        self, ctx: StepContext, rng: np.random.Generator
    ) -> np.ndarray:
        """One draw: pick a component then sample its diagonal Gaussian."""
        return self._sample_feature(self._design(ctx.z, ctx.u_next), rng)
