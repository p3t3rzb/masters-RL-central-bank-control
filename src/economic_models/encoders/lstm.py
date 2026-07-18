"""A nonlinear recurrent state encoder: an LSTM predictive filter.

The nonlinear sibling of
:class:`~economic_models.encoders.kalman.KalmanEncoder`: a gated recurrent
network learns an arbitrary nonlinear transition and observation in place of the
linear-Gaussian one. Each period the LSTM folds the newest standardized
``(f_t, u_t)`` into a hidden state ``h_t`` (the conditioning latent), and a small
readout head maps ``(h_t, u_{t+1})`` to the next standardized feature row:

    h_t          = LSTM([f_t, u_t], h_{t-1})           (nonlinear transition)
    f_{t+1}_hat  = Head([h_t, u_{t+1}])                (nonlinear observation)

Recurrence and head are trained end-to-end by minimising next-step prediction
error, with the exogenous inputs driving both so ``h`` integrates their history.
The encoder is *generative*: per-feature predictive residual scales are recorded
after training so :meth:`~LSTMEncoder.forecast` can draw stochastic samples. The
gated recurrence buys regime-dependent dynamics a single linear transition
cannot, and its bounded gates keep multi-step rollouts from compounding off a
cliff under stress (boom-bust, deflation spirals).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from economic_models._torch import resolve_device
from economic_models.encoders.base import GenerativeEncoder


@dataclass
class _Belief:
    """An LSTM hidden belief: the ``(h, c)`` state carried between steps.

    Both are ``(num_layers, hidden)`` numpy arrays, kept opaque to the proxy;
    only this encoder folds new observations into them (:meth:`advance`) and reads
    the conditioning latent off ``h`` (:meth:`latent`).
    """

    h: np.ndarray  # (num_layers, hidden)
    c: np.ndarray  # (num_layers, hidden)


class _PredictiveLSTM(nn.Module):
    """LSTM recurrence + MLP readout head, trained on next-step prediction.

    The LSTM maps the standardized ``[f_t, u_t]`` sequence to hidden states; the
    head maps ``[h_t, u_{t+1}]`` to the standardized next feature row ``f_{t+1}``.
    Kept deliberately small -- the design is a compact learned filter, not a deep
    sequence model.
    """

    def __init__(
        self,
        dy: int,
        ku: int,
        hidden: int,
        num_layers: int,
        head_hidden: int,
        dropout: float,
    ) -> None:
        """Build the recurrence and readout head.

        ``dy``/``ku`` are the observation and exog widths; ``hidden`` the LSTM
        hidden size (the latent dim); ``num_layers`` the stacked LSTM depth;
        ``head_hidden`` the readout MLP width; ``dropout`` the inter-layer LSTM
        dropout (applied only when ``num_layers > 1``).
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dy + ku,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + ku, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, dy),
        )

    def encode(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Hidden states for a ``(batch, T, dy+ku)`` input; returns ``(H, (h,c))``."""
        return self.lstm(x, state)

    def read(self, h: torch.Tensor, u_next: torch.Tensor) -> torch.Tensor:
        """Next standardized feature row from hidden state(s) and next exog."""
        return self.head(torch.cat([h, u_next], dim=-1))


class LSTMEncoder(GenerativeEncoder):
    """A nonlinear recurrent (LSTM) predictive-filter state encoder."""

    def __init__(
        self,
        latent_dim: int = 16,
        num_layers: int = 1,
        head_hidden: int = 64,
        epochs: int = 300,
        lr: float = 5e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.0,
        grad_clip: float = 1.0,
        min_window: int = 1,
        seed: int | None = 0,
        device: str | None = None,
    ) -> None:
        """Configure the LSTM predictive-filter encoder.

        ``latent_dim`` is the LSTM hidden size (the conditioning-latent width);
        ``num_layers`` the stacked LSTM depth; ``head_hidden`` the readout MLP
        width. Training is controlled by ``epochs``, ``lr``, ``weight_decay``,
        ``dropout`` and ``grad_clip`` (max gradient norm). ``min_window`` is the
        hard floor of warm-start rows; ``seed`` seeds Torch; ``device`` selects
        the compute device (``None``/``"cpu"`` -> CPU, ``"auto"`` -> best
        accelerator, else passed through, via :func:`resolve_device`).
        """
        self._m = latent_dim
        self.num_layers = num_layers
        self.head_hidden = head_hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.grad_clip = grad_clip
        self._min_window = min_window
        self.seed = seed
        # ``None``/``"cpu"`` -> CPU (reproducible default); ``"auto"`` picks
        # CUDA, then Apple MPS, then CPU; any explicit device passes through.
        self.device = resolve_device(device)

        self.net_: _PredictiveLSTM | None = None
        # Column standardisers for the observations (f) and exog inputs (u),
        # fit on the pooled features by :meth:`fit`.
        self._y_scaler: StandardScaler | None = None
        self._u_scaler: StandardScaler | None = None
        # Per-feature predictive residual std in standardized space, for stochastic
        # forecasts -- the LSTM's analogue of the Kalman encoder's predictive
        # observation covariance.
        self._resid_std: np.ndarray | None = None
        self.loss_: list[float] = []  # per-epoch masked next-step MSE

    # -- interface ----------------------------------------------------------

    @property
    def latent_dim(self) -> int:
        """Latent (LSTM hidden) dimension ``m``."""
        return self._m

    @property
    def min_window(self) -> int:
        """Hard floor of feature rows needed to warm-start a belief."""
        return self._min_window

    # -- standardisation ----------------------------------------------------

    def _sy(self, F: np.ndarray) -> np.ndarray:
        """Column-standardise observation rows ``F``."""
        return self._y_scaler.transform(F)

    def _su(self, U: np.ndarray) -> np.ndarray:
        """Column-standardise exog input rows ``U``."""
        return self._u_scaler.transform(U)

    def _inv_y(self, Y: np.ndarray) -> np.ndarray:
        """Map standardised observations ``Y`` back to original feature units."""
        return self._y_scaler.inverse_transform(Y)

    def _t(self, A: np.ndarray) -> torch.Tensor:
        """Convert an array to a float32 tensor on the encoder's device."""
        return torch.as_tensor(A, dtype=torch.float32, device=self.device)

    # -- fitting ------------------------------------------------------------

    def fit(self, feature_runs: list[tuple[np.ndarray, np.ndarray]]) -> Self:
        """Train the recurrence and head on next-step prediction; cache scalers."""
        if self.seed is not None:
            torch.manual_seed(self.seed)

        Fall = np.vstack([F for F, _ in feature_runs])
        Uall = np.vstack([U for _, U in feature_runs])
        self._y_scaler = StandardScaler().fit(Fall)
        self._u_scaler = StandardScaler().fit(Uall)
        dy, ku = Fall.shape[1], Uall.shape[1]

        # Only runs with at least one transition contribute training pairs.
        runs = [
            (self._sy(F), self._su(U))
            for F, U in feature_runs
            if len(F) >= 2
        ]
        if not runs:
            raise ValueError("LSTMEncoder.fit needs at least one run of length >= 2")

        # Pad the standardized (f, u) sequences to a common length and build a mask
        # over the T-1 next-step targets of each run.
        lengths = np.array([len(Y) for Y, _ in runs])
        Tmax = int(lengths.max())
        n = len(runs)
        X = np.zeros((n, Tmax, dy + ku), dtype=np.float32)
        Ytgt = np.zeros((n, Tmax - 1, dy), dtype=np.float32)
        Unext = np.zeros((n, Tmax - 1, ku), dtype=np.float32)
        mask = np.zeros((n, Tmax - 1), dtype=np.float32)
        for i, (Y, Uc) in enumerate(runs):
            L = len(Y)
            X[i, :L] = np.hstack([Y, Uc])
            Ytgt[i, : L - 1] = Y[1:]
            Unext[i, : L - 1] = Uc[1:]
            mask[i, : L - 1] = 1.0

        Xt, Ytt = self._t(X), self._t(Ytgt)
        Unt, mt = self._t(Unext), self._t(mask)

        self.net_ = _PredictiveLSTM(
            dy, ku, self._m, self.num_layers, self.head_hidden, self.dropout
        ).to(self.device)
        opt = torch.optim.Adam(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.loss_ = []
        self.net_.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            H, _ = self.net_.encode(Xt)  # (n, Tmax, m)
            pred = self.net_.read(H[:, :-1, :], Unt)  # (n, Tmax-1, dy)
            sq = ((pred - Ytt) ** 2).sum(dim=-1) * mt
            loss = sq.sum() / mt.sum()
            loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.net_.parameters(), self.grad_clip)
            opt.step()
            self.loss_.append(float(loss.detach()))

        # Record per-feature predictive residual std for stochastic forecasts.
        self.net_.eval()
        with torch.no_grad():
            H, _ = self.net_.encode(Xt)
            pred = self.net_.read(H[:, :-1, :], Unt).cpu().numpy()
        m3 = mask[:, :, None].astype(bool)
        resid = (Ytgt - pred)[np.broadcast_to(m3, Ytgt.shape)].reshape(-1, dy)
        self._resid_std = resid.std(axis=0)
        self._resid_std[self._resid_std == 0] = 1e-6
        self._fitted = True
        return self

    # -- batch encode -------------------------------------------------------

    def encode_run(self, F: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Causal hidden-state latent per period for one run, ``(len(F), m)``."""
        self._require_fitted()
        X = np.hstack([self._sy(F), self._su(U)])[None, :, :]
        self.net_.eval()
        with torch.no_grad():
            H, _ = self.net_.encode(self._t(X))
        return H[0].cpu().numpy()

    # -- online -------------------------------------------------------------

    def init_belief(self, F_win: np.ndarray, U_win: np.ndarray) -> _Belief:
        """Warm-start the ``(h, c)`` state by running the LSTM over the window."""
        self._require_fitted()
        X = np.hstack([self._sy(F_win), self._su(U_win)])[None, :, :]
        self.net_.eval()
        with torch.no_grad():
            _, (h, c) = self.net_.encode(self._t(X))
        return _Belief(h=h[:, 0, :].cpu().numpy(), c=c[:, 0, :].cpu().numpy())

    def latent(self, belief: _Belief) -> np.ndarray:
        """Return the top layer's hidden state as the conditioning latent."""
        # The top layer's hidden state is the conditioning latent.
        return belief.h[-1].copy()

    def advance(self, belief: _Belief, f_next: np.ndarray, u_next: np.ndarray) -> _Belief:
        """Fold one realised ``(f_next, u_next)`` in via one LSTM step."""
        x = np.hstack([self._sy(f_next[None, :]), self._su(u_next[None, :])])[None, :, :]
        state = (
            self._t(belief.h)[:, None, :].contiguous(),
            self._t(belief.c)[:, None, :].contiguous(),
        )
        self.net_.eval()
        with torch.no_grad():
            _, (h, c) = self.net_.encode(self._t(x), state)
        return _Belief(h=h[:, 0, :].cpu().numpy(), c=c[:, 0, :].cpu().numpy())

    # -- generative forecast ------------------------------------------------

    def forecast(
        self,
        belief: _Belief,
        u_next: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """One-step head forecast from the belief; a draw if ``rng`` is given."""
        h = self._t(belief.h[-1][None, :])
        u = self._t(self._su(u_next[None, :]))
        self.net_.eval()
        with torch.no_grad():
            y = self.net_.read(h, u).cpu().numpy()[0]
        if rng is not None:
            y = y + self._resid_std * rng.standard_normal(y.shape[0])
        return self._inv_y(y[None, :])[0]

    def forecast_means(self, Z: np.ndarray, U_next: np.ndarray) -> np.ndarray:
        """Batch predictive means: next feature row from latents + next exog.

        ``Z`` are the hidden latents returned by :meth:`encode_run` and ``U_next``
        the raw exog features of each target period; returns the deterministic head
        forecast in original feature units.
        """
        h = self._t(Z)
        u = self._t(self._su(U_next))
        self.net_.eval()
        with torch.no_grad():
            Ystd = self.net_.read(h, u).cpu().numpy()
        return self._inv_y(Ystd)
