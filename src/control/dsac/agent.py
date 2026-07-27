"""DSAC: a maximum-entropy actor over twin distributional critics.

Soft actor-critic with the point critic replaced by a quantile one. The actor,
the replay and the auto-tuned entropy temperature are SAC's; what changes is what
the critic learns and how the actor reads it:

* **critic** -- each critic outputs ``N`` quantiles of the soft return ``Z(s, a)``
  and is fit by the quantile Huber loss against the distributional Bellman target
  ``r + gamma * (1 - d) * (Z'(s', a') - alpha * log pi(a'|s'))``. The target
  quantiles come from whichever twin has the smaller *mean* -- the distributional
  reading of SAC's ``min``, applied to the whole vector so the target stays a
  coherent distribution rather than an elementwise mixture of two.
* **actor** -- maximises ``risk(Z(s, a)) - alpha * log pi(a|s)``. With
  :class:`~control.dsac.risk.MeanRisk` this is exactly SAC; with
  :class:`~control.dsac.risk.CVaRRisk` it is a risk-averse policy, which is the
  reason to carry a distribution in the first place.

Modelling the spread also curbs the value overestimation a point critic patches
with twin minima, which matters here because the transition noise is genuinely
heavy-tailed: the proxies emit crisis draws.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from economic_models._torch import resolve_device

from control.dsac.networks import QuantileCritic, SquashedGaussianPolicy
from control.dsac.replay import Batch
from control.dsac.risk import MeanRisk, RiskMeasure


class DSACAgent:
    """A distributional soft actor-critic over a continuous action box."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        n_quantiles: int = 32,
        hidden: int = 256,
        n_layers: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        huber_kappa: float = 1.0,
        target_entropy: float | None = None,
        risk: RiskMeasure | None = None,
        seed: int | None = 0,
        device: str | None = None,
    ) -> None:
        """Build the actor, the twin quantile critics and the temperature.

        ``obs_dim``/``action_dim`` size the networks; each critic outputs
        ``n_quantiles`` quantiles from a body of ``n_layers`` layers of width
        ``hidden``. ``gamma`` discounts, ``tau`` is the Polyak rate of the target
        critics, ``lr`` the Adam learning rate shared by actor, critics and
        temperature, and ``huber_kappa`` the quantile Huber threshold.
        ``target_entropy`` defaults to ``-action_dim`` (the usual heuristic).
        ``risk`` is the functional the actor maximises over the return
        distribution (defaults to the risk-neutral mean). ``seed`` seeds torch and
        the sampler; ``device`` selects the compute device.
        """
        # ``None``/``"cpu"`` -> CPU (reproducible default); ``"auto"`` picks
        # CUDA, then Apple MPS, then CPU; any explicit device passes through.
        self.device = resolve_device(device)
        if seed is not None:
            torch.manual_seed(seed)

        self.gamma = gamma
        self.tau = tau
        self.huber_kappa = huber_kappa
        self.action_dim = action_dim
        self.risk = risk or MeanRisk()
        self.target_entropy = (
            float(-action_dim) if target_entropy is None else target_entropy
        )

        self.actor = SquashedGaussianPolicy(obs_dim, action_dim, hidden, n_layers).to(
            self.device
        )
        self.critics = nn.ModuleList(
            QuantileCritic(obs_dim, action_dim, n_quantiles, hidden, n_layers)
            for _ in range(2)
        ).to(self.device)
        self.targets = nn.ModuleList(
            QuantileCritic(obs_dim, action_dim, n_quantiles, hidden, n_layers)
            for _ in range(2)
        ).to(self.device)
        self.targets.load_state_dict(self.critics.state_dict())
        for p in self.targets.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=lr)
        # The temperature is optimised in log space so it stays positive.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        # Quantile midpoints tau_hat = (i + 0.5) / N, the fractions the critic's
        # outputs are estimates of.
        self._tau_hat = (
            (torch.arange(n_quantiles, device=self.device, dtype=torch.float32) + 0.5)
            / n_quantiles
        ).view(1, -1, 1)

    # -- acting --------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """The current entropy temperature."""
        return float(self.log_alpha.detach().exp())

    def act(self, obs: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        """A normalised action for one observation.

        ``deterministic`` returns the squashed mean (evaluation); otherwise the
        action is sampled from the policy (training and exploration).
        """
        with torch.no_grad():
            x = self._t(obs).unsqueeze(0)
            action, _, mean = self.actor.sample(x)
            chosen = mean if deterministic else action
        return chosen.squeeze(0).cpu().numpy()

    # -- learning ------------------------------------------------------------

    def update(self, batch: Batch) -> dict[str, float]:
        """One gradient step on the critics, the actor and the temperature."""
        obs = self._t(batch.obs)
        action = self._t(batch.action)
        reward = self._t(batch.reward)
        next_obs = self._t(batch.next_obs)
        done = self._t(batch.done)
        alpha = self.log_alpha.exp().detach()

        # -- critics: quantile regression on the distributional soft target --
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_obs)
            target_quantiles = self._pessimistic(self.targets, next_obs, next_action)
            soft = target_quantiles - alpha * next_logp
            y = reward + self.gamma * (1.0 - done) * soft  # (n, N)

        critic_loss = sum(
            self._quantile_huber(critic(obs, action), y) for critic in self.critics
        )
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # -- actor: maximise risk(Z) with the entropy bonus -------------------
        fresh, logp, _ = self.actor.sample(obs)
        score = self.risk.aggregate(self._pessimistic(self.critics, obs, fresh))
        actor_loss = (alpha * logp - score).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # -- temperature: hold the policy at the target entropy ---------------
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._polyak()
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": self.alpha,
            "entropy": float(-logp.mean().detach()),
        }

    def _pessimistic(
        self, critics: nn.ModuleList, obs: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """The quantile vector of whichever twin values ``(obs, action)`` lower.

        Selected by mean and taken whole: mixing the two critics elementwise
        would return a vector that is no longer a coherent return distribution.
        """
        left, right = critics[0](obs, action), critics[1](obs, action)
        take_left = (left.mean(dim=-1, keepdim=True) <= right.mean(dim=-1, keepdim=True))
        return torch.where(take_left, left, right)

    def _quantile_huber(self, predicted: torch.Tensor, target: torch.Tensor
                        ) -> torch.Tensor:
        """The quantile Huber loss between predicted and target quantiles.

        ``predicted`` and ``target`` are ``(n, N)``; the pairwise temporal
        differences are weighted by ``|tau_hat - 1{td < 0}|``, summed over the
        predicted quantiles and averaged over the target ones and the batch.
        """
        td = target.unsqueeze(1) - predicted.unsqueeze(2)  # (n, N_pred, N_target)
        k = self.huber_kappa
        huber = torch.where(
            td.abs() <= k, 0.5 * td.pow(2), k * (td.abs() - 0.5 * k)
        )
        weight = (self._tau_hat - (td.detach() < 0).float()).abs()
        return (weight * huber / k).sum(dim=1).mean()

    def _polyak(self) -> None:
        """Soft-update the target critics toward the live ones."""
        with torch.no_grad():
            for target, critic in zip(self.targets.parameters(), self.critics.parameters()):
                target.mul_(1.0 - self.tau).add_(self.tau * critic)

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the actor, critics and temperature to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critics": self.critics.state_dict(),
                "targets": self.targets.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Restore a checkpoint written by :meth:`save`."""
        blob = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(blob["actor"])
        self.critics.load_state_dict(blob["critics"])
        self.targets.load_state_dict(blob["targets"])
        with torch.no_grad():
            self.log_alpha.copy_(blob["log_alpha"].to(self.device))

    def _t(self, A: np.ndarray) -> torch.Tensor:
        """Convert an array to a float32 tensor on this agent's device."""
        return torch.as_tensor(A, dtype=torch.float32, device=self.device)
