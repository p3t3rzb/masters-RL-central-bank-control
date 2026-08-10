"""The two networks DSAC needs: a squashed Gaussian actor and a quantile critic.

The actor is the standard maximum-entropy policy -- a diagonal Gaussian in
pre-squash space, pushed through ``tanh`` into the normalised action box, with the
change-of-variables correction on its log-density so the entropy term stays exact.

The critic is what makes this *distributional*: instead of regressing the expected
soft return it outputs ``n_quantiles`` values, the inverse CDF of the return
distribution ``Z(s, a)`` evaluated at the midpoints ``(i + 0.5) / N``. A point
critic is the special case ``N = 1``; everything downstream (the quantile Huber
loss, the risk measure the actor optimises) reads that vector rather than a mean.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

#: Clamp on the policy's log standard deviation -- keeps the Gaussian from
#: collapsing to a point mass or diffusing over the whole box.
_LOG_STD_MIN, _LOG_STD_MAX = -20.0, 2.0

#: How far inside ``(-1, 1)`` an action is clamped before the squash is inverted
#: (:meth:`SquashedGaussianPolicy.log_prob`) -- ``atanh`` is unbounded at the
#: edges, which ``tanh`` reaches only in the limit but floating point reaches
#: exactly.
_TANH_EPS = 1e-6


def _mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int) -> nn.Sequential:
    """A plain ReLU MLP of ``n_layers`` hidden layers of width ``hidden``."""
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(dim, hidden), nn.ReLU()]
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class SquashedGaussianPolicy(nn.Module):
    """A tanh-squashed diagonal Gaussian policy over the normalised action box."""

    def __init__(
        self, obs_dim: int, action_dim: int, hidden: int = 256, n_layers: int = 2
    ) -> None:
        """Build the shared body and the mean / log-std heads.

        ``obs_dim`` and ``action_dim`` size the input and output; the body is
        ``n_layers`` ReLU layers of width ``hidden``.
        """
        super().__init__()
        self.body = _mlp(obs_dim, 2 * action_dim, hidden, n_layers)
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The pre-squash mean and log standard deviation for ``obs``."""
        mu, log_std = self.body(obs).chunk(2, dim=-1)
        return mu, log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A reparameterised action, its log-density, and the greedy action.

        The log-density carries the ``tanh`` change-of-variables correction
        ``log(1 - tanh(u)^2)``, computed in the numerically stable form.
        """
        mu, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        u = normal.rsample()
        log_prob = self._squashed_log_prob(normal, u)
        return torch.tanh(u), log_prob, torch.tanh(mu)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """The log-density this policy assigns to an action *another* one chose.

        :meth:`sample` hands back the density of its own draw, which is all the
        entropy term needs; comparing two policies on the same action needs this
        instead. Inverts the squash to recover the pre-squash point and scores it
        there, clamping just inside ``(-1, 1)`` because ``atanh`` diverges at the
        edges the ``tanh`` only approaches.
        """
        mu, log_std = self(obs)
        normal = torch.distributions.Normal(mu, log_std.exp())
        u = torch.atanh(action.clamp(-1.0 + _TANH_EPS, 1.0 - _TANH_EPS))
        return self._squashed_log_prob(normal, u)

    @staticmethod
    def _squashed_log_prob(
        normal: torch.distributions.Normal, u: torch.Tensor
    ) -> torch.Tensor:
        """Density of ``tanh(u)`` under the pre-squash Gaussian, summed over levers."""
        # log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u)), stable for large |u|.
        correction = 2.0 * (np.log(2.0) - u - nn.functional.softplus(-2.0 * u))
        return (normal.log_prob(u) - correction).sum(dim=-1, keepdim=True)


class QuantileCritic(nn.Module):
    """A critic that outputs the return distribution as ``n_quantiles`` values."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_quantiles: int = 32,
        hidden: int = 256,
        n_layers: int = 2,
    ) -> None:
        """Build the MLP mapping ``[obs, action]`` to a quantile vector."""
        super().__init__()
        self.body = _mlp(obs_dim + action_dim, n_quantiles, hidden, n_layers)
        self.n_quantiles = n_quantiles

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Quantiles of ``Z(s, a)``, shape ``(batch, n_quantiles)``."""
        return self.body(torch.cat([obs, action], dim=-1))
