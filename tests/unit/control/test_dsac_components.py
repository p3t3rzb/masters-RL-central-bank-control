"""The DSAC pieces that need no environment: risk, replay, networks and the agent.

Small networks on synthetic batches -- enough to pin shapes, the tanh squash,
the zero-initialised gain correction the live deployment rests on, and a lossless
checkpoint round trip.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from control.dsac.agent import DSACAgent
from control.dsac.networks import GainCorrectedPolicy, QuantileCritic, SquashedGaussianPolicy
from control.dsac.replay import Batch, ReplayBuffer
from control.dsac.risk import CVaRRisk, MeanRisk

OBS_DIM, ACTION_DIM = 6, 3


# -- risk measures ---------------------------------------------------------


def test_mean_risk_is_the_expectation():
    """The null-object risk measure is a point critic's ``Q``."""
    q = torch.tensor([[1.0, 2.0, 3.0, 6.0]])
    assert MeanRisk().aggregate(q).tolist() == [[3.0]]


def test_cvar_averages_the_worst_tail_of_unsorted_quantiles():
    """CVaR@0.2 of ten outcomes is the mean of the lowest two, whatever the order."""
    q = torch.tensor([[5.0, 1.0, 9.0, 2.0, 7.0, 3.0, 10.0, 4.0, 8.0, 6.0]])
    assert CVaRRisk(0.2).aggregate(q).item() == pytest.approx(1.5)


def test_cvar_keeps_the_batch_dimension():
    """One score per row, as a column, so it broadcasts against the entropy term."""
    q = torch.randn(7, 32)
    assert CVaRRisk(0.1).aggregate(q).shape == (7, 1)
    torch.testing.assert_close(CVaRRisk(1.0).aggregate(q), MeanRisk().aggregate(q))


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_cvar_rejects_an_empty_or_impossible_tail(alpha):
    """The tail fraction must be in ``(0, 1]``."""
    with pytest.raises(ValueError):
        CVaRRisk(alpha)


def test_risk_names_label_the_attitude():
    """The log label carries the tail fraction."""
    assert MeanRisk().name == "MeanRisk"
    assert CVaRRisk(0.1).name == "CVaR@0.1"


# -- replay ------------------------------------------------------------------


def _fill(buffer: ReplayBuffer, n: int) -> None:
    """Add ``n`` transitions whose every field encodes its insertion index."""
    for i in range(n):
        buffer.add(
            np.full(OBS_DIM, i), np.full(ACTION_DIM, i), float(i), np.full(OBS_DIM, i), i % 2 == 0
        )


def test_replay_grows_then_saturates_at_capacity():
    """Length counts stored transitions, never more than the ring holds."""
    buffer = ReplayBuffer(5, OBS_DIM, ACTION_DIM)
    _fill(buffer, 3)
    assert len(buffer) == 3
    _fill(buffer, 4)
    assert len(buffer) == 5


def test_replay_overwrites_the_oldest_when_full():
    """A full ring keeps the most recent ``capacity`` transitions."""
    buffer = ReplayBuffer(4, OBS_DIM, ACTION_DIM)
    _fill(buffer, 10)
    batch = buffer.sample(500, np.random.default_rng(0))
    assert set(batch.reward.ravel().tolist()) == {6.0, 7.0, 8.0, 9.0}


def test_replay_samples_only_what_was_stored():
    """A partly filled buffer never hands back an empty slot."""
    buffer = ReplayBuffer(100, OBS_DIM, ACTION_DIM)
    _fill(buffer, 3)
    batch = buffer.sample(200, np.random.default_rng(0))
    assert set(batch.obs[:, 0].tolist()) <= {0.0, 1.0, 2.0}


def test_replay_batch_is_row_aligned():
    """Every field of a sampled row comes from the same transition."""
    buffer = ReplayBuffer(10, OBS_DIM, ACTION_DIM)
    _fill(buffer, 10)
    batch = buffer.sample(64, np.random.default_rng(1))
    assert batch.obs.shape == (64, OBS_DIM)
    assert batch.action.shape == (64, ACTION_DIM)
    assert batch.reward.shape == batch.done.shape == (64, 1)
    np.testing.assert_array_equal(batch.obs[:, 0], batch.reward[:, 0])
    np.testing.assert_array_equal(batch.done[:, 0], (batch.reward[:, 0] % 2 == 0).astype(np.float32))


# -- networks ----------------------------------------------------------------


def test_policy_actions_live_inside_the_box():
    """The tanh squash keeps every sampled and greedy action in ``(-1, 1)``."""
    torch.manual_seed(0)
    policy = SquashedGaussianPolicy(OBS_DIM, ACTION_DIM, hidden=16)
    action, log_prob, greedy = policy.sample(10.0 * torch.randn(64, OBS_DIM))
    assert action.shape == greedy.shape == (64, ACTION_DIM)
    assert log_prob.shape == (64, 1)
    assert torch.all(action.abs() <= 1.0) and torch.all(greedy.abs() <= 1.0)


def test_log_prob_rescores_the_policys_own_draw():
    """``log_prob`` of a sampled action agrees with the density ``sample`` returned."""
    torch.manual_seed(0)
    policy = SquashedGaussianPolicy(OBS_DIM, ACTION_DIM, hidden=16)
    obs = torch.randn(32, OBS_DIM)
    action, log_prob, _ = policy.sample(obs)
    keep = (action.abs() < 0.99).all(dim=-1)
    torch.testing.assert_close(
        policy.log_prob(obs, action)[keep], log_prob[keep], atol=1e-3, rtol=1e-3
    )


def test_gain_correction_starts_as_the_offline_policy():
    """Initialised at exactly zero, the corrected policy *is* the frozen one."""
    torch.manual_seed(0)
    base = SquashedGaussianPolicy(OBS_DIM, ACTION_DIM, hidden=16)
    corrected = GainCorrectedPolicy(base, features=[0, 2, 4])
    obs = torch.randn(16, OBS_DIM)
    for got, want in zip(corrected(obs), base(obs)):
        torch.testing.assert_close(got, want)


def test_gain_correction_freezes_the_base():
    """Only the gain is trainable, so a stray optimiser cannot move the offline actor."""
    base = SquashedGaussianPolicy(OBS_DIM, ACTION_DIM, hidden=16)
    corrected = GainCorrectedPolicy(base, features=[1])
    trainable = {name for name, p in corrected.named_parameters() if p.requires_grad}
    assert trainable == {"gain.weight", "gain.bias"}
    with pytest.raises(ValueError):
        GainCorrectedPolicy(base, features=[])


def test_critic_returns_one_quantile_vector_per_pair():
    """The critic maps ``(obs, action)`` to ``n_quantiles`` values."""
    critic = QuantileCritic(OBS_DIM, ACTION_DIM, n_quantiles=8, hidden=16)
    out = critic(torch.randn(5, OBS_DIM), torch.rand(5, ACTION_DIM) * 2 - 1)
    assert out.shape == (5, 8)


# -- the agent ---------------------------------------------------------------


@pytest.fixture
def agent() -> DSACAgent:
    """A tiny agent, seeded."""
    return DSACAgent(OBS_DIM, ACTION_DIM, n_quantiles=8, hidden=16, seed=0)


def _batch(n: int = 32, seed: int = 0) -> Batch:
    """A random batch of plausibly shaped transitions."""
    rng = np.random.default_rng(seed)
    f32 = np.float32
    return Batch(
        obs=rng.normal(size=(n, OBS_DIM)).astype(f32),
        action=rng.uniform(-1, 1, size=(n, ACTION_DIM)).astype(f32),
        reward=rng.normal(size=(n, 1)).astype(f32),
        next_obs=rng.normal(size=(n, OBS_DIM)).astype(f32),
        done=(rng.uniform(size=(n, 1)) < 0.1).astype(f32),
    )


def test_agent_acts_inside_the_box(agent):
    """Both exploration and evaluation actions are normalised lever steps."""
    obs = np.random.default_rng(0).normal(size=OBS_DIM)
    for deterministic in (False, True):
        action = agent.act(obs, deterministic=deterministic)
        assert action.shape == (ACTION_DIM,)
        assert np.all(np.abs(action) <= 1.0)


def test_greedy_action_is_deterministic(agent):
    """Evaluation must not depend on the sampler's state."""
    obs = np.ones(OBS_DIM)
    np.testing.assert_array_equal(agent.act(obs, deterministic=True), agent.act(obs, deterministic=True))


def test_update_moves_every_network_and_reports_finite_losses(agent):
    """One gradient step touches actor, critics, targets and temperature."""
    before = {
        "actor": [p.clone() for p in agent.actor.parameters()],
        "critics": [p.clone() for p in agent.critics.parameters()],
        "targets": [p.clone() for p in agent.targets.parameters()],
    }
    alpha = agent.alpha
    stats = agent.update(_batch())

    assert set(stats) >= {"critic_loss", "actor_loss", "alpha", "entropy", "anchor_kl"}
    assert all(np.isfinite(v) for v in stats.values())
    assert stats["anchor_kl"] == 0.0
    assert agent.alpha != alpha
    for name, params in before.items():
        now = getattr(agent, name).parameters()
        assert any(not torch.equal(a, b) for a, b in zip(params, now)), name


def test_anchor_to_itself_has_zero_kl(agent):
    """An agent anchored to an identical copy pays no KL at the first step."""
    clone = agent.clone()
    stats = agent.update(_batch(), anchor=clone, anchor_weight=1.0)
    assert stats["anchor_kl"] == pytest.approx(0.0, abs=1e-5)


def test_checkpoint_round_trip_restores_the_policy(agent, tmp_path):
    """A saved agent loaded into a fresh one acts identically."""
    agent.update(_batch())
    path = tmp_path / "agent.pt"
    agent.save(path)
    fresh = DSACAgent(OBS_DIM, ACTION_DIM, n_quantiles=8, hidden=16, seed=123)
    fresh.load(path)

    obs = np.random.default_rng(0).normal(size=(4, OBS_DIM))
    for row in obs:
        np.testing.assert_array_equal(
            fresh.act(row, deterministic=True), agent.act(row, deterministic=True)
        )
    assert fresh.alpha == pytest.approx(agent.alpha)
