"""The training pipeline end to end: world, proxy, observer, envs, and the DSAC loop.

Run at a size that finishes in seconds, so what is checked is that the pieces
fit together and the loop is reproducible -- not that the agent learns anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from control.dsac.agent import DSACAgent
from control.dsac.train import TrainConfig, train
from economic_models.ground_truth.registry import MODELS

TINY_RUN = dict(
    history_steps=120,
    horizon=8,
    n_train_futures=6,
    n_eval_futures=3,
    excitation="default",
    total_steps=120,
    warmup=40,
    batch=32,
    capacity=1_000,
    hidden=32,
    n_quantiles=8,
    eval_every=60,
    eval_episodes=2,
    final_repeats=1,
    log_every=10_000,
)


@pytest.fixture(scope="module", params=list(MODELS))
def config(request) -> TrainConfig:
    """A tiny run against one ground truth."""
    return TrainConfig(model=request.param, **TINY_RUN)


@pytest.fixture(scope="module")
def result(config):
    """That run, trained once."""
    return train(config, verbose=False)


def test_training_records_episodes_and_evaluations(result, config):
    """The loop finishes episodes and evaluates on the held-out futures."""
    assert result.episodes_
    assert len(result.evals_) == config.total_steps // config.eval_every
    for sweep in result.evals_:
        assert np.isfinite(sweep["return"])
    assert result.world.config.model == config.model


def test_trained_agent_acts_in_the_box(result):
    """The deployed policy emits normalised lever steps for a real observation."""
    obs, _ = result.eval_env.reset(seed=0)
    action = result.agent.act(obs, deterministic=True)
    assert action.shape == (result.eval_env.action_dim,)
    assert np.all(np.abs(action) <= 1.0)


def test_training_is_reproducible(result, config):
    """The same config retrains to the same evaluation curve."""
    again = train(config, verbose=False)
    assert [e["return"] for e in again.evals_] == pytest.approx(
        [e["return"] for e in result.evals_]
    )


def test_checkpoint_reloads_into_a_rebuilt_agent(result, config, tmp_path):
    """A saved actor dropped into a fresh agent acts identically on the same stack."""
    path = tmp_path / "agent.pt"
    result.agent.save(path)
    env = result.eval_env
    fresh = DSACAgent(
        env.obs_dim, env.action_dim, n_quantiles=config.n_quantiles, hidden=config.hidden, seed=99
    )
    fresh.load(path)
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(
        fresh.act(obs, deterministic=True), result.agent.act(obs, deterministic=True)
    )
