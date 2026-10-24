"""The central-bank environment, over a fitted proxy, for every ground truth.

The delta instrument and the collapse rule are the two pieces of the
environment the agent's whole problem is shaped by, so their arithmetic is
checked against the trailing exogenous block the environment itself records.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from control.dsac.train import (
    build_truth_env,
    calibration_actions,
    constant_policy,
    evaluate,
    free_instrument,
    rollout,
)
from control.env import CentralBankEnv


@pytest.fixture
def env(stack) -> CentralBankEnv:
    """The stack's training environment, reset onto a fixed episode."""
    stack.env.reset(seed=0)
    return stack.env


def _position(env: CentralBankEnv) -> np.ndarray:
    """Where the levers stand now, in the normalised box."""
    return env.to_normalised(env.interface.actions.from_row(env.exog[-1][-env.action_dim :]).to_dict())


def _fresh(stack, **config) -> CentralBankEnv:
    """A new environment over the stack's pieces, with config overrides."""
    env = stack.env
    return CentralBankEnv(
        env.driver,
        env.episodes,
        env.reward,
        env.observer,
        env.interface,
        dataclasses.replace(env.config, **config),
        seed=0,
    )


# -- the episode -------------------------------------------------------------


def test_reset_starts_at_the_episodes_branch(stack):
    """The first observation is finite and the episode starts at its fork row."""
    obs, info = stack.env.reset(seed=0)
    assert obs.shape == (stack.env.obs_dim,)
    assert np.all(np.isfinite(obs))
    assert info["branch"] == len(stack.world.history) - 1
    assert stack.env.step_index == 0


def test_episode_runs_to_its_horizon(env):
    """A surviving episode truncates at the horizon and never terminates."""
    for t in range(env.horizon):
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_dim))
        assert not terminated
        assert truncated == (t == env.horizon - 1)
        assert np.isfinite(reward) and reward <= 0.0
        assert reward == pytest.approx(
            sum(env.reward.weights[k] * v for k, v in info["reward_terms"].items())
        )
    assert env.step_index == env.horizon


def test_observation_is_clipped_and_standardised(env):
    """No feature exceeds the observer's clip, however the economy moves."""
    for _ in range(env.horizon):
        obs, *_ = env.step(np.ones(env.action_dim))
        assert np.all(np.abs(obs) <= env.observer.clip)


# -- the delta instrument ----------------------------------------------------


def test_zero_step_holds_the_levers(env):
    """A zero request leaves every lever exactly where it stood."""
    before = env.exog[-1].copy()
    *_, info = env.step(np.zeros(env.action_dim))
    n = env.action_dim
    np.testing.assert_allclose(env.exog[-1][-n:], before[-n:])
    np.testing.assert_array_equal(info["action"], np.zeros(n))


def test_applied_step_is_the_position_change(env):
    """The stored action is the move the instrument made, box edge included."""
    clipped = False
    for _ in range(env.horizon):
        before = _position(env)
        *_, info = env.step(np.ones(env.action_dim))
        after = _position(env)
        np.testing.assert_allclose(info["action"], (after - before) / env.delta_step, atol=1e-9)
        assert np.all(np.abs(after) <= 1.0 + 1e-12)
        assert np.all(info["action"] <= 1.0 + 1e-12)
        clipped |= bool(np.any(info["action"] < 1.0 - 1e-9))
    assert clipped, "eight full-speed steps should reach some lever's edge"


def test_delta_rate_is_annualised(stack):
    """One period's maximum move is ``2 * delta_rate * dt`` box units."""
    env = stack.env
    assert env.delta_step == pytest.approx(2.0 * env.config.delta_rate * stack.world.dt)


def test_level_mapping_round_trips(env):
    """``to_actions`` and ``to_normalised`` are inverse inside the box."""
    levels = calibration_actions(env.driver.world.config.model)
    back = env.to_actions(env.to_normalised(levels)).to_dict()
    assert back == pytest.approx(levels)
    edges = {name: high + 1.0 for name, (low, high) in env.config.action_bounds.items()}
    np.testing.assert_array_equal(env.to_normalised(edges), np.ones(env.action_dim))


def test_toward_moves_at_most_one_step(env):
    """A level target is approached at instrument speed and never overshot."""
    target = {name: high for name, (low, high) in env.config.action_bounds.items()}
    step = env.toward(target)
    assert np.all(np.abs(step) <= 1.0)
    before = _position(env)
    env.step(step)
    after = _position(env)
    assert np.all(after >= before - 1e-12)
    assert np.all(after - before <= env.delta_step + 1e-12)


def test_toward_needs_an_episode(stack):
    """A rule stated in levels reads the episode's levers, so it needs a reset."""
    with pytest.raises(RuntimeError, match="reset"):
        _fresh(stack).toward(calibration_actions(stack.world.config.model))


# -- collapse --------------------------------------------------------------


def test_collapse_costs_every_remaining_step(stack):
    """Leaving the corridor ends the episode at the penalty times the steps left."""
    env = _fresh(stack, er_bounds=(5.0, 6.0))
    obs, _ = env.reset(seed=0)
    next_obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_dim))
    assert terminated and not truncated
    assert reward == pytest.approx(env.config.collapse_penalty * env.horizon)
    assert "collapse" in info
    np.testing.assert_array_equal(next_obs, obs)


def test_plausibility_is_exposed_for_other_models(env):
    """The corridor a step applies is the one ``plausible`` answers."""
    *_, info = env.step(np.zeros(env.action_dim))
    assert env.plausible(info["state"])


# -- branching ---------------------------------------------------------------


def test_branch_step_draws_k_and_commits_one(env):
    """``k`` draws from one position, and the clock advances once."""
    draws = env.branch_step(np.zeros(env.action_dim), k=3)
    assert len(draws) == 3
    assert env.step_index == 1
    np.testing.assert_array_equal(env.exog, draws[-1].exog)
    assert len({d.reward for d in draws}) > 1


def test_branch_step_rejects_zero_draws(env):
    """At least one draw is always taken."""
    with pytest.raises(ValueError):
        env.branch_step(np.zeros(env.action_dim), k=0)


def test_ground_truth_cannot_branch(stack):
    """A deterministic solver has nothing to redraw; asking for ``k > 1`` is refused."""
    env = build_truth_env(stack.world, stack.observer, stack.env.reward, stack.config)
    env.reset(seed=0)
    with pytest.raises(RuntimeError, match="cannot branch"):
        env.branch_step(np.zeros(env.action_dim), k=2)


# -- scoring ---------------------------------------------------------------


def test_evaluation_is_paired(stack):
    """Two policies scored at one seed see the same futures and the same noise."""
    env = stack.eval_env
    policy = constant_policy(env, calibration_actions(stack.world.config.model))
    first, again = evaluate(env, policy, 2, seed=0), evaluate(env, policy, 2, seed=0)
    assert first == again
    assert first["collapse_rate"] == 0.0
    assert {"term_unemployment", "term_growth", "term_inflation"} <= set(first)


def test_resting_policy_survives_the_ground_truth(stack):
    """The calibration levers keep the structural economy inside its corridor."""
    config = free_instrument(stack.config)
    env = build_truth_env(stack.world, stack.observer, stack.env.reward, config)
    score = evaluate(env, constant_policy(env, calibration_actions(stack.world.config.model)), 2)
    assert score["collapse_rate"] == 0.0
    assert score["length"] == env.horizon
    assert np.isfinite(score["return"])


def test_rollout_records_one_row_per_future(stack):
    """The picture has a row per future and a column per step, no collapses."""
    env = stack.eval_env
    series = rollout(env, constant_policy(env, calibration_actions(stack.world.config.model)), 2)
    for name in (*env.interface.actions.names(), "growth", "potential", "ER", "PI"):
        assert series[name].shape == (2, env.horizon), name
        assert np.all(np.isfinite(series[name])), name
