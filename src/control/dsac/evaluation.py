"""Scoring and recording a policy on a fixed, paired set of futures.

:func:`evaluate` is the number and :func:`rollout` the picture, and they are kept
together because they must agree: the same futures in the same order under the
same per-future seeds, so a score and a recorded path taken at the same ``first``
and ``seed`` are the *same* rolls, and two policies scored here differ only in
what they did.
"""

from __future__ import annotations

import numpy as np

from control.dsac.policies import Policy, reset_policy
from control.env import CentralBankEnv
from control.world import Episode

#: How far apart the seeds of two repeats of one future are. Wider than any bank,
#: so repeat ``r`` of future ``i`` never reuses the seed of another future's roll.
_REPEAT_SEED_STRIDE = 10_000


# -- evaluation -------------------------------------------------------------


def evaluate(
    env: CentralBankEnv,
    policy: Policy,
    n_episodes: int,
    *,
    first: int = 0,
    seed: int = 0,
    repeats: int = 1,
) -> dict[str, float]:
    """Score ``policy`` on ``n_episodes`` futures of ``env``'s bank from ``first``.

    The futures are the same ones every call, and each ``(future, repeat)`` pair
    is rolled under a fixed seed, so two policies scored here are **paired**: they
    see the same forcing and the same transition-noise stream, and the difference
    between their returns is not an artefact of the draw.

    An episode's return varies for two reasons -- which future it is, and how the
    proxy happened to react -- and one roll per future cannot separate them.
    ``repeats`` rolls each future that many times: ``return_std`` is then the
    spread *across futures* (of their means) and ``noise_std`` the spread the
    proxy's own stochasticity contributes *within* a future. Repeats cost
    ``n_episodes * horizon * repeats`` proxy steps, which is the same currency as
    training, so keep them for the headline rather than for every curve point.

    ``first`` skips that many futures before taking ``n_episodes``, exactly as in
    :func:`rollout` and with the same seeding, so a score and a recorded path taken
    at the same ``first`` and ``seed`` are the *same* rolls -- which is what lets a
    figure carry its own return without paying for it twice on a deterministic
    model.

    ``return`` is the episode total. Per **step**, divide it by the horizon rather
    than by ``length``: a collapse is charged once for every period the episode
    would still have run, so the horizon is what that charge was priced against.
    """
    episodes = _futures(env, first, n_episodes)
    per_future, lengths, collapses, rolls = [], [], 0, 0
    terms: dict[str, list[float]] = {}
    for i, episode in enumerate(episodes):
        returns = []
        for r in range(repeats):
            total, steps, terminated = _score_episode(
                env, policy, episode, seed + _REPEAT_SEED_STRIDE * r + first + i, terms
            )
            collapses += terminated
            returns.append(total)
            lengths.append(steps)
            rolls += 1
        per_future.append(returns)

    means = np.mean(per_future, axis=1)
    result = {
        "return": float(np.mean(means)),
        "return_std": float(np.std(means)),
        "noise_std": float(np.mean(np.std(per_future, axis=1))) if repeats > 1 else 0.0,
        "length": float(np.mean(lengths)),
        "collapse_rate": collapses / rolls,
        "repeats": float(repeats),
    }
    result.update({f"term_{k}": float(np.mean(v)) for k, v in terms.items()})
    return result


def _futures(env: CentralBankEnv, first: int, n_episodes: int) -> list[Episode]:
    """The ``n_episodes`` futures of ``env``'s bank from ``first``, never none."""
    episodes = env.episodes.all()[first : first + n_episodes]
    if not episodes:
        raise ValueError(
            f"no futures at [{first}:{first + n_episodes}] of a bank of "
            f"{len(env.episodes)}"
        )
    return episodes


def _score_episode(
    env: CentralBankEnv,
    policy: Policy,
    episode: Episode,
    seed: int,
    terms: dict[str, list[float]],
) -> tuple[float, int, bool]:
    """Roll ``policy`` through ``episode`` once under ``seed``.

    Returns ``(return, steps, terminated)`` and appends every step's reward
    terms to ``terms``.
    """
    obs, _ = env.reset(seed=seed, episode=episode)
    reset_policy(policy)
    total, steps = 0.0, 0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        steps += 1
        for name, value in info.get("reward_terms", {}).items():
            terms.setdefault(name, []).append(value)
        if terminated or truncated:
            return total, steps, terminated


def rollout(
    env: CentralBankEnv,
    policy: Policy,
    n_episodes: int,
    *,
    first: int = 0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Record what ``policy`` set and what the economy did, period by period.

    Scores nothing -- :func:`evaluate` is the number -- this is the *picture*: one
    row per future and one column per step, of every lever the policy moved and of
    the three observables the mandate moves them for (annualised real growth, the
    employment rate, inflation), plus the potential growth that leg is scored
    against. The futures, their order and their per-future seeds are the ones
    :func:`evaluate` uses at ``repeats=1``, so two policies rolled here are paired
    with each other and with their scores.

    ``first`` skips that many futures before taking ``n_episodes`` of them, and a
    future keeps the seed of its position in the bank, so ``first`` chooses *which*
    futures without changing what any one of them is. It is how the futures no
    sweep ever touches are reached: :func:`evaluate` always reads the head of the
    bank, so ``first=eval_episodes`` is the first future the run has never scored.

    Real growth and potential growth both need the period *before* the first one,
    which is the history row the episode branches from -- so the two series line up
    from step zero rather than starting one period in.

    A collapsed episode has no plausible state to record, so its row is ``NaN``
    from the collapse onwards and a column-wise average is over the survivors.
    """
    action_names = env.interface.actions.names()
    episodes = _futures(env, first, n_episodes)
    # An episode is the shorter of its future and the environment's cap (see
    # :attr:`~control.env.EnvConfig.horizon`), so the panels are that wide --
    # sizing them by the future alone would draw a run of trailing ``NaN`` as
    # though every episode had collapsed at the cap.
    cap = env.config.horizon or max(len(e) for e in episodes)
    horizon = min(max(len(e) for e in episodes), cap)
    series = {
        name: np.full((len(episodes), horizon), np.nan)
        for name in (*action_names, "growth", "potential", "ER", "PI")
    }
    for i, episode in enumerate(episodes):
        _record_episode(env, policy, episode, seed + first + i, cap, series, row=i)
    return series


def _record_episode(
    env: CentralBankEnv,
    policy: Policy,
    episode: Episode,
    seed: int,
    cap: int,
    series: dict[str, np.ndarray],
    *,
    row: int,
) -> None:
    """Roll ``policy`` through ``episode`` under ``seed``, filling ``row`` of ``series``.

    At most ``cap`` steps are rolled; the row is left ``NaN`` from a collapse on.
    """
    world = env.driver.world
    action_names = env.interface.actions.names()
    yk = env.interface.state.names().index("Yk")
    nfe, grpr = (env.interface.parameters.names().index(n) for n in ("Nfe", "GRpr"))

    obs, _ = env.reset(seed=seed, episode=episode)
    reset_policy(policy)
    branch_row = episode.branch.row
    prev_yk = float(world.history.states[branch_row][yk])
    prev_nfe = float(world.history.params[branch_row][nfe])
    for t in range(min(len(episode), cap)):
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if terminated:
            break
        state, actions = info["state"], info["actions"]
        for name in action_names:
            series[name][row, t] = getattr(actions, name)
        exog = episode.params[t]
        series["growth"][row, t] = np.log(state.Yk / prev_yk) / world.dt
        series["potential"][row, t] = (
            exog[grpr] + np.log(exog[nfe] / prev_nfe) / world.dt
        )
        series["ER"][row, t] = state.ER
        series["PI"][row, t] = state.PI
        prev_yk, prev_nfe = state.Yk, float(exog[nfe])
        if truncated:
            break
