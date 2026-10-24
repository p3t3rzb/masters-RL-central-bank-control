"""The training loop: fit a proxy on one history, then learn a policy inside it.

Start-up is three steps -- build the world (one solved history plus its bank of
exogenous futures), fit a proxy on that history, fit the observation on it too
(its standardisation statistics and, if the policy is to have a memory, its state
encoder) -- after which the agent never touches the structural model again. Both
halves of that third step are then frozen: the observation is one fixed function
of the trajectory for the whole run, which is what lets the policy be moved to the
ground truth afterwards. What costs is asking the surrogate what happens next, so the budget is
``total_steps * branch_k`` proxy draws: ``total_steps`` transitions actually
followed, each drawn ``branch_k`` times. Neither the number of futures nor the
replay ratio is in that budget -- futures are free to draw, and replay re-reads
transitions already paid for.

Evaluation runs on the held-out futures against a *copy* of the fitted proxy, so a
mid-training evaluation cannot disturb the rollout state of the training
environment (the copy shares the expensive fitted estimator and keeps its own
belief). Three reference policies are scored on the same futures: uniform-random
actions, the book's calibration baseline, and a Taylor rule on the bill rate --
which is the bar that actually matters, since it is a *reactive* policy rather
than a constant one, and beating it is the only version of "the agent learned
something" worth reporting.

This module is the loop and the start-up around it. The pieces it composes live
beside it -- :mod:`~control.dsac.config` (the run's terms),
:mod:`~control.dsac.builders` (proxy, encoders, environments),
:mod:`~control.dsac.policies` (the agent's policy and the references),
:mod:`~control.dsac.evaluation` (paired scoring and recorded rollouts) and
:mod:`~control.dsac.cli` (the command line) -- and every public name of theirs is
re-exported here, which is where callers have always imported them from.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from control.dsac.agent import DSACAgent
from control.dsac.builders import (
    ENCODERS,
    PROXIES,
    build_encoder,
    build_env,
    build_obs_encoder,
    build_proxy,
    build_truth_env,
)
from control.dsac.config import TrainConfig
from control.dsac.evaluation import evaluate, rollout
from control.dsac.policies import (
    Policy,
    agent_policy,
    calibration_actions,
    constant_policy,
    free_instrument,
    random_policy,
    reset_policy,
    taylor_policy,
)
from control.dsac.replay import ReplayBuffer
from control.env import CentralBankEnv
from control.observation import Observer
from control.rewards import MandateReward
from control.world import TrainingWorld, build_world
from economic_models.proxy import BaseProxyModel

__all__ = [
    # defined here
    "TrainingResult",
    "setup",
    "train",
    "main",
    # re-exported from the modules this one was split into, so every name that
    # was ever imported from ``control.dsac.train`` still is
    "TrainConfig",
    "PROXIES",
    "ENCODERS",
    "build_encoder",
    "build_proxy",
    "build_obs_encoder",
    "build_env",
    "build_truth_env",
    "Policy",
    "reset_policy",
    "free_instrument",
    "random_policy",
    "constant_policy",
    "taylor_policy",
    "agent_policy",
    "calibration_actions",
    "evaluate",
    "rollout",
]


@dataclass
class TrainingResult:
    """A finished run: the artefacts plus the histories worth plotting."""

    agent: DSACAgent
    world: TrainingWorld
    proxy: BaseProxyModel
    observer: Observer
    env: CentralBankEnv
    eval_env: CentralBankEnv
    config: TrainConfig
    #: one entry per finished training episode
    episodes_: list[dict[str, float]] = field(default_factory=list)
    #: one entry per logged optimisation step
    updates_: list[dict[str, float]] = field(default_factory=list)
    #: one entry per evaluation sweep on the held-out futures
    evals_: list[dict[str, float]] = field(default_factory=list)
    #: reference returns on the same held-out futures (one roll each, for the plot)
    baselines_: dict[str, dict[str, float]] = field(default_factory=dict)
    #: the evaluation sweep :attr:`agent` was restored to, when
    #: :attr:`~control.dsac.config.TrainConfig.restore_best` is set: ``{"step", "return"}``
    best_: dict[str, float] = field(default_factory=dict)
    #: the headline: agent and references re-scored with ``final_repeats`` rolls
    #: per future, so the across-future and proxy-noise spreads are separated
    final_: dict[str, dict[str, float]] = field(default_factory=dict)


# -- training ---------------------------------------------------------------


def setup(config: TrainConfig | None = None, *, verbose: bool = True) -> TrainingResult:
    """Everything a run needs before the first gradient step, and nothing more.

    Builds the world, fits the proxy on its history, fits the observation on the
    same history, and constructs the two environments and an untrained agent --
    returning them in a :class:`TrainingResult` whose histories are all still
    empty. :func:`train` calls this and then runs the loop.

    It is separate because the loop is the only expensive part. A caller that
    already has a trained actor on disk -- the deployment script of
    :mod:`control.live`, which needs the *same* world, the *same* proxy and above
    all the *same* observation the checkpoint was trained against -- rebuilds the
    stack here in seconds and loads the weights into
    :attr:`TrainingResult.agent`, rather than paying for the training run again
    to recover the things around it.

    Sets torch's thread count process-wide before anything else (see
    :attr:`~control.dsac.config.TrainConfig.threads`), which is a global side effect but the right
    scope for it: the setting has to be in place before the first tensor op, and
    a training run owns its process.
    """
    config = config or TrainConfig()
    if config.threads:
        torch.set_num_threads(config.threads)

    world = build_world(config.world_config(), verbose=verbose)

    if verbose:
        print(
            f"proxy: fitting {config.proxy} ({config.encoder} encoder, "
            f"latent {config.latent}) on {len(world.history)} steps..."
        )
    proxy = build_proxy(
        config.proxy,
        interface=world.config.spec.interface,
        encoder=config.encoder,
        latent=config.latent,
        seed=config.seed,
        epistemic=config.epistemic,
    )
    proxy.fit([world.history])
    observer = Observer(
        world.config.spec.interface,
        encoder=build_obs_encoder(
            config.obs_encoder, latent=config.obs_latent, seed=config.seed
        ),
    ).fit(world.history, branches=world.start_rows)
    reward = MandateReward(
        config.pi_target, employment_target=world.config.spec.employment_target
    )
    if verbose:
        print(
            f"observation: {observer.dim} features "
            f"({config.obs_encoder} encoder, "
            f"latent {observer.encoder.latent_dim})"
        )

    env = build_env(
        proxy, world, observer, reward, config, split="train", seed=config.seed
    )
    # A shallow copy shares the fitted estimator but keeps its own rollout state,
    # so evaluating mid-training cannot disturb the training environment.
    eval_env = build_env(
        copy.copy(proxy),
        world,
        observer,
        reward,
        config,
        split="eval",
        seed=config.seed + 1,
    )

    agent = DSACAgent(
        env.obs_dim,
        env.action_dim,
        n_quantiles=config.n_quantiles,
        hidden=config.hidden,
        n_layers=config.n_layers,
        gamma=config.gamma,
        tau=config.tau,
        lr=config.lr,
        risk=config.risk_measure(),
        seed=config.seed,
        device=config.device,
    )
    return TrainingResult(
        agent=agent,
        world=world,
        proxy=proxy,
        observer=observer,
        env=env,
        eval_env=eval_env,
        config=config,
    )


def train(
    config: TrainConfig | None = None,
    *,
    result: TrainingResult | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Build the world, fit the proxy and train a DSAC agent inside it.

    ``result`` hands in a stack built elsewhere instead of calling :func:`setup`.
    One kind of caller wants that: an experiment that trains the same agent, in
    the same world, against a *different world model* -- a proxy read through an
    online correction, say -- which is a substitution of one object and needs
    none of the rest rebuilt. Everything else about the run, including which
    environment the loop steps and which one it evaluates in, comes from the
    stack it is given.
    """
    config = config or TrainConfig()
    result = setup(config, verbose=verbose) if result is None else result
    env = result.env

    rng = np.random.default_rng(config.seed)
    buffer = ReplayBuffer(config.capacity, env.obs_dim, env.action_dim)

    explore = random_policy(env.action_dim, rng)
    ref_env, references = _references(result, config, explore)
    result.baselines_ = {
        name: evaluate(
            ref_env, policy, config.eval_episodes, repeats=config.eval_repeats
        )
        for name, policy in references.items()
    }
    if verbose:
        _print_plan(result, config)

    best_agent = _train_loop(result, config, buffer, explore, rng, verbose=verbose)

    # Rewind to the best-scoring checkpoint *before* the final scoring, so the
    # headline, the figures and anything the caller goes on to deploy are all the
    # same agent -- the one that was kept, not the one training happened to end on.
    if best_agent is not None:
        result.agent = best_agent
        if verbose and result.best_["step"] != config.total_steps:
            print(
                f"restored the best checkpoint: step {result.best_['step']:.0f} "
                f"(eval {result.best_['return']:.1f}), discarding the "
                f"{config.total_steps - result.best_['step']:.0f} steps after it"
            )

    # The headline, paid for once: every future rolled ``final_repeats`` times,
    # so the spread across futures and the spread the proxy's own noise adds are
    # reported apart rather than summed into one misleading number.
    if config.final_repeats > 1:
        _final_scoring(result, config, ref_env, references, verbose=verbose)

    return result


def _train_loop(
    result: TrainingResult,
    config: TrainConfig,
    buffer: ReplayBuffer,
    explore: Policy,
    rng: np.random.Generator,
    *,
    verbose: bool,
) -> DSACAgent | None:
    """Run the ``total_steps`` collect / update / evaluate loop on ``result``'s agent.

    ``explore`` acts through the warm-up. Fills the episode, update and
    evaluation histories of ``result`` and returns the best-scoring checkpoint
    (``None`` unless :attr:`~control.dsac.config.TrainConfig.restore_best` is set).
    """
    env, agent = result.env, result.agent
    obs, _ = env.reset(seed=config.seed)
    tally = _EpisodeTally()
    stats: dict[str, float] = {}
    best_agent: DSACAgent | None = None

    for step in range(1, config.total_steps + 1):
        action = explore(obs) if step <= config.warmup else agent.act(obs)
        next_obs, reward, terminated, truncated, info = _collect(
            env, buffer, obs, action, config.branch_k
        )
        tally.add(reward, info)
        obs = next_obs

        if terminated or truncated:
            result.episodes_.append(tally.entry(step, terminated))
            obs, _ = env.reset()
            tally = _EpisodeTally()

        stats = _update(agent, buffer, config, rng, step, stats)

        if stats and step % config.log_every == 0:
            result.updates_.append({"step": step, **stats})

        if step % config.eval_every == 0 or step == config.total_steps:
            clone = _checkpoint(result, config, agent, step, stats, verbose=verbose)
            if clone is not None:
                best_agent = clone
            # The evaluation ran on a copy, but the training environment's own
            # episode is mid-flight; leave it running.
    return best_agent


def _update(
    agent: DSACAgent,
    buffer: ReplayBuffer,
    config: TrainConfig,
    rng: np.random.Generator,
    step: int,
    stats: dict[str, float],
) -> dict[str, float]:
    """Run this step's gradient updates, once warm and the buffer holds a batch.

    Returns the last update's statistics, or ``stats`` unchanged when no update
    ran.
    """
    if step <= config.warmup or len(buffer) < config.batch:
        return stats
    for _ in range(config.updates_per_step):
        stats = agent.update(buffer.sample(config.batch, rng))
    return stats


@dataclass
class _EpisodeTally:
    """The running totals of the training episode in flight."""

    episode_return: float = 0.0
    episode_steps: int = 0
    episode_terms: dict[str, list[float]] = field(default_factory=dict)

    def add(self, reward: float, info: Mapping[str, Any]) -> None:
        """Fold one followed step's ``reward`` and its reward terms into the tally."""
        self.episode_return += reward
        self.episode_steps += 1
        for name, value in info.get("reward_terms", {}).items():
            self.episode_terms.setdefault(name, []).append(value)

    def entry(self, step: int, terminated: bool) -> dict[str, float]:
        """The :attr:`TrainingResult.episodes_` entry for the episode ending at ``step``."""
        entry = {
            "step": step,
            "return": self.episode_return,
            "length": self.episode_steps,
            "collapsed": float(terminated),
        }
        entry.update(
            {f"term_{k}": float(np.mean(v)) for k, v in self.episode_terms.items()}
        )
        return entry


def _references(
    result: TrainingResult, config: TrainConfig, explore: Policy
) -> tuple[CentralBankEnv, dict[str, Policy]]:
    """The reference policies and the environment they are scored in.

    ``explore`` is the warm-up policy, which doubles as the random reference.
    """
    # The references act on a free instrument (see :func:`~control.dsac.policies.free_instrument`) --
    # an environment of their own over the same futures under the same seed, so
    # every comparison stays paired while only the agent is speed-constrained.
    ref_env = build_env(
        copy.copy(result.proxy),
        result.world,
        result.observer,
        result.env.reward,
        free_instrument(config),
        split="eval",
        seed=config.seed + 1,
    )
    model = result.world.config.model
    references: dict[str, Policy] = {
        "random": explore,
        "calibration": constant_policy(ref_env, calibration_actions(model)),
        "taylor": taylor_policy(
            ref_env,
            result.observer,
            dt=config.dt,
            pi_target=config.pi_target,
            model=model,
        ),
    }
    return ref_env, references


def _print_plan(result: TrainingResult, config: TrainConfig) -> None:
    """Print the reference scores and the training budget about to be spent."""
    for name, scores in result.baselines_.items():
        print(
            f"baseline {name:>12}: return {scores['return']:9.1f} "
            f"+- {scores['return_std']:7.1f}  "
            f"collapse {scores['collapse_rate']:.0%}"
        )
    stored = config.total_steps * config.branch_k
    print(
        f"train: {config.total_steps} steps x {config.branch_k} draws "
        f"= {stored} proxy draws "
        f"({config.risk_measure().name} risk, device={result.agent.device})"
    )
    if stored > config.capacity:
        print(
            f"  note: {stored} transitions into a {config.capacity} buffer -- "
            f"the oldest {stored - config.capacity} are evicted before the end"
        )


def _collect(
    env: CentralBankEnv,
    buffer: ReplayBuffer,
    obs: np.ndarray,
    action: np.ndarray,
    branch_k: int,
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    """Take one training step from ``obs``, storing every draw of it in ``buffer``.

    Returns the followed draw as a ``(obs, reward, terminated, truncated, info)``
    step.
    """
    # Every draw of this transition is stored; only the last is followed, so
    # the trajectory is unchanged and the buffer gains ``branch_k`` samples of
    # the kernel at this state rather than one.
    draws = env.branch_step(action, branch_k)
    # What the instrument *applied* -- the effective step, which is the
    # requested one only short of the box's edge (see
    # :meth:`~control.env.CentralBankEnv.resolve_action`). Every draw from
    # this position applied the same one; storing the request instead would
    # pair a reward with an action that did not earn it.
    applied = draws[-1].info.get("action", action)
    # ``done`` marks a genuine terminal state only: a horizon truncation must
    # still bootstrap, or the agent learns the world ends every episode.
    for draw in draws:
        buffer.add(obs, applied, draw.reward, draw.obs, draw.terminated)
    return draws[-1].as_step()


def _checkpoint(
    result: TrainingResult,
    config: TrainConfig,
    agent: DSACAgent,
    step: int,
    stats: Mapping[str, float],
    *,
    verbose: bool,
) -> DSACAgent | None:
    """Score ``agent`` on the held-out futures at ``step`` and log the sweep.

    Returns a clone of ``agent`` when it is the best-scoring checkpoint so far
    and :attr:`~control.dsac.config.TrainConfig.restore_best` is set, otherwise ``None``.
    """
    scores = evaluate(
        result.eval_env,
        agent_policy(agent),
        config.eval_episodes,
        repeats=config.eval_repeats,
    )
    result.evals_.append({"step": step, **scores})
    best_agent = None
    if config.restore_best and (
        not result.best_ or scores["return"] > result.best_["return"]
    ):
        result.best_ = {"step": float(step), "return": scores["return"]}
        best_agent = agent.clone()
    if verbose:
        recent = result.episodes_[-20:]
        train_return = np.mean([e["return"] for e in recent]) if recent else np.nan
        print(
            f"  [{step:>7}/{config.total_steps}] "
            f"train {train_return:9.1f} | eval {scores['return']:9.1f} "
            f"+- {scores['return_std']:7.1f} | "
            f"collapse {scores['collapse_rate']:.0%} | "
            f"alpha {stats.get('alpha', float('nan')):.3f}"
        )
    return best_agent


def _final_scoring(
    result: TrainingResult,
    config: TrainConfig,
    ref_env: CentralBankEnv,
    references: Mapping[str, Policy],
    *,
    verbose: bool,
) -> None:
    """Re-score the agent and the references with ``final_repeats`` rolls each.

    Fills :attr:`TrainingResult.final_`.
    """
    if verbose:
        print(
            f"final scoring: {config.eval_episodes} futures x "
            f"{config.final_repeats} rolls..."
        )
    scored = {"agent": (result.eval_env, agent_policy(result.agent))}
    scored.update({name: (ref_env, policy) for name, policy in references.items()})
    result.final_ = {
        name: evaluate(
            scoring_env, policy, config.eval_episodes, repeats=config.final_repeats
        )
        for name, (scoring_env, policy) in scored.items()
    }
    if verbose:
        for name, scores in result.final_.items():
            print(
                f"  {name:>12}: {scores['return']:9.1f} "
                f"+- {scores['return_std']:6.1f} across futures, "
                f"+- {scores['noise_std']:6.1f} from proxy noise "
                f"| collapse {scores['collapse_rate']:.0%}"
            )


def main() -> None:
    """Train an agent from the command line; see :func:`control.dsac.cli.main`."""
    from control.dsac.cli import main as cli_main

    cli_main()
