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
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass, field, fields, replace
from typing import Callable, Mapping

import numpy as np
import torch

from economic_models.encoders import (
    KalmanEncoder,
    LSTMEncoder,
    NullEncoder,
    StateEncoder,
)
from economic_models.ground_truth import GROWTH_INTERFACE, GrowthCalibration
from economic_models.proxy import (
    BaseProxyModel,
    DRFProxy,
    KNNProxy,
    MDNProxy,
    RandomWalkProxy,
    VARXProxy,
)

from control.dsac.agent import DSACAgent
from control.dsac.replay import ReplayBuffer
from control.dsac.risk import CVaRRisk, MeanRisk, RiskMeasure
from control.drivers import GroundTruthDriver, ModelDriver, ProxyDriver
from control.env import CentralBankEnv, EnvConfig
from control.observation import Observer
from control.rewards import MandateReward, RewardFunction
from control.world import TrainingWorld, WorldConfig, build_world

#: The proxy families this trainer can use as a world model.
PROXIES = ("varx", "drf", "mdn", "knn", "random-walk")

#: The state encoders, by name. Two are learned filters of the same shape -- the
#: linear-Gaussian :class:`~economic_models.encoders.kalman.KalmanEncoder` and its
#: nonlinear sibling the :class:`~economic_models.encoders.lstm.LSTMEncoder`, both
#: taking a latent width and a seed and nothing else -- so either can play either
#: of the two encoder roles here. ``"none"`` is the zero-width
#: :class:`~economic_models.encoders.base.NullEncoder`: as the *proxy's* encoder
#: it leaves a forecaster conditioned on the exogenous block alone (which is what
#: the random walk uses), and as the *observation's* it is the memoryless agent.
ENCODERS = ("none", "kalman", "lstm")

#: A policy maps one observation to a normalised action in ``[-1, 1]``. Most are
#: stateless functions of the observation; one that carries within-episode memory
#: (the Taylor rule's smoothed growth gap) additionally offers a ``reset()``,
#: which :func:`reset_policy` calls at every episode boundary.
Policy = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TrainConfig:
    """Everything that defines one training run.

    The world fields mirror :class:`~control.world.WorldConfig`; ``proxy``,
    ``encoder`` and ``latent`` pick the world model and the conditioning latent it
    forecasts from; the remainder are the agent's optimisation knobs.
    ``total_steps`` is the number of environment steps and
    ``total_steps * branch_k`` the number of proxy draws, which is what dominates
    the wall clock.

    The two encoders are configured **independently**: ``encoder``/``latent`` is
    the filter the *world model* conditions on, ``obs_encoder``/``obs_latent`` the
    filter the *policy* carries as its memory. Nothing requires them to match --
    the observation must be computable on the ground-truth side, where there is no
    proxy to borrow a latent from, so the two are separate instances holding
    separate beliefs even when set to the same family. Crossing them (a Kalman
    world model read by an LSTM-memory policy, say) is a supported ablation, not a
    misconfiguration.
    """

    # -- world --
    dt: float = 0.25  #: length of one step in years
    history_steps: int = 500  #: steps of the single run the proxy is fit on
    horizon: int = 50  #: steps per episode
    n_train_futures: int = 2000  #: exogenous paths for training episodes
    n_eval_futures: int = 64  #: held-out exogenous paths
    excitation: str = "realistic"  #: excitation preset for the world
    world_seed: int = 0  #: seeds the history and its futures

    # -- world model --
    proxy: str = "varx"  #: which proxy family is the environment
    #: the encoder whose filtered latent the proxy forecasts from (one of
    #: :data:`ENCODERS`). ``"kalman"`` is the linear-Gaussian filter, ``"lstm"``
    #: its nonlinear sibling; ``"none"`` leaves the proxy conditioned on the
    #: exogenous block alone, which is what the random walk is.
    encoder: str = "kalman"
    latent: int = 10  #: latent width of the proxy's state encoder

    # -- what the policy remembers --
    #: the encoder whose filtered latent is appended to the observation (one of
    #: :data:`ENCODERS`). At ``"none"`` the agent is memoryless: it sees only
    #: the current stationary features, which are Markov in a world that is not,
    #: since the excitation's hidden structural parameters persist unobserved.
    #: Anything else gives it a recursive summary of the episode so far, fit once
    #: on the history and frozen.
    obs_encoder: str = "none"
    obs_latent: int = 10  #: latent width of the observation encoder

    # -- mandate --
    pi_target: float = 0.02  #: annual inflation target
    collapse_penalty: float = -25.0  #: reward per remaining step of a collapsed episode

    # -- agent --
    total_steps: int = 100_000  #: environment steps (== proxy steps)
    warmup: int = 2_000  #: uniform-random steps before learning starts
    batch: int = 256  #: minibatch size
    capacity: int = 200_000  #: replay capacity
    gamma: float = 0.99  #: discount per step
    tau: float = 0.005  #: Polyak rate of the target critics
    lr: float = 3e-4  #: Adam learning rate
    n_quantiles: int = 32  #: quantiles per critic
    hidden: int = 256  #: width of every network body
    n_layers: int = 2  #: hidden layers per network body
    updates_per_step: int = 1  #: gradient steps per environment step
    #: draws of each transition (see
    #: :meth:`~control.env.CentralBankEnv.branch_step`). At ``1`` the buffer holds
    #: one sample of the kernel per visited state and the critic recovers its
    #: spread only by smoothing across neighbouring states; above ``1`` it holds
    #: ``branch_k`` samples at the state itself, at ``branch_k`` times the proxy
    #: draws for the same ``total_steps`` of trajectory. Worth its cost only where
    #: that smoothing is thin, which is the tail ``risk="cvar"`` optimises over.
    branch_k: int = 1
    risk: str = "mean"  #: "mean" (risk-neutral) or "cvar"
    cvar_alpha: float = 0.1  #: tail fraction when ``risk="cvar"``

    # -- bookkeeping --
    eval_every: int = 5_000  #: environment steps between evaluations
    eval_episodes: int = 32  #: held-out futures per evaluation
    #: rolls per future during training evaluations. One is enough for a curve
    #: (the sweeps are paired, so the comparison is fair) and costs the same
    #: currency as training: a sweep is ``eval_episodes * horizon * repeats``
    #: proxy steps.
    eval_repeats: int = 1
    #: rolls per future for the final scoring, where separating "which future"
    #: from "how the proxy reacted" is worth paying for.
    final_repeats: int = 8
    log_every: int = 1_000  #: environment steps between progress lines
    seed: int = 0  #: seeds the agent, the replay sampling and the environments
    device: str | None = None  #: torch device (``None``/``"cpu"``/``"auto"``)
    #: intra-op torch threads, applied process-wide by :func:`train` (``0`` leaves
    #: torch's default alone). One is the *fast* setting, not a throttle: the
    #: networks here are small enough that a batch of 256 through a width-256 MLP
    #: costs less than synchronising the threads that split it, so torch's default
    #: of one thread per core makes an update roughly twice as slow as running it
    #: on a single core. Staying single-threaded also leaves the other cores free,
    #: which is what makes a sweep of runs scale: independent processes at one
    #: thread each are far and away the cheapest speed-up available here.
    threads: int = 1

    def world_config(self) -> WorldConfig:
        """The :class:`~control.world.WorldConfig` this run's world is built from."""
        return WorldConfig(
            dt=self.dt,
            history_steps=self.history_steps,
            horizon=self.horizon,
            n_train_futures=self.n_train_futures,
            n_eval_futures=self.n_eval_futures,
            excitation=self.excitation,
            seed=self.world_seed,
        )

    def risk_measure(self) -> RiskMeasure:
        """The risk functional the actor maximises."""
        if self.risk == "mean":
            return MeanRisk()
        if self.risk == "cvar":
            return CVaRRisk(self.cvar_alpha)
        raise ValueError(f"risk must be 'mean' or 'cvar', got {self.risk!r}")


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
    #: the headline: agent and references re-scored with ``final_repeats`` rolls
    #: per future, so the across-future and proxy-noise spreads are separated
    final_: dict[str, dict[str, float]] = field(default_factory=dict)


# -- construction ----------------------------------------------------------


def build_encoder(
    name: str, *, latent: int = 10, seed: int = 0, role: str = "encoder"
) -> StateEncoder:
    """An unfitted state encoder of family ``name`` (one of :data:`ENCODERS`).

    The one place an encoder is named, for both roles: the latent a proxy
    forecasts from and the latent the observation carries. ``latent`` is its
    width, ``seed`` its initialisation, and ``role`` only names the field in the
    error message, since the two are configured separately and either can be
    wrong.

    Nothing beyond the width and the seed is exposed. The encoders' remaining
    knobs (the LSTM's depth, epochs and learning rate, say) are the encoder
    family's own business and are left at their defaults here, which is what keeps
    the two families interchangeable at this seam.
    """
    if name not in ENCODERS:
        raise ValueError(f"{role} must be one of {ENCODERS}, got {name!r}")
    if name == "kalman":
        return KalmanEncoder(latent_dim=latent, seed=seed)
    if name == "lstm":
        return LSTMEncoder(latent_dim=latent, seed=seed)
    return NullEncoder()


def build_proxy(
    name: str, *, encoder: str = "kalman", latent: int = 10, seed: int = 0
) -> BaseProxyModel:
    """An unfitted proxy of family ``name`` over the GROWTH interface.

    ``encoder`` picks the family of the conditioning latent it forecasts from and
    ``latent`` that latent's width -- a free choice, since a proxy reads its
    encoder through the :class:`~economic_models.encoders.base.StateEncoder`
    interface alone. Each proxy gets its **own** instance, never a shared one: an
    encoder is fitted state, and two proxies fitted on the same history must not
    end up sharing one belief.

    The random walk takes no encoder (it is the encoder-free baseline) and ignores
    both arguments.
    """
    if name not in PROXIES:
        raise ValueError(f"proxy must be one of {PROXIES}, got {name!r}")

    def new() -> StateEncoder:
        return build_encoder(encoder, latent=latent, seed=seed, role="encoder")

    if name == "varx":
        return VARXProxy(GROWTH_INTERFACE, encoder=new())
    if name == "drf":
        return DRFProxy(GROWTH_INTERFACE, encoder=new(), seed=seed)
    if name == "mdn":
        return MDNProxy(GROWTH_INTERFACE, encoder=new(), seed=seed)
    if name == "knn":
        return KNNProxy(GROWTH_INTERFACE, encoder=new())
    return RandomWalkProxy(GROWTH_INTERFACE)


def build_obs_encoder(
    name: str, *, latent: int = 10, seed: int = 0
) -> StateEncoder:
    """An unfitted state encoder for the observation's latent block.

    Deliberately a fresh instance rather than the proxy's, even where both are the
    same family: the observation has to be computable on the ground-truth side
    too, where there is no proxy to borrow a component from. Both end up fit on
    the same history, so they agree.
    """
    return build_encoder(name, latent=latent, seed=seed, role="obs_encoder")


def _env(
    driver: ModelDriver,
    world: TrainingWorld,
    observer: Observer,
    reward: RewardFunction,
    config: TrainConfig,
    *,
    split: str,
    seed: int,
) -> CentralBankEnv:
    """An environment over ``driver``, drawing from the train or eval futures."""
    episodes = world.train_futures if split == "train" else world.eval_futures
    return CentralBankEnv(
        driver,
        episodes,
        reward,
        observer,
        GROWTH_INTERFACE,
        EnvConfig(collapse_penalty=config.collapse_penalty),
        seed=seed,
    )


def build_env(
    proxy: BaseProxyModel,
    world: TrainingWorld,
    observer: Observer,
    reward: RewardFunction,
    config: TrainConfig,
    *,
    split: str,
    seed: int,
) -> CentralBankEnv:
    """An environment over ``proxy`` drawing from the train or eval futures."""
    return _env(ProxyDriver(proxy, world), world, observer, reward, config,
                split=split, seed=seed)


def build_truth_env(
    world: TrainingWorld,
    observer: Observer,
    reward: RewardFunction,
    config: TrainConfig,
    *,
    split: str = "eval",
    seed: int = 0,
) -> CentralBankEnv:
    """The same environment over the **structural** model instead of a proxy.

    Everything a policy touches is unchanged -- the same observer, the same
    mandate, the same action box, the same futures -- so a policy trained in the
    proxy can be dropped straight in and what differs is only who answers "what
    happens next". That is the whole point: the gap between a policy's return here
    and in the proxy it was trained in *is* the surrogate's error, expressed in the
    units the mandate is written in.

    Expensive by comparison -- every step solves the full system rather than
    sampling a fitted conditional -- and not branchable, so it is for evaluation
    and never for training.
    """
    return _env(GroundTruthDriver(world), world, observer, reward, config,
                split=split, seed=seed)


# -- policies ---------------------------------------------------------------


def reset_policy(policy: Policy) -> None:
    """Clear a policy's within-episode memory at an episode boundary.

    A no-op for the stateless policies, which is all of them but the Taylor rule
    -- rather than making every policy carry an empty ``reset``, this asks for one
    and does nothing when there is none, so a plain function stays a valid
    :data:`Policy`. Called by :func:`evaluate` and :func:`rollout` right after
    every :meth:`~control.env.CentralBankEnv.reset`.
    """
    reset = getattr(policy, "reset", None)
    if reset is not None:
        reset()


def random_policy(action_dim: int, rng: np.random.Generator) -> Policy:
    """A policy that samples uniformly from the action box."""
    return lambda obs: rng.uniform(-1.0, 1.0, size=action_dim)


def constant_policy(env: CentralBankEnv, levels: Mapping[str, float]) -> Policy:
    """A policy that always applies the given action *levels* (not normalised)."""
    action = env.to_normalised(levels)
    return lambda obs: action


def taylor_policy(
    env: CentralBankEnv,
    observer: Observer,
    *,
    dt: float,
    pi_target: float = 0.02,
    phi_pi: float = 0.5,
    phi_y: float = 0.5,
) -> Policy:
    """The textbook Taylor rule on the bill rate, calibration on the other levers.

    ``Rbbar = i* + (1 + phi_pi)(PI - pi_target) + phi_y * growth_gap``, where
    ``i*`` is the calibration bill rate -- so at target inflation and at potential
    growth the rule *is* the calibration baseline, and every difference between the
    two references is the rule reacting to the cycle. The response to inflation is
    ``1 + phi_pi`` because the nominal rate must move with ``PI`` one-for-one
    before it moves the *real* rate at all (the Taylor principle); the defaults
    ``phi_pi = phi_y = 0.5`` are Taylor's original pair.

    The gap term is a **growth** gap, not a level output gap: the same
    ``realised - potential`` annualised growth the mandate scores (see
    :class:`~control.rewards.mandate.MandateReward`), because potential *output*
    is not identified from an observation whereas potential *growth* is
    (``GRpr`` plus full-employment labour growth). ``NCAR`` and ``ro`` are held at
    calibration: this is a monetary rule, and leaving the other two levers fixed
    is what makes it comparable to the baseline.

    The gap is **smoothed over roughly a year** before it is acted on, by the same
    exponential recursion the model itself uses to build ``PI`` from the one-period
    price change (11.10: ``x = dt*x_raw + (1 - dt)*x(-1)``). Without it the rule is
    not frequency-invariant and destabilises the economy at sub-quarterly ``dt``:
    ``dlog(Yk)/dt`` annualises a *single* period's log-difference, whose transitory
    component scales as ``sqrt(dt)`` rather than ``dt``, so the effective feedback
    gain on it grows as ``1/sqrt(dt)`` while the transmission lag stays fixed in
    calendar time -- high gain against an unchanged delay, which shows up as a
    period-2 flip oscillation that the action clip then sustains as a limit cycle.
    The smoother's gain at that flip frequency is ``dt/(2 - dt)``, which cancels the
    ``1/dt`` annualisation almost exactly, so ``phi_y`` means the same thing at
    quarterly and monthly steps. ``PI`` needs no such treatment: it arrives already
    smoothed this way, which is why only the growth leg had to be fixed.

    Carrying that smoothed gap makes the rule **stateful**: it holds one scalar
    between steps and :meth:`reset` clears it at an episode boundary, which
    :func:`reset_policy` does for every roll. It starts each episode at zero -- the
    branch point sits at the book's calibration, which grows at potential -- so the
    rule opens at the calibration rate and earns its deviations from data.

    The rule reads the standardised observation and inverts it, so it consumes
    exactly what the agent does and can be scored on the same futures. Rates
    outside :attr:`~control.env.EnvConfig.action_bounds` are clipped by
    :meth:`~control.env.CentralBankEnv.to_normalised`, and that lower bound
    **binds often**: the book's calibration settles with inflation around 1%, so a
    rule aiming at 2% cuts into the floor and spends much of an episode there. The
    rule is therefore a reference, not a well-tuned policy -- which is the point of
    scoring it next to the constant baseline rather than instead of it.
    """
    return _TaylorRule(
        env,
        observer,
        dt=dt,
        pi_target=pi_target,
        phi_pi=phi_pi,
        phi_y=phi_y,
    )


class _TaylorRule:
    """The stateful body of :func:`taylor_policy`; see it for the rule itself.

    A class rather than a closure because the rule carries the smoothed growth gap
    between steps and has to be told where an episode ends -- state a plain
    function would have to hide in a cell and expose by attribute.
    """

    def __init__(
        self,
        env: CentralBankEnv,
        observer: Observer,
        *,
        dt: float,
        pi_target: float,
        phi_pi: float,
        phi_y: float,
    ) -> None:
        """Bind the rule to an environment and resolve its feature columns."""
        self._env = env
        self._observer = observer
        self._dt = dt
        self._pi_target = pi_target
        self._phi_pi = phi_pi
        self._phi_y = phi_y
        self._levels = dict(calibration_actions())
        self._i_star = self._levels["Rbbar"]
        self._pi_at, self._growth_at, self._potential_at, self._labour_at = (
            observer.index(name) for name in ("PI", "dlog(Yk)", "GRpr", "dlog(Nfe)")
        )
        self._gap = 0.0

    def reset(self) -> None:
        """Clear the smoothed growth gap; the next episode opens at potential."""
        self._gap = 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """The bill rate the rule sets, given one standardised observation."""
        raw = self._observer.unstandardise(obs)
        dt = self._dt
        gap = raw[self._growth_at] / dt - (
            raw[self._potential_at] + raw[self._labour_at] / dt
        )
        # 11.10's smoother, applied to the growth gap: a ~1-year time constant at
        # any dt, which is what keeps phi_y frequency-invariant.
        self._gap = dt * gap + (1.0 - dt) * self._gap
        rate = (
            self._i_star
            + (1.0 + self._phi_pi) * (raw[self._pi_at] - self._pi_target)
            + self._phi_y * self._gap
        )
        return self._env.to_normalised({**self._levels, "Rbbar": rate})


def agent_policy(agent: DSACAgent, *, deterministic: bool = True) -> Policy:
    """A policy that queries the agent's actor."""
    return lambda obs: agent.act(obs, deterministic=deterministic)


def calibration_actions() -> dict[str, float]:
    """The book's baseline policy levers, the reference a policy must beat."""
    baselines = GrowthCalibration.baseline().baselines()
    return {name: float(baselines[name]) for name in GROWTH_INTERFACE.actions.names()}


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
    episodes = env.episodes.all()[first : first + n_episodes]
    if not episodes:
        raise ValueError(
            f"no futures at [{first}:{first + n_episodes}] of a bank of "
            f"{len(env.episodes)}"
        )
    per_future, lengths, collapses, rolls = [], [], 0, 0
    terms: dict[str, list[float]] = {}
    for i, episode in enumerate(episodes):
        returns = []
        for r in range(repeats):
            obs, _ = env.reset(seed=seed + 10_000 * r + first + i, episode=episode)
            reset_policy(policy)
            total, steps = 0.0, 0
            while True:
                obs, reward, terminated, truncated, info = env.step(policy(obs))
                total += reward
                steps += 1
                for name, value in info.get("reward_terms", {}).items():
                    terms.setdefault(name, []).append(value)
                if terminated or truncated:
                    collapses += terminated
                    break
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
    which is the history's last row for every episode -- every future branches from
    the same state, so the two series line up from step zero rather than starting
    one period in.

    A collapsed episode has no plausible state to record, so its row is ``NaN``
    from the collapse onwards and a column-wise average is over the survivors.
    """
    world = env.driver.world
    action_names = env.interface.actions.names()
    yk = env.interface.state.names().index("Yk")
    nfe, grpr = (env.interface.parameters.names().index(n) for n in ("Nfe", "GRpr"))

    episodes = env.episodes.all()[first : first + n_episodes]
    if not episodes:
        raise ValueError(
            f"no futures at [{first}:{first + n_episodes}] of a bank of "
            f"{len(env.episodes)}"
        )
    horizon = max(len(e) for e in episodes)
    series = {
        name: np.full((len(episodes), horizon), np.nan)
        for name in (*action_names, "growth", "potential", "ER", "PI")
    }
    for i, episode in enumerate(episodes):
        obs, _ = env.reset(seed=seed + first + i, episode=episode)
        reset_policy(policy)
        prev_yk = float(world.history.states[-1][yk])
        prev_nfe = float(world.history.params[-1][nfe])
        for t in range(len(episode)):
            obs, _, terminated, truncated, info = env.step(policy(obs))
            if terminated:
                break
            state, actions = info["state"], info["actions"]
            for name in action_names:
                series[name][i, t] = getattr(actions, name)
            exog = episode.params[t]
            series["growth"][i, t] = np.log(state.Yk / prev_yk) / world.dt
            series["potential"][i, t] = (
                exog[grpr] + np.log(exog[nfe] / prev_nfe) / world.dt
            )
            series["ER"][i, t] = state.ER
            series["PI"][i, t] = state.PI
            prev_yk, prev_nfe = state.Yk, float(exog[nfe])
            if truncated:
                break
    return series


# -- training ---------------------------------------------------------------


def train(config: TrainConfig | None = None, *, verbose: bool = True) -> TrainingResult:
    """Build the world, fit the proxy and train a DSAC agent inside it.

    Sets torch's thread count process-wide before anything else (see
    :attr:`TrainConfig.threads`), which is a global side effect but the right
    scope for it: the setting has to be in place before the first tensor op, and
    a training run owns its process.
    """
    config = config or TrainConfig()
    if config.threads:
        torch.set_num_threads(config.threads)
    rng = np.random.default_rng(config.seed)

    world = build_world(config.world_config(), verbose=verbose)

    if verbose:
        print(
            f"proxy: fitting {config.proxy} ({config.encoder} encoder, "
            f"latent {config.latent}) on {len(world.history)} steps..."
        )
    proxy = build_proxy(
        config.proxy, encoder=config.encoder, latent=config.latent, seed=config.seed
    )
    proxy.fit([world.history])
    observer = Observer(
        GROWTH_INTERFACE,
        encoder=build_obs_encoder(
            config.obs_encoder, latent=config.obs_latent, seed=config.seed
        ),
    ).fit(world.history)
    reward = MandateReward(config.pi_target)
    if verbose:
        print(
            f"observation: {observer.dim} features "
            f"({config.obs_encoder} encoder, "
            f"latent {observer.encoder.latent_dim})"
        )

    env = build_env(proxy, world, observer, reward, config, split="train",
                    seed=config.seed)
    # A shallow copy shares the fitted estimator but keeps its own rollout state,
    # so evaluating mid-training cannot disturb the training environment.
    eval_env = build_env(copy.copy(proxy), world, observer, reward, config,
                         split="eval", seed=config.seed + 1)

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
    buffer = ReplayBuffer(config.capacity, env.obs_dim, env.action_dim)
    result = TrainingResult(
        agent=agent, world=world, proxy=proxy, observer=observer,
        env=env, eval_env=eval_env, config=config,
    )

    explore = random_policy(env.action_dim, rng)
    references: dict[str, Policy] = {
        "random": explore,
        "calibration": constant_policy(eval_env, calibration_actions()),
        "taylor": taylor_policy(eval_env, observer, dt=config.dt,
                                pi_target=config.pi_target),
    }
    result.baselines_ = {
        name: evaluate(eval_env, policy, config.eval_episodes,
                       repeats=config.eval_repeats)
        for name, policy in references.items()
    }
    if verbose:
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
            f"({config.risk_measure().name} risk, device={agent.device})"
        )
        if stored > config.capacity:
            print(
                f"  note: {stored} transitions into a {config.capacity} buffer -- "
                f"the oldest {stored - config.capacity} are evicted before the end"
            )

    obs, _ = env.reset(seed=config.seed)
    episode_return, episode_steps = 0.0, 0
    episode_terms: dict[str, list[float]] = {}
    stats: dict[str, float] = {}

    for step in range(1, config.total_steps + 1):
        action = explore(obs) if step <= config.warmup else agent.act(obs)
        # Every draw of this transition is stored; only the last is followed, so
        # the trajectory is unchanged and the buffer gains ``branch_k`` samples of
        # the kernel at this state rather than one.
        draws = env.branch_step(action, config.branch_k)
        # ``done`` marks a genuine terminal state only: a horizon truncation must
        # still bootstrap, or the agent learns the world ends every episode.
        for draw in draws:
            buffer.add(obs, action, draw.reward, draw.obs, draw.terminated)
        next_obs, reward, terminated, truncated, info = draws[-1].as_step()
        episode_return += reward
        episode_steps += 1
        for name, value in info.get("reward_terms", {}).items():
            episode_terms.setdefault(name, []).append(value)
        obs = next_obs

        if terminated or truncated:
            entry = {
                "step": step,
                "return": episode_return,
                "length": episode_steps,
                "collapsed": float(terminated),
            }
            entry.update(
                {f"term_{k}": float(np.mean(v)) for k, v in episode_terms.items()}
            )
            result.episodes_.append(entry)
            obs, _ = env.reset()
            episode_return, episode_steps, episode_terms = 0.0, 0, {}

        if step > config.warmup and len(buffer) >= config.batch:
            for _ in range(config.updates_per_step):
                stats = agent.update(buffer.sample(config.batch, rng))

        if stats and step % config.log_every == 0:
            result.updates_.append({"step": step, **stats})

        if step % config.eval_every == 0 or step == config.total_steps:
            scores = evaluate(
                eval_env, agent_policy(agent), config.eval_episodes,
                repeats=config.eval_repeats,
            )
            result.evals_.append({"step": step, **scores})
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
            # The evaluation ran on a copy, but the training environment's own
            # episode is mid-flight; leave it running.

    # The headline, paid for once: every future rolled ``final_repeats`` times,
    # so the spread across futures and the spread the proxy's own noise adds are
    # reported apart rather than summed into one misleading number.
    if config.final_repeats > 1:
        if verbose:
            print(
                f"final scoring: {config.eval_episodes} futures x "
                f"{config.final_repeats} rolls..."
            )
        result.final_ = {
            name: evaluate(eval_env, policy, config.eval_episodes,
                           repeats=config.final_repeats)
            for name, policy in {"agent": agent_policy(agent), **references}.items()
        }
        if verbose:
            for name, scores in result.final_.items():
                print(
                    f"  {name:>12}: {scores['return']:9.1f} "
                    f"+- {scores['return_std']:6.1f} across futures, "
                    f"+- {scores['noise_std']:6.1f} from proxy noise "
                    f"| collapse {scores['collapse_rate']:.0%}"
                )

    return result


# -- CLI ---------------------------------------------------------------------


def _parse_args() -> TrainConfig:
    """Build a :class:`TrainConfig` from the command line."""
    defaults = TrainConfig()
    ap = argparse.ArgumentParser(
        description="Train a DSAC central bank inside a fitted proxy economy."
    )
    for f in fields(defaults):
        flag = "--" + f.name.replace("_", "-")
        current = getattr(defaults, f.name)
        kind = str if current is None else type(current)
        ap.add_argument(flag, type=kind, default=current, help=f.metadata.get("help", ""))
    args = ap.parse_args()
    return replace(defaults, **{f.name: getattr(args, f.name) for f in fields(defaults)})


def main() -> None:
    """Train an agent from the command line and report the final scores.

    A training run is hours long and reports as it goes, but Python block-buffers
    stdout whenever it is not a terminal -- so redirected to a log file (which is
    how a run this long is usually started) the progress lines sit in the buffer
    and the file stays empty until the process exits. Line buffering costs
    nothing at one line per few thousand steps and makes ``tail -f`` work.
    """
    sys.stdout.reconfigure(line_buffering=True)
    config = _parse_args()
    result = train(config)
    scores = result.final_ or {
        "agent": result.evals_[-1], **result.baselines_
    }
    print(
        f"final: agent {scores['agent']['return']:.1f} "
        f"(taylor {scores['taylor']['return']:.1f}, "
        f"calibration {scores['calibration']['return']:.1f}, "
        f"random {scores['random']['return']:.1f})"
    )
