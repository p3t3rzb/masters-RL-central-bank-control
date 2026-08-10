"""The live loop: one unrepeatable run through the ground truth, learned from.

The agent arrives having been trained entirely inside a proxy, and is placed in
the structural economy for **one** pass. No reset, no second episode, no bank of
futures, and roughly two hundred transitions against the hundred thousand it was
trained on. Online reinforcement learning from two hundred samples is not a
thing, so the pass is not spent on the policy. It is spent on the *model error*:

1. every real step is recorded in :class:`~control.live.buffer.RealBuffer`,
   strictly apart from the synthetic one;
2. the same step's residual -- what the proxy predicted against what the economy
   did -- updates the correction of :mod:`control.live.residual`, which starts
   already anchored on the historic run;
3. every few steps, short on-policy rollouts are branched off recently visited
   *real* states through the **corrected** model, and a small, guarded batch of
   DSAC updates is taken on minibatches mixed from the two buffers.

That is MBPO (Janner et al., 2019) with the two modifications this setting
forces: the world model is a pre-trained proxy plus an online residual rather
than a model learned from scratch, and the whole thing runs under the discipline
of a policy that cannot be rolled back.

**The shadow proxy** is what makes step 3 possible. A proxy rolled open-loop
drifts away from the economy within a few periods, so its rollout state stops
describing where the real economy is. The deployment therefore keeps one
teacher-forced against the realised run
(:meth:`~economic_models.proxy.base.BaseProxyModel.absorb`) and stores a snapshot
of it with every transition: a branch point is then a position the truth actually
occupied, not a position the model wandered to.

**The guardrails are load-bearing.** An update that is merely usually good is not
good enough when there is no reset: the actor is anchored to the offline policy
and stepped a tenth as fast, every candidate is scored against the incumbent
under CVaR before it is allowed to act, the out-of-distribution monitor freezes
learning when the economy leaves the region the apparatus was calibrated in, and
a run that goes badly enough hands control back to the frozen offline agent.
Reverting to the policy we already had is a successful outcome for a safe system,
and is reported as one. Each is separately switchable (:meth:`LiveConfig.guarding`)
so the ablation can say which one paid for itself.

The levers' **slew limit is not on that list**, and deliberately. It used to be:
a clip applied to the agent's action inside this loop and nowhere else. But a
constraint that exists only at deployment is one the agent never trained against,
never was evaluated under, and does not appear in the rollouts the acceptance
test scores -- so the policy being optimised, the policy being tested and the
policy being run were three different controllers, and two policies saturating
the limit in the same direction emitted identical actions no matter how much
their parameters differed. It now lives in
:attr:`~control.env.EnvConfig.action_rate`, as a property of the instrument
rather than a restraint on one agent, and training, evaluation, the references
and the synthetic rollouts all read it from there.

Everything here is testable before the live run by *rehearsing* it: the ground
truth is unavailable to the bank but available in the lab, so :func:`rehearse`
replays the entire procedure on held-out futures and returns a distribution over
one-shot deployments, which is the right object to set hyperparameters against.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from economic_models.proxy import BaseProxyModel
from economic_models.run import Run
from economic_models.variables import Actions, Parameters, State

from control.dsac.agent import DSACAgent
from control.dsac.replay import Batch, ReplayBuffer
from control.dsac.train import (
    TrainConfig,
    TrainingResult,
    agent_policy,
    build_proxy,
    build_truth_env,
    calibration_actions,
    constant_policy,
    reset_policy,
    taylor_policy,
)
from control.env import CentralBankEnv
from control.live.buffer import BranchPoint, RealBuffer, seed_from_run
from control.live.corrected import CorrectedProxy
from control.live.forcing import (
    BlockBootstrapForcing,
    ForcingModel,
    OracleForcing,
    VARForcing,
)
from control.live.monitor import OODMonitor
from control.live.residual import (
    BlockBootstrapResidualNoise,
    NullResidual,
    Residual,
    ResidualModel,
    cross_fitted_residuals,
    design,
    in_sample_residuals,
)
from control.rewards import RewardContext
from control.world import Episode, NullStabilizer, TrainingWorld

#: The forcing families a deployment can forecast its exogenous environment with.
FORCINGS = ("var", "bootstrap", "oracle")


@dataclass(frozen=True)
class LiveConfig:
    """Everything about the deployment that is not the offline run.

    Every one of these has to be frozen **before** the live run starts. Tuning on
    it is not merely bad practice here, it is impossible: there is one run and no
    held-out anything to tune against. :func:`rehearse` is where they are chosen,
    on futures disjoint from the ones the result is reported on.
    """

    # -- the run --
    live_steps: int = 200  #: 50 years at dt=0.25
    warmup: int = 20  #: live steps before the first policy update
    update_every: int = 4  #: real steps between policy updates (a year)
    gradient_steps: int = 100  #: DSAC updates per policy update
    branch_points: int = 400  #: rollouts started per policy update
    rollout_length: int = 3  #: nominal k
    rollout_min: int = 1  #: k when the correction is uncertain
    rollout_max: int = 5  #: k when it has been predicting well
    generations: int = 5  #: policy updates of synthetic data retained
    real_fraction: float = 0.2  #: rho, the real share of a minibatch
    batch: int = 256  #: minibatch size
    recency_decay: float = 0.99  #: geometric weight on branch-point age
    #: what a *synthetic* collapse is charged against. The environment prices a
    #: collapse per remaining step of an episode, and a live run has no episode
    #: length; charging synthetic rollouts against the training horizon keeps the
    #: penalty the critic was trained on and the one it now sees on one scale.
    synthetic_horizon: int = 50
    #: re-apply GROWTH's fiscal response inside *synthetic* rollouts. Off by
    #: default, and the default is the claim: the response is a rule on the
    #: previous period's employment rate, so switching it on assumes the bank
    #: knows the fiscal reaction function -- which it plausibly does (automatic
    #: stabilizers are legislation, not a hidden structural parameter), but which
    #: is an assumption, and the deployment reads better without one it can do
    #: without. Off, ``GRg`` is forecast as an opaque series and nothing re-adds
    #: the response, so rollouts under-state how much a slump repairs itself --
    #: conservative, which is the safe direction for a CVaR actor. On, the
    #: forecast strips the historic response on the way in
    #: (:meth:`~control.world.FiscalStabilizer.strip`) so it is not counted
    #: twice. The *environment* keeps its stabilizer either way: that one is the
    #: economy, not a belief about it.
    rollout_stabilizer: bool = False

    # -- the correction --
    cross_fit: bool = True  #: cross-fit the historic residuals (see §4.4)
    folds: int = 5  #: rolling-origin folds when cross-fitting
    history_weight: float = 0.5  #: trust in a historic row against a live one
    tau: float = 1.0  #: prior scale on the correction's coefficients
    forgetting: float = 0.999  #: RLS forgetting factor
    kappa: float = 0.2  #: bias-state EWMA rate (~1 year at dt=0.25)
    bootstrap_noise: bool = False  #: block-bootstrap the spread instead of Gaussian
    correct: bool = True  #: at False the model is the raw proxy (the ablation)

    # -- the exogenous forecast --
    forcing: str = "var"  #: one of :data:`FORCINGS`

    # -- the guardrails --
    #
    # Separable things, and the §8.2 ablation is more informative when they can be
    # removed one at a time: "the guardrails cost 12 points" says much less than
    # which of them did. :attr:`guardrails` is the master switch -- off, all of
    # them are off whatever they individually say -- and each may also be turned
    # off on its own. :meth:`guarding` is how the loop asks.
    guardrails: bool = True  #: master switch; off disables every guard below
    guard_actor_lr: bool = True  #: take the online step at :attr:`actor_lr`
    guard_anchor: bool = True  #: pull the actor toward the offline policy
    guard_accept: bool = True  #: score a candidate before letting it act
    guard_monitor: bool = True  #: let the monitor freeze learning when it trips
    guard_fallback: bool = True  #: hand control back after enough trouble
    #: run the live environment under the instrument's slew limit. Unlike the
    #: others this is not a restraint on *learning* -- it is a property of the
    #: levers, shared with training, evaluation and the references -- so turning
    #: it off is a deliberate train/deploy mismatch: a policy optimised under a
    #: rate-limited instrument, run without one. Kept switchable because that
    #: mismatch is exactly what the ablation is asking about.
    guard_rate_limit: bool = True
    #: run the Taylor rule and the calibration baseline under the slew limit too.
    #: On by default because the limit is a property of the *instrument*: a
    #: reference exempt from it operates levers nobody has, and the deployment
    #: would be paying a cost its own baseline does not. It is not a free choice
    #: either way -- limiting Taylor materially **helps** it, by damping the
    #: high-gain oscillation :func:`~control.dsac.train.taylor_policy` documents
    #: at sub-quarterly ``dt``, so this makes the bar harder rather than easier.
    #: Off recovers the textbook rules unmodified, at the cost of comparing two
    #: policies operating different instruments.
    limit_references: bool = True
    #: the instrument's slew limit *for the live run*, as a fraction of the box
    #: per year. ``None`` inherits whatever the offline run was trained under
    #: (:attr:`~control.dsac.train.TrainConfig.action_rate`), which is the
    #: coherent setting: a limit the agent trained against is one it learned to
    #: work with, and the acceptance test then scores the controller that will
    #: actually act. A number here deliberately breaks that tie -- deploying
    #: under a tighter instrument than the one trained for -- and is an
    #: assumption to report rather than a default to rely on.
    action_rate: float | None = None
    actor_lr: float = 3e-5  #: ten times below the training-time rate
    anchor_weight: float = 1.0  #: strength of the KL pull toward the deployed policy
    accept_branches: int = 128  #: branch points the acceptance test scores over
    #: the functional the acceptance test compares the two policies under, one of
    #: ``"mean"`` or ``"cvar"``. ``None`` inherits the actor's own
    #: (:attr:`~control.dsac.train.TrainConfig.risk`), which is the setting that
    #: makes the test coherent: scoring a candidate on the lower tail while the
    #: actor was optimising the mean asks it to have improved something it never
    #: tried to, and under ``--risk mean`` that mismatch rejects sound updates
    #: indefinitely. Setting it explicitly deploys under a stricter measure than
    #: was trained for, which is an assumption to report rather than a default.
    accept_risk: str | None = None
    #: how much worse a candidate may score and still be accepted, in standard
    #: errors of the test's own estimate. Not a fudge: the two errors are not
    #: symmetric. A wrongly *rejected* candidate forfeits the only learning a
    #: live run can do and, run against :attr:`reject_patience`, eventually hands
    #: control back altogether; a wrongly *accepted* one is bounded by the anchor
    #: to the offline policy, moves at :attr:`actor_lr`, and faces the same test
    #: again in :attr:`update_every` periods. At zero the test demands a strict
    #: improvement on a noisy estimate, which is a bias toward never learning.
    accept_tolerance: float = 1.0
    #: steps per acceptance-test rollout. Kept at :attr:`rollout_length` by
    #: default rather than run longer: the corrected model is trusted for exactly
    #: as far as the data-generating rollouts trust it, and a test conducted past
    #: that horizon is scoring the correction's divergence rather than the policy.
    accept_steps: int | None = None
    accept_alpha: float = 0.1  #: tail fraction the test compares under
    #: consecutive rejections before handing control back. A rejection is a
    #: *no-op* -- the incumbent keeps acting and nothing was risked -- so this is
    #: a much weaker signal of trouble than the monitor tripping, and treating a
    #: short run of them as terminal is what froze most of the runs this
    #: apparatus was built for. Read against :attr:`update_every`: at 2 and 8
    #: this is sixteen live periods of a candidate never beating the incumbent.
    reject_patience: int = 8
    #: readings the monitor takes before it will trip. Derived from
    #: :attr:`warmup` rather than fixed, because the invariant is that it sits
    #: *inside* the policy's own warmup -- a monitor that can trip before it has
    #: a reference to be unusual against will trip on the opening transient, and
    #: the sticky window then clears :attr:`reject_patience` before a single
    #: update has been taken. ``None`` keeps that tie; a number breaks it
    #: deliberately.
    monitor_warmup: int | None = None
    monitor_variance_factor: float = 4.0  #: variance spike, in units of its own recent level
    monitor_residual_z: float = 4.0  #: residual spike, in units of its own recent scale
    monitor_clip_fraction: float = 0.25  #: observation features at the standardisation clip

    seed: int = 0  #: seeds the rollouts, the sampling and the acceptance test

    def __post_init__(self) -> None:
        """Reject a configuration that cannot be run, before anything is built."""
        if self.forcing not in FORCINGS:
            raise ValueError(f"forcing must be one of {FORCINGS}, got {self.forcing!r}")
        if self.accept_risk not in (None, "mean", "cvar"):
            raise ValueError(
                f"accept_risk must be 'mean', 'cvar' or None, got {self.accept_risk!r}"
            )
        if not 0.0 <= self.real_fraction <= 1.0:
            raise ValueError(f"real_fraction must be in [0, 1], got {self.real_fraction}")
        if not self.rollout_min <= self.rollout_length <= self.rollout_max:
            raise ValueError(
                f"rollout_length {self.rollout_length} is outside "
                f"[{self.rollout_min}, {self.rollout_max}]"
            )
        if self.monitor_warmup is not None and self.monitor_warmup > self.warmup:
            raise ValueError(
                f"monitor_warmup {self.monitor_warmup} must sit inside the policy "
                f"warmup {self.warmup}; see LiveConfig.monitor_warmup"
            )

    def guarding(self, which: str) -> bool:
        """Whether guardrail ``which`` is active: the master switch **and** its own.

        ``which`` is the suffix of a ``guard_*`` field (``"accept"``,
        ``"anchor"``, ...). Unknown names raise rather than quietly reading as
        off, since a typo would silently remove a guardrail from a run whose
        whole claim is that it has them.
        """
        try:
            own = getattr(self, f"guard_{which}")
        except AttributeError:
            raise ValueError(f"no guardrail called {which!r}") from None
        return self.guardrails and bool(own)

    @property
    def accept_horizon(self) -> int:
        """Steps per acceptance-test rollout, defaulting to :attr:`rollout_length`."""
        return self.rollout_length if self.accept_steps is None else self.accept_steps

    def build_monitor(self) -> OODMonitor:
        """The out-of-distribution monitor these settings describe."""
        return OODMonitor(
            variance_factor=self.monitor_variance_factor,
            residual_z=self.monitor_residual_z,
            clip_fraction=self.monitor_clip_fraction,
            warmup=(
                max(1, (4 * self.warmup) // 5)
                if self.monitor_warmup is None
                else self.monitor_warmup
            ),
        )


@dataclass
class RunRecord:
    """One policy's pass through one live future, period by period.

    ``steps_`` is the log every figure is drawn from: one row per period, with
    the mandate and its legs, the levers, the three observables, and -- for a run
    that adapted -- the model diagnostics. ``events_`` marks the discrete things
    that happened to the policy (updates accepted, updates rejected, the monitor
    tripping, control handed back).
    """

    name: str
    episode: int
    steps_: list[dict[str, float]] = field(default_factory=list)
    events_: list[dict[str, Any]] = field(default_factory=list)
    collapsed: bool = False
    fallback_at: int | None = None

    def __len__(self) -> int:
        """Periods survived."""
        return len(self.steps_)

    @property
    def total(self) -> float:
        """Cumulative mandate penalty over the run, collapse included."""
        return float(sum(s["reward"] for s in self.steps_))

    def series(self, key: str) -> np.ndarray:
        """One logged quantity as an array, ``NaN`` where the row lacks it."""
        return np.array([s.get(key, np.nan) for s in self.steps_], dtype=float)


@dataclass
class RehearsalResult:
    """A whole rehearsal: the deployment and its references, future by future.

    ``runs_`` is keyed by policy name (``"adapting"``, ``"frozen"``, ``"taylor"``,
    ``"calibration"``), each a list of :class:`RunRecord` aligned by position --
    entry ``j`` of every list is the *same* future under the *same* seed, which is
    what makes every comparison in :mod:`scripts.deploy_agent` paired.
    """

    runs_: dict[str, list[RunRecord]] = field(default_factory=dict)
    config: LiveConfig = field(default_factory=LiveConfig)

    def totals(self, name: str) -> np.ndarray:
        """The per-future cumulative penalty of one policy."""
        return np.array([r.total for r in self.runs_[name]], dtype=float)

    def improvement(self, over: str = "frozen", of: str = "adapting") -> np.ndarray:
        """The paired difference in cumulative penalty, future by future."""
        return self.totals(of) - self.totals(over)

    def collapse_rate(self, name: str) -> float:
        """The fraction of rehearsed deployments of ``name`` that collapsed."""
        runs = self.runs_[name]
        return float(np.mean([r.collapsed for r in runs])) if runs else 0.0


# -- the world model, corrected ---------------------------------------------


def residual_seed(
    result: TrainingResult, live: LiveConfig, *, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """The historic residual rows every deployment starts from.

    Computed once and reused across rehearsals: it is a function of the proxy
    family and the history alone, and cross-fitting it means fitting the proxy
    ``folds`` times, which is worth paying once rather than per future.
    """
    config = result.config
    if not live.cross_fit:
        if verbose:
            print("residual: in-sample historic residuals (the §8.2 ablation)")
        return in_sample_residuals(result.proxy, result.world.history)

    if verbose:
        print(
            f"residual: cross-fitting {live.folds} rolling-origin folds of "
            f"{config.proxy} over {len(result.world.history)} historic steps..."
        )
    return cross_fitted_residuals(
        # A copy of the *deployed* encoder, not a fresh one of the same family:
        # the fold proxies must agree with each other and with the live shadow on
        # what a latent means, or the pooled design has no consistent columns.
        lambda: build_proxy(
            config.proxy,
            encoder=result.proxy.encoder,
            latent=config.latent,
            seed=config.seed,
        ),
        result.world.history,
        folds=live.folds,
        verbose=verbose,
    )


def build_residual(
    seed_rows: tuple[np.ndarray, np.ndarray], live: LiveConfig, n_outputs: int
) -> Residual:
    """A correction primed on the historic rows, or the null one."""
    if not live.correct:
        return NullResidual(n_outputs)
    noise = (
        BlockBootstrapResidualNoise(n_outputs)
        if live.bootstrap_noise
        else None
    )
    model = ResidualModel(
        n_outputs,
        tau=live.tau,
        forgetting=live.forgetting,
        kappa=live.kappa,
        noise=noise,
    )
    X, E = seed_rows
    return model.seed(X, E, weight=live.history_weight)


def build_forcing(
    live: LiveConfig, env: CentralBankEnv, history: Run, episode: Episode
) -> ForcingModel:
    """The exogenous forecast a synthetic rollout will be driven by.

    The stabilizer reaches the forecast only when the rollouts are going to
    re-apply it, and for the one purpose of taking it back out of the observed
    rows first -- the two halves of that decision have to travel together or the
    fiscal response is counted twice or not at all.
    """
    stabilizer = env.driver.world.stabilizer if live.rollout_stabilizer else None
    if live.forcing == "oracle":
        model: ForcingModel = OracleForcing(
            env.interface, episode.params, stabilizer=stabilizer
        )
    elif live.forcing == "bootstrap":
        model = BlockBootstrapForcing(env.interface, stabilizer=stabilizer)
    else:
        model = VARForcing(env.interface, stabilizer=stabilizer)
    return model.fit(history)


# -- short rollouts through the corrected model ------------------------------


class SyntheticRollouts:
    """Short on-policy rollouts from real branch points, through a world model.

    MBPO's core move, and it earns its place for a specific reason: a one-step
    residual model is a one-step object, its error compounds over a rollout, and
    after enough steps the corrected model's trajectory says nothing about the
    truth. But the *policy* only needs short-horizon on-policy data -- the critic
    supplies the long horizon by bootstrapping -- so the rollouts can be kept
    inside the range the correction is good for.

    Each rollout reproduces exactly what a real step does, in the same
    coordinates, or the two buffers would not be mixable: the same observer, the
    same mandate, the same fiscal stabilizer applied against the *simulated*
    employment rate, and the same plausibility corridor. A rollout that leaves
    that corridor is truncated there and the collapse transition **is** stored,
    with its terminal flag and its penalty. It is the only way the agent learns
    where the cliff is without walking off it for real, and it is the single most
    valuable kind of synthetic transition the corrected model can produce.
    """

    def __init__(
        self,
        model: BaseProxyModel,
        env: CentralBankEnv,
        forcing: ForcingModel,
        live: LiveConfig,
    ) -> None:
        """Bind to the corrected ``model`` and the ``env`` whose rules it copies."""
        self.model = model
        self.env = env
        self.forcing = forcing
        self.live = live
        # Taken from the config rather than from the environment: the
        # environment's stabilizer is the economy's, and this one is the bank's
        # belief about it, which is a different object even when it is the same
        # number (see LiveConfig.rollout_stabilizer).
        self.stabilizer = (
            env.driver.world.stabilizer if live.rollout_stabilizer else NullStabilizer()
        )
        self.dt = env.driver.world.dt
        self._er = env.interface.state.names().index("ER")
        self._n_params = len(env.interface.parameters.names())

    def roll(
        self,
        branch: BranchPoint,
        policy: Callable[[np.ndarray], np.ndarray],
        k: int,
        rng: np.random.Generator,
    ) -> tuple[float, list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]]:
        """Roll ``k`` steps from ``branch``; return the total and the transitions."""
        interface = self.env.interface
        self.model.restore(branch.handle)
        belief = branch.belief
        states, exog = branch.states.copy(), branch.exog.copy()
        obs = branch.obs
        # Cumulated onto the branch point's own parameter row. A path re-based
        # anywhere else makes this rollout's first exogenous feature the drift
        # between the two positions rather than a one-period change.
        path = self.forcing.sample(k, rng, previous=exog[-1][: self._n_params])

        rows: list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        total = 0.0
        for j in range(k):
            # Through the same instrument the real step goes through, measured
            # from the same trailing row. Without this the rollouts imagine an
            # unconstrained controller and the acceptance test scores a policy
            # that cannot be run -- the two buffers would not be mixable, which
            # is the one thing this class exists to guarantee.
            action = self.env.apply_action_rate(policy(obs), exog[-1])
            params = self.stabilizer.apply(path[j], float(states[-1][self._er]))
            parameters = interface.parameters.from_row(params)
            actions = self.env.to_actions(action)

            penalty = self.env.config.collapse_penalty * max(
                1, self.live.synthetic_horizon - j
            )
            try:
                state = self.model.step(parameters, actions, rng=rng)
                if not self.env.plausible(state):
                    raise ValueError("state left the plausibility corridor")
                levels = self.env.levels_of(state)
                nxt = np.vstack([states[-1], levels])
                exog_now = np.hstack(
                    [params, [getattr(actions, n) for n in interface.actions.names()]]
                )
                nxt_exog = np.vstack([exog[-1], exog_now])
                next_obs, belief = self.env.observer.observe(nxt, nxt_exog, belief)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                rows.append((obs, action, penalty, obs, True))
                total += penalty
                break

            r = self.env.reward(
                RewardContext(
                    state=state,
                    prev_state=interface.state.from_row(states[-1]),
                    parameters=parameters,
                    prev_parameters=interface.parameters.from_row(
                        exog[-1][: len(interface.parameters.names())]
                    ),
                    actions=actions,
                    prev_actions=interface.actions.from_row(
                        exog[-1][len(interface.parameters.names()) :]
                    ),
                    dt=self.dt,
                )
            )
            rows.append((obs, action, r, next_obs, False))
            total += r
            states, exog, obs = nxt, nxt_exog, next_obs
        return total, rows


# -- one live deployment -----------------------------------------------------


def deploy(
    result: TrainingResult,
    episode: Episode,
    live: LiveConfig | None = None,
    *,
    seed_rows: tuple[np.ndarray, np.ndarray] | None = None,
    seed: int = 0,
    adapt: bool = True,
    verbose: bool = False,
) -> RunRecord:
    """Put the offline agent in the ground truth for one run, and adapt it there.

    ``result`` supplies the whole offline stack -- the world, the fitted proxy,
    the frozen observation and the trained agent -- and ``episode`` is the single
    future the economy will actually take. ``seed_rows`` are the historic
    residuals from :func:`residual_seed`, passed in rather than recomputed
    because they do not depend on the future. ``adapt=False`` runs the identical
    loop with the policy update switched off, which is the paired frozen-agent
    reference: same environment, same seeds, same diagnostics, one difference.

    Returns the period-by-period record; the policy itself is deliberately not
    returned, because a deployment that ends is a deployment whose policy has no
    remaining use.
    """
    live = live or LiveConfig()
    config = result.config
    world, observer = result.world, result.observer
    rng = np.random.default_rng(live.seed + seed)

    # The slew limit is the environment's, not a filter on the agent's output, so
    # the deployment, its synthetic rollouts and the references all act under the
    # same instrument -- and under the one the offline run was trained against
    # unless this configuration deliberately says otherwise.
    env = build_truth_env(
        world, observer, result.env.reward, config, seed=seed,
        horizon=live.live_steps, action_rate=_action_rate(live, config),
    )
    n_feat = len(result.proxy.transform.state_feature_names)

    # The policy being deployed, the policy it is anchored to, and the policy it
    # falls back on are three references to two objects: the clone adapts, the
    # original never moves and is both the anchor and the parachute.
    #
    # The anchor is deliberately the *offline* policy rather than the
    # most-recently-accepted one. Anchoring to the incumbent bounds each update
    # but not their sum, so a run of small accepted steps can walk the policy
    # arbitrarily far from a thing that took 10^5 training steps to build, on the
    # evidence of a few hundred live samples. Anchoring to where it started bounds
    # the total drift, which is the quantity that actually matters when there is
    # no reset. (The acceptance test below *is* against the incumbent -- that one
    # is asking "is this step an improvement", which is a different question.)
    frozen = result.agent
    agent = frozen.clone()
    if live.guarding("actor_lr"):
        agent.set_lr(actor=live.actor_lr)
    anchor_weight = live.anchor_weight if live.guarding("anchor") else 0.0

    # The shadow proxy and the corrected model are shallow copies: they share the
    # expensive fitted estimator and the fitted encoder, and keep their own
    # rollout state, so filtering the live run and imagining a future off it
    # cannot disturb each other.
    shadow: BaseProxyModel = copy.copy(result.proxy)
    residual = build_residual(
        seed_rows if seed_rows is not None else residual_seed(result, live, verbose=verbose),
        live,
        n_feat,
    )
    model = CorrectedProxy(copy.copy(result.proxy), residual)

    real = RealBuffer(
        len(world.history) + live.live_steps + 8, env.obs_dim, env.action_dim
    )
    seed_from_run(
        real, world.history, env, observer, env.reward, shadow,
        interface=env.interface, dt=world.dt, weight=live.history_weight,
    )
    # Retention is by capacity on a ring buffer, so ``generations`` is exact only
    # if every generation contributes the same number of rows -- and it does not,
    # because k is chosen per update (see :func:`_rollout_length`). Sized on the
    # nominal k: a stretch of cautious short rollouts retains a few generations
    # more, a stretch of confident long ones a few less. Sizing on rollout_max
    # instead would make the *nominal* case retain 5/3 of what it claims to.
    synthetic = ReplayBuffer(
        max(1, live.branch_points * live.rollout_length * live.generations),
        env.obs_dim,
        env.action_dim,
    )
    forcing = build_forcing(live, env, world.history, episode)
    rollouts = SyntheticRollouts(model, env, forcing, live)
    monitor = live.build_monitor()
    er = env.interface.state.names().index("ER")

    record = RunRecord(name="adapting" if adapt else "frozen", episode=episode.index)
    obs, _ = env.reset(seed=seed, episode=episode)
    tracker = _OutcomeTracker(env, world)
    rejections = 0

    steps = min(live.live_steps, len(episode))
    for t in range(1, steps + 1):
        requested = agent.act(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(requested)
        # What the instrument actually did, which is what the buffer must record.
        action = info.get("action", requested)

        if terminated:
            # Stored, not just recorded. It is the one real transition that shows
            # the critic where the cliff is, and the run produces at most one of
            # them ever. No branch point: there is no plausible state to continue
            # from, which is what collapsing means.
            real.add(obs, action, reward, next_obs, True, branch=None, weight=1.0)
            record.steps_.append({"step": t, "reward": reward, "collapsed": 1.0})
            record.collapsed = True
            if verbose:
                print(f"  [{t:>3}] COLLAPSED: {info.get('collapse')}")
            break

        # -- what did the proxy think would happen? --------------------------
        state, params, actions = info["state"], info["parameters"], info["actions"]
        ctx = shadow.context(params, actions)
        x = design(ctx.z, ctx.u_next, ctx.f_prev)
        raw_mean = shadow.predict_mean(ctx)
        # ``env.states`` is exactly (the levels this step came from, the levels it
        # produced), which is the pair the feature row is a difference of.
        eps = shadow.transform.transform_states(env.states)[0] - raw_mean
        # Measured *before* the correction sees this row, so both numbers are
        # genuine one-step-ahead errors rather than fits.
        corrected_error = eps - residual.correction(x)
        variance = residual.variance(x)
        residual.update(x, eps)

        # -- move the shadow onto the period that actually happened ----------
        shadow.absorb(state, params, actions)
        # ``env.states[0]`` is the row this step started from, whose employment
        # rate is the one the stabilizer responded to when it built ``params``.
        forcing.absorb(
            np.array(list(params.to_dict().values()), dtype=float),
            er_prev=float(env.states[0][er]),
        )
        if isinstance(forcing, OracleForcing):
            forcing.advance()

        reading = monitor.update(t, variance, eps, next_obs, clip=observer.clip)
        real.add(
            obs, action, reward, next_obs, False,
            branch=BranchPoint(
                obs=next_obs,
                belief=env.belief,
                handle=shadow.snapshot(),
                states=env.states.copy(),
                exog=env.exog.copy(),
            ),
            weight=1.0,
        )
        record.steps_.append(
            {
                "step": t,
                "reward": reward,
                **{f"term_{k}": v for k, v in info["reward_terms"].items()},
                **tracker.observe(state, params, actions),
                "error_raw": float(np.linalg.norm(eps)),
                "error_corrected": float(np.linalg.norm(corrected_error)),
                "residual_var": reading.variance,
                "bias": float(np.linalg.norm(residual.bias)),
                "clipped": reading.clipped,
                "tripped": float(reading.tripped),
                "collapsed": 0.0,
            }
        )

        # -- every N steps: regenerate the synthetic data and update ----------
        # The monitor freezes learning only when the guardrails are on; with them
        # off it still reads, and its readings still reach the record, but it is
        # not allowed to stop anything -- otherwise the §8.2 ablation would be
        # "no guardrails except this one".
        frozen_by_monitor = live.guarding("monitor") and monitor.tripped()
        due = adapt and t > live.warmup and t % live.update_every == 0
        if due and record.fallback_at is None and not frozen_by_monitor:
            # The forcing model has been absorbing rows every step but is only
            # re-estimated here: its *state* has to be current for a rollout to
            # start from the right place, its *coefficients* do not, and this is
            # the clock a deployment can afford to pay estimation on.
            forcing.refit()
            k = _rollout_length(live, monitor)
            _generate(rollouts, real, synthetic, agent, live, k, rng)
            candidate = agent.clone()
            for _ in range(live.gradient_steps):
                candidate.update(
                    _mix(real, synthetic, live, rng),
                    anchor=frozen,
                    anchor_weight=anchor_weight,
                )
            test = (
                _accept(
                    candidate, agent, rollouts, real, live,
                    risk=live.accept_risk or config.risk,
                    rng_seed=live.seed + t,
                )
                if live.guarding("accept")
                else AcceptTest(margin=0.0, error=0.0, accepted=True)
            )
            accepted = test.accepted
            record.events_.append(
                {
                    "step": t,
                    "kind": "accepted" if accepted else "rejected",
                    "k": k,
                    "margin": test.margin,
                    "error": test.error,
                }
            )
            if accepted:
                agent, rejections = candidate, 0
            else:
                rejections += 1
        elif due and record.fallback_at is None:
            record.events_.append({"step": t, "kind": "frozen-by-monitor"})

        if live.guarding("fallback") and record.fallback_at is None and (
            monitor.failed() or rejections >= live.reject_patience
        ):
            agent = frozen
            record.fallback_at = t
            record.events_.append({"step": t, "kind": "fallback"})
            if verbose:
                print(f"  [{t:>3}] control handed back to the frozen offline agent")

        obs = next_obs
        if truncated:
            break

    if verbose:
        print(
            f"  {record.name}: return {record.total:9.1f} over {len(record)} steps"
            + ("  COLLAPSED" if record.collapsed else "")
        )
    return record


def run_reference(
    result: TrainingResult,
    episode: Episode,
    policy_name: str,
    live: LiveConfig,
    *,
    seed: int = 0,
) -> RunRecord:
    """A reference policy through the same live future, logged the same way.

    The Taylor rule and the calibration baseline need none of the machinery
    above, but they do need the same environment, the same seed and the same
    per-period record, or the figures would be comparing a deployment against a
    differently-measured thing.

    That now includes the instrument's slew limit, which it did not while the
    limit was a filter on the agent's output. A rate limit that lives in the
    environment is a fact about the levers rather than a constraint on one
    policy, so a reference exempt from it would be a reference operating an
    instrument nobody has -- and the deployment would be paying a cost its
    baseline does not. :attr:`LiveConfig.limit_references` switches that off for
    anyone who wants the textbook rules unmodified instead.

    Note that a constant baseline is only *nominally* unaffected: it holds its
    levers still, but it does not start at them. The economy hands over wherever
    the history left it, so a limited calibration run spends its first few
    periods travelling to its own setting.
    """
    env = build_truth_env(
        result.world, result.observer, result.env.reward, result.config, seed=seed,
        horizon=live.live_steps,
        action_rate=(
            _action_rate(live, result.config) if live.limit_references else None
        ),
    )
    policies = {
        "taylor": lambda: taylor_policy(
            env, result.observer, dt=result.config.dt,
            pi_target=result.config.pi_target,
        ),
        "calibration": lambda: constant_policy(env, calibration_actions()),
    }
    if policy_name not in policies:
        raise ValueError(f"unknown reference policy {policy_name!r}")
    policy = policies[policy_name]()

    record = RunRecord(name=policy_name, episode=episode.index)
    obs, _ = env.reset(seed=seed, episode=episode)
    reset_policy(policy)
    tracker = _OutcomeTracker(env, result.world)
    for t in range(1, min(live.live_steps, len(episode)) + 1):
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        if terminated:
            record.steps_.append({"step": t, "reward": reward, "collapsed": 1.0})
            record.collapsed = True
            break
        record.steps_.append(
            {
                "step": t,
                "reward": reward,
                **{f"term_{k}": v for k, v in info["reward_terms"].items()},
                **tracker.observe(info["state"], info["parameters"], info["actions"]),
                "collapsed": 0.0,
            }
        )
        if truncated:
            break
    return record


# -- the rehearsal protocol --------------------------------------------------


def rehearse(
    result: TrainingResult,
    live: LiveConfig | None = None,
    *,
    futures: int = 8,
    first: int | None = None,
    references: tuple[str, ...] = ("taylor", "calibration"),
    verbose: bool = True,
) -> RehearsalResult:
    """Replay the whole deployment on held-out futures: a distribution, not a run.

    There is exactly one live run, so every hyperparameter has to be frozen
    before it starts and there is nothing to tune against. The escape is that the
    ground truth is unavailable to the *bank* but available in the *lab*: take the
    held-out futures training never touched, treat each as one complete
    deployment, and run the entire online algorithm against it. The live run is
    then a single draw from the distribution this returns, which is the right
    object to choose a configuration against -- on the median and the tenth
    percentile and the collapse rate, never on the mean, because a configuration
    with a better mean and a worse tail is worse for a policy that gets one
    attempt.

    Which bank is rehearsed on depends on whether the world drew a **deployment**
    one (:attr:`~control.world.WorldConfig.deploy_excitation`). If it did, that
    bank is used from its start: it is disjoint from training and evaluation by
    construction and, when the preset differs, is the whole experiment -- the
    offline stack meeting an economy it was not fitted in.

    Otherwise the eval bank is used, and ``first`` defaults to
    :attr:`~control.dsac.train.TrainConfig.eval_episodes` -- the first future the
    offline run has never scored. That default is not cosmetic. With
    :attr:`~control.dsac.train.TrainConfig.restore_best` the returned agent *is*
    the checkpoint that scored best on futures ``[0, eval_episodes)``, so
    rehearsing from zero would deploy the agent on the very futures it was
    selected against and report the selection back as a result.

    The remaining rule is the caller's to enforce: the futures used for tuning
    the :class:`LiveConfig` and the futures the result is reported on must
    themselves be disjoint. The references are rolled through the *same* futures
    under the *same* seeds, so every comparison is paired.
    """
    live = live or LiveConfig()
    dedicated = len(result.world.deploy_futures) > 0
    pool = result.world.deploy_futures if dedicated else result.world.eval_futures
    if first is None:
        first = 0 if dedicated else result.config.eval_episodes
    bank = pool.all()[first : first + futures]
    if not bank:
        raise ValueError(
            f"no futures at [{first}:{first + futures}] of a "
            f"{'deployment' if dedicated else 'eval'} bank of {len(pool)}"
        )
    seeds = residual_seed(result, live, verbose=verbose)
    out = RehearsalResult(config=live)
    out.runs_ = {name: [] for name in ("adapting", "frozen", *references)}

    for j, episode in enumerate(bank):
        if verbose:
            print(f"rehearsal {j + 1}/{len(bank)}: future #{episode.index}")
        for adapt in (True, False):
            record = deploy(
                result, episode, live, seed_rows=seeds, seed=first + j,
                adapt=adapt, verbose=verbose,
            )
            out.runs_[record.name].append(record)
        for name in references:
            record = run_reference(result, episode, name, live, seed=first + j)
            out.runs_[name].append(record)
            if verbose:
                print(f"  {name}: return {record.total:9.1f} over {len(record)} steps")
    return out


# -- the pieces of one update ------------------------------------------------


def _generate(
    rollouts: SyntheticRollouts,
    real: RealBuffer,
    synthetic: ReplayBuffer,
    agent: DSACAgent,
    live: LiveConfig,
    k: int,
    rng: np.random.Generator,
) -> None:
    """Fill the synthetic buffer with fresh on-policy data from real states."""
    policy = agent_policy(agent, deterministic=False)
    for branch in real.branches(live.branch_points, rng, decay=live.recency_decay):
        _, rows = rollouts.roll(branch, policy, k, rng)
        for obs, action, reward, next_obs, done in rows:
            synthetic.add(obs, action, reward, next_obs, done)


def _mix(
    real: RealBuffer, synthetic: ReplayBuffer, live: LiveConfig, rng: np.random.Generator
) -> Batch:
    """A minibatch mixed from the two buffers, never a merged buffer.

    MBPO's own default is around 5% real; here the real data are far scarcer and
    also far more precious, and the argument cuts both ways -- too low and the
    update ignores the only unbiased data there is, too high and the critic
    overfits a few hundred transitions and the model-based apparatus is
    pointless. The sensitivity to this number is itself a result, because it
    measures how much the correction is actually being trusted.
    """
    n_real = int(round(live.real_fraction * live.batch))
    n_model = live.batch - n_real
    if len(synthetic) == 0 or n_model == 0:
        return real.sample(live.batch, rng)
    if n_real == 0:
        return synthetic.sample(live.batch, rng)
    a, b = real.sample(n_real, rng), synthetic.sample(n_model, rng)
    return Batch(
        obs=np.vstack([a.obs, b.obs]),
        action=np.vstack([a.action, b.action]),
        reward=np.vstack([a.reward, b.reward]),
        next_obs=np.vstack([a.next_obs, b.next_obs]),
        done=np.vstack([a.done, b.done]),
    )


@dataclass(frozen=True)
class AcceptTest:
    """What the acceptance test found, rather than only what it decided.

    ``margin`` is how much better the candidate scored than the incumbent under
    the measure being optimised, and ``error`` the sampling error of that
    estimate. Both go into the record, because "rejected" alone cannot tell a
    candidate that was marginally behind from one that was crushed, and those
    two ask for opposite changes: more branches in the first case, a different
    update in the second.
    """

    margin: float
    error: float
    accepted: bool


def _accept(
    candidate: DSACAgent,
    deployed: DSACAgent,
    rollouts: SyntheticRollouts,
    real: RealBuffer,
    live: LiveConfig,
    *,
    risk: str,
    rng_seed: int,
) -> AcceptTest:
    """Score a candidate policy against the incumbent before letting it act.

    Both are rolled through the corrected model from the *same* branch points
    under the *same* seeded noise stream, and compared under **the functional the
    actor is optimising** (:attr:`LiveConfig.accept_risk`). That last point is
    the whole design of this test and the easiest thing to get wrong. A test
    scoring the lower tail while the actor maximised the mean is asking the
    candidate to have improved something it never tried to improve; it rejects
    sound updates for as long as the run lasts, and run against
    :attr:`LiveConfig.reject_patience` it hands control back to the frozen policy
    on no evidence at all.

    The same trap has a subtler form worth naming, since it is the natural thing
    to reach for once the pairing is noticed: the branches *are* paired, so it is
    tempting to difference them and take the CVaR of the improvement. That
    statistic is not a risk-averse comparison, it is an impossible bar. The worst
    tenth of a difference distribution is negative whenever the difference has
    any spread at all, however far ahead the candidate is on average -- a
    candidate better on every branch by a full standard deviation still fails
    it. CVaR belongs to the *return* distribution, which is what the actor
    optimises; a difference of returns is a location problem and wants a location
    statistic.

    So: under mean risk the estimate is the paired mean improvement, which is
    exactly paired and has the branch's own difficulty divided out. Under CVaR
    risk it is the difference of the two marginal CVaRs -- there is no paired
    analogue, because a CVaR is a functional of a marginal -- with its error
    estimated by resampling branches under a **common** index set, so what the
    two tails share cancels. Either way the candidate is accepted when it is
    ahead, or behind by less than :attr:`LiveConfig.accept_tolerance` of that
    error.

    Scored **deterministically**, unlike the data-generating rollouts: what is
    being compared is the policy that would actually be deployed, and the
    exploration noise the buffer wants would only add variance to a comparison
    whose whole value is that it is paired.

    Each branch gets its **own** generator rather than one threaded through all
    of them. Threading desynchronises the pairing the moment the two policies
    consume different numbers of draws -- which is precisely what happens when
    one of them collapses early and stops consuming, i.e. in the case the test
    exists to catch.
    """
    rng = np.random.default_rng(rng_seed)
    branches = real.branches(live.accept_branches, rng, decay=live.recency_decay)
    scores = {}
    for name, agent in (("deployed", deployed), ("candidate", candidate)):
        policy = agent_policy(agent, deterministic=True)
        scores[name] = np.array(
            [
                rollouts.roll(
                    b, policy, live.accept_horizon,
                    np.random.default_rng(rng_seed + 1 + i),
                )[0]
                for i, b in enumerate(branches)
            ]
        )
    new, old = scores["candidate"], scores["deployed"]

    if risk == "cvar":
        margin = _cvar(new, live.accept_alpha) - _cvar(old, live.accept_alpha)
        boot = np.random.default_rng(rng_seed)
        draws = boot.integers(len(new), size=(200, len(new)))
        error = float(
            np.std(
                [
                    _cvar(new[i], live.accept_alpha) - _cvar(old[i], live.accept_alpha)
                    for i in draws
                ]
            )
        )
    else:
        delta = new - old
        margin = float(delta.mean())
        error = (
            float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0.0
        )
    return AcceptTest(
        margin=margin,
        error=error,
        accepted=bool(margin >= -live.accept_tolerance * error),
    )


def _cvar(values: np.ndarray, alpha: float) -> float:
    """The mean of the worst ``alpha`` fraction of ``values``."""
    n = max(1, int(round(alpha * len(values))))
    return float(np.sort(values)[:n].mean())


def _rollout_length(live: LiveConfig, monitor: OODMonitor) -> int:
    """How far to trust the corrected model right now, in steps.

    MBPO schedules ``k`` on wall-clock training progress. Here the gate is
    uncertainty instead, because there is so little real data that a fixed
    schedule cannot know when it has earned a longer horizon: shrink toward
    ``rollout_min`` when the correction's predictive variance is running above
    its own recent level, stretch toward ``rollout_max`` when it has been
    predicting better than usual.
    """
    if not monitor.readings_ or monitor.variance_ref_ in (None, 0.0):
        return live.rollout_length
    ratio = monitor.readings_[-1].variance / max(monitor.variance_ref_, 1e-12)
    if ratio > 2.0:
        return live.rollout_min
    if ratio < 0.5:
        return live.rollout_max
    return live.rollout_length


def _action_rate(live: LiveConfig, config: TrainConfig) -> float | None:
    """The slew limit the live environment should run under.

    ``None`` on :class:`LiveConfig` inherits the offline run's, which is the
    coherent case -- the agent trained against that instrument, so the policy
    being deployed is the policy that was optimised. Switching the guardrail off
    removes the limit for the live run only, which is the ablation and is a
    genuine train/deploy mismatch rather than a milder setting.
    """
    if not live.guarding("rate_limit"):
        return None
    return config.action_rate if live.action_rate is None else live.action_rate


class _OutcomeTracker:
    """The three observables the mandate scores, plus the potential it wants.

    Carries the two previous levels the growth rates are differences of, which is
    the only state a per-period log needs and the reason this is a small object
    rather than a function.
    """

    def __init__(self, env: CentralBankEnv, world: TrainingWorld) -> None:
        """Start from the history's last row, where every future branches."""
        names = env.interface.state.names()
        params = env.interface.parameters.names()
        self._yk = names.index("Yk")
        self._nfe, self._grpr = params.index("Nfe"), params.index("GRpr")
        self._action_names = env.interface.actions.names()
        self.dt = world.dt
        self.prev_yk = float(world.history.states[-1][self._yk])
        self.prev_nfe = float(world.history.params[-1][self._nfe])

    def observe(
        self, state: State, parameters: Parameters, actions: Actions
    ) -> dict[str, float]:
        """One period's levers and outcomes, ready to go into a log row."""
        pars = parameters.to_dict()
        growth = np.log(state.Yk / self.prev_yk) / self.dt
        labour = np.log(pars["Nfe"] / self.prev_nfe) / self.dt
        row = {
            "growth": float(growth),
            "potential": float(pars["GRpr"] + labour),
            "ER": float(state.ER),
            "PI": float(state.PI),
            **{name: float(getattr(actions, name)) for name in self._action_names},
        }
        self.prev_yk, self.prev_nfe = float(state.Yk), float(pars["Nfe"])
        return row
