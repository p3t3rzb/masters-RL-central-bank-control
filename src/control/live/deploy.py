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

The levers' **finite speed is not on that list**, and deliberately. It is not a
guardrail at all but the instrument itself: the policy's action is a *step*, the
levers integrate it, and :attr:`~control.env.EnvConfig.delta_rate` fixes the
maximum speed -- during training, in the synthetic rollouts, in the acceptance
test and here, all read from one place. (An earlier design applied a slew clip
to the agent's output inside this loop and nowhere else, which meant the policy
being optimised, the policy being tested and the policy being run were three
different controllers; the delta parametrization retired both the clip and the
mismatch.) The references are the one deliberate exemption: they run under a
:func:`~control.dsac.train.free_instrument` configuration, whose instrument is
fast enough that :meth:`~control.env.CentralBankEnv.toward` lands on a rule's
target within the period -- the textbook rules are the fixed bars, and the
speed limit is the agent's problem alone.

Everything here is testable before the live run by *rehearsing* it: the ground
truth is unavailable to the bank but available in the lab, so :func:`rehearse`
replays the entire procedure on held-out futures and returns a distribution over
one-shot deployments, which is the right object to set hyperparameters against.
"""

from __future__ import annotations

import copy
import os
from concurrent.futures import as_completed
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import torch

from economic_models.proxy import BaseProxyModel
from economic_models.run import Run
from economic_models.variables import Actions, Parameters, State
from parallel import managed_pool

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
    free_instrument,
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
    action_columns,
    NullResidual,
    Residual,
    ResidualModel,
    cross_fitted_residuals,
    design,
    fitted_gate,
    in_sample_residuals,
)
from control.rewards import RewardContext
from control.world import Episode, NullStabilizer, TrainingWorld

if TYPE_CHECKING:
    from control.live.oracle import OracleConfig

#: The forcing families a deployment can forecast its exogenous environment with.
FORCINGS = ("var", "bootstrap", "oracle")

#: What the online phase may fine-tune. See :attr:`LiveConfig.online_policy`.
ONLINE_POLICIES = ("full", "gain")


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
    #: rollouts started per policy update. Read together with
    #: :attr:`rollout_length`: the synthetic data a generation contributes is
    #: their *product*, so a comparison of two rollout horizons at a fixed
    #: ``branch_points`` is confounded with the amount of synthetic data, and the
    #: honest sweep holds the product fixed.
    branch_points: int = 400
    rollout_length: int = 3  #: nominal k
    rollout_min: int = 1  #: k when the correction is uncertain
    rollout_max: int = 5  #: k when it has been predicting well
    generations: int = 5  #: policy updates of synthetic data retained
    real_fraction: float = 0.2  #: rho, the real share of a minibatch
    batch: int = 256  #: minibatch size
    #: deliberate lever excitation, as a standard deviation in **normalised**
    #: action units (the box is [-1, 1]), added to what the policy asks for.
    #:
    #: Off by default, and switching it on is a real decision rather than a
    #: tuning knob: it spends mandate return to buy identification. The
    #: correction is a function of a design that carries the action block, and a
    #: deployment that holds its levers on one path teaches that block nothing --
    #: the coefficients stay at their prior and the corrected model's *response*
    #: to a lever is whatever the offline proxy already believed. That is
    #: precisely the direction in which a surrogate fitted on one history is
    #: least trustworthy and the direction the policy gradient lives in, so a run
    #: that never moves the instrument off-policy cannot learn the thing it most
    #: needs to. Excitation is the classical answer and the classical cost.
    #:
    #: Applied identically to the adapting and the frozen arm, off a stream of
    #: its own, so the two remain paired step for step and the comparison
    #: isolates *learning from* the excited data rather than the excitation
    #: itself. The levers still integrate at instrument speed: this perturbs the
    #: requested step, it does not bypass the instrument.
    explore: float = 0.0
    #: persistence of the excitation, as an AR(1) coefficient.
    #:
    #: At zero the perturbation is white noise on the requested *step*, which is
    #: what earlier runs used and which is a very weak instrument -- because the
    #: instrument is a delta. The levers integrate the step, so the design's
    #: action columns carry the lever *level*, and i.i.d. noise on the step
    #: contributes a random walk whose low-frequency power is tiny: measured over
    #: a deployment, excitation at 0.30 normalised units moves the action block's
    #: variance inflation only from 74x to 20x and the corrected model's cosine
    #: against the structural economy's own lever response from +0.00 to +0.03.
    #: Identification of a level response needs excursions that *last*, which is
    #: also what a deliberate policy experiment looks like when a central bank
    #: runs one. Read against ``dt``: 0.9 at dt=0.083 is a perturbation with a
    #: memory of about ten months.
    #:
    #: Scaled so that :attr:`explore` remains the standard deviation of the
    #: perturbation whatever the persistence, which is the only way a sweep over
    #: one of them is not secretly a sweep over both.
    explore_rho: float = 0.0
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
    #: let the correction read the **action block** of its design.
    #:
    #: Off is the better default and the reason is identification, not taste. A
    #: live run moves its levers by policy, so the action columns are a near
    #: exact function of the rest of the design -- measured over a deployment,
    #: a variance inflation of order a thousand against the twenty-odd the
    #: excited history carries. Their coefficients are then settled by the prior
    #: rather than by the data, and the direction they are wrong in is the one
    #: direction a policy is made of: the model's response to the instrument.
    #: What that buys in forecast is negligible (deleting the block moves the
    #: one-step error from 0.161 to 0.184 of the raw proxy's, under a thirtieth
    #: of the correction's total gain) and what it costs is the whole of the
    #: control signal -- the corrected response to a lever has been measured
    #: *orthogonal* to the structural economy's while the raw proxy's still
    #: carries a positive alignment with it.
    #:
    #: On restores the earlier behaviour, and is worth keeping for the ablation:
    #: "the correction may not touch the levers" is a claim, and a run with it
    #: switched on is what the claim is measured against.
    correct_actions: bool = False
    #: how much an imagined step is charged for what the model does not know.
    #:
    #: The rollout reward becomes ``r - uncertainty_penalty * u``, where ``u`` is
    #: the correction's own predictive standard deviation in the mandate's three
    #: columns, expressed in the units the mandate divides by -- so ``u = 1``
    #: means "one leg's worth of reference deviation of pure model uncertainty"
    #: and the penalty is on the same scale as the legs themselves.
    #:
    #: This is MOPO's term (Yu et al., 2020) and it is the piece the apparatus
    #: was missing. The predictive variance was already computed every step and
    #: spent on two things -- choosing ``k`` and tripping the monitor -- while the
    #: quantity a policy actually optimises never saw it. Inside a rollout the
    #: variance only scaled a **mean-zero** draw, which adds spread without
    #: adding caution: under ``risk="mean"`` the actor is indifferent to it, and
    #: an imagined state the model has no idea about is worth exactly as much as
    #: one it is sure of.
    #:
    #: Why it matters here specifically. The guardrails answer "an update might
    #: be wrong" by making the update *small* -- a tenth of the learning rate, a
    #: KL anchor to the offline policy -- and that is measurably a dead end in
    #: both directions: at the default the step lands below the acceptance test's
    #: own resolution, and removing the anchor (``--no-anchor --actor-lr 3e-4``)
    #: moves the policy eight times as far for no gain in the mean and a tail
    #: that runs to -31 points. A small step is not a safe step, it is an
    #: undirected one made shorter. Pessimism is the other way to be safe: keep
    #: the step large and make the model refuse to recommend the places it cannot
    #: see. Off by default so the pair is the ablation.
    uncertainty_penalty: float = 0.0
    #: what the online phase is allowed to fine-tune.
    #:
    #: ``"full"`` is the whole actor, which is what MBPO would do and what every
    #: earlier run here did. ``"gain"`` restricts it to a linear reaction on
    #: :attr:`gain_features` on top of the frozen offline mean
    #: (:meth:`~control.dsac.agent.DSACAgent.restrict_actor`) -- a dozen
    #: parameters instead of seventy thousand.
    #:
    #: The case for ``"gain"`` is that "online RL from two hundred samples is not
    #: a thing" is a claim about parameter counts, and the guardrails answer it
    #: in the wrong currency: a tenth of the learning rate and a KL anchor make
    #: the step *small* without making it identifiable, and the acceptance test
    #: then has to resolve a step measured at three hundredths of the instrument
    #: against a sampling error of the same size -- which it cannot, and its
    #: 45% acceptance rate is the coin flip that follows. Twelve parameters is
    #: inside what a few hundred transitions identify, so the step can be large
    #: enough for the test to see and still bounded by construction.
    online_policy: str = "full"
    #: the observation channels the ``"gain"`` reaction reads, by name.
    #:
    #: The mandate's own three by default: those are the only signals the reward
    #: is a function of, so a reaction to anything else cannot be scored on this
    #: run's evidence. Named rather than positional because the observation's
    #: width depends on the encoder and its economic block does not.
    gain_features: tuple[str, ...] = ("dlog(Yk)", "ER", "PI")
    #: shrink the correction **per output column**, at a trust fitted out of
    #: sample on the historic rows (:func:`~control.live.residual.fitted_gate`).
    #:
    #: On is the better default and the reason is the mandate. The correction is
    #: one estimator against a sixteen-wide target, and it is measurably good on
    #: some of those columns and measurably harmful on others: over the GROWTH
    #: history, ``dlog(Yk)`` scores an out-of-fold R-squared of +0.24 at this
    #: prior scale while ``PI`` scores -2.18 and ``ER`` -5.00. Three of the
    #: sixteen columns are the only ones the reward reads, and two of those three
    #: are in the second group -- so the un-gated correction cuts the *state*
    #: error to under half (almost all of it in the equity price, which carries
    #: 99.5% of the squared error norm and which the mandate never looks at)
    #: while making the two mandate columns worse. A policy improved against that
    #: model has been handed a better forecast of something it is not scored on.
    #:
    #: Off is the default because the size of the effect is an empirical
    #: question and the answer, measured, is small: run prequentially over the
    #: GROWTH history the way a deployment runs it, the correction turns out to
    #: have *some* skill in every column (``dlog(Yk)`` R-squared +0.77, ``PI``
    #: +0.15, ``ER`` -0.07), so the fitted trust lands between 0.80 and 0.97
    #: nearly everywhere and closes nothing. The uneven skill is real and worth
    #: reporting -- the correction removes 86% of the whole-state error and only
    #: 31% of the error in the three columns the reward reads -- but it is not
    #: something a per-column shrinkage can repair, because the columns are not
    #: *harmful*, only unhelpful. Kept as the switch that establishes that.
    gate: bool = False
    gate_floor: float = 0.0  #: the smallest trust a column may be given

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
    actor_lr: float = 3e-5  #: ten times below the training-time rate
    #: the rate the ``"gain"`` reaction is fine-tuned at when the actor-rate
    #: guardrail is off. A dozen parameters starting from exactly zero can take
    #: the training-time step without the drift that makes a slow rate necessary
    #: for the full actor, so the guardrail being off means *this*, not nothing.
    gain_lr: float = 3e-4
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
    #: score candidates on branch states the synthetic rollouts did **not** start
    #: from, by splitting the real buffer on arrival parity
    #: (:meth:`~control.live.buffer.RealBuffer.branches`).
    #:
    #: Off, the acceptance test is in-sample in every respect but the draw: the
    #: candidate is improved on rollouts from a pool of states and then scored on
    #: rollouts from that same pool, through the same model, under the same
    #: forcing. That test can say a candidate is better where it was fitted; it
    #: cannot say it will be better anywhere else, which is the question a
    #: deployment is asking. On, the two pools are disjoint, so the margin is a
    #: held-out quantity -- the only held-out quantity a run with one economy and
    #: no reset can construct.
    #:
    #: It is not a cure for a biased model: both pools are scored *through* the
    #: corrected proxy, so a model that ranks policies wrongly ranks them wrongly
    #: on either half. What it catches is the other failure -- a candidate that
    #: has learned the particular states it was rolled from.
    holdout_branches: bool = False
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
        if self.online_policy not in ONLINE_POLICIES:
            raise ValueError(
                f"online_policy must be one of {ONLINE_POLICIES}, "
                f"got {self.online_policy!r}"
            )
        if self.forcing not in FORCINGS:
            raise ValueError(f"forcing must be one of {FORCINGS}, got {self.forcing!r}")
        if self.accept_risk not in (None, "mean", "cvar"):
            raise ValueError(
                f"accept_risk must be 'mean', 'cvar' or None, got {self.accept_risk!r}"
            )
        if not -1.0 < self.explore_rho < 1.0:
            raise ValueError(
                f"explore_rho must be a stationary AR(1) coefficient in "
                f"(-1, 1), got {self.explore_rho}"
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
            interface=result.world.config.spec.interface,
            encoder=result.proxy.encoder,
            latent=config.latent,
            seed=config.seed,
        ),
        result.world.history,
        folds=live.folds,
        verbose=verbose,
    )


def build_residual(
    seed_rows: tuple[np.ndarray, np.ndarray],
    live: LiveConfig,
    n_outputs: int,
    *,
    ignore: Sequence[int] | None = None,
    verbose: bool = False,
) -> Residual:
    """A correction primed on the historic rows, or the null one.

    ``ignore`` are design columns the correction may not read; a caller passes
    the action block here when :attr:`LiveConfig.correct_actions` is off.
    """
    if not live.correct:
        return NullResidual(n_outputs)
    noise = (
        BlockBootstrapResidualNoise(n_outputs)
        if live.bootstrap_noise
        else None
    )
    X, E = seed_rows
    shared = dict(
        tau=live.tau,
        forgetting=live.forgetting,
        kappa=live.kappa,
        ignore=ignore,
    )
    # Fitted before the model it gates, on the same rows and by the same
    # estimator, so the trust per column is the history's own verdict on the
    # correction rather than a hyperparameter.
    gate = (
        fitted_gate(
            X, E, floor=live.gate_floor, weight=live.history_weight,
            verbose=verbose, **shared,
        )
        if live.gate
        else None
    )
    model = ResidualModel(n_outputs, noise=noise, gate=gate, **shared)
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


def mandate_columns(proxy: BaseProxyModel, reward: Any) -> np.ndarray:
    """The state-feature columns the reward reads, or every column if unknown.

    The state features are positionally aligned with the state variables, so a
    reward that declares which variables it reads
    (:attr:`~control.rewards.mandate.MandateReward.STATES`) names its own
    columns. A reward that declares nothing gets all of them, which reduces the
    mandate-restricted error to the ordinary one rather than to an empty slice.
    """
    names = proxy.transform.state_names
    wanted = getattr(reward, "STATES", None)
    if not wanted:
        return np.arange(len(names))
    return np.array(
        [names.index(n) for n in wanted if n in names], dtype=int
    )


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
        self._penalty_cols, self._penalty_scale = _uncertainty_scale(
            model, env.reward, self.dt
        )

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
            position, action = self.env.resolve_action(policy(obs), exog[-1])
            params = self.stabilizer.apply(path[j], float(states[-1][self._er]))
            parameters = interface.parameters.from_row(params)
            actions = self.env.to_actions(position)

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

            # Charged for what the model does not know, before anything else
            # sees this reward: the same number goes into the replay buffer, the
            # critic and the acceptance test, so a policy is optimised, scored
            # and admitted under one consistent degree of caution.
            penalty = 0.0
            if self.live.uncertainty_penalty:
                variance = getattr(self.model, "last_variance_", None)
                if variance is not None and len(self._penalty_cols):
                    u = float(
                        np.linalg.norm(
                            np.sqrt(np.maximum(variance[self._penalty_cols], 0.0))
                            / self._penalty_scale
                        )
                    )
                    penalty = self.live.uncertainty_penalty * u
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
            r -= penalty
            rows.append((obs, action, r, next_obs, False))
            total += r
            states, exog, obs = nxt, nxt_exog, next_obs
        return total, rows


def _uncertainty_scale(
    model: BaseProxyModel, reward: Any, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """The mandate's feature columns and the divisor that puts them in leg units.

    An uncertainty penalty is only interpretable if it is measured in the same
    units as the thing it is subtracted from. Each mandate leg is a squared ratio
    ``(gap / scale)``, so a predictive standard deviation of ``s`` in that leg's
    feature contributes ``s / scale`` of normalised deviation -- except for real
    growth, which the mandate reads as an annualised rate while the feature is a
    one-period log difference, so its divisor carries an extra ``dt``.

    Returns ``(columns, divisors)``, both empty when the reward declares no
    states, which switches the penalty off rather than guessing at a scale.
    """
    columns = mandate_columns(model, reward)
    names = model.transform.state_names
    wanted = getattr(reward, "STATES", ())
    #: leg scale per declared state, and whether the feature is a rate per period
    per_state = {
        "Yk": (getattr(reward, "g_scale", 0.02), True),
        "ER": (getattr(reward, "u_scale", 0.02), False),
        "PI": (getattr(reward, "pi_scale", 0.02), False),
    }
    divisors = []
    for name in wanted:
        if name not in names:
            continue
        scale, per_period = per_state.get(name, (0.02, False))
        divisors.append(scale * dt if per_period else scale)
    if not divisors:
        return np.array([], dtype=int), np.array([], dtype=float)
    return columns, np.array(divisors, dtype=float)


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

    # The instrument's speed is the environment's, not a filter on the agent's
    # output, so the deployment, its synthetic rollouts and the references all
    # act under the same instrument -- the one the offline run was trained
    # against.
    env = build_truth_env(
        world, observer, result.env.reward, config, seed=seed,
        horizon=live.live_steps,
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
    if live.online_policy == "gain":
        # Restricted on the clone only: ``frozen`` stays the plain offline
        # policy, so it is still both a coherent anchor and a parachute that
        # carries none of the online phase's parameters.
        missing = [n for n in live.gain_features if n not in observer.names]
        if missing:
            raise ValueError(
                f"gain_features {missing} are not observation channels; "
                f"this observation has {observer.names}"
            )
        agent.restrict_actor(
            [observer.index(n) for n in live.gain_features],
            lr=live.actor_lr if live.guarding("actor_lr") else live.gain_lr,
        )
    elif live.guarding("actor_lr"):
        agent.set_lr(actor=live.actor_lr)
    anchor_weight = live.anchor_weight if live.guarding("anchor") else 0.0

    # The shadow proxy and the corrected model are shallow copies: they share the
    # expensive fitted estimator and the fitted encoder, and keep their own
    # rollout state, so filtering the live run and imagining a future off it
    # cannot disturb each other.
    shadow: BaseProxyModel = copy.copy(result.proxy)
    rows = (
        seed_rows if seed_rows is not None
        else residual_seed(result, live, verbose=verbose)
    )
    residual = build_residual(
        rows,
        live,
        n_feat,
        ignore=(
            None if live.correct_actions
            else action_columns(result.proxy, rows[0].shape[1])
        ),
        verbose=verbose,
    )
    model = CorrectedProxy(copy.copy(result.proxy), residual)
    # The three columns the mandate reads, so the record can carry the error
    # that matters to a policy next to the error over the whole state.
    mandate = mandate_columns(result.proxy, result.env.reward)
    # The same conversion the rollout's uncertainty penalty uses, so the record
    # can report how large that charge actually is per step -- a penalty whose
    # scale is not known is a penalty whose coefficient cannot be chosen.
    penalty_cols, penalty_scale = _uncertainty_scale(
        result.proxy, result.env.reward, world.dt
    )

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
    # A stream of its own, keyed to the deployment rather than to the arm: the
    # frozen reference has no updates to draw for and would otherwise consume the
    # main stream at a different rate, so a shared generator would hand the two
    # arms different excitation and quietly unpair them.
    exciter = np.random.default_rng(live.seed + 7919 + seed)
    # The excitation's own AR(1) state, shared by the two arms for the same
    # reason the stream is: they must be perturbed identically or the comparison
    # stops being about learning from the excited data.
    shock = np.zeros(env.action_dim)
    innovation = float(np.sqrt(max(0.0, 1.0 - live.explore_rho**2)))
    # Which half of the branch states trains the candidate and which scores it.
    # ``None`` on both is the unsplit behaviour: one pool, used for both.
    fit_half = 0 if live.holdout_branches else None
    test_half = 1 if live.holdout_branches else None

    steps = min(live.live_steps, len(episode))
    for t in range(1, steps + 1):
        requested = agent.act(obs, deterministic=True)
        if live.explore:
            shock = live.explore_rho * shock + innovation * exciter.standard_normal(
                env.action_dim
            )
            requested = np.clip(
                requested + live.explore * shock, -1.0, 1.0
            )
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
                "error_raw_mandate": float(np.linalg.norm(eps[mandate])),
                "error_corrected_mandate": float(
                    np.linalg.norm(corrected_error[mandate])
                ),
                "residual_var": reading.variance,
                "uncertainty": (
                    float(
                        np.linalg.norm(
                            np.sqrt(np.maximum(variance[penalty_cols], 0.0))
                            / penalty_scale
                        )
                    )
                    if len(penalty_cols)
                    else 0.0
                ),
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
            _generate(rollouts, real, synthetic, agent, live, k, rng, half=fit_half)
            candidate = agent.clone()
            for _ in range(live.gradient_steps):
                candidate.update(
                    _mix(real, synthetic, live, rng),
                    anchor=frozen,
                    anchor_weight=anchor_weight,
                )
            risk = live.accept_risk or config.risk
            test = (
                _accept(
                    candidate, agent, rollouts, real, live,
                    risk=risk, rng_seed=live.seed + t, half=test_half,
                )
                if live.guarding("accept")
                else AcceptTest(margin=0.0, error=0.0, accepted=True)
            )
            # The same comparison on the states the candidate was rolled from.
            # It decides nothing -- it is the in-sample twin of the margin that
            # does, and the gap between the two is what "the candidate learned
            # these particular states" looks like when it is measured rather
            # than assumed.
            insample = (
                _accept(
                    candidate, agent, rollouts, real, live,
                    risk=risk, rng_seed=live.seed + t, half=fit_half,
                ).margin
                if live.holdout_branches and live.guarding("accept")
                else None
            )
            accepted = test.accepted
            # What the test was asked to resolve, alongside its verdict: a step
            # smaller than the test's own error is not a rejected improvement,
            # it is an improvement nobody could have measured.
            step_size, drift = _displacement(
                candidate, agent, frozen, real, live, rng
            )
            record.events_.append(
                {
                    "step": t,
                    "kind": "accepted" if accepted else "rejected",
                    "k": k,
                    "margin": test.margin,
                    **({} if insample is None else {"margin_insample": insample}),
                    "error": test.error,
                    "move": step_size,
                    "drift": drift,
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


#: The feedback gains each named Taylor reference is run at.
#:
#: The rule is one bar with two settings worth drawing, and which of them is
#: *the* bar is exactly the question :mod:`control.tuning` was written to answer.
#: ``"taylor"`` is Taylor's original pair, taken on faith everywhere in this
#: repository before that package existed; ``"taylor-tuned"`` is the pair
#: ``scripts/tune_policy.py`` selected over the ``data/`` ensemble, chosen on
#: held-out economies and confirmed on a test split it had never seen
#: (``scripts/tune_taylor_signed.json``).
#:
#: The tuned ``phi_pi`` is **negative**, and deliberately so. The rule's response
#: to inflation is ``1 + phi_pi``, so the searchable range has to span zero for
#: the rule to reach its own baseline: ``phi_pi = -1`` with ``phi_y = 0`` leaves
#: ``rate = i*`` at every period, which *is* the calibration constant. It matters
#: here because GROWTH settles near 0.6% inflation against a 2% target, so a
#: one-for-one response costs a permanent ~1.4-point rate offset that no choice
#: of ``phi_y`` can undo. Read the tuned pair as the rule buying back that level,
#: not as evidence against inflation feedback -- at ``1 + phi_pi = 0.094`` it
#: violates the Taylor principle, and beats the constant baseline by only 1.4%.
#:
#: Both are kept, and a figure should show both: an agent that beats the untuned
#: rule and loses to the tuned one has not beaten the Taylor rule, and the only
#: way to see that is to draw the two separately.
TAYLOR_GAINS: Mapping[str, tuple[float, float]] = {
    "taylor": (0.5, 0.5),
    "taylor-tuned": (-0.9065, 0.3161),
}


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

    The environment is the same in everything but the instrument's speed: a
    reference is a rule stated in lever levels, and it is run under a
    :func:`~control.dsac.train.free_instrument` configuration -- one period's
    move spans the whole box, so
    :meth:`~control.env.CentralBankEnv.toward` lands on the rule's target within
    the period and the textbook rule is compared unmodified. The speed limit is
    the *agent's* problem, deliberately: the references are the fixed bars, not
    competitors under its constraints.
    """
    env = build_truth_env(
        result.world, result.observer, result.env.reward,
        free_instrument(result.config), seed=seed, horizon=live.live_steps,
    )
    model = result.world.config.model
    gains = result.world.config.spec.reference_gains
    if policy_name in gains:
        phi_pi, phi_y = gains[policy_name]
        policy = taylor_policy(
            env, result.observer, dt=result.config.dt,
            pi_target=result.config.pi_target, phi_pi=phi_pi, phi_y=phi_y,
            model=model,
        )
    elif policy_name == "calibration":
        policy = constant_policy(env, calibration_actions(model))
    else:
        raise ValueError(
            f"unknown reference policy {policy_name!r} for model {model!r}; "
            f"expected one of {(*gains, 'calibration')}"
        )

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


def _rehearse_one(
    result: TrainingResult,
    episode: Episode,
    live: LiveConfig,
    seeds: tuple[np.ndarray, np.ndarray],
    references: tuple[str, ...],
    oracle: "OracleConfig | None",
    *,
    seed: int,
    verbose: bool,
) -> dict[str, RunRecord]:
    """One future's complete rehearsal: both arms, the references, the oracle.

    The unit the futures loop repeats and the unit a parallel rehearsal ships to
    a worker, so it must depend on nothing but its arguments. Torch's global
    stream is re-seeded per future for the same reason: the adapting arm's
    updates draw from it, and a future whose draws depend on which futures ran
    before it (or beside it, in a pool) would make the two execution modes
    different experiments. Seeded, they are the same experiment run in either
    order.
    """
    torch.manual_seed(live.seed + seed)
    out: dict[str, RunRecord] = {}
    for adapt in (True, False):
        record = deploy(
            result, episode, live, seed_rows=seeds, seed=seed,
            adapt=adapt, verbose=verbose,
        )
        out[record.name] = record
    for name in references:
        record = run_reference(result, episode, name, live, seed=seed)
        out[name] = record
        if verbose:
            print(f"  {name}: return {record.total:9.1f} over {len(record)} steps")
    if oracle is not None:
        # Imported here rather than at the top: the oracle module builds on
        # this one, and this call is the single place the dependency points
        # back the other way.
        from control.live.oracle import optimal_run

        record = optimal_run(
            result, episode, live, oracle, seed=seed, verbose=verbose
        )
        out["optimal"] = record
        if verbose:
            print(f"  optimal: return {record.total:9.1f} over {len(record)} steps")
    return out


#: One worker's share of a parallel rehearsal, set once by :func:`_worker_init`.
#: A module global because a process pool has no other place for per-worker
#: state; nothing outside these two functions may touch it.
_WORKER: tuple[TrainingResult, LiveConfig, tuple, tuple, Any] | None = None


def _worker_init(
    result: TrainingResult,
    live: LiveConfig,
    seeds: tuple[np.ndarray, np.ndarray],
    references: tuple[str, ...],
    oracle: "OracleConfig | None",
) -> None:
    """Receive the (pickled) offline stack once per worker, not once per future."""
    global _WORKER
    # One torch thread per worker: the workers are the parallelism, and a pool
    # of them each spawning intra-op threads oversubscribes the machine (see
    # the DSAC dispatch-bound finding -- one thread is no slower per update).
    torch.set_num_threads(1)
    _WORKER = (result, live, seeds, references, oracle)


def _worker_rehearse(j: int, episode: Episode, seed: int) -> tuple[int, dict[str, RunRecord]]:
    """One future's rehearsal inside a worker, tagged with its position."""
    assert _WORKER is not None, "worker used before _worker_init"
    result, live, seeds, references, oracle = _WORKER
    return j, _rehearse_one(
        result, episode, live, seeds, references, oracle, seed=seed, verbose=False
    )


def rehearse(
    result: TrainingResult,
    live: LiveConfig | None = None,
    *,
    futures: int = 8,
    first: int | None = None,
    references: tuple[str, ...] = ("taylor", "calibration"),
    oracle: "OracleConfig | None" = None,
    workers: int = 1,
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

    ``oracle`` additionally computes the **optimal run** on every future -- the
    lever path a genetic search finds with the ground truth in hand
    (:func:`~control.live.oracle.optimal_run`), logged as one more paired
    policy named ``"optimal"``. Lab-only, like the oracle forcing, and by far
    the dearest thing here: every fitness evaluation is a full ground-truth
    run, which is why it is off unless asked for and why
    :attr:`~control.live.oracle.OracleConfig.cache` exists.

    ``workers`` rehearses that many futures concurrently, in spawned processes
    each holding its own copy of the offline stack (0 sizes the pool to the
    machine). The futures are independent by construction -- that is the whole
    premise of a rehearsal -- and torch's stream is seeded per future in either
    mode, so the parallel run is the same experiment as the sequential one:
    the torch-free runs (the references and the oracle) land bit-identical,
    the two agent arms to float noise (torch's CPU kernels are not
    bit-reproducible across process boundaries), and each future's adapting
    and frozen arms share one process either way, so their *paired* difference
    is never crossed by that noise. Inside a pool the oracle's own worker pool
    is collapsed to one process: the futures are already the parallelism, and
    nested pools oversubscribe.
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
    names = (
        "adapting", "frozen", *references,
        *(("optimal",) if oracle is not None else ()),
    )
    out.runs_ = {name: [] for name in names}

    if workers != 1 and len(bank) > 1:
        pool_size = min(
            len(bank), workers if workers else max(1, (os.cpu_count() or 2) - 2)
        )
    else:
        pool_size = 1

    if pool_size == 1:
        rehearsed = []
        for j, episode in enumerate(bank):
            if verbose:
                print(f"rehearsal {j + 1}/{len(bank)}: future #{episode.index}")
            rehearsed.append(
                _rehearse_one(
                    result, episode, live, seeds, references, oracle,
                    seed=first + j, verbose=verbose,
                )
            )
    else:
        # The futures are the parallelism, so the oracle inside each worker
        # keeps to one process -- a pool of pools oversubscribes the machine.
        oracle_one = None if oracle is None else replace(oracle, workers=1)
        rehearsed = [None] * len(bank)
        with managed_pool(
            pool_size,
            initializer=_worker_init,
            initargs=(result, live, seeds, references, oracle_one),
        ) as pool:
            pending = {
                pool.submit(_worker_rehearse, j, episode, first + j)
                for j, episode in enumerate(bank)
            }
            for done in as_completed(pending):
                j, records = done.result()
                rehearsed[j] = records
                if verbose:
                    line = ", ".join(
                        f"{name} {records[name].total:.1f}" for name in names
                    )
                    print(f"rehearsal of future #{bank[j].index}: {line}")

    for records in rehearsed:
        for name in names:
            out.runs_[name].append(records[name])
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
    *,
    half: int | None = None,
) -> None:
    """Fill the synthetic buffer with fresh on-policy data from real states."""
    policy = agent_policy(agent, deterministic=False)
    for branch in real.branches(
        live.branch_points, rng, decay=live.recency_decay, half=half
    ):
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


def _displacement(
    candidate: DSACAgent,
    incumbent: DSACAgent,
    offline: DSACAgent,
    real: RealBuffer,
    live: LiveConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """How far the candidate has moved, in the actor's own normalised units.

    The acceptance margin says whether an update helped; it cannot say whether
    there *was* an update. This is the missing half: the mean absolute
    difference between what two policies request at the same real states, in the
    normalised step the actor emits -- so 1.0 is the whole instrument, one
    period's travel being :attr:`~control.env.CentralBankEnv.delta_step` of the
    box, and a difference held for a whole deployment moves the levers by
    ``move * delta_step * live_steps / 2`` of it.

    Two numbers, because they answer different questions. Against the
    **incumbent** it is the size of this step, which is what the acceptance test
    is trying to resolve -- a step far below the test's own sampling error is
    one the test cannot see whatever its tolerance. Against the **offline**
    policy it is the total drift, which is what the anchor bounds and what
    ultimately has to be large enough to matter: a deployment whose policy ends
    a hundredth of the box from where it started cannot have closed a gap that
    a tenth of the box wide.

    Measured deterministically on states the run actually visited, since a
    difference of two squashed means is what would actually reach the levers.
    """
    batch = real.sample(min(256, len(real)), rng)
    with torch.no_grad():
        moves = []
        for other in (incumbent, offline):
            gap = np.abs(
                np.stack([candidate.act(o, deterministic=True) for o in batch.obs])
                - np.stack([other.act(o, deterministic=True) for o in batch.obs])
            )
            moves.append(float(gap.mean()))
    return moves[0], moves[1]


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
    half: int | None = None,
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
    branches = real.branches(
        live.accept_branches, rng, decay=live.recency_decay, half=half
    )
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
