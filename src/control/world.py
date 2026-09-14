"""The training world: one historic run and the futures branching off it.

The whole control experiment is set in a single economy. A ground-truth
:class:`~economic_models.run.Run` is simulated once -- the *history*
-- and is the only data the proxy is ever fit on. From a state that history
passes through, :meth:`~economic_models.ground_truth.excitation.base.ExcitedRunGenerator.generate_branched`
forks the excitation process into many independent *futures*: pure exogenous
forcing paths (:class:`~economic_models.run.Scenario`), drawn without
solving any model, because the states depend on the actions an agent has not
chosen yet. Those futures are this package's episodes.

Which state they fork from is a :class:`BranchPoint`, and by default there is one
-- the history's end. The evaluation and deployment banks always use it, so a
score means the same thing in every configuration. The *training* bank may be
spread over several rows instead (:attr:`WorldConfig.n_starts`), each future
carrying the branch it was drawn for, so an agent is optimised from a
distribution of initial conditions rather than from whichever corner of the state
space this history happened to stop in.

Generating a future is cheap; *rolling through* one is not -- every step of it
costs one proxy step at training time. The bank of futures is therefore sized for
variety, not for the training budget.

One piece of the model's dynamics cannot live in a frozen forcing path: GROWTH's
countercyclical government-spending stabilizer reads the realised employment rate,
which only exists once the agent's economy has run. A scenario is drawn at full
employment (``ER = 1``, so the stabilizer contributes nothing);
:class:`FiscalStabilizer` puts that response back at rollout, against the state
the agent actually produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np

from economic_models.ground_truth.excitation.base import ExcitationConfig
from economic_models.ground_truth.registry import (
    DEFAULT_MODEL,
    GroundTruthSpec,
    ground_truth,
)
from economic_models.ground_truth.excitation.base import BranchCapture
from economic_models.ground_truth.excitation.specs import ExcitationJitter
from economic_models.ground_truth.excitation.specs import SpendingResponse
from economic_models.run import Run, Scenario

#: The excitation presets a world may be built with.
EXCITATIONS = ("default", "calm", "realistic")


@dataclass(frozen=True)
class WorldConfig:
    """How the history and its futures are drawn.

    ``dt`` is the length of one step in years and ``history_steps`` the length of
    the single run the proxy is fit on; ``horizon`` is how many steps one episode
    lasts. ``n_train_futures`` and ``n_eval_futures`` size the two disjoint banks
    of exogenous paths (the eval bank is never seen during training).
    ``excitation`` names one of the excitation presets of :attr:`model`'s family
    preset, and ``seed`` fixes the whole draw.

    ``deploy_excitation`` and ``n_deploy_futures`` add a **third** bank, drawn
    under a different preset: see :attr:`deploy_excitation`.
    """

    dt: float = 0.25  #: length of one step in years (0.25 quarterly, 1/12 monthly)
    history_steps: int = 500  #: recorded steps of the run the proxy is fit on
    horizon: int = 50  #: steps per episode
    n_train_futures: int = 2000  #: exogenous paths available for training episodes
    n_eval_futures: int = 100  #: held-out exogenous paths, disjoint seeds
    #: which ground-truth economy the world is built on, a key of
    #: :data:`~economic_models.ground_truth.registry.MODELS`.
    model: str = DEFAULT_MODEL
    excitation: str = "realistic"  #: which excitation preset of that model's family
    #: how many futures to draw for the third, *deployment* bank. Zero leaves the
    #: bank empty and a deployment falls back on the eval one, which is the
    #: original behaviour.
    n_deploy_futures: int = 0
    #: the preset the deployment bank is drawn under, ``None`` meaning
    #: :attr:`excitation` like everything else. Setting it to something harsher
    #: is the misspecification experiment: the history the proxy is fitted on and
    #: the futures the agent is trained across come from one economy, and the
    #: deployment happens in another. The break lands **at the branch** -- every
    #: future forks from the same state under the same accumulated drift, and
    #: only the weather from there on differs -- so what a deployment measures is
    #: an online correction closing a real gap rather than absorbing sampling
    #: noise. Deploying onto futures drawn from the very generator the proxy was
    #: fit on leaves it nothing to correct, which is why the paired improvement
    #: over a frozen agent comes out near zero there.
    deploy_excitation: str | None = None
    #: how far each deployment future's economy is redrawn around its preset,
    #: as a multiplier on :class:`~economic_models.ground_truth.excitation.specs.ExcitationJitter`.
    #: Zero deploys onto the preset itself.
    #:
    #: This is the *other* half of the misspecification question, and the half
    #: :attr:`deploy_excitation` cannot ask. Leaving the preset alone and drawing
    #: more futures from it does not produce a new economy: an agent trained
    #: across thousands of futures and a proxy fitted on a history from the same
    #: generator have between them already seen that economy in full, so a
    #: deployment onto a further draw of it tests sampling luck. Jitter gives
    #: each deployment a *neighbouring* economy of comparable difficulty --
    #: median-neutral by construction, so the bank is not quietly a harder bank
    #: -- which the offline stack has genuinely never seen.
    #:
    #: Perturbed **per future**, not once for the bank: which economy is drawn is
    #: then part of what a rehearsal is a draw of, and the spread over rehearsals
    #: is a spread over worlds as well as over shocks, which is the honest object
    #: when the live run is one draw from it.
    deploy_jitter: float = 0.0
    #: how many rows of the history training episodes may fork from. One is the
    #: original behaviour -- every episode starts from the state the history ends
    #: in. Above one, the training bank is spread evenly over ``n_starts`` rows
    #: ending at that same last one, and an episode starts wherever its future
    #: was forked from.
    #:
    #: The end of a run is *one draw* from the economy's state distribution, and
    #: being one draw it is reliably an extreme of something -- across world seeds
    #: it lands at the 95th percentile of the historic bill rate, or the 99th of
    #: inflation, or the very minimum of ``ro``. Every training episode starting
    #: there conditions the policy on that corner, and under the delta instrument
    #: it conditions it on that corner's *lever positions* too, since the levers
    #: integrate from wherever the economy hands them over. Fifty steps cannot
    #: undo it: the persistent features (the rate block, ``GD/Y``, ``theta``)
    #: move through less than half their historic spread inside an episode, so a
    #: single-start agent never sees a low-rate or high-debt regime at all.
    #:
    #: Only the **training** bank is spread. The evaluation and deployment banks
    #: fork off the history's end as they always have, so a score is comparable
    #: across settings of this and the live deployment still starts where the
    #: history stops.
    n_starts: int = 1
    #: rows at the head of the history no training episode forks from. The proxy
    #: rollout and the observation both start an episode from a belief filtered
    #: over the prefix, and a filter given a short prefix is still at its prior;
    #: this is the burn-in that buys convergence before a row is usable as a start.
    start_burn_in: int = 100
    seed: int = 0  #: seeds the history and (offset) every future

    def __post_init__(self) -> None:
        """Validate the configuration eagerly, before any expensive generation."""
        for name in ("excitation", "deploy_excitation"):
            value = getattr(self, name)
            if value is not None and value not in EXCITATIONS:
                raise ValueError(
                    f"{name} must be one of {EXCITATIONS}, got {value!r}"
                )
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        for name in ("history_steps", "horizon", "n_train_futures"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        for name in ("n_eval_futures", "n_deploy_futures"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.deploy_jitter < 0.0:
            raise ValueError(f"deploy_jitter must be non-negative, got {self.deploy_jitter}")
        if self.n_starts < 1:
            raise ValueError(f"n_starts must be at least 1, got {self.n_starts}")
        if self.start_burn_in < 0:
            raise ValueError(
                f"start_burn_in must be non-negative, got {self.start_burn_in}"
            )
        if self.n_starts > 1 and self.start_burn_in >= self.history_steps - 1:
            raise ValueError(
                f"a {self.history_steps}-step history has no rows left to start from "
                f"after a burn-in of {self.start_burn_in}"
            )
        if self.n_starts > self.n_train_futures:
            raise ValueError(
                f"{self.n_starts} start points but only {self.n_train_futures} "
                "training futures to spread over them"
            )
        if self.deploys_differently and self.n_deploy_futures < 1:
            raise ValueError(
                "a deployment economy was configured but no bank was asked for: "
                "set n_deploy_futures"
            )

    @property
    def deploys_differently(self) -> bool:
        """Whether the deployment bank is drawn from anything but the plain preset."""
        return bool(
            self.deploy_jitter > 0.0
            or (
                self.deploy_excitation is not None
                and self.deploy_excitation != self.excitation
            )
        )

    @property
    def spec(self) -> GroundTruthSpec:
        """The ground-truth model this world is built on."""
        return ground_truth(self.model)

    def excitation_config(self) -> ExcitationConfig:
        """The excitation preset named by :attr:`excitation`."""
        return self.spec.excitation_config(self.excitation)

    def deploy_configs(self) -> list[ExcitationConfig] | None:
        """One excitation config per deployment future, or ``None`` for the preset.

        The preset named by :attr:`deploy_excitation` (or the world's own),
        redrawn once per future at :attr:`deploy_jitter`. Seeded off
        :attr:`seed` in a block of its own so the economies are reproducible and
        do not move when the number of training futures does.
        """
        if not self.deploys_differently:
            return None
        preset = self.spec.excitation_config(self.deploy_excitation or self.excitation)
        if self.deploy_jitter <= 0.0:
            return [preset] * self.n_deploy_futures
        jitter = ExcitationJitter().scaled(self.deploy_jitter)
        base = self.seed * 1_000_000 + 900_000
        return [
            preset.perturbed(np.random.default_rng(base + j), jitter)
            for j in range(self.n_deploy_futures)
        ]


@dataclass(frozen=True)
class BranchPoint:
    """A row of the history an episode may start from, and the state it is in.

    ``row`` indexes the history; ``state`` is the full internal ground-truth
    state there, which is what pins a solver, and the trailing level rows of the
    history up to ``row`` are what warm-starts a proxy and an observation. A
    future forked here (:class:`~economic_models.ground_truth.excitation.base.BranchCapture`)
    resumes the excitation exactly where this row left it, which is why a future
    belongs to its branch point and cannot be spliced onto another.
    """

    row: int  #: index into the history
    state: Mapping[str, float]  #: full internal ground-truth state at that row


@dataclass(frozen=True)
class Episode:
    """One exogenous future the agent is asked to steer through.

    ``params`` are the bank-visible :class:`~economic_models.variables.Parameters`
    per step and ``hidden`` the hidden structural parameters (invisible to the
    agent, but needed to drive the ground-truth model); ``index`` identifies the
    future for logging and replay; ``branch`` is where the economy stands when
    this future begins.

    Carrying the branch **on the episode** is what lets a bank be spread over
    several start points without anything downstream changing: a future and the
    state it forks from are drawn together, and every caller that already hands
    the environment a particular episode (an evaluation sweep, a recorded
    rollout, a deployment) goes on getting the start that episode was drawn for.
    """

    params: np.ndarray  # (horizon, n_params) visible exogenous path
    hidden: np.ndarray | None  # (horizon, n_hidden) hidden structural path
    index: int
    branch: BranchPoint

    def __len__(self) -> int:
        """The number of steps this episode's forcing lasts."""
        return len(self.params)


# -- the fiscal stabilizer -------------------------------------------------


class FiscalStabilizer:
    """GROWTH's countercyclical spending response, applied at rollout.

    A frozen :class:`~economic_models.run.Scenario` records ``GRg``
    drawn at full employment, so the automatic response to a slump is missing
    from it. This adds it back from the realised employment rate, exactly as
    :class:`~economic_models.ground_truth.models.growth.excitation.generator.GrowthExcitationProcess`
    would have::

        GRg = clip(GRg_frozen + gain * (1 - ER_prev), *bounds)

    (The frozen level is already clipped, so this differs from the generator only
    when the un-stabilized draw had itself hit a bound -- a rare, small offset.)
    Without this the agent trains in an economy stripped of its only fast
    stabilizer, which is neither the model nor the world the proxy was fit in.
    """

    def __init__(self, spec: SpendingResponse, param_names: tuple[str, ...]) -> None:
        """Wire the stabilizer to ``spec``'s response, bounds and target parameter.

        ``param_names`` is the column order of a parameter row, used once to find
        the column the response moves. Which column that is, and how the response
        is computed, are the spec's business -- so this works for any ground truth
        whose excitation carries one.
        """
        self.spec = spec
        self._target = param_names.index(spec.target)

    def apply(self, params: np.ndarray, er_prev: float) -> np.ndarray:
        """A copy of the parameter row with ``GRg`` responding to ``er_prev``."""
        stabilized = params.astype(float).copy()
        stabilized[self._target] = float(
            np.clip(stabilized[self._target] + self.spec.support(er_prev), *self.spec.bounds)
        )
        return stabilized

    def strip(self, params: np.ndarray, er_prev: float) -> np.ndarray:
        """The inverse of :meth:`apply`: back out the response ``er_prev`` caused.

        A *realised* run records ``GRg`` with the response already in it, because
        the generator put it there. A *frozen* future does not. Anything that
        learns from realised rows and is then replayed through :meth:`apply` --
        the exogenous forecast of :mod:`control.live.forcing` -- has to remove it
        on the way in, or the response is counted twice: once as the level the
        estimate was fit around, once as the term added back at rollout.

        Inexact by exactly the amount :meth:`apply` is: the clip is not
        invertible, so a row whose stabilized draw hit a bound comes back short.
        Rare and small, and the same rare and small offset in both directions.
        """
        raw = params.astype(float).copy()
        raw[self._target] -= self.spec.support(er_prev)
        return raw


class NullStabilizer:
    """The null object stabilizer: exogenous forcing passes through untouched.

    For ablations that ask what the agent does when it is the economy's *only*
    stabilizer -- and for a deployment that declines to assume it knows the
    fiscal reaction function at all (see
    :attr:`~control.live.deploy.LiveConfig.rollout_stabilizer`).
    """

    def apply(self, params: np.ndarray, er_prev: float) -> np.ndarray:
        """Return ``params`` unchanged."""
        return params.astype(float)

    def strip(self, params: np.ndarray, er_prev: float) -> np.ndarray:
        """Return ``params`` unchanged: nothing was added, nothing comes off."""
        return params.astype(float)


# -- the world -------------------------------------------------------------


class EpisodeBank:
    """A fixed pool of exogenous futures to draw episodes from."""

    def __init__(
        self,
        futures: list[Scenario],
        offset: int = 0,
        branches: Sequence[BranchPoint] | BranchPoint | None = None,
    ) -> None:
        """Hold ``futures`` as episodes numbered from ``offset``.

        ``branches`` says where each future forks from: one branch point for the
        whole bank (the usual case -- an evaluation or deployment bank all forks
        off the history's end) or one per future (a training bank spread over
        several start points).
        """
        if branches is None or isinstance(branches, BranchPoint):
            branches = [branches] * len(futures)
        elif len(branches) != len(futures):
            raise ValueError(
                f"{len(branches)} branch points for {len(futures)} futures"
            )
        self._episodes = [
            Episode(params=s.params, hidden=s.hidden, index=offset + j, branch=b)
            for j, (s, b) in enumerate(zip(futures, branches))
        ]

    @property
    def branch_rows(self) -> tuple[int, ...]:
        """The distinct history rows this bank's futures fork from, in order."""
        rows = {e.branch.row for e in self._episodes if e.branch is not None}
        return tuple(sorted(rows))

    def __len__(self) -> int:
        """The number of futures in the pool."""
        return len(self._episodes)

    def draw(self, rng: np.random.Generator) -> Episode:
        """One future drawn uniformly at random (training)."""
        return self._episodes[rng.integers(len(self._episodes))]

    def all(self) -> list[Episode]:
        """Every future in pool order (deterministic evaluation sweeps)."""
        return list(self._episodes)


@dataclass(frozen=True)
class TrainingWorld:
    """One history, the state it ends in, and the futures branching off it.

    ``history`` is the run the proxy is fit on and whose tail warm-starts every
    episode; ``branch`` is the full internal ground-truth state at its end (the
    common starting point); ``train_futures`` and ``eval_futures`` are the two
    disjoint banks of exogenous paths; ``deploy_futures`` is the third, possibly
    empty, bank a live deployment is rehearsed on
    (:attr:`WorldConfig.deploy_excitation`); ``stabilizer`` re-applies the fiscal
    response a frozen future cannot carry.
    """

    history: Run
    branch: Mapping[str, float]
    train_futures: EpisodeBank
    eval_futures: EpisodeBank
    stabilizer: FiscalStabilizer | NullStabilizer
    hidden_names: tuple[str, ...]
    config: WorldConfig
    deploy_futures: EpisodeBank = field(default_factory=lambda: EpisodeBank([]))

    @property
    def dt(self) -> float:
        """Length of one step in years."""
        return self.config.dt

    @property
    def branch_point(self) -> BranchPoint:
        """The history's end: the branch an evaluation or deployment starts from."""
        return BranchPoint(row=len(self.history) - 1, state=self.branch)

    @property
    def start_rows(self) -> tuple[int, ...]:
        """The history rows training episodes fork from, in order."""
        return self.train_futures.branch_rows

    def window(self, rows: int, end: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The ``rows`` level rows of the history ending at ``end``: a warm start.

        ``end`` is an *exclusive* bound defaulting to the whole history, so the
        default is the trailing window it has always been; a branch point passes
        its own row plus one and gets the prefix ending there instead.
        """
        end = len(self.history) if end is None else end
        if not 0 < rows <= end <= len(self.history):
            raise ValueError(
                f"a {len(self.history)}-row history cannot supply {rows} rows "
                f"ending at {end}"
            )
        h = self.history
        return (
            h.states[end - rows : end],
            h.params[end - rows : end],
            h.actions[end - rows : end],
        )


def start_rows(config: WorldConfig) -> tuple[int, ...]:
    """The history rows training futures fork from, ending at the last one.

    ``n_starts`` rows spread evenly over the history past its burn-in, the last
    of which is always the history's end -- so raising ``n_starts`` *adds* start
    points to the one that was always there rather than moving it, and
    ``n_starts=1`` is exactly the original single branch.
    """
    end = config.history_steps - 1
    if config.n_starts == 1:
        return (end,)
    rows = np.linspace(config.start_burn_in, end, config.n_starts)
    return tuple(sorted({int(round(r)) for r in rows}))


def build_world(config: WorldConfig | None = None, *, verbose: bool = True) -> TrainingWorld:
    """Simulate the history once and fork the futures that branch off its end.

    Only the history solves the structural model; the futures are drawn from the
    excitation process alone. ``verbose`` prints a one-line progress report.

    With :attr:`WorldConfig.n_starts` above one the *training* futures are dealt
    round-robin over several rows of the history (:func:`start_rows`) and forked
    from the excitation as it stood at each -- the evaluation and deployment
    banks still fork off the end, and their seeds do not move, so a score is
    comparable across settings of ``n_starts``.
    """
    config = config or WorldConfig()
    excitation = config.excitation_config()
    generator = config.spec.generator(excitation, dt=config.dt, on_collapse="truncate")

    n_offline = config.n_train_futures + config.n_eval_futures
    n_futures = n_offline + config.n_deploy_futures
    # Disjoint seed blocks: the first block trains, the next evaluates, the tail
    # deploys, and all three are a deterministic function of ``config.seed``.
    base = config.seed * 1_000_000
    seeds = [base + 1 + j for j in range(n_futures)]
    # The deployment bank is the only one that may break: reconfiguring at the
    # branch leaves every future forking from the same drift under new weather.
    deploy_configs = config.deploy_configs()
    configs = (
        None if deploy_configs is None else [None] * n_offline + list(deploy_configs)
    )
    # Dealt round-robin rather than in contiguous blocks, so that any prefix of
    # the training bank is spread over every start point -- a bank truncated for
    # a short run is then still a multi-start bank.
    rows = start_rows(config)
    end = config.history_steps - 1
    branch_at = [rows[j % len(rows)] for j in range(config.n_train_futures)]
    branch_at += [end] * (n_futures - config.n_train_futures)

    if verbose:
        print(
            f"world: simulating {config.history_steps} steps at dt={config.dt} "
            f"({config.excitation} excitation, seed={config.seed})..."
        )
    history, futures, captures = generator.generate_branched(
        config.history_steps,
        config.horizon,
        n_futures,
        branch_at=branch_at,
        seed=config.seed,
        continuation_seeds=seeds,
        continuation_configs=configs,
    )
    branch = captures[end].state
    branches = [BranchPoint(row=t, state=captures[t].state) for t in branch_at]
    if len(history) < config.history_steps:
        raise RuntimeError(
            f"the history collapsed after {len(history)}/{config.history_steps} steps "
            f"(seed={config.seed}); pick another seed or a calmer excitation"
        )
    if verbose:
        jittered = (
            f", each redrawn at jitter {config.deploy_jitter:g}"
            if config.deploy_jitter
            else ""
        )
        deployed = (
            ""
            if not config.n_deploy_futures
            else f" / {config.n_deploy_futures} deploy futures "
                 f"({config.deploy_excitation or config.excitation} excitation"
                 f"{jittered})"
        )
        started = (
            ""
            if len(rows) == 1
            else f" from {len(rows)} start rows (history {rows[0]}..{rows[-1]})"
        )
        print(
            f"world: history of {len(history)} steps "
            f"({history.dampened_steps} dampened), "
            f"{config.n_train_futures} train{started} / {config.n_eval_futures} "
            f"eval futures{deployed}"
        )

    terminal = BranchPoint(row=end, state=branch)
    return TrainingWorld(
        history=history,
        branch=branch,
        train_futures=EpisodeBank(
            futures[: config.n_train_futures],
            branches=branches[: config.n_train_futures],
        ),
        eval_futures=EpisodeBank(
            futures[config.n_train_futures : n_offline],
            offset=config.n_train_futures,
            branches=terminal,
        ),
        deploy_futures=EpisodeBank(
            futures[n_offline:], offset=n_offline, branches=terminal
        ),
        stabilizer=FiscalStabilizer(
            excitation.gov_spending, config.spec.interface.parameters.names()
        ),
        hidden_names=excitation.hidden_names,
        config=config,
    )


def unstabilized(world: TrainingWorld) -> TrainingWorld:
    """The same world with the fiscal stabilizer switched off (ablation)."""
    return replace(world, stabilizer=NullStabilizer())
