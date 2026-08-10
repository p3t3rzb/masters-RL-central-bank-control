"""The training world: one historic run and the futures branching off its end.

The whole control experiment is set in a single economy. A ground-truth
:class:`~economic_models.run.Run` is simulated once -- the *history*
-- and is the only data the proxy is ever fit on. From the state that history
ends in, :meth:`~economic_models.ground_truth.excitation.base.ExcitedRunGenerator.generate_with_continuations`
forks the excitation process into many independent *futures*: pure exogenous
forcing paths (:class:`~economic_models.run.Scenario`), drawn without
solving any model, because the states depend on the actions an agent has not
chosen yet. Those futures are this package's episodes.

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
from typing import Mapping

import numpy as np

from economic_models.ground_truth import (
    GROWTH_INTERFACE,
    GrowthExcitationConfig,
    GrowthRunGenerator,
)
from economic_models.ground_truth.excitation.specs import ExcitationJitter
from economic_models.ground_truth.models.growth.excitation.specs import GovSpendingSpec
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
    ``excitation`` names the :class:`~economic_models.ground_truth.GrowthExcitationConfig`
    preset, and ``seed`` fixes the whole draw.

    ``deploy_excitation`` and ``n_deploy_futures`` add a **third** bank, drawn
    under a different preset: see :attr:`deploy_excitation`.
    """

    dt: float = 0.25  #: length of one step in years (0.25 quarterly, 1/12 monthly)
    history_steps: int = 500  #: recorded steps of the run the proxy is fit on
    horizon: int = 50  #: steps per episode
    n_train_futures: int = 2000  #: exogenous paths available for training episodes
    n_eval_futures: int = 100  #: held-out exogenous paths, disjoint seeds
    excitation: str = "realistic"  #: which GrowthExcitationConfig preset
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

    def excitation_config(self) -> GrowthExcitationConfig:
        """The excitation preset named by :attr:`excitation`."""
        return getattr(GrowthExcitationConfig, self.excitation)()

    def deploy_configs(self) -> list[GrowthExcitationConfig] | None:
        """One excitation config per deployment future, or ``None`` for the preset.

        The preset named by :attr:`deploy_excitation` (or the world's own),
        redrawn once per future at :attr:`deploy_jitter`. Seeded off
        :attr:`seed` in a block of its own so the economies are reproducible and
        do not move when the number of training futures does.
        """
        if not self.deploys_differently:
            return None
        preset = getattr(
            GrowthExcitationConfig, self.deploy_excitation or self.excitation
        )()
        if self.deploy_jitter <= 0.0:
            return [preset] * self.n_deploy_futures
        jitter = ExcitationJitter().scaled(self.deploy_jitter)
        base = self.seed * 1_000_000 + 900_000
        return [
            preset.perturbed(np.random.default_rng(base + j), jitter)
            for j in range(self.n_deploy_futures)
        ]


@dataclass(frozen=True)
class Episode:
    """One exogenous future the agent is asked to steer through.

    ``params`` are the bank-visible :class:`~economic_models.variables.Parameters`
    per step and ``hidden`` the hidden structural parameters (invisible to the
    agent, but needed to drive the ground-truth model); ``index`` identifies the
    future for logging and replay.
    """

    params: np.ndarray  # (horizon, n_params) visible exogenous path
    hidden: np.ndarray | None  # (horizon, n_hidden) hidden structural path
    index: int

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

    def __init__(self, spec: GovSpendingSpec, param_names: tuple[str, ...]) -> None:
        """Wire the stabilizer to ``spec``'s gain and bounds.

        ``param_names`` is the column order of a parameter row, used once to find
        the ``GRg`` column.
        """
        self.spec = spec
        self._grg = param_names.index("GRg")

    def apply(self, params: np.ndarray, er_prev: float) -> np.ndarray:
        """A copy of the parameter row with ``GRg`` responding to ``er_prev``."""
        stabilized = params.astype(float).copy()
        stabilized[self._grg] = float(
            np.clip(
                stabilized[self._grg] + self.spec.stabilizer * (1.0 - er_prev),
                *self.spec.bounds,
            )
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
        raw[self._grg] -= self.spec.stabilizer * (1.0 - er_prev)
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

    def __init__(self, futures: list[Scenario], offset: int = 0) -> None:
        """Hold ``futures`` as episodes numbered from ``offset``."""
        self._episodes = [
            Episode(params=s.params, hidden=s.hidden, index=offset + j)
            for j, s in enumerate(futures)
        ]

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

    def window(self, rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The trailing ``rows`` level rows of the history: the warm start."""
        if rows > len(self.history):
            raise ValueError(
                f"history has {len(self.history)} rows, need {rows} to warm-start"
            )
        h = self.history
        return h.states[-rows:], h.params[-rows:], h.actions[-rows:]


def build_world(config: WorldConfig | None = None, *, verbose: bool = True) -> TrainingWorld:
    """Simulate the history once and fork the futures that branch off its end.

    Only the history solves the structural model; the futures are drawn from the
    excitation process alone. ``verbose`` prints a one-line progress report.
    """
    config = config or WorldConfig()
    excitation = config.excitation_config()
    generator = GrowthRunGenerator(
        excitation, dt=config.dt, on_collapse="truncate"
    )

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

    if verbose:
        print(
            f"world: simulating {config.history_steps} steps at dt={config.dt} "
            f"({config.excitation} excitation, seed={config.seed})..."
        )
    history, futures, branch = generator.generate_with_continuations(
        config.history_steps,
        config.horizon,
        n_futures,
        seed=config.seed,
        continuation_seeds=seeds,
        continuation_configs=configs,
    )
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
        print(
            f"world: history of {len(history)} steps "
            f"({history.dampened_steps} dampened), "
            f"{config.n_train_futures} train / {config.n_eval_futures} eval futures"
            f"{deployed}"
        )

    return TrainingWorld(
        history=history,
        branch=branch,
        train_futures=EpisodeBank(futures[: config.n_train_futures]),
        eval_futures=EpisodeBank(
            futures[config.n_train_futures : n_offline], offset=config.n_train_futures
        ),
        deploy_futures=EpisodeBank(futures[n_offline:], offset=n_offline),
        stabilizer=FiscalStabilizer(
            excitation.gov_spending, GROWTH_INTERFACE.parameters.names()
        ),
        hidden_names=excitation.hidden_names,
        config=config,
    )


def unstabilized(world: TrainingWorld) -> TrainingWorld:
    """The same world with the fiscal stabilizer switched off (ablation)."""
    return replace(world, stabilizer=NullStabilizer())
