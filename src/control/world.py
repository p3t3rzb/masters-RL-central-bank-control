"""The training world: one historic run and the futures branching off its end.

The whole control experiment is set in a single economy. A ground-truth
:class:`~economic_models.ground_truth.run.Run` is simulated once -- the *history*
-- and is the only data the proxy is ever fit on. From the state that history
ends in, :meth:`~economic_models.ground_truth.excitation.base.ExcitedRunGenerator.generate_with_continuations`
forks the excitation process into many independent *futures*: pure exogenous
forcing paths (:class:`~economic_models.ground_truth.run.Scenario`), drawn without
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

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from economic_models.ground_truth import (
    GROWTH_INTERFACE,
    GrowthExcitationConfig,
    GrowthRunGenerator,
    Run,
    Scenario,
)
from economic_models.ground_truth.growth.excitation.specs import GovSpendingSpec

#: The excitation presets a world may be built with.
EXCITATIONS = ("default", "realistic")


@dataclass(frozen=True)
class WorldConfig:
    """How the history and its futures are drawn.

    ``dt`` is the length of one step in years and ``history_steps`` the length of
    the single run the proxy is fit on; ``horizon`` is how many steps one episode
    lasts. ``n_train_futures`` and ``n_eval_futures`` size the two disjoint banks
    of exogenous paths (the eval bank is never seen during training).
    ``excitation`` names the :class:`~economic_models.ground_truth.GrowthExcitationConfig`
    preset, and ``seed`` fixes the whole draw.
    """

    dt: float = 0.25  #: length of one step in years (0.25 quarterly, 1/12 monthly)
    history_steps: int = 500  #: recorded steps of the run the proxy is fit on
    horizon: int = 50  #: steps per episode
    n_train_futures: int = 2000  #: exogenous paths available for training episodes
    n_eval_futures: int = 100  #: held-out exogenous paths, disjoint seeds
    excitation: str = "realistic"  #: which GrowthExcitationConfig preset
    seed: int = 0  #: seeds the history and (offset) every future

    def __post_init__(self) -> None:
        """Validate the configuration eagerly, before any expensive generation."""
        if self.excitation not in EXCITATIONS:
            raise ValueError(
                f"excitation must be one of {EXCITATIONS}, got {self.excitation!r}"
            )
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        for name in ("history_steps", "horizon", "n_train_futures"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.n_eval_futures < 0:
            raise ValueError(f"n_eval_futures must be non-negative, got {self.n_eval_futures}")

    def excitation_config(self) -> GrowthExcitationConfig:
        """The excitation preset named by :attr:`excitation`."""
        return getattr(GrowthExcitationConfig, self.excitation)()


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

    A frozen :class:`~economic_models.ground_truth.run.Scenario` records ``GRg``
    drawn at full employment, so the automatic response to a slump is missing
    from it. This adds it back from the realised employment rate, exactly as
    :class:`~economic_models.ground_truth.growth.excitation.generator.GrowthExcitationProcess`
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


class NullStabilizer:
    """The null object stabilizer: exogenous forcing passes through untouched.

    For ablations that ask what the agent does when it is the economy's *only*
    stabilizer.
    """

    def apply(self, params: np.ndarray, er_prev: float) -> np.ndarray:
        """Return ``params`` unchanged."""
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
    disjoint banks of exogenous paths; ``stabilizer`` re-applies the fiscal
    response a frozen future cannot carry.
    """

    history: Run
    branch: Mapping[str, float]
    train_futures: EpisodeBank
    eval_futures: EpisodeBank
    stabilizer: FiscalStabilizer | NullStabilizer
    hidden_names: tuple[str, ...]
    config: WorldConfig

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

    n_futures = config.n_train_futures + config.n_eval_futures
    # Disjoint seed blocks: the first block trains, the tail block evaluates, and
    # both are a deterministic function of ``config.seed``.
    base = config.seed * 1_000_000
    seeds = [base + 1 + j for j in range(n_futures)]

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
    )
    if len(history) < config.history_steps:
        raise RuntimeError(
            f"the history collapsed after {len(history)}/{config.history_steps} steps "
            f"(seed={config.seed}); pick another seed or a calmer excitation"
        )
    if verbose:
        print(
            f"world: history of {len(history)} steps "
            f"({history.dampened_steps} dampened), "
            f"{config.n_train_futures} train / {config.n_eval_futures} eval futures"
        )

    return TrainingWorld(
        history=history,
        branch=branch,
        train_futures=EpisodeBank(futures[: config.n_train_futures]),
        eval_futures=EpisodeBank(
            futures[config.n_train_futures :], offset=config.n_train_futures
        ),
        stabilizer=FiscalStabilizer(
            excitation.gov_spending, GROWTH_INTERFACE.parameters.names()
        ),
        hidden_names=excitation.hidden_names,
        config=config,
    )


def unstabilized(world: TrainingWorld) -> TrainingWorld:
    """The same world with the fiscal stabilizer switched off (ablation)."""
    return replace(world, stabilizer=NullStabilizer())
