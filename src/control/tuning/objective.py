"""What a candidate is worth: rolling it through the ground truth, economy by economy.

A hyperparameter is only as good as the return it earns, and the return that
matters is the one the **structural** model pays -- not a proxy's estimate of it.
So the objective here is the plainest thing it could be: build the environment a
dataset group describes, run the candidate policy through that group's futures
with :func:`~control.dsac.train.evaluate`, and average over groups. The loss is
the negative of that, because optimisers minimise.

Two properties of the arrangement are worth stating, because the whole protocol
rests on them:

* **It is paired.** A :class:`Fold` fixes which groups and how many futures, and
  every candidate is scored on that same fold in that same order. The ground truth
  is a deterministic solver, so two candidates differ only in what they did, never
  in what they were asked to survive.
* **It is separable by group.** A group is an independent economy with its own
  history, so scoring one is an independent unit of work. That is what lets the
  fold be walked in blocks (the pruner's resource) and spread over processes (the
  wall-clock), without either changing the number that comes out.

The expensive part is the solver: roughly ten milliseconds a period, so a
fifty-period episode costs about half a second and everything else here -- reading
a group off disk, fitting the observer on its history -- disappears against it.
Worlds are cached per worker anyway, since a fold is walked once per candidate and
a hundred candidates walk the same prefix.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import numpy as np

from economic_models.ground_truth import GROWTH_INTERFACE
from parallel import guarded_pool, terminate

from control.dataset import group_count, load_world
from control.drivers import GroundTruthDriver
from control.dsac.train import evaluate
from control.env import CentralBankEnv, EnvConfig
from control.rewards import MandateReward
from control.world import TrainingWorld

if TYPE_CHECKING:  # a policy calls into this module's types, not the other way
    from control.tuning.policies import TunablePolicy


# -- configuration ---------------------------------------------------------


@dataclass(frozen=True)
class TuningConfig:
    """Everything about *how* a candidate is scored, as opposed to *which* one.

    ``root`` is the dataset directory and ``futures`` how many of each group's
    continuations an episode bank is cut to. ``block`` is how many groups are
    scored between two reports to the pruner -- the granularity at which a hopeless
    candidate can be abandoned. ``seed`` fixes the fold draw and the per-group
    evaluation seeds; ``workers`` is the process count, ``0`` meaning "all but two
    cores" and ``1`` meaning "in this process", which is what a debugger wants.

    The remaining fields are the environment's, and default to the values the rest
    of the control stack uses: ``horizon`` overrides the episode cap (``None``
    taking the dataset's own continuation length), ``collapse_penalty`` is charged
    per remaining step of a collapsed episode, ``delta_rate`` is the instrument
    speed for a policy that does *not* ask for a free instrument, ``pi_target`` is
    the mandate's inflation target and ``iterations`` the solver's iteration cap.
    """

    root: str = "data"  #: dataset directory, as written by :mod:`data_generation`
    futures: int = 8  #: continuations per group an episode bank is cut to
    block: int = 8  #: groups scored between two reports to the pruner
    seed: int = 0  #: fixes the fold draw and every per-group evaluation seed
    workers: int = 0  #: processes; 0 = all but two cores, 1 = in-process
    horizon: int | None = None  #: episode cap, or the dataset's continuation length
    collapse_penalty: float = -25.0  #: charged per remaining step of a collapse
    delta_rate: float = 0.4  #: instrument speed for a non-free-instrument policy
    pi_target: float = 0.02  #: the mandate's inflation target
    iterations: int = 1000  #: solver iteration cap per period

    def __post_init__(self) -> None:
        """Validate eagerly, before a fold is drawn or a solver is built."""
        for name in ("futures", "block", "iterations"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.workers < 0:
            raise ValueError(f"workers must be non-negative, got {self.workers}")
        if self.horizon is not None and self.horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {self.horizon}")

    def resolved_workers(self) -> int:
        """The process count to use: ``workers``, or all but two cores when zero.

        Two are left alone because the parent process and the machine both keep
        working while a study runs; the solver is single-threaded NumPy scalar
        arithmetic, so there is nothing to gain from oversubscribing.
        """
        if self.workers:
            return self.workers
        return max(1, (os.cpu_count() or 1) - 2)


# -- folds -----------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """A fixed set of dataset groups, in a fixed order, at a fixed bank size.

    The unit every score in this package is taken over. Order matters: a fold is
    walked in blocks and truncated by pruning, so two candidates that were
    abandoned at different depths are still comparable on the prefix they share.
    """

    split: str  #: which dataset split the groups are read from
    groups: tuple[int, ...]  #: group indices, in walk order
    futures: int  #: continuations per group

    def __len__(self) -> int:
        """The number of groups in the fold."""
        return len(self.groups)

    def blocks(self, size: int) -> Iterator["Fold"]:
        """The fold cut into consecutive chunks of ``size`` groups.

        Disjoint rather than cumulative: a caller walking these accumulates the
        per-group returns it has seen, so no group is ever scored twice, and the
        running mean after ``k`` chunks is the score of the fold's first ``k *
        size`` groups.
        """
        if size < 1:
            raise ValueError(f"block size must be at least 1, got {size}")
        for start in range(0, len(self.groups), size):
            yield replace(self, groups=self.groups[start : start + size])


def make_folds(
    config: TuningConfig, *, groups: int, val_groups: int
) -> tuple[Fold, Fold]:
    """The search sequence and the validation fold, drawn from the training split.

    ``groups + val_groups`` distinct training groups are drawn without replacement
    and shuffled once under :attr:`TuningConfig.seed`; the first ``groups`` become
    the sequence every candidate walks, and the rest become the fold the finalists
    are re-scored on. They are disjoint by construction, which is the point: the
    best of many noisy estimates is optimistic, so the configuration that goes on
    to the test split has to be chosen on economies the search never saw.
    """
    if groups < 1 or val_groups < 1:
        raise ValueError(
            f"both folds need at least one group, got {groups} and {val_groups}"
        )
    available = group_count(config.root, "train")
    if groups + val_groups > available:
        raise ValueError(
            f"asked for {groups} + {val_groups} training groups but the dataset "
            f"has {available}"
        )
    rng = np.random.default_rng(config.seed)
    drawn = rng.choice(available, size=groups + val_groups, replace=False)
    return (
        Fold("train", tuple(int(i) for i in drawn[:groups]), config.futures),
        Fold("train", tuple(int(i) for i in drawn[groups:]), config.futures),
    )


def make_fold(
    config: TuningConfig, split: str, groups: int, *, futures: int | None = None
) -> Fold:
    """A fold of ``groups`` groups drawn from ``split``, for a one-off score.

    How the test fold is built. Drawn under a seed derived from the split's name
    rather than :attr:`TuningConfig.seed` alone, so the test groups do not shadow
    the training draw when the two splits happen to be the same size.
    """
    available = group_count(config.root, split)
    if not 1 <= groups <= available:
        raise ValueError(
            f"asked for {groups} {split} groups but the dataset has {available}"
        )
    rng = np.random.default_rng([config.seed, *(ord(c) for c in split)])
    drawn = rng.choice(available, size=groups, replace=False)
    return Fold(split, tuple(int(i) for i in drawn), futures or config.futures)


# -- scoring one group -----------------------------------------------------


def build_env(
    world: TrainingWorld, observer: Any, config: TuningConfig, *, free_instrument: bool
) -> CentralBankEnv:
    """The ground-truth environment a group's futures are scored in.

    ``free_instrument`` gives the levers a speed of one box per period
    (``delta_rate = 1/dt``), which is how the reference rules are scored throughout
    this repository: the instrument's speed is part of the *agent's* problem, and a
    rule stated in levels slowed to the agent's instrument would be a different
    rule than the one in the book. See
    :func:`~control.dsac.train.free_instrument`.
    """
    return CentralBankEnv(
        GroundTruthDriver(world, iterations=config.iterations),
        world.eval_futures,
        MandateReward(config.pi_target),
        observer,
        GROWTH_INTERFACE,
        EnvConfig(
            collapse_penalty=config.collapse_penalty,
            horizon=config.horizon or world.config.horizon,
            delta_rate=1.0 / world.dt if free_instrument else config.delta_rate,
        ),
        seed=config.seed,
    )


def score_group(
    policy: "TunablePolicy",
    prepared: Any,
    split: str,
    index: int,
    futures: int,
    config: TuningConfig,
) -> dict[str, float]:
    """Score one candidate on one economy: the atom of every number here.

    Loads the group (cached per process), asks the policy for its observer and its
    built form, and hands both to :func:`~control.dsac.train.evaluate` over the
    group's whole bank. ``repeats`` stays at one because the ground truth is a
    deterministic solver -- a second roll of the same future would agree exactly.
    """
    world = _world(config.root, split, index, futures)
    observer = policy.observer(prepared, world)
    env = build_env(world, observer, config, free_instrument=policy.free_instrument)
    built = policy.build(prepared, env, observer, world, dt=world.dt)
    result = evaluate(env, built, futures, seed=config.seed + 1000 * index, repeats=1)
    return {
        "return": result["return"],
        "collapse_rate": result["collapse_rate"],
    }


# -- scoring a fold --------------------------------------------------------


class Evaluator:
    """Scores candidates over folds, on a pool of processes it owns.

    Built once per study and reused: the pool's workers cache the worlds they have
    loaded, so the second candidate to walk a prefix pays no disk at all, and a
    study walks the same prefix once per trial.

    A context manager, and safe to use as a plain object followed by
    :meth:`close`. With ``workers = 1`` there is no pool and everything runs in the
    calling process, which is what makes a failing candidate debuggable.
    """

    def __init__(self, config: TuningConfig) -> None:
        """Start the pool ``config`` asks for, or none at all when it asks for one."""
        self.config = config
        self._workers = config.resolved_workers()
        self._pool: ProcessPoolExecutor | None = None
        if self._workers > 1:
            self._pool = guarded_pool(self._workers)

    @property
    def workers(self) -> int:
        """How many processes this evaluator scores on."""
        return self._workers

    def __enter__(self) -> "Evaluator":
        """Enter the context; the pool is already up."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Shut the pool down on the way out."""
        self.close()

    def close(self) -> None:
        """Stop the pool. Idempotent.

        :func:`~parallel.pool.terminate` rather than ``shutdown``: a study being
        torn down has the scores it asked for already, and one unwinding from an
        interrupt would otherwise sit here scoring the rest of the fold.
        """
        if self._pool is not None:
            terminate(self._pool)
            self._pool = None

    def score(
        self, policy: "TunablePolicy", prepared: Any, fold: Fold
    ) -> dict[str, Any]:
        """Score one candidate over ``fold``: the mean return and its breakdown.

        ``loss`` is the negative mean return (optimisers minimise), ``return_std``
        the spread *across economies* of their per-group returns -- which is the
        honest error bar on a fold this size -- and ``per_group`` the raw returns
        in fold order, kept so a study's report can show where a candidate won and
        where it lost.
        """
        args = [
            (policy, prepared, fold.split, index, fold.futures, self.config)
            for index in fold.groups
        ]
        if self._pool is None:
            results = [score_group(*a) for a in args]
        else:
            results = list(self._pool.map(_score_group_task, args))

        returns = np.array([r["return"] for r in results], dtype=float)
        collapses = np.array([r["collapse_rate"] for r in results], dtype=float)
        return {
            "loss": -float(returns.mean()),
            "return": float(returns.mean()),
            "return_std": float(returns.std()),
            "collapse_rate": float(collapses.mean()),
            "groups": len(fold),
            "per_group": {int(i): float(r) for i, r in zip(fold.groups, returns)},
        }


def score(
    policy: "TunablePolicy", prepared: Any, fold: Fold, config: TuningConfig
) -> dict[str, Any]:
    """Score one candidate over ``fold`` on a pool built for this call alone.

    The convenience form of :meth:`Evaluator.score`, for a one-off number. A study
    should hold an :class:`Evaluator` open instead, so its workers keep the worlds
    they have already loaded.
    """
    with Evaluator(config) as evaluator:
        return evaluator.score(policy, prepared, fold)


# -- worker internals ------------------------------------------------------

#: One loaded world per ``(root, split, index, futures)`` this process has seen.
#: A fold is walked once per candidate and a study runs many, so a group is read
#: off disk once per worker and its history is fitted against thereafter.
_WORLDS: dict[tuple[str, str, int, int], TrainingWorld] = {}


def _world(root: str, split: str, index: int, futures: int) -> TrainingWorld:
    """Group ``index`` of ``split``, from this process's cache or from disk."""
    key = (root, split, index, futures)
    world = _WORLDS.get(key)
    if world is None:
        world = load_world(root, split, index, futures=futures)
        _WORLDS[key] = world
    return world


def _score_group_task(args: Sequence[Any]) -> dict[str, float]:
    """Unpack a worker task and score it; :meth:`Evaluator.score`'s pool entry."""
    return score_group(*args)
