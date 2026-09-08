"""The optimal run: what the mandate could have earned with the truth in hand.

Every policy in the deployment figures is handicapped by not knowing the
economy: the agent by having been trained inside a proxy, the references by
being rules. This module computes the bar none of them can be measured without
-- the return a policymaker could have earned on the *same* future with full
knowledge of the ground truth -- by simply searching for it. Like
:class:`~control.live.forcing.OracleForcing`, it is lab-only information and a
**bound, not a method**: the live bank cannot rerun its economy, but the lab
can, and reusing the ground truth is exactly what makes the bound computable.

The search is well-posed for a specific reason. The structural solver is
deterministic and an episode freezes its forcing (the fiscal stabilizer reacts
to the realised employment rate, but deterministically), so an entire future is
a pure function of the lever path. An open-loop path is therefore a complete
policy here -- feedback has nothing to add when nothing is random -- and "the
optimal run" is a finite-dimensional optimisation over lever paths rather than
a reinforcement-learning problem. A genetic algorithm suits its shape: the
objective is a black box with cliffs (the collapse corridor), gradients are
unavailable, and good solutions are smooth paths that a crossover at a point in
*time* recombines meaningfully.

Two choices keep the genome small and the bound honest:

* the path is parametrized by **knots** every few periods, interpolated
  linearly between them. Levers with a finite speed have no use for white
  noise, so the interpolation is a smoothness prior that divides the search
  space by the knot spacing rather than a restriction that costs return;
* the instrument is the references' free one by default
  (:func:`~control.dsac.train.free_instrument`): a bound on *any* policy should
  not inherit the agent's speed limit. ``instrument="agent"`` computes the
  tighter bound of what the agent's own instrument could have reached, which is
  the fairer number to hold the *agent* against.

The result is logged as a :class:`~control.live.deploy.RunRecord` named
``"optimal"`` on the same future under the same seed, so every figure and every
paired comparison treats it as one more policy.

The search is by far the dearest thing in a rehearsal -- every fitness
evaluation is a full ground-truth run. Two things keep it affordable. A
generation's genomes are independent, so they are evaluated concurrently
across a process pool (:attr:`OracleConfig.workers`) -- concurrency the pool
can exploit fully because the search itself never leaves the parent, which is
also why the result is identical at any worker count. And results can be
cached (:attr:`OracleConfig.cache`): the optimum depends on the future, the horizon
and the mandate, never on the agent, so it survives any change to the
deployment configuration. The cache key carries the world, the episode, the
horizon and the search settings; it does **not** see the mandate's weights or
the action box, so a change to either means clearing the cache by hand.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from parallel import guarded_pool, terminate

from control.dsac.train import (
    build_truth_env,
    calibration_actions,
    free_instrument,
    reset_policy,
    taylor_policy,
)
from control.env import CentralBankEnv
from control.live.deploy import LiveConfig, RunRecord, _OutcomeTracker
from control.world import Episode, TrainingWorld

if TYPE_CHECKING:
    from control.dsac.train import TrainingResult

#: The instrument the bound is computed under: the references' free one, or the
#: agent's own (see the module docstring).
INSTRUMENTS = ("free", "agent")


@dataclass(frozen=True)
class OracleConfig:
    """The search's knobs: budget, genome shape, and the operators' scales.

    The budget is ``population * generations`` ground-truth runs per future in
    the worst case, which dwarfs everything else in a rehearsal -- the four
    existing policies cost four runs per future between them. The defaults are
    a compromise sized to plateau on 200-step futures; :attr:`patience` is what
    keeps a converged search from spending the rest of its budget standing
    still.
    """

    population: int = 32  #: genomes per generation
    generations: int = 40  #: generations before the search stops regardless
    #: periods between knots. The genome holds the lever positions at the knots
    #: and the path is linear between them, so this is the smoothness prior: at
    #: 4 the levers turn at most yearly (dt=0.25), and the genome for a
    #: 200-step future is 51 knots x 3 levers.
    knot_every: int = 4
    elite: int = 4  #: best genomes copied unchanged into the next generation
    tournament: int = 3  #: genomes drawn per parent selection
    crossover: float = 0.9  #: probability a child crosses two parents at all
    mutation: float = 0.3  #: per-knot probability of a Gaussian nudge
    sigma: float = 0.3  #: opening mutation scale, in normalised lever units
    sigma_final: float = 0.05  #: the scale the anneal closes on
    #: generations without improvement before the search stops early. The
    #: objective is deterministic, so "no improvement" is exact rather than a
    #: noisy read, and waiting longer buys nothing but wall clock.
    patience: int = 10
    instrument: str = "free"  #: one of :data:`INSTRUMENTS`
    seed: int = 0  #: seeds the operators; the objective itself has no noise
    #: processes evaluating a generation's genomes concurrently. The
    #: evaluations are independent full ground-truth runs and the solver is
    #: pure Python, so this is the one dial that moves the wall clock by
    #: multiples. 0 sizes the pool to the machine (two cores spared), 1 runs
    #: in-process. **Not** part of the cache key, deliberately: every random
    #: decision is taken in the parent off one stream, so the search finds the
    #: same optimum at any worker count and the two are the same experiment.
    workers: int = 0
    #: directory to cache best-found runs in, or ``None`` to search every time.
    #: See the module docstring for what the key does and does not protect.
    cache: str | None = None

    def __post_init__(self) -> None:
        """Reject a configuration that cannot be run, before anything is built."""
        if self.instrument not in INSTRUMENTS:
            raise ValueError(
                f"instrument must be one of {INSTRUMENTS}, got {self.instrument!r}"
            )
        if self.elite >= self.population:
            raise ValueError(
                f"elite {self.elite} must be smaller than population {self.population}"
            )
        if self.knot_every < 1:
            raise ValueError(f"knot_every must be at least 1, got {self.knot_every}")


# -- the genome and its evaluation -------------------------------------------


def _knot_times(steps: int, knot_every: int) -> np.ndarray:
    """The step indices the genome's knots sit at: every ``knot_every``, plus the end.

    The last knot is pinned to the final step so the interpolation never
    extrapolates; on a horizon that is not a multiple of the spacing the last
    segment is simply shorter.
    """
    times = np.arange(0, steps, knot_every)
    if times[-1] != steps - 1:
        times = np.append(times, steps - 1)
    return times


def _positions(genome: np.ndarray, knot_times: np.ndarray, steps: int) -> np.ndarray:
    """The per-step normalised lever path a genome describes, ``(steps, levers)``."""
    knots = genome.reshape(len(knot_times), -1)
    t = np.arange(steps)
    return np.column_stack(
        [np.interp(t, knot_times, knots[:, j]) for j in range(knots.shape[1])]
    )


def _run(
    env: CentralBankEnv,
    world: TrainingWorld,
    episode: Episode,
    positions: np.ndarray,
    *,
    seed: int,
) -> RunRecord:
    """One ground-truth run down a lever path, logged like every other policy.

    The path is stated in normalised positions and driven through
    :meth:`~control.env.CentralBankEnv.toward`, the same way a rule stated in
    levels is: under the free instrument each period lands on its knot exactly,
    under the agent's it travels there at instrument speed -- either way the
    run scores what the instrument actually did, which is what makes the bound
    honest under both settings.
    """
    record = RunRecord(name="optimal", episode=episode.index)
    obs, _ = env.reset(seed=seed, episode=episode)
    tracker = _OutcomeTracker(env, world)
    for t in range(1, len(positions) + 1):
        levels = env.to_actions(positions[t - 1]).to_dict()
        obs, reward, terminated, truncated, info = env.step(env.toward(levels))
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


# -- evaluating a generation, concurrently -----------------------------------
#
# A generation's genomes are independent full ground-truth runs and the solver
# is pure Python, so processes -- not threads -- are what buys wall clock. The
# genetic algorithm itself stays in the parent: workers only ever map a lever
# path to its logged run, so the search's random stream, and therefore its
# result, is identical at any worker count.

#: The worker's environment, built once per process by :func:`_worker_init`.
#: A module global because a process pool has no other place to keep per-worker
#: state; nothing outside these two functions may touch it.
_WORKER: tuple[CentralBankEnv, TrainingWorld, Episode, int] | None = None


def _worker_init(
    world: TrainingWorld,
    observer,
    reward,
    config,
    horizon: int,
    episode: Episode,
    seed: int,
) -> None:
    """Build one worker's own ground-truth environment, once.

    The expensive part of an evaluation is the run, not the environment, but an
    environment is stateful and cannot be shared -- so each worker gets its
    own, built from the same (pickled) world, observer and mandate the parent's
    was, and resets it per evaluation exactly as the parent would.
    """
    global _WORKER
    _WORKER = (
        build_truth_env(world, observer, reward, config, seed=seed, horizon=horizon),
        world,
        episode,
        seed,
    )


def _worker_run(positions: np.ndarray) -> RunRecord:
    """One fitness evaluation inside a worker: a lever path to its logged run."""
    assert _WORKER is not None, "worker used before _worker_init"
    env, world, episode, seed = _WORKER
    return _run(env, world, episode, positions, seed=seed)


class _Evaluator:
    """Maps lever paths to their logged runs, in-process or across a pool.

    Owns the parent's environment either way -- the Taylor warm start and the
    final re-run of the best genome go through it -- and, above one worker, a
    spawned process pool whose workers each hold an environment of their own.
    ``close`` must be called (``optimal_run`` does, in a ``finally``) or the
    pool's processes outlive the search -- though a worker whose parent dies
    before it can (a ``SIGKILL``) stops itself, see :mod:`parallel.pool`.
    """

    def __init__(
        self,
        env: CentralBankEnv,
        world: TrainingWorld,
        episode: Episode,
        oracle: OracleConfig,
        *,
        observer,
        reward,
        config,
        horizon: int,
        seed: int,
    ) -> None:
        """Bind to the parent's ``env`` and spin up the pool when asked to."""
        self.env = env
        self.world = world
        self.episode = episode
        self.seed = seed
        workers = oracle.workers if oracle.workers else max(1, (os.cpu_count() or 2) - 2)
        workers = min(workers, oracle.population)
        self.pool: ProcessPoolExecutor | None = None
        if workers > 1:
            self.pool = guarded_pool(
                workers,
                initializer=_worker_init,
                initargs=(world, observer, reward, config, horizon, episode, seed),
            )

    def evaluate(self, paths: list[np.ndarray]) -> list[RunRecord]:
        """The logged run of every path, in order."""
        if self.pool is None:
            return [
                _run(self.env, self.world, self.episode, p, seed=self.seed)
                for p in paths
            ]
        return list(self.pool.map(_worker_run, paths))

    def close(self) -> None:
        """Stop the pool, if one was started.

        :func:`~parallel.pool.terminate` rather than ``shutdown``: a search that
        is being torn down has its evaluations in hand already, and a search that
        is unwinding from an interrupt would otherwise sit here evaluating the
        rest of the generation it no longer has a use for.
        """
        if self.pool is not None:
            terminate(self.pool)
            self.pool = None


# -- seeding the population --------------------------------------------------


def _taylor_path(
    env: CentralBankEnv,
    result: TrainingResult,
    episode: Episode,
    steps: int,
    *,
    seed: int,
) -> np.ndarray:
    """The lever path the Taylor rule actually walks, as normalised positions.

    One extra ground-truth run, spent on the best warm start available: the
    rule is the strongest reference, so a population seeded with its realised
    path starts the search at the bar instead of below it. If the rule's run
    collapses, the path holds its last position from there -- a usable genome
    even when the rule itself was not.
    """
    policy = taylor_policy(
        env, result.observer, dt=result.config.dt,
        pi_target=result.config.pi_target,
    )
    obs, _ = env.reset(seed=seed, episode=episode)
    reset_policy(policy)
    path = np.tile(env.to_normalised(calibration_actions()), (steps, 1))
    for t in range(steps):
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if terminated:
            path[t:] = path[t - 1] if t else path[t]
            break
        path[t] = env.to_normalised(info["actions"].to_dict())
        if truncated:
            path[t + 1 :] = path[t]
            break
    return path


def _seed_population(
    env: CentralBankEnv,
    result: TrainingResult,
    episode: Episode,
    oracle: OracleConfig,
    knot_times: np.ndarray,
    steps: int,
    rng: np.random.Generator,
    *,
    seed: int,
) -> list[np.ndarray]:
    """The opening population: the references' paths, their perturbations, noise.

    Two exact seeds (the calibration constants and the Taylor rule's realised
    path), then perturbed copies of them, then uniform genomes -- so the search
    opens already at the references' level and spends its budget on beating
    them rather than on rediscovering that the box has a sensible middle.
    """
    calibration = np.tile(
        env.to_normalised(calibration_actions()), (len(knot_times), 1)
    ).ravel()
    taylor = _taylor_path(env, result, episode, steps, seed=seed)[knot_times].ravel()
    population = [calibration, taylor]
    while len(population) < max(2, (oracle.population + 1) // 2):
        base = population[rng.integers(2)]
        population.append(
            np.clip(base + rng.normal(0.0, oracle.sigma, size=base.shape), -1.0, 1.0)
        )
    while len(population) < oracle.population:
        population.append(rng.uniform(-1.0, 1.0, size=calibration.shape))
    return population[: oracle.population]


# -- the genetic algorithm ---------------------------------------------------


def _select(
    fitness: np.ndarray, oracle: OracleConfig, rng: np.random.Generator
) -> int:
    """Tournament selection: the fittest of ``tournament`` uniform draws."""
    picks = rng.integers(len(fitness), size=oracle.tournament)
    return int(picks[np.argmax(fitness[picks])])


def _child(
    population: list[np.ndarray],
    fitness: np.ndarray,
    oracle: OracleConfig,
    sigma: float,
    n_knots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One offspring: time-coherent crossover, then knot-wise Gaussian mutation.

    The crossover point is a *knot*, so a child is one parent's opening decades
    grafted onto the other's remainder -- the recombination that makes sense
    for a control path, where a good beginning and a good end are separable
    things. Mutation nudges whole knots rather than single genes for the same
    reason: a lever position at one period is not meaningful apart from its
    neighbours.
    """
    levers = len(population[0]) // n_knots
    parent = population[_select(fitness, oracle, rng)]
    genome = parent.reshape(n_knots, levers).copy()
    if oracle.crossover > 0 and rng.random() < oracle.crossover and n_knots > 1:
        other = population[_select(fitness, oracle, rng)].reshape(n_knots, levers)
        cut = int(rng.integers(1, n_knots))
        genome[cut:] = other[cut:]
    mutate = rng.random(n_knots) < oracle.mutation
    genome[mutate] += rng.normal(0.0, sigma, size=(int(mutate.sum()), levers))
    return np.clip(genome.ravel(), -1.0, 1.0)


def _optimise(
    env: CentralBankEnv,
    evaluator: _Evaluator,
    result: TrainingResult,
    episode: Episode,
    oracle: OracleConfig,
    steps: int,
    *,
    seed: int,
    verbose: bool,
) -> tuple[np.ndarray, RunRecord]:
    """Search for the best lever path; return it and its logged run.

    Elitist and deterministic in its objective: the best genome is carried
    unchanged and never re-evaluated, so the best fitness is monotone and the
    early stop (:attr:`OracleConfig.patience`) reads exact stagnation rather
    than noise. The generation-by-generation best goes into the returned
    record's events, which is where the figures' machinery already looks for a
    run's discrete history.
    """
    rng = np.random.default_rng(oracle.seed + seed)
    knot_times = _knot_times(steps, oracle.knot_every)
    population = _seed_population(
        env, result, episode, oracle, knot_times, steps, rng, seed=seed
    )
    records = evaluator.evaluate(
        [_positions(g, knot_times, steps) for g in population]
    )
    fitness = np.array([r.total for r in records])
    best = int(np.argmax(fitness))
    best_genome, best_fit = population[best].copy(), float(fitness[best])
    events = [{"step": 0, "kind": "generation", "generation": 0, "best": best_fit}]
    stale = 0

    for g in range(1, oracle.generations + 1):
        # A geometric anneal from sigma to sigma_final: wide moves while the
        # search is placing the path, fine ones once it is shaping it.
        frac = g / max(1, oracle.generations - 1)
        sigma = oracle.sigma * (oracle.sigma_final / oracle.sigma) ** min(1.0, frac)
        order = np.argsort(fitness)[::-1][: oracle.elite]
        children = [
            _child(population, fitness, oracle, sigma, len(knot_times), rng)
            for _ in range(oracle.population - oracle.elite)
        ]
        child_records = evaluator.evaluate(
            [_positions(c, knot_times, steps) for c in children]
        )
        population = [population[i] for i in order] + children
        fitness = np.concatenate(
            [fitness[order], [r.total for r in child_records]]
        )

        top = int(np.argmax(fitness))
        if fitness[top] > best_fit + 1e-9:
            best_genome, best_fit = population[top].copy(), float(fitness[top])
            stale = 0
        else:
            stale += 1
        events.append(
            {"step": 0, "kind": "generation", "generation": g, "best": best_fit}
        )
        if verbose and (g % 5 == 0 or stale == 0):
            print(
                f"    oracle gen {g:>3}/{oracle.generations}: "
                f"best {best_fit:9.1f}  median {np.median(fitness):9.1f}"
            )
        if stale >= oracle.patience:
            if verbose:
                print(f"    oracle converged after {g} generations")
            break

    record = _run(
        env, result.world, episode, _positions(best_genome, knot_times, steps),
        seed=seed,
    )
    record.events_ = events
    return best_genome, record


# -- the cache ---------------------------------------------------------------


def _cache_file(
    oracle: OracleConfig,
    result: TrainingResult,
    episode: Episode,
    steps: int,
    seed: int,
) -> tuple[Path, str]:
    """Where a search this shaped, on this future, would have been stored.

    The key is everything the optimum is a function of: the world that drew the
    episode, the episode itself, the horizon, the instrument, and the search's
    own settings and seed (a different budget finds a different optimum, and
    conflating them would report the cheap search's bound as the dear one's).
    Deliberately absent is everything about the *agent* -- proxy, encoders,
    training budget -- which is what makes the cache worth having: every
    deployment configuration compared on the same futures shares one optimum.
    """
    cfg = result.config
    key = "|".join(
        str(v)
        for v in (
            cfg.excitation, cfg.deploy_excitation, cfg.deploy_jitter,
            cfg.world_seed, cfg.dt, cfg.history_steps, cfg.delta_rate,
            episode.index, steps, seed,
            oracle.population, oracle.generations, oracle.knot_every,
            oracle.elite, oracle.tournament, oracle.crossover, oracle.mutation,
            oracle.sigma, oracle.sigma_final, oracle.patience,
            oracle.instrument, oracle.seed,
        )
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    assert oracle.cache is not None
    return Path(oracle.cache) / f"optimal_ep{episode.index}_{digest}.json", key


def _load(path: Path) -> RunRecord | None:
    """The cached best run at ``path``, or ``None`` when there is none."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    record = RunRecord(name="optimal", episode=payload["episode"])
    record.steps_ = payload["steps"]
    record.events_ = payload["events"]
    record.collapsed = payload["collapsed"]
    return record


def _store(path: Path, key: str, genome: np.ndarray, record: RunRecord) -> None:
    """Write the best run (and the genome that produced it) under ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "key": key,
                "episode": record.episode,
                "genome": genome.tolist(),
                "steps": record.steps_,
                "events": record.events_,
                "collapsed": record.collapsed,
            },
            default=float,
        )
    )


# -- the public entry point --------------------------------------------------


def optimal_run(
    result: TrainingResult,
    episode: Episode,
    live: LiveConfig,
    oracle: OracleConfig | None = None,
    *,
    seed: int = 0,
    verbose: bool = False,
) -> RunRecord:
    """The best run a GA can find on ``episode`` with the ground truth in hand.

    Same environment builder, same seed and same per-period record as
    :func:`~control.live.deploy.run_reference`, so the returned ``"optimal"``
    record pairs with every other policy on this future. The environment is
    rebuilt fresh here -- the search resets it hundreds of times, and nothing
    another caller holds may be disturbed by that.
    """
    oracle = oracle or OracleConfig()
    steps = min(live.live_steps, len(episode))

    path = key = None
    if oracle.cache is not None:
        path, key = _cache_file(oracle, result, episode, steps, seed)
        cached = _load(path)
        if cached is not None:
            if verbose:
                print(f"  optimal: cached ({path.name})")
            return cached

    config = (
        result.config
        if oracle.instrument == "agent"
        else free_instrument(result.config)
    )
    env = build_truth_env(
        result.world, result.observer, result.env.reward, config,
        seed=seed, horizon=live.live_steps,
    )
    evaluator = _Evaluator(
        env, result.world, episode, oracle,
        observer=result.observer, reward=result.env.reward, config=config,
        horizon=live.live_steps, seed=seed,
    )
    try:
        genome, record = _optimise(
            env, evaluator, result, episode, oracle, steps, seed=seed,
            verbose=verbose,
        )
    finally:
        evaluator.close()
    if path is not None and key is not None:
        _store(path, key, genome, record)
    return record
