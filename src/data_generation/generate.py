"""Generate a GROWTH-model dataset of main histories and branching continuations.

The dataset is an ensemble of ``N`` excited *main* runs of the ground-truth
GROWTH model (fully simulated histories); from the end of each main run, ``M``
alternative-future *continuation scenarios* branch off. A scenario is only the
exogenous forcing (params + hidden) an in-control agent would act within -- the
agent supplies the actions and the model produces the states at rollout, so a
scenario fixes neither. Each group therefore stores the main run, the full
internal *branch state* the main run ends in (the shared start every scenario is
rolled out from), and one scenario per continuation (see
:meth:`~economic_models.ground_truth.growth.excitation.generator.GrowthRunGenerator.generate_with_continuations`).
The ``N`` groups are split into training and testing data at the group level, so
a main run and all its continuations always stay on the same side of the split.

Each group is an independent unit of work, so groups are generated in parallel
across processes. Only the main runs solve the GROWTH model -- a ``pysolve``
fixed-point solver over NumPy scalars, CPU-bound and not a fit for the GPU (which
the learned proxies, not this ground-truth generation, use). The continuation
scenarios need no solve at all (pure excitation draws), so multiprocessing across
groups is the lever that scales generation here.

Run it as a module::

    uv run python -m data_generation                       # default dataset
    uv run python -m data_generation --n-runs 100 --n-continuations 32
    uv run python -m data_generation --excitation realistic --dt 0.0833

The output lands under ``data/`` (next to ``src/``): ``data/train`` and
``data/test``, one ``group_XXXX/`` directory each, plus a ``data/manifest.json``
describing the whole dataset. Reload a run with
:func:`data_generation.storage.load_run`.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace

import numpy as np

from economic_models.ground_truth import GrowthExcitationConfig, GrowthRunGenerator

from data_generation import storage
from data_generation.config import DatasetConfig


#: Attempts a group may burn under ``on_collapse="resample"`` before giving up.
#: Collapses are a small-probability event per run, so exhausting this many
#: independent seeds signals a systematically unstable configuration, not bad luck.
MAX_RESAMPLE_ATTEMPTS = 25


@dataclass(frozen=True)
class _GroupTask:
    """The reproducible recipe for one group's generation, shipped to a worker.

    Carries the group's split and index (which fix its output directory) and its
    global run index (which fixes its seed stream, see :func:`_attempt_seeds`),
    so a worker needs only this plus the shared :class:`DatasetConfig` to
    produce the group.
    """

    index: int  #: the group's index within its split
    split: str  #: "train" or "test"
    run_index: int  #: the group's global index across both splits (seed-tree key)


@dataclass(frozen=True)
class _GroupResult:
    """A worker's summary of one finished group (the arrays are written to disk)."""

    index: int
    split: str
    main_length: int
    main_collapsed: bool
    n_scenarios: int
    scenario_length: int
    dampened_steps: int
    resamples: int  #: collapsed attempts discarded before this group succeeded


def _build_generator(config: DatasetConfig) -> GrowthRunGenerator:
    """Construct the run generator for ``config``'s excitation regime and timestep."""
    excitation = (
        GrowthExcitationConfig.realistic()
        if config.excitation == "realistic"
        else GrowthExcitationConfig.default()
    )
    return GrowthRunGenerator(
        excitation,
        dt=config.dt,
        burn_in=config.burn_in,
        # Under "resample" the retry loop in generate_group needs the collapse
        # surfaced as a truncated run's flag, not an exception.
        on_collapse="truncate" if config.on_collapse == "resample" else config.on_collapse,
    )


def plan_tasks(config: DatasetConfig) -> list[_GroupTask]:
    """Derive every group's split, per-split index and global run index."""
    n_train = config.n_train
    tasks: list[_GroupTask] = []
    for run_index in range(config.n_runs):
        split = "train" if run_index < n_train else "test"
        index = run_index if split == "train" else run_index - n_train
        tasks.append(_GroupTask(index, split, run_index))
    return tasks


def _attempt_seeds(
    config: DatasetConfig, run_index: int, attempt: int
) -> tuple[int, list[int]]:
    """The main-run seed and continuation seeds for one attempt at one group.

    Seeds come from a :class:`numpy.random.SeedSequence` tree rooted at
    ``config.base_seed``: the node at ``spawn_key=(run_index, attempt)`` yields
    the main-run seed and spawns one child per continuation. This gives
    well-separated, high-quality seed streams that are fully reproducible from
    the base seed alone and independent of the worker count, completion order
    and how many attempts other groups burned -- a resample simply moves to the
    next ``attempt`` node of the same group.
    """
    seq = np.random.SeedSequence(config.base_seed, spawn_key=(run_index, attempt))
    main_seed = int(seq.generate_state(1, dtype=np.uint32)[0])
    cont_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seq.spawn(config.n_continuations)
    ]
    return main_seed, cont_seeds


def generate_group(
    task: _GroupTask, config: DatasetConfig, generator: GrowthRunGenerator
) -> _GroupResult:
    """Generate one group with ``generator`` and write its runs under ``config.output_dir``.

    Saves three kinds of artifact in the group's directory: the fully-simulated
    main run (``main.npz``), the full internal branch state it ends in
    (``branch.npz``, the shared start every scenario rolls out from), and one
    ``scenario_XXX.npz`` per continuation holding just the exogenous forcing.
    Returns a small summary; the heavy arrays stay on disk rather than travelling
    back to the parent process.

    Under ``on_collapse="resample"`` a collapsed main run is discarded and the
    whole group is regenerated from the next attempt's seed node until a
    full-length run comes out -- a collapsed end state cannot be stepped from,
    so it would make a useless branch point for the group's continuations.
    """
    resample = config.on_collapse == "resample"
    for attempt in range(MAX_RESAMPLE_ATTEMPTS if resample else 1):
        main_seed, cont_seeds = _attempt_seeds(config, task.run_index, attempt)
        main_run, scenarios, branch_state = generator.generate_with_continuations(
            config.main_length,
            config.continuation_length,
            config.n_continuations,
            seed=main_seed,
            continuation_seeds=cont_seeds,
        )
        if not (resample and main_run.collapsed):
            break
    else:
        raise RuntimeError(
            f"{task.split}/group_{task.index:04d} collapsed on all "
            f"{MAX_RESAMPLE_ATTEMPTS} resample attempts; the configuration "
            f"looks systematically unstable"
        )

    out, split, index = config.output_dir, task.split, task.index
    storage.save_run(storage.main_path(out, split, index), main_run)
    storage.save_branch_state(storage.branch_path(out, split, index), branch_state)
    for j, scenario in enumerate(scenarios):
        storage.save_scenario(storage.scenario_path(out, split, index, j), scenario)

    return _GroupResult(
        index=task.index,
        split=task.split,
        main_length=len(main_run),
        main_collapsed=main_run.collapsed,
        n_scenarios=len(scenarios),
        scenario_length=len(scenarios[0]) if scenarios else 0,
        dampened_steps=main_run.dampened_steps,
        resamples=attempt,
    )


# One generator per worker process, built lazily on first use and reused for
# every group that process handles -- building the pysolve model is the one
# non-trivial per-model cost, so it is paid once per worker, not once per group.
_WORKER_GENERATOR: GrowthRunGenerator | None = None
_WORKER_CONFIG: DatasetConfig | None = None


def _worker_init(config: DatasetConfig) -> None:
    """Process-pool initialiser: stash the config and build this worker's generator."""
    global _WORKER_GENERATOR, _WORKER_CONFIG
    _WORKER_CONFIG = config
    _WORKER_GENERATOR = _build_generator(config)


def _worker_generate(task: _GroupTask) -> _GroupResult:
    """Pool task entry point: generate ``task`` with this worker's cached generator."""
    assert _WORKER_GENERATOR is not None and _WORKER_CONFIG is not None
    return generate_group(task, _WORKER_CONFIG, _WORKER_GENERATOR)


def generate_dataset(config: DatasetConfig, *, verbose: bool = True) -> dict:
    """Generate the whole dataset described by ``config`` and write it to disk.

    Fans the group tasks out across ``config.resolved_workers()`` processes (or
    runs them inline when a single worker is requested), then writes the manifest
    and returns it. The manifest records the config, the interface column order
    and a per-group summary, so the dataset is self-describing on disk.
    """
    from economic_models.ground_truth.growth import (
        GrowthActions as Actions,
        GrowthParameters as Parameters,
        GrowthState as State,
    )

    tasks = plan_tasks(config)
    workers = min(config.resolved_workers(), len(tasks))
    if verbose:
        print(
            f"Generating {config.n_runs} main runs x {config.n_continuations} "
            f"continuations ({config.n_train} train / {config.n_test} test groups), "
            f"main {config.main_length} steps, continuation {config.continuation_length} "
            f"steps, dt={config.dt:g}, excitation={config.excitation}, "
            f"{workers} worker(s)."
        )

    start = time.perf_counter()
    results: list[_GroupResult] = []
    if workers <= 1:
        _worker_init(config)
        for task in tasks:
            results.append(_worker_generate(task))
            if verbose:
                _report(results[-1], len(results), len(tasks))
    else:
        # 'spawn' keeps macOS/Windows behaviour identical to Linux and avoids
        # inheriting a half-initialised interpreter; workers rebuild their own
        # generator in _worker_init.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(config,),
        ) as pool:
            futures = {pool.submit(_worker_generate, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if verbose:
                    _report(result, len(results), len(tasks))

    elapsed = time.perf_counter() - start
    manifest = _build_manifest(config, results, elapsed, State, Parameters, Actions)
    storage.write_manifest(config.output_dir, manifest)
    if verbose:
        collapsed = sum(r.main_collapsed for r in results)
        resamples = sum(r.resamples for r in results)
        print(
            f"Done in {elapsed:.1f}s. Wrote {len(results)} groups to "
            f"{config.output_dir}/ ({collapsed} main run(s) collapsed early, "
            f"{resamples} collapsed attempt(s) resampled)."
        )
    return manifest


def _report(result: _GroupResult, done: int, total: int) -> None:
    """Print a one-line progress note for a finished group."""
    flag = " [collapsed]" if result.main_collapsed else ""
    if result.resamples:
        flag += f" [resampled x{result.resamples}]"
    print(
        f"  [{done:>{len(str(total))}}/{total}] {result.split}/group_"
        f"{result.index:04d}: main {result.main_length} steps, "
        f"{result.n_scenarios} scenarios x {result.scenario_length} steps{flag}"
    )


def _build_manifest(
    config: DatasetConfig,
    results: list[_GroupResult],
    elapsed: float,
    State,
    Parameters,
    Actions,
) -> dict:
    """Assemble the self-describing dataset manifest from the finished groups."""
    excitation = _build_generator(config).config
    by_split: dict[str, list[dict]] = {"train": [], "test": []}
    for result in sorted(results, key=lambda r: (r.split, r.index)):
        by_split[result.split].append(
            {
                "index": result.index,
                "main_length": result.main_length,
                "main_collapsed": result.main_collapsed,
                "n_scenarios": result.n_scenarios,
                "scenario_length": result.scenario_length,
                "dampened_steps": result.dampened_steps,
                "resamples": result.resamples,
            }
        )
    return {
        "config": storage.config_to_dict(config),
        # Per group: a fully-simulated main run (states/params/actions/hidden), the
        # full internal branch state it ends in, and one scenario per continuation
        # holding only the exogenous forcing (params + hidden) -- the agent supplies
        # the actions and the model produces the states at rollout, so a scenario
        # fixes neither. Continuation scenarios are generated without solving the
        # model; their GRg has no employment-feedback stabilizer term (see Scenario).
        "artifacts": {
            "main": "Run: states, params, actions, hidden (+ diagnostics)",
            "branch": "full internal model state at the main run's end (name->value)",
            "scenario": "Scenario: params, hidden (+ diagnostics); no states/actions",
        },
        "columns": {
            "states": list(State.names()),
            "params": list(Parameters.names()),
            "actions": list(Actions.names()),
            "hidden": list(excitation.hidden_names),
        },
        "splits": {
            "train": {"n_groups": config.n_train, "groups": by_split["train"]},
            "test": {"n_groups": config.n_test, "groups": by_split["test"]},
        },
        "elapsed_seconds": elapsed,
    }


def _parse_args(argv: list[str] | None = None) -> DatasetConfig:
    """Parse the CLI into a :class:`DatasetConfig` (defaults mirror the dataclass)."""
    defaults = DatasetConfig()
    ap = argparse.ArgumentParser(
        prog="python -m data_generation",
        description="Generate a GROWTH-model dataset of excited main runs and "
        "their branching continuations, split into train/test data.",
    )
    ap.add_argument("--n-runs", type=int, default=defaults.n_runs,
                    help="number of independent main (historic) runs")
    ap.add_argument("--n-continuations", type=int, default=defaults.n_continuations,
                    help="continuations branching from the end of each main run")
    ap.add_argument("--main-length", type=int, default=defaults.main_length,
                    help="recorded steps in each main run")
    ap.add_argument("--continuation-length", type=int, default=defaults.continuation_length,
                    help="recorded steps in each continuation")
    ap.add_argument("--dt", type=float, default=defaults.dt,
                    help="step length in years (1 annual, 0.0833 monthly)")
    ap.add_argument("--excitation", choices=("default", "realistic"),
                    default=defaults.excitation,
                    help="'default' calm calibration or 'realistic' with volatility/crises")
    ap.add_argument("--burn-in", type=int, default=defaults.burn_in,
                    help="years run unrecorded to settle each main run onto its path")
    ap.add_argument("--on-collapse", choices=("raise", "truncate", "resample"),
                    default=defaults.on_collapse,
                    help="what to do on an unrecoverable solver failure "
                    "('resample' regenerates the group with fresh seeds)")
    ap.add_argument("--train-fraction", type=float, default=defaults.train_fraction,
                    help="fraction of main runs (with their continuations) used for training")
    ap.add_argument("--base-seed", type=int, default=defaults.base_seed,
                    help="root seed; all run/continuation seeds derive from it")
    ap.add_argument("--output-dir", default=defaults.output_dir,
                    help="directory the dataset is written under (next to src/)")
    ap.add_argument("--workers", type=int, default=defaults.workers,
                    help="worker processes (default: CPU count)")
    args = ap.parse_args(argv)
    return replace(
        defaults,
        n_runs=args.n_runs,
        n_continuations=args.n_continuations,
        main_length=args.main_length,
        continuation_length=args.continuation_length,
        dt=args.dt,
        excitation=args.excitation,
        burn_in=args.burn_in,
        on_collapse=args.on_collapse,
        train_fraction=args.train_fraction,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        workers=args.workers,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parse arguments and generate the dataset."""
    config = _parse_args(argv)
    generate_dataset(config)


if __name__ == "__main__":
    main()
