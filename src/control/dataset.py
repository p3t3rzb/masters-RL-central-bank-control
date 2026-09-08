"""Reading a :mod:`data_generation` group back as a :class:`TrainingWorld`.

:mod:`control.world` builds its setting by *simulating* one history and forking
futures off it, which is what an offline training run wants: a single economy,
studied in depth. Anything that asks whether a policy generalises wants the
opposite -- many economies, each shallow -- and that is exactly what
:mod:`data_generation` has already written to disk: a thousand independent main
runs, each with the internal model state it ends in and a hundred continuation
:class:`~economic_models.run.Scenario`\\ s branching from there, split
into training and testing groups at the group level.

The two are the same object seen twice. A dataset group holds a history, a branch
state and a bank of futures; a :class:`~control.world.TrainingWorld` *is* a
history, a branch state and a bank of futures. So nothing in the control stack has
to learn about ``data/``: this module composes the pieces
:func:`~data_generation.storage.load_run` and friends return into the world the
environment, the drivers and :func:`~control.dsac.train.evaluate` already speak,
and a group off disk drives them unchanged.

What the dataset cannot store, the manifest identifies: the fiscal stabilizer and
the hidden-parameter column order both belong to the excitation preset the dataset
was generated under, which is recorded there by name and rebuilt here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from economic_models.ground_truth import GROWTH_INTERFACE, GrowthExcitationConfig
from data_generation.storage import (
    branch_path,
    load_branch_state,
    load_run,
    load_scenario,
    main_path,
    read_manifest,
    scenario_path,
)

from control.world import (
    BranchPoint,
    EpisodeBank,
    FiscalStabilizer,
    TrainingWorld,
    WorldConfig,
)

#: The dataset splits a group may be read from.
SPLITS = ("train", "test")


# -- the manifest ----------------------------------------------------------


@lru_cache(maxsize=8)
def dataset_manifest(root: str | Path) -> dict[str, Any]:
    """The dataset's ``manifest.json``, read once per root.

    Every group of a dataset shares its timestep, its lengths and its excitation
    preset, so the manifest is read once and cached rather than re-parsed for each
    of a thousand groups (it is a few hundred kilobytes of per-group summaries).
    """
    return read_manifest(root)


def group_count(root: str | Path, split: str) -> int:
    """How many groups ``split`` holds, from the manifest."""
    _check_split(split)
    return int(dataset_manifest(str(root))["splits"][split]["n_groups"])


def dataset_dt(root: str | Path) -> float:
    """The dataset's timestep in years, from the manifest."""
    return float(dataset_manifest(str(root))["config"]["dt"])


# -- one group as a world --------------------------------------------------


def load_world(
    root: str | Path, split: str, index: int, *, futures: int | None = None
) -> TrainingWorld:
    """Group ``index`` of ``split``, as a world the control stack can run.

    ``futures`` caps how many of the group's continuations are loaded, which is
    the knob that matters: a group's scenarios are exchangeable draws off one
    branch state, so a prefix of them is a smaller bank of the same thing, and
    reading eight instead of a hundred is eight per cent of the file I/O.

    The futures land in :attr:`~control.world.TrainingWorld.eval_futures` and the
    training bank is left empty. A dataset group has exactly one branch point --
    the end of its main run -- and no notion of a train/eval divide *within*
    itself: the divide is between groups, and it is the dataset's own directory
    split. Anything scoring a policy here therefore asks for the ``"eval"`` split
    of the environment, which is what
    :func:`~control.dsac.train.build_truth_env` defaults to.

    The returned :attr:`~control.world.TrainingWorld.config` describes the group
    faithfully except for ``n_train_futures``, which is one because
    :class:`~control.world.WorldConfig` requires at least one and nothing reads it
    here; ``dt`` is the field that is actually consulted at rollout.
    """
    _check_split(split)
    root = str(root)
    manifest = dataset_manifest(root)
    config = manifest["config"]

    history = load_run(main_path(root, split, index))
    branch = load_branch_state(branch_path(root, split, index))
    available = int(config["n_continuations"])
    n = available if futures is None else min(int(futures), available)
    if n < 1:
        raise ValueError(f"a world needs at least one future, asked for {futures}")
    scenarios = [load_scenario(scenario_path(root, split, index, j)) for j in range(n)]

    excitation = _excitation(manifest)
    terminal = BranchPoint(row=len(history) - 1, state=branch)
    return TrainingWorld(
        history=history,
        branch=branch,
        train_futures=EpisodeBank([]),
        eval_futures=EpisodeBank(scenarios, branches=terminal),
        stabilizer=FiscalStabilizer(
            excitation.gov_spending, GROWTH_INTERFACE.parameters.names()
        ),
        hidden_names=excitation.hidden_names,
        config=WorldConfig(
            dt=float(config["dt"]),
            history_steps=len(history),
            horizon=int(config["continuation_length"]),
            n_train_futures=1,
            n_eval_futures=n,
            excitation=str(config["excitation"]),
            seed=int(config["base_seed"]),
        ),
    )


# -- internals -------------------------------------------------------------


def _check_split(split: str) -> None:
    """Reject a split the dataset does not have, by name rather than by path."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")


def _excitation(manifest: dict[str, Any]) -> GrowthExcitationConfig:
    """The excitation preset the dataset was generated under.

    Two things a group's ``.npz`` files do not carry come from here: the fiscal
    stabilizer's gain and bounds, and the names of the hidden-parameter columns.
    The latter is checked against the column order the manifest recorded, because
    a mismatch would not fail -- it would quietly feed the solver one structural
    parameter's path under another's name.
    """
    name = str(manifest["config"]["excitation"])
    excitation: GrowthExcitationConfig = getattr(GrowthExcitationConfig, name)()
    recorded = tuple(manifest["columns"]["hidden"])
    if excitation.hidden_names != recorded:
        raise ValueError(
            f"the {name!r} preset records hidden parameters "
            f"{excitation.hidden_names}, but the dataset was written with "
            f"{recorded}; it was generated by a different version of the model"
        )
    return excitation
