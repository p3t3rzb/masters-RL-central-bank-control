"""Generation of GROWTH-model datasets: main histories and branching scenarios.

This package drives the ground-truth
:class:`~economic_models.ground_truth.ExcitedRunGenerator` to produce an ensemble
of excited *main* runs (fully simulated histories) and, from the end of each, a
set of alternative-future *continuation scenarios* -- the exogenous forcing an
in-control agent would act within -- storing them under ``data/`` split into
training and testing data. :class:`~data_generation.config.DatasetConfig` fixes
*which* dataset; :func:`~data_generation.generate.generate_dataset` produces it
(in parallel across groups); :mod:`~data_generation.storage` reads and writes the
runs, branch states and scenarios on disk.
"""

from economic_models.ground_truth.run import Run, Scenario

from data_generation.config import DatasetConfig
from data_generation.generate import generate_dataset, plan_tasks
from data_generation.storage import (
    load_branch_state,
    load_run,
    load_scenario,
    read_manifest,
    save_branch_state,
    save_run,
    save_scenario,
)

__all__ = [
    "DatasetConfig",
    "Run",
    "Scenario",
    "generate_dataset",
    "plan_tasks",
    "load_run",
    "load_scenario",
    "load_branch_state",
    "save_run",
    "save_scenario",
    "save_branch_state",
    "read_manifest",
]
