"""The on-disk format: every artifact round-trips losslessly through ``save``/``load``.

Optional diagnostics are stored only when present and come back as ``None``
when absent, so both the fully populated and the bare forms are checked.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from data_generation import storage
from data_generation.config import DatasetConfig
from economic_models.run import Run, Scenario


def _run(*, full: bool) -> Run:
    """A run with every optional field set (``full``) or left at its default."""
    rng = np.random.default_rng(0)
    steps = 6
    optional = (
        dict(
            hidden=rng.normal(size=(steps, 2)),
            seed=17,
            dampened_steps=2,
            volatility=rng.uniform(size=steps),
            crisis_intensity=rng.uniform(size=steps),
            climate=0.3,
            collapsed=True,
        )
        if full
        else {}
    )
    return Run(
        states=rng.normal(size=(steps, 4)),
        params=rng.normal(size=(steps, 3)),
        actions=rng.normal(size=(steps, 2)),
        dt=0.25,
        **optional,
    )


def _assert_same(a, b, fields):
    """Field-by-field equality, arrays compared exactly and ``None`` preserved."""
    for name in fields:
        x, y = getattr(a, name), getattr(b, name)
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y, err_msg=name)
        else:
            assert x == y, name


RUN_FIELDS = (
    "states", "params", "actions", "hidden", "dt", "seed", "dampened_steps",
    "volatility", "crisis_intensity", "climate", "collapsed",
)
SCENARIO_FIELDS = ("params", "hidden", "dt", "seed", "volatility", "crisis_intensity", "climate")


@pytest.mark.parametrize("full", [True, False], ids=["full", "bare"])
def test_run_round_trip(tmp_path, full):
    """A main run reconstructs exactly, diagnostics and metadata included."""
    run = _run(full=full)
    path = tmp_path / "nested" / "main.npz"
    storage.save_run(path, run)
    loaded = storage.load_run(path)
    _assert_same(run, loaded, RUN_FIELDS)
    assert len(loaded) == len(run)


@pytest.mark.parametrize("full", [True, False], ids=["full", "bare"])
def test_scenario_round_trip(tmp_path, full):
    """A continuation stores only its forcing, and reconstructs exactly."""
    run = _run(full=full)
    scenario = Scenario(
        params=run.params,
        hidden=run.hidden,
        dt=run.dt,
        seed=run.seed,
        volatility=run.volatility,
        crisis_intensity=run.crisis_intensity,
        climate=run.climate,
    )
    path = tmp_path / "scenario.npz"
    storage.save_scenario(path, scenario)
    _assert_same(scenario, storage.load_scenario(path), SCENARIO_FIELDS)
    with np.load(path) as data:
        assert "states" not in data and "actions" not in data


def test_branch_state_round_trip(tmp_path):
    """The full internal model state comes back name for name."""
    state = {"Y": 1.5e8, "ER": 0.98, "_hidden_thing": -3.0}
    path = tmp_path / "branch.npz"
    storage.save_branch_state(path, state)
    assert storage.load_branch_state(path) == state


def test_group_layout(tmp_path):
    """A group lives at ``root/<split>/group_<index>`` with fixed file names."""
    group = tmp_path / "train" / "group_0007"
    assert storage.group_dir(tmp_path, "train", 7) == group
    assert storage.main_path(tmp_path, "train", 7) == group / "main.npz"
    assert storage.branch_path(tmp_path, "train", 7) == group / "branch.npz"
    assert storage.scenario_path(tmp_path, "train", 7, 3) == group / "scenario_003.npz"


def test_manifest_round_trip_carries_the_config(tmp_path):
    """A dataset's manifest is plain JSON, config included."""
    config = DatasetConfig(n_runs=2, workers=1)
    manifest = {"config": storage.config_to_dict(config), "splits": {"train": {"n_groups": 2}}}
    storage.write_manifest(tmp_path / "ds", manifest)
    assert storage.read_manifest(tmp_path / "ds") == json.loads(json.dumps(manifest))
    assert storage.read_manifest(tmp_path / "ds")["config"]["n_runs"] == 2
