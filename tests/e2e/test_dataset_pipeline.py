"""Dataset generation end to end: written to disk, read back, and run as a world.

The generator writes groups the control stack later loads as
:class:`~control.world.TrainingWorld`\\ s, so the check that matters is the round
trip: a group on disk must drive the ground-truth environment it was made for.
"""

from __future__ import annotations

import numpy as np
import pytest

from control.dataset import dataset_dt, group_count, load_world
from control.dsac.train import (
    TrainConfig,
    build_truth_env,
    calibration_actions,
    constant_policy,
    evaluate,
    free_instrument,
)
from control.observation import Observer
from control.rewards import MandateReward
from data_generation import storage
from data_generation.config import DatasetConfig
from data_generation.generate import generate_dataset
from economic_models.ground_truth.registry import MODELS


@pytest.fixture(scope="module", params=list(MODELS))
def dataset(request, tmp_path_factory):
    """A four-group dataset of one model, written to a fresh directory."""
    model = request.param
    root = tmp_path_factory.mktemp(f"dataset_{model}")
    config = DatasetConfig(
        n_runs=4,
        n_continuations=2,
        main_length=24,
        continuation_length=6,
        dt=MODELS[model].fixed_dt or 1.0,
        model=model,
        train_fraction=0.5,
        output_dir=str(root),
        workers=1,
    )
    manifest = generate_dataset(config, verbose=False)
    return config, manifest, root


def test_every_group_is_on_disk(dataset):
    """Each split holds its groups, each with a main run, branch state and scenarios."""
    config, _, root = dataset
    for split, n in (("train", config.n_train), ("test", config.n_test)):
        assert group_count(root, split) == n
        for index in range(n):
            assert storage.main_path(root, split, index).exists()
            assert storage.branch_path(root, split, index).exists()
            for j in range(config.n_continuations):
                assert storage.scenario_path(root, split, index, j).exists()


def test_manifest_describes_the_dataset(dataset):
    """The manifest records the config and the column order it was written in."""
    config, manifest, root = dataset
    assert storage.read_manifest(root) == manifest
    assert manifest["config"]["model"] == config.model
    assert dataset_dt(root) == config.dt


def test_groups_have_the_configured_shape(dataset):
    """Main runs and continuations are as long as configured, at the configured ``dt``."""
    config, _, root = dataset
    run = storage.load_run(storage.main_path(root, "train", 0))
    scenario = storage.load_scenario(storage.scenario_path(root, "train", 0, 0))
    assert len(run) == config.main_length and run.dt == config.dt
    assert len(scenario) == config.continuation_length
    assert np.all(np.isfinite(run.states))


def test_generation_is_reproducible(dataset, tmp_path):
    """The same base seed writes the same histories, whatever directory it lands in."""
    config, _, root = dataset
    again = DatasetConfig(**{**storage.config_to_dict(config), "output_dir": str(tmp_path)})
    generate_dataset(again, verbose=False)
    first = storage.load_run(storage.main_path(root, "test", 1))
    second = storage.load_run(storage.main_path(tmp_path, "test", 1))
    np.testing.assert_array_equal(first.states, second.states)


def test_a_loaded_group_drives_the_ground_truth(dataset):
    """A group read back as a world runs the structural model to the horizon."""
    config, _, root = dataset
    world = load_world(root, "test", 0)
    spec = world.config.spec
    assert len(world.eval_futures) == config.n_continuations
    assert world.branch_point.row == config.main_length - 1

    observer = Observer(spec.interface).fit(world.history)
    reward = MandateReward(employment_target=spec.employment_target)
    train_config = free_instrument(
        TrainConfig(model=config.model, dt=config.dt, horizon=config.continuation_length)
    )
    env = build_truth_env(world, observer, reward, train_config)
    policy = constant_policy(env, calibration_actions(config.model))
    score = evaluate(env, policy, config.n_continuations)
    assert score["collapse_rate"] == 0.0
    assert score["length"] == config.continuation_length
