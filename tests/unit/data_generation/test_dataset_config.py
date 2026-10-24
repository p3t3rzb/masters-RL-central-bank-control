"""The dataset config: validated eagerly, before a thousand groups are generated."""

from __future__ import annotations

import pytest

from data_generation.config import DatasetConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"excitation": "calm"},
        {"on_collapse": "ignore"},
        {"train_fraction": 1.5},
        {"train_fraction": -0.1},
        {"n_runs": 0},
        {"n_continuations": 0},
        {"main_length": 0},
        {"continuation_length": 0},
        {"model": "godley"},
        {"model": "smets_wouters", "dt": 1.0},
    ],
)
def test_bad_configs_fail_at_construction(kwargs):
    """Every invalid knob is rejected up front, naming what was wrong."""
    with pytest.raises(ValueError):
        DatasetConfig(**kwargs)


def test_quarterly_model_accepts_its_own_timestep():
    """Smets-Wouters pins ``dt`` to a quarter, which the default already is."""
    assert DatasetConfig(model="smets_wouters").dt == 0.25


@pytest.mark.parametrize("n_runs, fraction", [(10, 0.8), (7, 0.5), (3, 0.0), (3, 1.0)])
def test_split_assigns_every_run_once(n_runs, fraction):
    """Train and test groups partition the main runs."""
    config = DatasetConfig(n_runs=n_runs, train_fraction=fraction)
    assert config.n_train + config.n_test == n_runs
    assert config.n_train == round(n_runs * fraction)


def test_explicit_worker_count_wins():
    """``workers`` overrides the CPU-count default."""
    assert DatasetConfig(workers=3).resolved_workers() == 3
    assert DatasetConfig().resolved_workers() >= 1
