"""The excited-run generators, for every registered ground truth.

A dataset is only reproducible if a run is a pure function of its seed, and a
continuation bank is only meaningful if its main run is exactly the run
:meth:`~economic_models.ground_truth.excitation.base.ExcitedRunGenerator.generate`
would have produced -- both are pinned here for each model.
"""

from __future__ import annotations

import numpy as np
import pytest

from economic_models.ground_truth.registry import MODELS

STEPS = 25


@pytest.fixture(scope="module", params=list(MODELS))
def spec(request):
    """One registered ground-truth spec per parametrization."""
    return MODELS[request.param]


@pytest.fixture(scope="module")
def generator(spec):
    """That model's generator under its default preset, at its own timestep."""
    return spec.generator(spec.excitation_config("default"), dt=spec.fixed_dt or 1.0)


@pytest.fixture(scope="module")
def run(generator):
    """One excited run, seed 3."""
    return generator.generate(STEPS, seed=3)


def test_run_has_the_interface_shape(spec, generator, run):
    """Columns follow the model's value spaces; every row is finite."""
    interface = spec.interface
    assert len(run) == STEPS
    assert run.states.shape == (STEPS, len(interface.state.names()))
    assert run.params.shape == (STEPS, len(interface.parameters.names()))
    assert run.actions.shape == (STEPS, len(interface.actions.names()))
    assert run.hidden.shape == (STEPS, len(generator.config.hidden_names))
    assert np.all(np.isfinite(run.states))
    assert run.seed == 3 and run.dt == generator.dt and not run.collapsed


def test_run_is_a_pure_function_of_its_seed(generator, run):
    """The same seed reproduces the run exactly; another seed does not."""
    again = generator.generate(STEPS, seed=3)
    np.testing.assert_array_equal(again.states, run.states)
    np.testing.assert_array_equal(again.params, run.params)
    np.testing.assert_array_equal(again.hidden, run.hidden)
    other = generator.generate(STEPS, seed=4)
    assert not np.allclose(other.params, run.params)


def test_unexcited_run_holds_inputs_at_baseline(generator):
    """``excite=False`` is the steady-growth reference: inputs never move."""
    reference = generator.generate(10, excite=False)
    np.testing.assert_allclose(
        reference.params, np.broadcast_to(reference.params[0], reference.params.shape)
    )
    np.testing.assert_allclose(
        reference.actions,
        np.broadcast_to(reference.actions[0], reference.actions.shape),
    )
    assert reference.volatility is None


def test_continuations_branch_off_the_same_main_run(generator, run):
    """The main run of a continuation bank is exactly ``generate`` for its seed."""
    main, scenarios, branch = generator.generate_with_continuations(
        STEPS, 6, 3, seed=3, continuation_seeds=[10, 11, 12]
    )
    np.testing.assert_array_equal(main.states, run.states)
    assert [len(s) for s in scenarios] == [6, 6, 6]
    assert all(
        s.hidden.shape == (6, len(generator.config.hidden_names)) for s in scenarios
    )
    assert all(s.dt == generator.dt for s in scenarios)
    assert branch and all(np.isfinite(v) for v in branch.values())


def test_continuations_are_reproducible_and_distinct(generator):
    """Seeded futures repeat exactly, and different seeds give different futures."""
    kwargs = dict(seed=1, continuation_seeds=[5, 6])
    _, first, _ = generator.generate_with_continuations(STEPS, 6, 2, **kwargs)
    _, again, _ = generator.generate_with_continuations(STEPS, 6, 2, **kwargs)
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a.params, b.params)
    assert not np.allclose(first[0].params, first[1].params)


def test_branched_futures_fork_at_the_requested_rows(generator):
    """Each requested row gets one capture, taken where its future forks."""
    rows = [5, 12, STEPS - 1]
    main, scenarios, captures = generator.generate_branched(
        STEPS, 4, 3, branch_at=rows, seed=2, continuation_seeds=[0, 1, 2]
    )
    assert sorted(captures) == rows
    assert all(captures[r].index == r for r in rows)
    assert len(scenarios) == 3


def test_branch_rows_must_match_the_continuation_count(generator):
    """One fork row per continuation, or the call is refused."""
    with pytest.raises(ValueError, match="branch_at"):
        generator.generate_branched(STEPS, 4, 3, branch_at=[1, 2])


def test_collapse_policy_is_validated(spec):
    """``on_collapse`` is ``'raise'`` or ``'truncate'`` and nothing else."""
    with pytest.raises(ValueError, match="on_collapse"):
        spec.generator(
            spec.excitation_config("default"),
            dt=spec.fixed_dt or 1.0,
            on_collapse="ignore",
        )
