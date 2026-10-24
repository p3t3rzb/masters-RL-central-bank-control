"""The world's bookkeeping: config validation, start rows, banks, the stabilizer.

Everything here is pure arithmetic on hand-built runs and scenarios -- the
simulated world itself is exercised in ``integration/``.
"""

from __future__ import annotations

import numpy as np
import pytest

from control.world import (
    BranchPoint,
    EpisodeBank,
    FiscalStabilizer,
    NullStabilizer,
    TrainingWorld,
    WorldConfig,
    start_rows,
)
from economic_models.ground_truth.models.growth.excitation.specs import GovSpendingSpec
from economic_models.ground_truth.models.growth.variables import GrowthParameters
from economic_models.run import Run, Scenario

PARAM_NAMES = GrowthParameters.names()
GRG = PARAM_NAMES.index("GRg")


# -- WorldConfig -----------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"excitation": "stormy"},
        {"deploy_excitation": "stormy", "n_deploy_futures": 1},
        {"dt": 0.0},
        {"history_steps": 0},
        {"horizon": 0},
        {"n_train_futures": 0},
        {"n_eval_futures": -1},
        {"deploy_jitter": -0.1, "n_deploy_futures": 1},
        {"n_starts": 0},
        {"start_burn_in": -1},
        {"n_starts": 3, "history_steps": 50, "start_burn_in": 49},
        {"n_starts": 5, "n_train_futures": 4},
        {"deploy_excitation": "calm"},
        {"deploy_jitter": 0.5},
    ],
)
def test_world_config_rejects_impossible_worlds(kwargs):
    """Every bad combination fails at construction, before any simulation."""
    with pytest.raises(ValueError):
        WorldConfig(**kwargs)


def test_deploying_differently_needs_a_different_economy():
    """Naming the world's own preset is not a misspecification experiment."""
    assert not WorldConfig(deploy_excitation="realistic").deploys_differently
    assert WorldConfig(deploy_excitation="calm", n_deploy_futures=2).deploys_differently
    assert WorldConfig(deploy_jitter=0.5, n_deploy_futures=2).deploys_differently


def test_deploy_configs_are_one_per_future_and_reproducible():
    """Jittered deployment economies are seeded off the world seed, per future."""
    config = WorldConfig(deploy_jitter=1.0, n_deploy_futures=3, seed=4)
    first, again = config.deploy_configs(), config.deploy_configs()
    assert len(first) == 3
    assert first == again
    assert first[0] != first[1]
    assert first[0] != config.excitation_config()
    assert WorldConfig().deploy_configs() is None


# -- start rows --------------------------------------------------------------


def test_single_start_is_the_history_end():
    """``n_starts=1`` is exactly the original single branch."""
    assert start_rows(WorldConfig(history_steps=200)) == (199,)


def test_multiple_starts_span_burn_in_to_end():
    """Raising ``n_starts`` adds rows behind the end rather than moving it."""
    rows = start_rows(WorldConfig(history_steps=200, n_starts=5, start_burn_in=100))
    assert rows[0] == 100 and rows[-1] == 199
    assert list(rows) == sorted(set(rows)) and len(rows) == 5


# -- banks -------------------------------------------------------------------


def _scenarios(n: int, steps: int = 4) -> list[Scenario]:
    """``n`` scenarios whose parameter rows encode their position in the list."""
    return [Scenario(params=np.full((steps, len(PARAM_NAMES)), float(j))) for j in range(n)]


def test_bank_numbers_episodes_from_its_offset():
    """Episode indices identify a future across banks for logging and replay."""
    branch = BranchPoint(row=9, state={})
    bank = EpisodeBank(_scenarios(3), offset=10, branches=branch)
    episodes = bank.all()
    assert [e.index for e in episodes] == [10, 11, 12]
    assert all(e.branch is branch for e in episodes)
    assert [e.params[0, 0] for e in episodes] == [0.0, 1.0, 2.0]
    assert len(bank) == 3 and len(episodes[0]) == 4


def test_bank_spread_over_branches_reports_its_rows():
    """A multi-start bank knows which history rows its futures fork from."""
    branches = [BranchPoint(row=r, state={}) for r in (50, 10, 50, 30)]
    assert EpisodeBank(_scenarios(4), branches=branches).branch_rows == (10, 30, 50)


def test_bank_rejects_a_branch_count_mismatch():
    """One branch per future, or one for all -- nothing in between."""
    with pytest.raises(ValueError):
        EpisodeBank(_scenarios(3), branches=[BranchPoint(row=0, state={})] * 2)


def test_bank_draws_its_own_futures():
    """A training draw always comes out of the pool."""
    bank = EpisodeBank(_scenarios(5), offset=100)
    rng = np.random.default_rng(0)
    assert {bank.draw(rng).index for _ in range(200)} == set(range(100, 105))


# -- stabilizers -------------------------------------------------------------


@pytest.fixture
def stabilizer() -> FiscalStabilizer:
    """GROWTH's stabilizer at gain 0.5, clipped to ``[-2%, 8%]``."""
    spec = GovSpendingSpec(gap_sigma=0.01, gap_clip=0.02, stabilizer=0.5, bounds=(-0.02, 0.08))
    return FiscalStabilizer(spec, PARAM_NAMES)


def _row(grg: float = 0.03) -> np.ndarray:
    """A parameter row with ``GRg`` set and every other column distinct."""
    row = np.arange(1.0, len(PARAM_NAMES) + 1.0)
    row[GRG] = grg
    return row


def test_stabilizer_moves_only_government_spending(stabilizer):
    """A slump raises ``GRg``; every other parameter passes through."""
    row = _row()
    out = stabilizer.apply(row, er_prev=0.96)
    assert out[GRG] == pytest.approx(0.03 + 0.5 * 0.04)
    np.testing.assert_array_equal(np.delete(out, GRG), np.delete(row, GRG))
    np.testing.assert_array_equal(row, _row())


def test_stabilizer_respects_its_bounds(stabilizer):
    """A deep slump cannot push spending growth past its hard clip."""
    assert stabilizer.apply(_row(), er_prev=0.5)[GRG] == pytest.approx(0.08)


def test_strip_inverts_apply_off_the_bounds(stabilizer):
    """A realised row is backed out to its frozen draw before it is replayed."""
    row = _row()
    for er in (0.95, 1.0, 1.04):
        np.testing.assert_allclose(stabilizer.strip(stabilizer.apply(row, er), er), row)


def test_null_stabilizer_passes_everything_through():
    """The ablation leaves forcing untouched in both directions."""
    row = _row()
    np.testing.assert_array_equal(NullStabilizer().apply(row, 0.5), row)
    np.testing.assert_array_equal(NullStabilizer().strip(row, 0.5), row)


# -- the world's window ------------------------------------------------------


@pytest.fixture
def world() -> TrainingWorld:
    """A world around a 10-row synthetic history whose rows encode their index."""
    steps = 10
    rows = np.arange(steps, dtype=float)[:, None]
    history = Run(
        states=np.repeat(rows, 3, axis=1),
        params=np.repeat(rows, 2, axis=1),
        actions=np.repeat(rows, 1, axis=1),
    )
    return TrainingWorld(
        history=history,
        branch={"x": 1.0},
        train_futures=EpisodeBank([]),
        eval_futures=EpisodeBank([]),
        stabilizer=NullStabilizer(),
        hidden_names=(),
        config=WorldConfig(history_steps=steps, dt=1.0),
    )


def test_window_defaults_to_the_trailing_rows(world):
    """The warm start an episode off the history's end reads."""
    states, params, actions = world.window(3)
    assert states[:, 0].tolist() == [7.0, 8.0, 9.0]
    assert params.shape == (3, 2) and actions.shape == (3, 1)


def test_window_end_is_exclusive(world):
    """A branch at row ``r`` passes ``r + 1`` and gets the prefix ending there."""
    states, _, _ = world.window(2, end=5)
    assert states[:, 0].tolist() == [3.0, 4.0]


@pytest.mark.parametrize("rows, end", [(0, None), (11, None), (4, 3), (2, 11)])
def test_window_rejects_rows_the_history_cannot_supply(world, rows, end):
    """Asking for more rows than precede ``end`` is an error, not a short window."""
    with pytest.raises(ValueError):
        world.window(rows, end=end)


def test_branch_point_is_the_last_row(world):
    """Evaluation and deployment start where the history stops."""
    assert world.branch_point.row == 9
    assert world.branch_point.state == {"x": 1.0}
    assert world.dt == 1.0
