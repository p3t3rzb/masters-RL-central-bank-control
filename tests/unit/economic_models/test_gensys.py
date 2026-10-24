"""Sims' ``gensys`` and the state-space solution it hands the Smets-Wouters model.

Checked on scalar systems whose solutions are known in closed form, so a
regression in the QZ ordering or the existence/uniqueness verdict shows up as a
wrong number rather than as a plausible-looking wrong simulation.

The forward-looking case is ``y_t = b E_t y_{t+1} + z_t``, written canonically
with the auxiliary ``x_t = E_t y_{t+1}`` and the forecast error ``eta``. Its
root is ``1/b``, so it is determinate exactly when ``|b| < 1``, with solution
``y_t = z_t``.
"""

from __future__ import annotations

import numpy as np
import pytest

from economic_models.ground_truth.solvers import gensys
from economic_models.ground_truth.solvers.state_space import StateSpace


def _forward(b: float):
    """The canonical form of ``y_t = b E_t y_{t+1} + z_t`` over ``(y, x)``."""
    g0 = np.array([[1.0, -b], [1.0, 0.0]])
    g1 = np.array([[0.0, 0.0], [0.0, 1.0]])
    psi = np.array([[1.0], [0.0]])
    pi = np.array([[0.0], [1.0]])
    return g0, g1, psi, pi


def test_backward_ar1_is_recovered_exactly():
    """With no expectations the solution is the system itself."""
    result = gensys(np.eye(1), np.array([[0.5]]), np.eye(1), np.zeros((1, 0)))
    assert result.determinate
    np.testing.assert_allclose(result.transition, [[0.5]])
    np.testing.assert_allclose(result.impact, [[1.0]])
    np.testing.assert_allclose(result.constant, [0.0])


def test_forward_looking_determinate_solution():
    """``|b| < 1``: one explosive root, one forecast error, and ``y_t = z_t``."""
    result = gensys(*_forward(0.5))
    assert result.determinate
    assert result.n_unstable == 1
    assert result.reason() == "determinate"
    np.testing.assert_allclose(result.transition, np.zeros((2, 2)), atol=1e-10)
    np.testing.assert_allclose(result.impact, [[1.0], [0.0]], atol=1e-10)


def test_forward_looking_indeterminate_solution():
    """``|b| > 1``: no explosive root to pin the forecast error -- sunspots."""
    result = gensys(*_forward(1.5))
    assert result.existence
    assert not result.uniqueness
    assert not result.determinate
    assert "indeterminate" in result.reason()


def test_explosive_system_without_forecast_errors_has_no_solution():
    """An explosive backward root cannot be purged: no stable solution exists."""
    result = gensys(np.eye(1), np.array([[1.5]]), np.eye(1), np.zeros((1, 0)))
    assert not result.existence
    assert "no stable solution" in result.reason()


def test_constant_is_the_steady_state_drift():
    """A backward system's constant passes straight into ``Thetac``: ``y_t = 0.5 y_{t-1} + 1``."""
    result = gensys(
        np.eye(1), np.array([[0.5]]), np.eye(1), np.zeros((1, 0)), c=np.array([1.0])
    )
    np.testing.assert_allclose(result.constant, [1.0])


def test_state_space_forward_iterates_the_recursion():
    """``forward`` is ``y <- A y + k`` with shocks at zero, ``horizon`` times."""
    verdict = gensys(np.eye(1), np.array([[0.5]]), np.eye(1), np.zeros((1, 0)))
    a = np.array([[0.5, 0.1], [0.0, 0.9]])
    k = np.array([0.1, -0.2])
    space = StateSpace(a=a, b=np.eye(2), k=k, verdict=verdict)
    y = np.array([1.0, 2.0])
    path = space.forward(y, 3)

    expected, cur = [], y
    for _ in range(3):
        cur = a @ cur + k
        expected.append(cur)
    np.testing.assert_allclose(path, expected)
    assert space.spectral_radius == pytest.approx(0.9)
