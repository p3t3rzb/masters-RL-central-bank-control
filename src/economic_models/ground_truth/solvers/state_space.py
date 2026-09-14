"""Turning a solved rational-expectations system into a plain state-space recursion.

:func:`gensys` answers the hard question -- which of the candidate paths is the
unique stable one -- but it answers it over an *extended* vector, one auxiliary
expectation variable per forward-looking term. What the simulation loop wants is
the recursion over the model's own variables,

.. math::

    y_t = A y_{t-1} + B z_t + k,

and what the occasionally-binding-constraint loop wants is the same thing plus
the structural matrices to re-solve a constrained period against.
:func:`solve_lead_form` produces ``A`` and ``B`` by collapsing the auxiliaries out
of the ``gensys`` solution, and then *verifies* them against the structural form
by checking the matrix quadratic they must satisfy:

.. math::

    G_0 A = G_1 + G_f A^2, \\qquad (G_0 - G_f A) B = \\Psi.

That check is the reason to derive ``A`` this way rather than by iterating on the
quadratic directly: the fixed point of the iteration can be a non-minimal-state
solution, whereas ``gensys`` picks the right one and the quadratic then confirms
the collapse was exact. Every solve is self-checking, so a transcription error in
the equations shows up here rather than as a plausible-looking wrong simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from economic_models.base import ModelStepFailure
from economic_models.ground_truth.solvers.gensys import GensysResult, gensys

#: How far the recovered ``A``/``B`` may violate the structural form before the
#: solve is rejected as inconsistent.
_TOLERANCE = 1e-8


class IndeterminateSystem(ModelStepFailure):
    """The parameter draw has no unique stable rational-expectations solution.

    Raised rather than returned because a caller that ignores it would simulate
    a meaningless path. The generator catches it and resamples, which is the
    Blanchard-Kahn analogue of GROWTH's solver-failure resampling.
    """


@dataclass(frozen=True)
class StateSpace:
    """The reduced-form law of motion of a determinate model.

    ``y_t = A y_{t-1} + B z_t + k``, over the model's own variables in the order
    the :class:`~...smets_wouters.system.LinearSystem` declared them.
    """

    a: np.ndarray  #: (n, n) autoregressive matrix
    b: np.ndarray  #: (n, n_shocks) impact matrix
    k: np.ndarray  #: (n,) constant
    verdict: GensysResult  #: the determinacy check this came from

    @property
    def spectral_radius(self) -> float:
        """Largest eigenvalue modulus of :attr:`a` -- below one on a stable solution."""
        return float(np.max(np.abs(np.linalg.eigvals(self.a))))

    def forward(self, y: np.ndarray, horizon: int) -> np.ndarray:
        """The expected path ``E_t y_{t+1..t+horizon}`` from state ``y``, shocks at zero.

        Used to price anything that depends on expected future variables -- the
        long rate, for one -- straight off the solution.
        """
        out = np.empty((horizon, y.size))
        cur = y
        for h in range(horizon):
            cur = self.a @ cur + self.k
            out[h] = cur
        return out


def solve_lead_form(
    g0: np.ndarray,
    g1: np.ndarray,
    gf: np.ndarray,
    psi: np.ndarray,
    c: np.ndarray,
    gensys_matrices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> StateSpace:
    """Solve ``g0 y_t = g1 y_{t-1} + gf E_t y_{t+1} + c + psi z_t`` for its state space.

    ``gensys_matrices`` is the same system in Sims' canonical form -- the
    ``(g0, g1, psi, pi)`` of
    :meth:`~...smets_wouters.system.LinearSystem.gensys_form` -- whose extended
    vector carries the model's variables first. Raises
    :class:`IndeterminateSystem` if the draw is not determinate, or if the
    recovered matrices fail the structural check.
    """
    n = g0.shape[0]
    verdict = gensys(*gensys_matrices)
    if not verdict.determinate:
        raise IndeterminateSystem(verdict.reason())

    # The auxiliary-defining rows of the canonical form carry the selector:
    # each says ``x_t = E[x]_{t-1} + eta``, so its core block is a row of S.
    selector = gensys_matrices[0][n:, :n]
    a, b, k = _collapse(verdict, n, selector)

    residual_a = np.max(np.abs(g0 @ a - g1 - gf @ a @ a))
    residual_b = np.max(np.abs((g0 - gf @ a) @ b - psi))
    residual_k = np.max(np.abs((g0 - gf @ a - gf) @ k - c))
    worst = max(residual_a, residual_b, residual_k)
    if worst > _TOLERANCE:
        raise IndeterminateSystem(
            f"recovered state space violates the structural form by {worst:.2e} "
            "(a transcription error, or a solution that is not minimal-state)"
        )

    return StateSpace(a=a, b=b, k=k, verdict=verdict)


# -- internals -------------------------------------------------------------


def _collapse(
    verdict: GensysResult, n: int, selector: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Substitute the auxiliary expectations out of a ``gensys`` solution.

    ``selector`` is the ``(p, n)`` matrix picking the expected variables out of
    ``y`` -- read straight off the auxiliary-defining rows of the canonical form.
    On the solution path ``E[y]_t = S E_t y_{t+1}``, and the core rows give
    ``E_t y_{t+1}`` in terms of ``y_t`` and ``E[y]_t``, so the auxiliaries satisfy
    a linear system in themselves. Solving it leaves ``E[y]_t = M y_t + m0``,
    which folds back into the core rows as a recursion in ``y`` alone.
    """
    own = verdict.transition[:n, :n]
    on_aux = verdict.transition[:n, n:]
    const = verdict.constant[:n]
    p = on_aux.shape[1]

    lhs = np.eye(p) - selector @ on_aux
    m = np.linalg.solve(lhs, selector @ own)
    m0 = np.linalg.solve(lhs, selector @ const)

    a = own + on_aux @ m
    b = verdict.impact[:n]
    k = const + on_aux @ m0
    return a, b, k
