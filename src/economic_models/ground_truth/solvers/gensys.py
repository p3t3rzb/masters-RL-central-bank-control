"""Sims' (2002) ``gensys`` solver for linear rational-expectations systems.

A linear rational-expectations model is a system in which today's variables
depend on *expectations* of tomorrow's, so it cannot be advanced by substitution
the way a backward-looking simultaneous system can. Written canonically as

.. math::

    \\Gamma_0 y_t = \\Gamma_1 y_{t-1} + C + \\Psi z_t + \\Pi \\eta_t

-- with ``z`` the exogenous innovations and ``eta`` the endogenous one-step
forecast errors -- it has a unique stable solution only when the number of
explosive roots of the pencil exactly matches the number of forecast errors that
can be used to purge them. :func:`gensys` returns that solution as a plain
state-space recursion

.. math::

    y_t = \\Theta_1 y_{t-1} + \\Theta_c + \\Theta_0 z_t

together with an existence/uniqueness verdict. That verdict *is* the
Blanchard-Kahn check: a model whose deep parameters wander into the indeterminate
region is rejected rather than simulated.

The implementation follows Sims' reference ``gensys.m``. Two conventions differ
between MATLAB's ``qz`` and :func:`scipy.linalg.ordqz` and are reconciled here
once, in :func:`_ordered_qz`: SciPy returns ``Q``/``Z`` with
:math:`\\Gamma_0 = Q S Z^H`, where Sims writes :math:`\\Gamma_0 = Q' \\Lambda Z'`,
so Sims' ``q`` is SciPy's ``Q`` conjugate-transposed while Sims' ``z`` is SciPy's
``Z`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import ordqz

#: Singular values and QZ diagonal entries below this count as zero.
_SMALL = 1e-6

#: Roots at or below this modulus are treated as stable. Sims nudges it down
#: when a root sits just above 1 so a near-unit root is not misclassified.
_DIV = 1.01


@dataclass(frozen=True)
class GensysResult:
    """The state-space solution of a linear rational-expectations system.

    :attr:`transition`, :attr:`constant` and :attr:`impact` are the
    :math:`\\Theta_1`, :math:`\\Theta_c` and :math:`\\Theta_0` of
    ``y_t = Theta1 y_{t-1} + Thetac + Theta0 z_t``. :attr:`existence` and
    :attr:`uniqueness` are Sims' two-part verdict (his ``eu``); only a system
    with both true has the unique stable solution that recursion represents.
    """

    transition: np.ndarray  #: (n, n) response to the previous state
    constant: np.ndarray  #: (n,) deterministic drift
    impact: np.ndarray  #: (n, n_shocks) response to this period's innovations
    existence: bool  #: a stable solution exists
    uniqueness: bool  #: it is the only one (no sunspot equilibria)
    n_unstable: int  #: explosive roots of the pencil
    n_forecast_errors: int  #: expectational errors available to purge them

    @property
    def determinate(self) -> bool:
        """Whether the system has exactly one stable solution."""
        return self.existence and self.uniqueness

    def reason(self) -> str:
        """A short human-readable verdict, for error messages and diagnostics."""
        if self.determinate:
            return "determinate"
        if not self.existence:
            return (
                f"no stable solution: {self.n_unstable} explosive roots but only "
                f"{self.n_forecast_errors} forecast errors"
            )
        return f"indeterminate: {self.n_unstable} explosive roots leave sunspots free"


def gensys(
    g0: np.ndarray,
    g1: np.ndarray,
    psi: np.ndarray,
    pi: np.ndarray,
    c: np.ndarray | None = None,
    *,
    div: float | None = None,
) -> GensysResult:
    """Solve ``g0 y_t = g1 y_{t-1} + c + psi z_t + pi eta_t`` for its stable law of motion.

    ``g0`` and ``g1`` are the ``(n, n)`` coefficient matrices on this period's and
    last period's variables, ``psi`` the ``(n, n_shocks)`` loading on the
    exogenous innovations, ``pi`` the ``(n, n_eta)`` loading on the endogenous
    forecast errors, and ``c`` an optional ``(n,)`` constant. ``div`` overrides the
    modulus at which a root counts as explosive; by default it is chosen just
    above one, nudged to avoid splitting a near-unit root arbitrarily.

    Returns a :class:`GensysResult` whose matrices are meaningful only when it is
    :attr:`~GensysResult.determinate` -- the caller decides whether to reject the
    parameter draw or raise.
    """
    g0 = np.asarray(g0, dtype=float)
    g1 = np.asarray(g1, dtype=float)
    psi = np.atleast_2d(np.asarray(psi, dtype=float))
    pi = np.atleast_2d(np.asarray(pi, dtype=float))
    n = g0.shape[0]
    c = np.zeros(n) if c is None else np.asarray(c, dtype=float).reshape(n)

    aa, bb, q, z, n_unstable = _ordered_qz(g0, g1, div)
    n_stable = n - n_unstable
    n_eta = pi.shape[1]

    q1, q2 = q[:n_stable], q[n_stable:]

    # The forecast errors visible to the explosive block. Existence needs every
    # explosive direction to be spanned by some combination of them.
    u2, d2, v2 = _truncated_svd(q2 @ pi)
    u1, d1, v1 = _truncated_svd(q1 @ pi)

    existence = len(d2) >= n_unstable
    uniqueness = _no_loose_errors(v1, v2, n)

    # Sims' tmat: the stable rows of the rotated system, with the explosive block's
    # forecast errors substituted out.
    if len(d2) and len(d1):
        m = u2 @ np.diag(1.0 / d2) @ v2.conj().T @ v1 @ np.diag(d1) @ u1.conj().T
    else:
        m = np.zeros((n_unstable, n_stable), dtype=complex)
    tmat = np.hstack([np.eye(n_stable, dtype=complex), -m.conj().T])

    top = tmat @ aa
    bottom = np.hstack(
        [np.zeros((n_unstable, n_stable), dtype=complex), np.eye(n_unstable, dtype=complex)]
    )
    big0 = np.vstack([top, bottom])
    big1 = np.vstack([tmat @ bb, np.zeros((n_unstable, n), dtype=complex)])
    big0_inv = np.linalg.inv(big0)

    # The explosive block is solved forward; with no anticipated future shocks its
    # only contribution is the deterministic constant.
    unstable = slice(n_stable, n)
    a2, b2 = aa[unstable, unstable], bb[unstable, unstable]
    forward_c = np.linalg.solve(a2 - b2, q2 @ c) if n_unstable else np.zeros(0, dtype=complex)

    transition = big0_inv @ big1
    constant = big0_inv @ np.concatenate([tmat @ q @ c, forward_c])
    impact = big0_inv @ np.vstack(
        [tmat @ q @ psi, np.zeros((n_unstable, psi.shape[1]), dtype=complex)]
    )

    return GensysResult(
        transition=np.real(z @ transition @ z.conj().T),
        constant=np.real(z @ constant),
        impact=np.real(z @ impact),
        existence=bool(existence),
        uniqueness=bool(uniqueness),
        n_unstable=int(n_unstable),
        n_forecast_errors=int(n_eta),
    )


# -- internals -------------------------------------------------------------


def _ordered_qz(
    g0: np.ndarray, g1: np.ndarray, div: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """QZ-decompose the pencil with the stable roots ordered first.

    Returns ``(S, T, q, z, n_unstable)`` in *Sims'* orientation --
    ``g0 = q^H S z^H`` -- so the caller can transcribe ``gensys.m`` directly.
    The root of row ``i`` is ``T[i, i] / S[i, i]``; ``div`` is the modulus above
    which it counts as explosive.
    """
    cutoff = _DIV if div is None else div
    if div is None:
        cutoff = _adaptive_div(g0, g1)

    def stable(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return np.abs(beta) <= cutoff * np.abs(alpha)

    aa, bb, alpha, beta, qs, zs = ordqz(g0, g1, sort=stable, output="complex")
    n_unstable = int(np.sum(np.abs(beta) > cutoff * np.abs(alpha)))

    coincident = (np.abs(alpha) < _SMALL) & (np.abs(beta) < _SMALL)
    if coincident.any():
        raise ValueError("coincident zeros in the pencil: the system is degenerate")

    return aa, bb, qs.conj().T, zs, n_unstable


def _adaptive_div(g0: np.ndarray, g1: np.ndarray) -> float:
    """Pick the stable/explosive cutoff, backing off from any near-unit root.

    Sims' rule: start just above one, and whenever a root sits in the sliver
    between one and the current cutoff, move the cutoff down to the midpoint so
    the classification never hinges on floating-point noise.
    """
    _, _, alpha, beta, _, _ = ordqz(g0, g1, sort="lhp", output="complex")
    div = _DIV
    for a, b in zip(alpha, beta):
        if abs(a) > 0:
            root = abs(b) / abs(a)
            if 1 + _SMALL < root <= div:
                div = 0.5 * (1 + root)
    return div


def _truncated_svd(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD of ``m`` with the numerically-zero singular values and their vectors dropped."""
    u, s, vh = np.linalg.svd(m, full_matrices=False)
    keep = s > _SMALL
    return u[:, keep], s[keep], vh[keep].conj().T


def _no_loose_errors(v1: np.ndarray, v2: np.ndarray, n: int) -> bool:
    """Whether every forecast error the stable block sees is pinned by the explosive one.

    Any direction of ``v1`` left over once ``v2``'s span is projected out is a
    forecast error nothing determines -- a sunspot, and so indeterminacy.
    """
    if v1.shape[1] == 0:
        return True
    loose = v1 - v2 @ v2.conj().T @ v1 if v2.shape[1] else v1
    return bool(np.sum(np.linalg.svd(loose, compute_uv=False) > _SMALL * n) == 0)
