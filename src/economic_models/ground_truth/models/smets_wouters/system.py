"""A readable way to state a linear rational-expectations model, in two forms at once.

The GROWTH model states its economics as pysolve equation strings, one per line,
each carrying its Godley-Lavoie equation number. A log-linearised DSGE has no
such luxury: it is a pile of coefficient matrices, and transcribing a paper into
them by hand is exactly where the errors go. :class:`LinearSystem` restores the
line-per-equation shape -- each equation is added by *name*, with dictionaries
of coefficients on lagged, current and expected-future variables -- and assembles
the matrices itself.

It emits **two** forms of the same system, because the two things that happen at
runtime need different ones:

* :meth:`~LinearSystem.lead_form` -- the structural form
  ``G0 y_t = G1 y_{t-1} + Gf E_t y_{t+1} + c + Psi z_t``. This is what the
  matrix-quadratic solve and the occasionally-binding-constraint recursion work
  on, and it keeps the variable vector minimal.
* :meth:`~LinearSystem.gensys_form` -- Sims' canonical form, which needs an
  auxiliary variable per expectation and a forecast error to go with it. Used for
  the determinacy verdict and as an independent check on the first.

Stating an equation once and deriving both is the point: the two solvers can
never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LeadForm:
    """The structural form ``G0 y_t = G1 y_{t-1} + Gf E_t y_{t+1} + c + Psi z_t``."""

    g0: np.ndarray  #: (n, n) coefficients on this period's variables
    g1: np.ndarray  #: (n, n) coefficients on last period's
    gf: np.ndarray  #: (n, n) coefficients on next period's expectation
    psi: np.ndarray  #: (n, n_shocks) loading on the innovations
    c: np.ndarray  #: (n,) constants
    variables: tuple[str, ...]  #: column order of ``y``
    shocks: tuple[str, ...]  #: column order of ``z``


@dataclass(frozen=True)
class GensysForm:
    """Sims' canonical form, with one auxiliary variable per expectation."""

    g0: np.ndarray
    g1: np.ndarray
    psi: np.ndarray
    pi: np.ndarray  #: (n, n_eta) loading on the endogenous forecast errors
    variables: tuple[str, ...]  #: the model's variables, then the ``E[...]`` auxiliaries
    shocks: tuple[str, ...]


@dataclass
class _Equation:
    """One structural equation, as coefficient maps over named variables."""

    name: str
    cur: Mapping[str, float]
    lag: Mapping[str, float]
    lead: Mapping[str, float]
    shock: Mapping[str, float]
    const: float


class LinearSystem:
    """Accumulates named equations over named variables, then assembles the matrices.

    Every equation is written in the residual convention **left-hand side equals
    zero**: the coefficient maps together state ``sum(coeffs * terms) + const = 0``.
    Writing each equation the way the paper prints it -- moving everything to one
    side -- keeps the transcription mechanical.
    """

    def __init__(self, variables: Sequence[str], shocks: Sequence[str]) -> None:
        """Fix the variable and shock vectors; equations are added afterwards."""
        self.variables = tuple(variables)
        self.shocks = tuple(shocks)
        self._index = {name: i for i, name in enumerate(self.variables)}
        self._shock_index = {name: i for i, name in enumerate(self.shocks)}
        self._equations: list[_Equation] = []
        if len(self._index) != len(self.variables):
            raise ValueError("duplicate variable name")

    # -- building ----------------------------------------------------------

    def add(
        self,
        name: str,
        *,
        cur: Mapping[str, float] | None = None,
        lag: Mapping[str, float] | None = None,
        lead: Mapping[str, float] | None = None,
        shock: Mapping[str, float] | None = None,
        const: float = 0.0,
    ) -> None:
        """Add the equation ``cur*y_t + lag*y_{t-1} + lead*E_t y_{t+1} + shock*z_t + const = 0``.

        ``name`` is for error messages and diagnostics only. Every key must be a
        declared variable (or, for ``shock``, a declared shock); an unknown name
        is a transcription slip and raises rather than being silently ignored.
        """
        cur, lag, lead, shock = cur or {}, lag or {}, lead or {}, shock or {}
        for group, keys in (("variable", (*cur, *lag, *lead)), ("shock", tuple(shock))):
            table = self._index if group == "variable" else self._shock_index
            unknown = [k for k in keys if k not in table]
            if unknown:
                raise ValueError(f"equation {name!r}: unknown {group}(s) {sorted(unknown)}")
        self._equations.append(_Equation(name, dict(cur), dict(lag), dict(lead), dict(shock), const))

    def exogenous(
        self, name: str, *, rho: float, shock: str, ma: str | None = None, ma_weight: float = 0.0
    ) -> None:
        """Declare ``name`` an AR(1) (or ARMA(1,1)) process driven by ``shock``.

        ``ma``, when given, names the variable holding last period's innovation,
        so the process reads ``x_t = rho*x_{t-1} + z_t - ma_weight*z_{t-1}`` --
        the form Smets and Wouters give the two mark-up disturbances.
        """
        self.add(
            name,
            cur={name: 1.0},
            lag={name: -rho, **({ma: ma_weight} if ma else {})},
            shock={shock: -1.0},
        )

    @property
    def equations(self) -> tuple[str, ...]:
        """The equation names, in the row order the matrices are assembled in."""
        return tuple(eq.name for eq in self._equations)

    def row(self, name: str) -> int:
        """The row ``name`` occupies -- how a model finds an equation to replace."""
        return self.equations.index(name)

    # -- assembly ----------------------------------------------------------

    def lead_form(self) -> LeadForm:
        """Assemble ``G0 y_t = G1 y_{t-1} + Gf E_t y_{t+1} + c + Psi z_t``.

        Note the sign flip against the residual convention equations are written
        in: everything but ``G0`` moves to the right-hand side.
        """
        n, m = self._require_square()
        g0 = np.zeros((n, n))
        g1 = np.zeros((n, n))
        gf = np.zeros((n, n))
        psi = np.zeros((n, m))
        c = np.zeros(n)
        for row, eq in enumerate(self._equations):
            for name, v in eq.cur.items():
                g0[row, self._index[name]] += v
            for name, v in eq.lag.items():
                g1[row, self._index[name]] -= v
            for name, v in eq.lead.items():
                gf[row, self._index[name]] -= v
            for name, v in eq.shock.items():
                psi[row, self._shock_index[name]] -= v
            c[row] -= eq.const
        return LeadForm(g0, g1, gf, psi, c, self.variables, self.shocks)

    def gensys_form(self) -> GensysForm:
        """Assemble Sims' canonical form, adding an auxiliary variable per expectation.

        For every variable ``x`` that some equation expects, an auxiliary
        ``E[x]`` is appended to the vector, its lead coefficient is re-pointed at
        that auxiliary, and the defining equation ``x_t = E[x]_{t-1} + eta_x`` is
        added. ``eta_x`` is then the one-step forecast error the solver uses to
        purge the explosive roots.
        """
        n, m = self._require_square()
        expected = tuple(
            name for name in self.variables if any(name in eq.lead for eq in self._equations)
        )
        aux = {name: n + i for i, name in enumerate(expected)}
        total = n + len(expected)

        g0 = np.zeros((total, total))
        g1 = np.zeros((total, total))
        psi = np.zeros((total, m))
        pi = np.zeros((total, len(expected)))

        for row, eq in enumerate(self._equations):
            for name, v in eq.cur.items():
                g0[row, self._index[name]] += v
            for name, v in eq.lag.items():
                g1[row, self._index[name]] -= v
            for name, v in eq.lead.items():
                g0[row, aux[name]] += v  # E_t x_{t+1} *is* the auxiliary, dated t
            for name, v in eq.shock.items():
                psi[row, self._shock_index[name]] -= v

        # x_t = E[x]_{t-1} + eta_x: the auxiliary was last period's forecast, and
        # the gap between it and the realisation is the forecast error.
        for k, name in enumerate(expected):
            row = n + k
            g0[row, self._index[name]] = 1.0
            g1[row, aux[name]] = 1.0
            pi[row, k] = 1.0

        return GensysForm(g0, g1, psi, pi, (*self.variables, *(f"E[{x}]" for x in expected)), self.shocks)

    # -- internals ---------------------------------------------------------

    def _require_square(self) -> tuple[int, int]:
        """Check the system is exactly determined, naming the mismatch if not."""
        n, m = len(self.variables), len(self.shocks)
        if len(self._equations) != n:
            raise ValueError(
                f"{len(self._equations)} equations for {n} variables: "
                f"{'missing' if len(self._equations) < n else 'too many'}"
            )
        return n, m
