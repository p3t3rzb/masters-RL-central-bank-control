"""The base for ground-truth models solved as linear rational-expectations systems.

A sibling of :class:`~economic_models.ground_truth.base.PysolveEconomicModel`, not
a subclass of it: both realise
:class:`~economic_models.base.BaseEconomicModel`, but a model whose agents form
expectations cannot be advanced by iterating a simultaneous system, so it needs
its own machinery. This class owns four pieces of it.

**The solution, and when to recompute it.** The reduced form depends on the deep
parameters, so a model whose structural parameters drift has to be re-solved as
they move. That is the honest thing -- and it is also what keeps a step's cost in
the same range as a pysolve model's, since the decomposition is most of the work.
Agents re-optimise each period treating the current parameters as permanent, the
anticipated-utility convention (Kreps 1998; Cogley and Sargent 2005) that
time-varying-parameter DSGEs are read under.

**Driving it from outside.** The excitation supplies the *level* of every
disturbance -- clipped, volatility-scaled, occasionally hit by a crisis -- while
the model's own equations say each follows an AR(1). Rather than choose between
them, this class asks each period which innovations would have produced exactly
the levels the excitation chose, and feeds those. The realised path is the
excitation's; expectations stay consistent with the process agents believe in;
and the gap between the two is precisely the sense in which the economy is harder
than anyone's model of it.

**The lower bound.** Nominal rates cannot fall far below zero, which is the one
nonlinearity that matters for monetary policy and this model's counterpart to
GROWTH's stability corridor. It is imposed the way Guerrieri and Iacoviello
(2015) impose an occasionally-binding constraint: guess how long the bound will
hold, solve the path backward under the constrained system with the unconstrained
solution as its terminal condition, and check the guess.

**Saving and restoring.** Everything the model needs to continue is a flat
``{name: float}`` mapping -- the deviation vector, the reconstructed levels, the
current deep parameters and the exogenous inputs. That is the contract branching
depends on, and satisfying it is why the branching, dataset and driver code needs
no changes to carry a second ground-truth model.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Mapping, Sequence

import numpy as np

from economic_models.base import BaseEconomicModel, ModelStepFailure
from economic_models.ground_truth.solvers.state_space import (
    IndeterminateSystem,
    StateSpace,
    solve_lead_form,
)

#: How many periods forward the constrained regime is allowed to run before the
#: bound is assumed never to lift. Well past any plausible episode.
_MAX_BINDING = 40

#: Iterations of the guess-and-verify loop before the regime sequence is declared
#: not to converge -- which is a collapse, not a solver setting to raise.
_MAX_OCCBIN_ITERATIONS = 20


class LowerBoundNotResolved(ModelStepFailure):
    """The regime sequence at the lower bound would not settle.

    The economy is in a deflationary spiral the bound cannot arrest: expectations
    of a bound that never lifts justify the demand shortfall that keeps it
    binding. A genuine failure mode of the economy rather than of the arithmetic,
    and treated as a collapse.
    """


def _constrain(
    lead: tuple[np.ndarray, ...], row: int, level: float, replacement: np.ndarray
) -> tuple[np.ndarray, ...]:
    """The structural form with its policy equation replaced by the binding bound."""
    g0, g1, gf, psi, c = (m.copy() for m in lead)
    g0[row] = replacement
    g1[row] = 0.0
    gf[row] = 0.0
    psi[row] = 0.0
    c[row] = level
    return g0, g1, gf, psi, c


class LinearREEconomicModel(BaseEconomicModel):
    """A ground-truth model advanced through its rational-expectations solution."""

    def __init__(self, *, dt: float = 0.25) -> None:
        """Set up the model at its calibration; ``dt`` is one period in years."""
        self.dt = dt
        self._variables = tuple(self._variable_names())
        self._index = {name: i for i, name in enumerate(self._variables)}
        self._driven = tuple(self._driven_names())
        self._driven_rows = np.array([self._index[n] for n in self._driven])
        self._y = np.zeros(len(self._variables))
        self._solution: StateSpace | None = None
        self._solved_at: tuple[float, ...] | None = None
        self._solutions: list[dict[str, float]] = []

    # -- subclass contract -------------------------------------------------

    @abstractmethod
    def _variable_names(self) -> Sequence[str]:
        """The model's variables, in the permanent order of the state vector."""

    @abstractmethod
    def _driven_names(self) -> Sequence[str]:
        """The variables whose level is supplied from outside each period.

        One innovation each, so the map between levels and innovations is square.
        """

    @abstractmethod
    def _matrices(self) -> tuple[np.ndarray, ...]:
        """The current parameters' structural and canonical matrices.

        Returns ``(g0, g1, gf, psi, c, cg0, cg1, cpsi, cpi)`` -- the lead form
        followed by the canonical form the determinacy check runs on.
        """

    @abstractmethod
    def _parameter_signature(self) -> tuple[float, ...]:
        """The deep parameters, as a tuple; the solution is reused while it is unchanged."""

    @abstractmethod
    def _targets(self) -> np.ndarray:
        """This period's desired level for each of :meth:`_driven_names`."""

    @abstractmethod
    def _record(self) -> dict[str, float]:
        """The full internal state as a flat mapping, for saving and restoring."""

    def _bound(self) -> tuple[int, float, np.ndarray] | None:
        """The lower bound, as ``(row, level, replacement)``; ``None`` to disable it.

        ``row`` is the equation replaced while the bound binds, ``level`` the value
        the constrained variable is pinned to, and ``replacement`` the row of
        coefficients that pins it.
        """
        return None

    # -- the solution ------------------------------------------------------

    @property
    def solution(self) -> StateSpace:
        """The current reduced form, recomputed only when the parameters have moved."""
        signature = self._parameter_signature()
        if self._solution is None or signature != self._solved_at:
            matrices = self._matrices()
            self._solution = solve_lead_form(
                *matrices[:5], (matrices[5], matrices[6], matrices[7], matrices[8])
            )
            self._solved_at = signature
        return self._solution

    # -- advancing ---------------------------------------------------------

    def step(self) -> None:
        """Advance one period: hit this period's exogenous levels and solve forward."""
        self._prepare()
        solution = self.solution
        innovations = self._innovations(solution, self._y)
        self._y = self._advance_state(solution, self._y, innovations)
        self._update_levels()
        self._solutions.append(self._record())

    def run(self, n_steps: int) -> None:
        """Advance the simulation by ``n_steps`` periods."""
        for _ in range(n_steps):
            self.step()

    def _innovations(self, solution: StateSpace, y_prev: np.ndarray) -> np.ndarray:
        """The innovations that land the driven variables exactly on their targets.

        The driven rows of the reduced form *are* those variables' own laws of
        motion -- they depend on nothing endogenous -- so inverting them is exact
        rather than an approximation.
        """
        rows = self._driven_rows
        free = solution.a[rows] @ y_prev + solution.k[rows]
        return np.linalg.solve(solution.b[rows], self._targets() - free)

    def _advance_state(
        self, solution: StateSpace, y_prev: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """One period of the law of motion, respecting the lower bound if it binds."""
        y = solution.a @ y_prev + solution.b @ z + solution.k
        bound = self._bound()
        if bound is None:
            return y
        row, level, replacement = bound
        if y[self._bound_variable] >= level:
            return y
        return self._bounded_step(y_prev, z, row, level, replacement)

    def _bounded_step(
        self, y_prev: np.ndarray, z: np.ndarray, row: int, level: float, replacement: np.ndarray
    ) -> np.ndarray:
        """Guess how long the bound binds, solve backward, and check the guess.

        The constrained system replaces the policy equation with one pinning the
        rate to the bound. Its terminal condition is the unconstrained solution,
        which is what makes the path piecewise linear rather than merely clipped:
        agents in the constrained periods know the bound will lift, and roughly
        when, and that expectation is what determines how deep the episode is.
        """
        reference = self.solution
        lead = self._matrices()[:5]
        constrained = _constrain(lead, row, level, replacement)

        duration = 1
        for _ in range(_MAX_OCCBIN_ITERATIONS):
            path = self._piecewise(y_prev, z, duration, constrained, reference)
            implied = self._implied_duration(path, y_prev, z, duration, level, lead)
            if implied == duration:
                return path[0]
            if implied == 0:
                return reference.a @ y_prev + reference.b @ z + reference.k
            if implied > _MAX_BINDING:
                raise LowerBoundNotResolved(
                    "the lower bound would bind beyond the horizon: the economy is in a "
                    "deflationary spiral policy cannot arrest"
                )
            duration = implied
        raise LowerBoundNotResolved("the regime sequence at the lower bound did not settle")

    def _piecewise(
        self,
        y_prev: np.ndarray,
        z: np.ndarray,
        duration: int,
        constrained: tuple[np.ndarray, ...],
        reference: StateSpace,
    ) -> np.ndarray:
        """The path implied by ``duration`` constrained periods followed by the reference.

        Backward first, because a constrained period's decision rule depends on
        the one that follows it; the recursion is seeded with the unconstrained
        solution, which is what the economy returns to once the bound lifts.
        """
        cg0, cg1, cgf, cpsi, cc = constrained
        a, k, b = reference.a, reference.k, reference.b
        rules: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for _ in range(duration):
            lhs = cg0 - cgf @ a
            a, k, b = (
                np.linalg.solve(lhs, cg1),
                np.linalg.solve(lhs, cc + cgf @ k),
                np.linalg.solve(lhs, cpsi),
            )
            rules.append((a, k, b))
        rules.reverse()  # rules[t] is the decision rule of constrained period t

        horizon = duration + _MAX_BINDING
        path = np.empty((horizon, len(self._variables)))
        state = y_prev
        for h in range(horizon):
            if h < duration:
                a_h, k_h, b_h = rules[h]
                state = a_h @ state + k_h + (b_h @ z if h == 0 else 0.0)
            else:
                state = reference.a @ state + reference.k + (
                    reference.b @ z if h == 0 else 0.0
                )
            path[h] = state
        return path

    def _implied_duration(
        self,
        path: np.ndarray,
        y_prev: np.ndarray,
        z: np.ndarray,
        guessed: int,
        level: float,
        lead: tuple[np.ndarray, ...],
    ) -> int:
        """How long the bound *should* bind along ``path``, given it was guessed at ``guessed``.

        Inside the guessed spell the rate is pinned, so the test is whether the
        rule would have wanted to go lower; outside it, whether the rate the rule
        actually set is below the bound. The answer is the last period failing
        that test, plus one -- the spell is taken to be contiguous, as an episode
        at the bound is.
        """
        shadow = self._shadow_rate(path, y_prev, z, lead)
        realised = path[:, self._bound_variable]
        binding = np.where(np.arange(len(path)) < guessed, shadow, realised) < level - 1e-10
        hit = np.flatnonzero(binding)
        return int(hit[-1]) + 1 if hit.size else 0

    def _shadow_rate(
        self,
        path: np.ndarray,
        y_prev: np.ndarray,
        z: np.ndarray,
        lead: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        """The rate the *unconstrained* rule would have set at each point of ``path``.

        Read straight off the policy equation of the structural form, using the
        realised path for its lag and the following period for its expectation --
        which along a deterministic continuation is what agents expect.
        """
        bound = self._bound()
        if bound is None:
            return np.full(len(path), np.inf)
        row, _, _ = bound
        g0, g1, gf, psi, c = lead
        col = self._bound_variable
        own = g0[row, col]

        lagged = np.vstack([y_prev, path[:-1]])
        expected = np.vstack([path[1:], path[-1]])
        rhs = lagged @ g1[row] + expected @ gf[row] + c[row]
        rhs[0] += psi[row] @ z
        others = g0[row].copy()
        others[col] = 0.0
        return (rhs - path @ others) / own

    def _prepare(self) -> None:
        """Reconcile deep parameters with the exogenous inputs, before solving.

        Some observed exogenous quantities *are* deep parameters -- trend growth,
        for one -- so a model overrides this to fold them in while the solution can
        still be recomputed to match.
        """

    def _update_levels(self) -> None:
        """Reconstruct the observable levels from the new deviation vector."""

    # -- state -------------------------------------------------------------

    @property
    def solutions(self) -> list[dict[str, float]]:
        """The full internal state after each step, as pysolve exposes it."""
        return self._solutions

    def set_values(self, values: Mapping[str, float]) -> None:
        """Set any internal quantity by name, visible or hidden.

        The escape hatch branching restores through, and the channel the
        excitation drives the hidden disturbances and drifting parameters with.
        """
        self._restore(values)

    @abstractmethod
    def _restore(self, values: Mapping[str, float]) -> None:
        """Apply ``values`` to whichever part of the internal state each belongs to."""


__all__ = ["LinearREEconomicModel", "LowerBoundNotResolved", "IndeterminateSystem"]
