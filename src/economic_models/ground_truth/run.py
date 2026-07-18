"""The :class:`Run` dataset container: one simulated history of a ground-truth model.

A :class:`Run` is model-agnostic -- per-step levels of the visible interface
(:class:`~economic_models.variables.State` /
:class:`~economic_models.variables.Parameters` /
:class:`~economic_models.variables.Actions`) plus optional diagnostics -- and is
the dataset a :mod:`~economic_models.proxy` trains on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Run:
    """One simulated history: per-step levels of the visible interface.

    Row ``t`` of each array holds the values after simulation step ``t``;
    columns follow ``State.names()`` / ``Parameters.names()`` / ``Actions.names()``.
    """

    states: np.ndarray  # (T, n_state)
    params: np.ndarray  # (T, n_params)
    actions: np.ndarray  # (T, n_actions)
    #: hidden structural parameter paths (T, len(hidden_names)) -- diagnostics
    #: only: the bank cannot see these, so no proxy may train on them.
    hidden: np.ndarray | None = None
    dt: float = 1.0
    seed: int | None = field(default=None, kw_only=True)
    #: steps whose exogenous inputs were dampened to keep the solver converging
    dampened_steps: int = field(default=0, kw_only=True)
    #: per-step stochastic-volatility multiplier (diagnostics only), or ``None``
    volatility: np.ndarray | None = field(default=None, kw_only=True)
    #: per-step total crisis intensity (diagnostics only), or ``None``
    crisis_intensity: np.ndarray | None = field(default=None, kw_only=True)
    #: this run's drawn climate in ``[0, 1]`` (0 calm, 1 stormy), or ``None`` if
    #: the config has no climate spec (diagnostics only)
    climate: float | None = field(default=None, kw_only=True)
    #: ``True`` if the run was cut short by an unrecoverable solver failure (an
    #: economic collapse the excitation could not climb out of)
    collapsed: bool = field(default=False, kw_only=True)

    def __len__(self) -> int:
        """The number of recorded simulation steps in this run."""
        return len(self.states)
