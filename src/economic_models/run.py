"""The :class:`Run` and :class:`Scenario` dataset containers.

A :class:`Run` is one *realised* history of a ground-truth model -- per-step
levels of the visible interface (:class:`~economic_models.variables.State` /
:class:`~economic_models.variables.Parameters` /
:class:`~economic_models.variables.Actions`) plus optional diagnostics -- and is
the dataset a :mod:`~economic_models.proxy` trains on.

A :class:`Scenario` is instead only the *exogenous forcing* a model would be
driven with -- per-step :class:`~economic_models.variables.Parameters` and hidden
structural parameters, but **no** actions and **no** states. It is the "world" an
in-control agent acts within: the agent supplies the actions, and the states are
whatever the model then produces, so neither is fixed in advance.
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


@dataclass
class Scenario:
    """One exogenous forcing path: the "world" an in-control agent acts within.

    Holds only the model's exogenous inputs per step -- the visible
    :class:`~economic_models.variables.Parameters` and the hidden structural
    parameters -- with **no** actions and **no** states. The actions are the
    agent's to choose, and the states are whatever the model produces from
    (start state, this forcing, the agent's actions), so recording either would
    pin a particular policy. Row ``t`` of each array holds the inputs applied at
    step ``t``; ``params`` columns follow ``Parameters.names()``.

    A scenario is generated from the excitation process alone, without solving any
    model. The government-spending stabilizer's employment-feedback term is
    therefore evaluated at full employment (``ER = 1``, contributing nothing): a
    stateless scenario has no realised ``ER``, and the automatic countercyclical
    response is environment dynamics to apply at rollout against the agent's own
    economy, not something frozen into the forcing here.
    """

    params: np.ndarray  # (T, n_params)
    #: hidden structural parameter paths (T, len(hidden_names)); the bank cannot
    #: see these, but the ground-truth model needs them to be driven.
    hidden: np.ndarray | None = None
    dt: float = 1.0
    seed: int | None = field(default=None, kw_only=True)
    #: per-step stochastic-volatility multiplier (diagnostics only), or ``None``
    volatility: np.ndarray | None = field(default=None, kw_only=True)
    #: per-step total crisis intensity (diagnostics only), or ``None``
    crisis_intensity: np.ndarray | None = field(default=None, kw_only=True)
    #: the drawn climate in ``[0, 1]`` of the run this scenario branches from, or
    #: ``None`` if the config has no climate spec (diagnostics only)
    climate: float | None = field(default=None, kw_only=True)

    def __len__(self) -> int:
        """The number of exogenous-forcing steps in this scenario."""
        return len(self.params)
