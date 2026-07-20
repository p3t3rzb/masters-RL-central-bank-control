"""The per-model interface a proxy is pointed at to mimic a ground-truth model.

A proxy is model-agnostic machinery; to stand in for a particular ground-truth
model it needs that model's *interface*: its three value spaces (so it can build
typed :class:`~economic_models.variables.State` / ``Parameters`` / ``Actions``
observations and read their column order) and the stationarization rules its
learned feature space is built on (which trending columns to log-difference,
which to enter as a ratio). :class:`ModelInterface` bundles both; a model exposes
one constant (e.g. GROWTH's ``GROWTH_INTERFACE``) that callers hand to a proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

from economic_models.variables import Actions, Parameters, State


@dataclass(frozen=True)
class TransformSpec:
    """The stationarization rules a proxy's feature space is built on.

    Model levels trend, so a proxy is fit on a stationary view: the trending
    columns named here are differenced or ratioed, everything else passes through
    as its level. ``state_names`` / ``exog_names`` fix the column order the rules
    (and the run arrays) are aligned to.
    """

    state_names: tuple[str, ...]  #: endogenous columns, aligned with the State space
    exog_names: tuple[str, ...]  #: exogenous columns, aligned with Parameters then Actions
    #: state columns entered as one-step log-differences (growth rates per step)
    log_diff: tuple[str, ...]
    #: state columns entered as a ratio to the ``denominator`` column
    ratio_to: tuple[str, ...]
    #: the state column the ``ratio_to`` columns are divided by (e.g. nominal GDP)
    denominator: str
    #: exogenous columns entered as one-step log-differences
    exog_log_diff: tuple[str, ...]


@dataclass(frozen=True)
class ModelInterface:
    """A ground-truth model's value spaces plus its stationarization spec.

    Everything a proxy needs to stand in for the model: the typed
    :attr:`state` / :attr:`parameters` / :attr:`actions` spaces and the
    :attr:`transform_spec` its learned feature space is built on.
    """

    state: type[State]
    parameters: type[Parameters]
    actions: type[Actions]
    transform_spec: TransformSpec
