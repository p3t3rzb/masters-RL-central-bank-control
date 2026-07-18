"""Generate "historic" runs of the GROWTH model with excited exogenous inputs.

A proxy can only learn the causal effect of the policy levers if the historic
run actually moved them. Each bank-visible exogenous input therefore drifts as a
clipped AR(1) around its baseline (``Nfe``, a level, as a log random walk),
persistent enough to read as slow regime drift and varied enough to identify the
transmission channels. The hidden structural parameters drift too; the bank
neither observes nor records them, so their effect reaches the dataset only as
unexplained variance in the visible state.

The policy rate carries no feedback rule, only tightly clipped noise: in this
model its channel is so delayed (~15 years to inflation) that a Taylor-type
anchor is destabilizing. Stability comes instead from the one fast lever,
government spending growth, whose countercyclical response to the employment gap
keeps the economy inside its stable corridor around full employment.

An :class:`~economic_models.ground_truth.growth.excitation.presets.ExcitationConfig`
describes the excitation;
:class:`~economic_models.ground_truth.growth.excitation.generator.ExcitedRunGenerator`
turns one into reproducible :class:`~economic_models.ground_truth.run.Run`
datasets. Hidden parameter paths are recorded for diagnostics only -- a proxy
must never train on them. Split into :mod:`.specs`, :mod:`.presets` and
:mod:`.generator`; the public names are re-exported here.
"""

from economic_models.ground_truth.growth.excitation.generator import (
    ExcitedRunGenerator,
)
from economic_models.ground_truth.growth.excitation.presets import ExcitationConfig
from economic_models.ground_truth.growth.excitation.specs import (
    AR1Spec,
    ClimateSpec,
    CrisisSpec,
    GovSpendingSpec,
    RandomWalkSpec,
    StochasticVolatilitySpec,
    discretize_ar1,
)
from economic_models.ground_truth.run import Run

__all__ = [
    "AR1Spec",
    "RandomWalkSpec",
    "StochasticVolatilitySpec",
    "CrisisSpec",
    "ClimateSpec",
    "GovSpendingSpec",
    "ExcitationConfig",
    "ExcitedRunGenerator",
    "Run",
    "discretize_ar1",
]
