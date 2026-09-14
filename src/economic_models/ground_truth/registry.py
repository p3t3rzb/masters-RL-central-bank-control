"""The ground-truth models the pipeline can be pointed at, by name.

Until there were two of them, every consumer of a ground-truth model simply
imported GROWTH's classes: the driver built a ``GrowthModel``, the world built a
``GrowthRunGenerator``, the environment's action box was a module-level dictionary
of GROWTH's lever names. That is the right amount of machinery for one model and
the wrong amount for two.

:class:`GroundTruthSpec` gathers everything the control and data-generation
stacks need to know about a model into one object, and :data:`MODELS` maps a name
to it -- following the pattern the codebase already uses for excitation presets
and proxy estimators, where a string in a config selects the thing. The default
everywhere is ``"growth"``, so nothing that already worked changes.

What is *not* in here is as deliberate as what is. There is no hook for how a
model solves, what its hidden state looks like, or how it is calibrated: those
stay behind :class:`~economic_models.base.BaseEconomicModel` and the model's own
package. A spec is a lookup table, not an abstraction layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from economic_models.base import BaseEconomicModel
from economic_models.ground_truth.excitation.base import ExcitationConfig, ExcitedRunGenerator
from economic_models.ground_truth.models.growth import (
    GROWTH_INTERFACE,
    GrowthCalibration,
    GrowthExcitationConfig,
    GrowthModel,
    GrowthRunGenerator,
)
from economic_models.ground_truth.models.smets_wouters import (
    SW_INTERFACE,
    SwCalibration,
    SwModel,
)
from economic_models.ground_truth.models.smets_wouters.excitation import (
    SwExcitationConfig,
    SwRunGenerator,
)
from economic_models.interface import ModelInterface


@dataclass(frozen=True)
class GroundTruthSpec:
    """Everything the pipeline needs in order to run against one ground-truth model."""

    name: str  #: the name a config selects this model by
    interface: ModelInterface  #: the value spaces and stationarisation a proxy is handed
    model: type[BaseEconomicModel]  #: the structural model itself
    generator: type[ExcitedRunGenerator]  #: draws excited runs and scenario continuations
    excitation: type[ExcitationConfig]  #: the excitation family, whose presets are named
    calibration: type  #: the calibration, whose ``baseline()`` gives the resting levers
    excitations: tuple[str, ...]  #: the preset names this family offers
    #: The box the environment maps the agent's normalised action onto. Wider than
    #: the excitation's own bands, deliberately -- see :mod:`control.env`.
    action_bounds: Mapping[str, tuple[float, float]]
    #: The employment rate the economy rests at, which the mandate measures
    #: shortfalls against. One for GROWTH, whose labour market clears at full
    #: employment; below one wherever a steady-state mark-up leaves slack.
    employment_target: float
    er_bounds: tuple[float, float]  #: employment rates outside which the economy has collapsed
    pi_bounds: tuple[float, float]  #: inflation rates outside which the economy has collapsed
    #: The lever a rule-based reference policy moves, and the response to the
    #: inflation gap that lever already owes before any tuned gain is added.
    #:
    #: For GROWTH the lever *is* the nominal rate, so the response starts at one:
    #: the rate has to move with inflation one-for-one before it moves the real
    #: rate at all. For Smets-Wouters the lever is a *deviation* from a policy rule
    #: that is already inside the model and already reacts to inflation with a
    #: coefficient of two, so the response starts at zero and a reference rule adds
    #: to what the estimated Federal Reserve was already doing.
    reference_lever: str
    reference_inflation_response: float
    #: The feedback gains each named rule reference is run at. ``"calibration"``
    #: is always available and is not in here: it is the levers at rest.
    reference_gains: Mapping[str, tuple[float, float]]
    #: The period, in years, this model is written for; ``None`` if it can be run
    #: at any. Smets-Wouters is estimated quarterly and its nominal rigidities
    #: cannot be rescaled, so it pins ``dt``.
    fixed_dt: float | None = None

    def excitation_config(self, preset: str) -> ExcitationConfig:
        """The named preset of this model's excitation family."""
        if preset not in self.excitations:
            raise ValueError(
                f"unknown excitation {preset!r} for model {self.name!r}; "
                f"expected one of {self.excitations}"
            )
        return getattr(self.excitation, preset)()

    def baseline_actions(self) -> dict[str, float]:
        """Where the levers rest when the bank does nothing.

        For GROWTH that is the book's calibrated policy settings. For
        Smets-Wouters it is the zero deviation from the estimated policy rule --
        which is to say, doing nothing *is* the Federal Reserve's own estimated
        reaction function, and the baseline the agent has to beat costs nothing to
        compute and has no gains to tune.
        """
        baselines = self.calibration.baseline().baselines()
        return {name: float(baselines[name]) for name in self.interface.actions.names()}


#: GROWTH's action box, kept at module scope because it is what
#: :mod:`control.env` has always exported.
GROWTH_ACTION_BOUNDS: Mapping[str, tuple[float, float]] = {
    "Rbbar": (0.005, 0.075),
    "NCAR": (0.05, 0.175),
    "ro": (0.015, 0.12),
}

#: The Smets-Wouters action box. Like GROWTH's it is wider than the bands the
#: excitation moved the levers over in the recorded history, so the agent has
#: somewhere to go that the historical bank did not -- with the same caveat, that
#: out there it is past the surrogate's training support.
SW_ACTION_BOUNDS: Mapping[str, tuple[float, float]] = {
    "Rdev": (-0.030, 0.030),
    "PIstar": (0.010, 0.050),
    "QE": (0.0, 0.040),
}

MODELS: dict[str, GroundTruthSpec] = {
    "growth": GroundTruthSpec(
        name="growth",
        interface=GROWTH_INTERFACE,
        model=GrowthModel,
        generator=GrowthRunGenerator,
        excitation=GrowthExcitationConfig,
        calibration=GrowthCalibration,
        excitations=("default", "calm", "realistic"),
        action_bounds=GROWTH_ACTION_BOUNDS,
        employment_target=1.0,
        er_bounds=(0.5, 1.5),
        pi_bounds=(-0.2, 0.5),
        reference_lever="Rbbar",
        reference_inflation_response=1.0,
        # Taylor's original pair, and the pair ``scripts/tune_policy.py`` selected
        # over the ``data/`` ensemble.
        reference_gains={"taylor": (0.5, 0.5), "taylor-tuned": (-0.9065, 0.3161)},
    ),
    "smets_wouters": GroundTruthSpec(
        name="smets_wouters",
        interface=SW_INTERFACE,
        model=SwModel,
        generator=SwRunGenerator,
        excitation=SwExcitationConfig,
        calibration=SwCalibration,
        excitations=("default", "calm", "realistic"),
        action_bounds=SW_ACTION_BOUNDS,
        employment_target=0.95,
        # Employment rests at 0.95 rather than at one, and inflation is anchored
        # by a rule inside the model rather than by a fifteen-year channel, so the
        # corridor is tighter at the top and no looser at the bottom.
        er_bounds=(0.6, 1.15),
        pi_bounds=(-0.2, 0.5),
        reference_lever="Rdev",
        reference_inflation_response=0.0,
        # One rule reference, reacting on top of the estimated one. There is no
        # tuned pair here because there is nothing to tune *to*: the calibration
        # reference is already Smets and Wouters' estimated Federal Reserve
        # reaction function, so the bar the agent has to clear comes for free.
        reference_gains={"taylor": (0.5, 0.5)},
        fixed_dt=0.25,
    ),
}


def ground_truth(name: str) -> GroundTruthSpec:
    """The spec named ``name``, with a useful error when there is no such model."""
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown ground-truth model {name!r}; expected one of {sorted(MODELS)}"
        ) from None


#: The model used wherever none is named, so every existing config, dataset and
#: figure keeps meaning exactly what it meant before.
DEFAULT_MODEL = "growth"
