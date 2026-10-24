"""Model-agnostic excitation: the base config, per-step process and run generator.

Kept as the import path every model package, the data-generation pipeline and
the control stack already use, so that splitting the machinery by concern moved
no public name. The two halves live in their own modules:

* :mod:`~economic_models.ground_truth.excitation.process` --
  :class:`ExcitationConfig` and :class:`ExcitationProcess`, the spec and the
  stateful per-step draw of exogenous inputs;
* :mod:`~economic_models.ground_truth.excitation.generator` --
  :class:`ExcitedRunGenerator` and :class:`BranchCapture`, the driver that
  settles a model, steps it under the drawn inputs, dampens the solver on
  failure and records the result as a :class:`~economic_models.run.Run` (plus
  branching :class:`~economic_models.run.Scenario`\\ s).

A concrete model subclasses :class:`ExcitationProcess` and
:class:`ExcitedRunGenerator` (and usually :class:`ExcitationConfig`) to supply
its value spaces, its model factory, its per-step feedback signal and its
special inputs -- see :mod:`economic_models.ground_truth.models.growth.excitation`.
"""

from economic_models.ground_truth.excitation.generator import (
    BranchCapture,
    DrivenModel,
    ExcitedRunGenerator,
)
from economic_models.ground_truth.excitation.process import (
    ExcitationConfig,
    ExcitationProcess,
)

__all__ = [
    "BranchCapture",
    "DrivenModel",
    "ExcitationConfig",
    "ExcitationProcess",
    "ExcitedRunGenerator",
]
