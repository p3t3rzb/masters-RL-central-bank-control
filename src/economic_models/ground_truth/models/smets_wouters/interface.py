"""The interface a proxy is handed to stand in for the Smets-Wouters model.

The stationarisation rules are simpler here than GROWTH's, and for a reason that
is worth stating: the observables are reconstructed from a model that is already
written in stationary deviations, so the only trending series are the five levels
built by integrating a balanced growth path. Those are log-differenced; the
rates, the employment rate and the fiscal ratios enter as they are.
"""

from __future__ import annotations

from economic_models.ground_truth.models.smets_wouters.variables import (
    SwActions,
    SwParameters,
    SwState,
)
from economic_models.interface import ModelInterface, TransformSpec

SW_INTERFACE = ModelInterface(
    state=SwState,
    parameters=SwParameters,
    actions=SwActions,
    transform_spec=TransformSpec(
        state_names=SwState.names(),
        exog_names=(*SwParameters.names(), *SwActions.names()),
        # The trending levels: everything on the balanced growth path, plus the
        # price level. Hours are stationary around the labour force, so ``L``
        # enters as a level rather than a growth rate.
        log_diff=("Y", "Yk", "P", "Ck", "Ik", "Wk", "L"),
        ratio_to=("GD", "PSBR"),
        denominator="Y",
        exog_log_diff=("Nfe", "Penergy"),
    ),
)
