"""The Godley-Lavoie GROWTH model (Chapter 11) as a :class:`BaseEconomicModel`.

The concrete model wires together the five sector modules. It owns every hidden
internal (expectations, targets, sector stocks/flows, behavioural parameters)
directly; the shared visible interface (macro observables + policy levers) lives
on :class:`BaseEconomicModel` and is common to all models.
"""

from pysolve.model import Model

from src.economic_models.base import BaseEconomicModel
from src.economic_models.growth.modules.firms import (
    add_firms_equations,
    add_firms_params,
    add_firms_variables,
)
from src.economic_models.growth.modules.households import (
    add_households_equations,
    add_households_params,
    add_households_variables,
)
from src.economic_models.growth.modules.government import (
    add_government_equations,
    add_government_params,
    add_government_variables,
)
from src.economic_models.growth.modules.central_bank import (
    add_central_bank_equations,
    add_central_bank_params,
    add_central_bank_variables,
)
from src.economic_models.growth.modules.commercial_banks import (
    add_commercial_banks_equations,
    add_commercial_banks_params,
    add_commercial_banks_variables,
)


class GrowthModel(BaseEconomicModel):
    """Stochastic growth model of a monetary economy (Godley & Lavoie, ch. 11).

    The visible state/parameter interface is inherited unchanged from
    :class:`BaseEconomicModel`; this class only supplies the hidden internals
    and equations that realise that interface.
    """

    def __init__(self, dt: float = 1, **param_overrides: float) -> None:
        self.dt = dt
        super().__init__(**param_overrides)

    def build(self, model: Model) -> None:
        model.param("dt", desc="Length of one period in years", default=self.dt)

        add_firms_variables(model)
        add_firms_params(model)
        add_households_variables(model)
        add_households_params(model)
        add_government_variables(model)
        add_government_params(model)
        add_central_bank_variables(model)
        add_central_bank_params(model)
        add_commercial_banks_variables(model)
        add_commercial_banks_params(model)

        add_firms_equations(model)
        add_households_equations(model)
        add_government_equations(model)
        add_central_bank_equations(model)
        add_commercial_banks_equations(model)
