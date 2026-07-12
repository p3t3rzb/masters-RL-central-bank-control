from pysolve.model import Model
from pysolve.utils import is_close, round_solution

import matplotlib.pyplot as plt

from src.modules.firms import add_firms_equations, add_firms_params, add_firms_variables
from src.modules.households import (
    add_households_equations,
    add_households_params,
    add_households_variables,
)
from src.modules.government import (
    add_government_equations,
    add_government_params,
    add_government_variables,
)
from src.modules.central_bank import (
    add_central_bank_equations,
    add_central_bank_params,
    add_central_bank_variables,
)
from src.modules.commercial_banks import (
    add_commercial_banks_equations,
    add_commercial_banks_params,
    add_commercial_banks_variables,
)


def create_growth_model(dt=1):
    model = Model()

    model.set_var_default(0)
    model.param("dt", desc="Length of one period in years", default=dt)

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

    return model
