"""The Godley-Lavoie GROWTH model (Chapter 11) as a :class:`PysolveEconomicModel`.

Wires together the five sector modules and owns every hidden internal directly.
A freshly constructed model is fully seeded from its
:class:`~economic_models.ground_truth.growth.calibration.GrowthCalibration`, so
``GrowthModel(Rbbar=0.04)`` just works with no external seeding.
"""

from pysolve.model import Model

from economic_models.ground_truth.base import PysolveEconomicModel
from economic_models.ground_truth.growth.calibration import GrowthCalibration
from economic_models.ground_truth.growth.modules.firms import (
    add_firms_equations,
    add_firms_params,
    add_firms_variables,
)
from economic_models.ground_truth.growth.modules.households import (
    add_households_equations,
    add_households_params,
    add_households_variables,
)
from economic_models.ground_truth.growth.modules.government import (
    add_government_equations,
    add_government_params,
    add_government_variables,
)
from economic_models.ground_truth.growth.modules.central_bank import (
    add_central_bank_equations,
    add_central_bank_params,
    add_central_bank_variables,
)
from economic_models.ground_truth.growth.modules.commercial_banks import (
    add_commercial_banks_equations,
    add_commercial_banks_params,
    add_commercial_banks_variables,
)


class GrowthModel(PysolveEconomicModel):
    """Stochastic growth model of a monetary economy (Godley & Lavoie, ch. 11).

    The visible state/parameter interface is inherited unchanged from
    :class:`PysolveEconomicModel`; this class only supplies the hidden internals
    and equations that realise that interface, and seeds itself from
    ``calibration`` (the book's baseline by default) before any constructor
    ``param_overrides`` are applied.
    """

    def __init__(
        self,
        calibration: GrowthCalibration | None = None,
        *,
        dt: float = 1,
        iterations: int = 200,
        threshold: float = 1e-6,
        **param_overrides: float,
    ) -> None:
        """Construct and fully seed the GROWTH model.

        ``calibration`` supplies the baseline seeding (defaults to the book's
        baseline); ``dt`` is the length of one period in years; ``iterations`` and
        ``threshold`` are the solver's iteration cap and convergence tolerance; any
        ``param_overrides`` set visible exogenous parameters on top of the
        calibration.
        """
        self.dt = dt
        self.calibration = (
            calibration if calibration is not None else GrowthCalibration.baseline()
        )
        super().__init__(
            iterations=iterations, threshold=threshold, **param_overrides
        )

    def _seed(self, model: Model) -> None:
        """Seed the baseline calibration; constructor overrides are applied after."""
        self.calibration.seed(model)

    def build(self, model: Model) -> None:
        """Register the ``dt`` parameter and all five sector modules onto ``model``.

        Adds the variables, parameters and equations of the firms, households,
        government, central-bank and commercial-bank modules that make up the
        GROWTH model.
        """
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
