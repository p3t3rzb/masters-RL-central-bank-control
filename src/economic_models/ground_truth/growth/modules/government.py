"""Government sector of the GROWTH model (Godley & Lavoie, ch. 11).

Registers the fiscal block: equations 11.71-11.75 (nominal and real government
expenditures, the government deficit, new bill issues and total government
debt).
"""

from pysolve.model import Model

from economic_models.ground_truth.growth.modules.conventions import NOMINAL_YEAR_AGO
from economic_models.variables import Parameters, State

_DESC = {**State.describe(), **Parameters.describe()}


def add_government_equations(model: Model) -> None:
    """Register the government equations 11.71-11.75 onto ``model``."""
    model.add("G = Gk*P")  # 11.71 : Pure government expenditures
    model.add("Gk = Gk(-1)*(1 + GRg)**dt")  # 11.72 : Real government expenditures
    model.add(
        f"PSBR = G + (BLs(-1) + Rb(-1)*(Bbs(-1) + Bhs(-1)))*{NOMINAL_YEAR_AGO} - T"
    )  # 11.73 : Government deficit
    model.add(
        f"Bs - Bs(-1) = dt*(G - T + (Rb(-1)*(Bhs(-1) + Bbs(-1)) + BLs(-1))*{NOMINAL_YEAR_AGO}) - d(BLs)*Pbl"
    )  # 11.74 : New issues of bills
    model.add("GD = Bbs + Bhs + BLs*Pbl + Hs")  # 11.75 : Government debt


def add_government_params(model: Model) -> None:
    """Register the government's exogenous parameters onto ``model``."""
    model.param("GRg", desc=_DESC["GRg"])


def add_government_variables(model: Model) -> None:
    """Register the government's endogenous variables onto ``model``."""
    model.var("Bs", desc="Supply of government bills")
    model.var("G", desc="Government expenditures")
    model.var("Gk", desc="Real government expenditures")
    model.var("GD", desc=_DESC["GD"])
    model.var("PSBR", desc=_DESC["PSBR"])
