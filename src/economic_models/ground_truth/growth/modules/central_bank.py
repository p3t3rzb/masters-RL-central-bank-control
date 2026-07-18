"""Central bank sector of the GROWTH model (Godley & Lavoie, ch. 11).

Registers the monetary-authority block: equations 11.76-11.86 (central bank
profits, the accommodation of bond/bill/cash/reserve demands, the exogenous
policy rate on bills and the long-term rate and bond price it anchors).
"""

from pysolve.model import Model

from economic_models.ground_truth.growth.modules.conventions import NOMINAL_YEAR_AGO
from economic_models.variables import Actions, Parameters, State

_DESC = {**State.describe(), **Parameters.describe(), **Actions.describe()}


def add_central_bank_equations(model: Model) -> None:
    """Register the central bank equations 11.76-11.86 onto ``model``."""
    model.add(
        f"Fcb = Rb(-1)*Bcbd(-1)*{NOMINAL_YEAR_AGO}"
    )  # 11.76 : Central bank profits (interest on bills of one year ago)
    model.add("BLs = BLd")  # 11.77 : Bonds are supplied on demand
    model.add("Bhs = Bhd")  # 11.78 : Household bills supplied on demand
    model.add("Hhs = Hhd")  # 11.79 : Cash supplied on demand
    model.add("Hbs = Hbd")  # 11.80 : Reserves supplied on demand
    model.add("Hs = Hbs + Hhs")  # 11.81 : Total supply of cash
    model.add("Bcbd = Hs")  # 11.82 : Central bank demand for bills
    model.add("Bcbs = Bcbd")  # 11.83 : Supply of bills to Central bank
    model.add("Rb = Rbbar")  # 11.84 : Interest rate on bills set exogenously
    model.add("Rbl = Rb + ADDbl")  # 11.85 : Long term interest rate
    model.add("Pbl = 1/Rbl")  # 11.86 : Price of long-term bonds


def add_central_bank_params(model: Model) -> None:
    """Register the central bank's exogenous parameters onto ``model``."""
    model.param("ADDbl", desc=_DESC["ADDbl"])
    model.param("Rbbar", desc=_DESC["Rbbar"])


def add_central_bank_variables(model: Model) -> None:
    """Register the central bank's endogenous variables onto ``model``."""
    model.var("Bcbd", desc="Government bills demanded by Central bank")
    model.var("Bcbs", desc="Government bills supplied by Central bank")
    model.var("Bhs", desc="Government bills supplied to households")
    model.var("BLs", desc="Supply of government bonds")
    model.var("Fcb", desc='Central bank "profits"')
    model.var("Hbs", desc="Cash supplied to banks")
    model.var("Hhs", desc="Cash supplied to households")
    model.var("Hs", desc="Total supply of cash")
    model.var("Pbl", desc="Price of government bonds")
    model.var("Rb", desc=_DESC["Rb"])
    model.var("Rbl", desc=_DESC["Rbl"])
