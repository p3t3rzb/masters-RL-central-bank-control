from pysolve.model import Model

from src.modules.firms import NOMINAL_YEAR_AGO


def add_central_bank_equations(model: Model) -> None:
    model.add(
        f"Fcb = Rb(-1)*Bcbd(-1)*{NOMINAL_YEAR_AGO}"
    )  # 11.76 : Central bank profits (interest on bills of one year ago)
    model.add("BLs = BLd")  # 11.77 : Bonds are supplied on demand
    model.add("Bhs = Bhd")  # 11.78 : Household bills supplied on demand
    model.add("Hhs = Hhd")  # 11.79 : Cash supplied on demand
    model.add("Hbs = Hbd")  # 11.80 : Reserves supplied on demand
    model.add("Hs = Hbs + Hhs")  # 11.81 : Total supply of cash
    model.add("Bcbd = Hs")  # 11.82 : Central bankd
    model.add("Bcbs = Bcbd")  # 11.83 : Supply of bills to Central bank
    model.add("Rb = Rbbar")  # 11.84 : Interest rate on bills set exogenously
    model.add("Rbl = Rb + ADDbl")  # 11.85 : Long term interest rate
    model.add("Pbl = 1/Rbl")  # 11.86 : Price of long-term bonds


def add_central_bank_params(model: Model) -> None:
    model.param("ADDbl", desc="Spread between long-term interest rate and rate on bills")
    model.param("epsrb", desc="Speed of adjustment in the real interest rate on bills")
    model.param("Rbbar", desc="Interest rate on bills, set exogenously")
    model.param("RRb", desc="Real interest rate on bills")


def add_central_bank_variables(model: Model) -> None:
    model.var("Bcbd", desc="Government bills demanded by Central bank")
    model.var("Bcbs", desc="Government bills supplied by Central bank")
    model.var("Bhs", desc="Government bills supplied to households")
    model.var("BLs", desc="Supply of government bonds")
    model.var("Fcb", desc='Central bank "profits"')
    model.var("Hbs", desc="Cash supplied to banks")
    model.var("Hhs", desc="Cash supplied to households")
    model.var("Hs", desc="Total supply of cash")
    model.var("Pbl", desc="Price of government bonds")
    model.var("Rb", desc="Interest rate on government bills")
    model.var("Rbl", desc="Interest rate on bonds")
