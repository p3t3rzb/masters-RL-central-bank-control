from pysolve.model import Model

from src.modules.firms import NOMINAL_YEAR_AGO


def add_government_equations(model: Model) -> None:
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
    model.param("GRg", desc="growth_mod of real government expenditures")


def add_government_variables(model: Model) -> None:
    model.var("Bs", desc="Supply of government bills")
    model.var("G", desc="Government expenditures")
    model.var("Gk", desc="Real government expenditures")
    model.var("GD", desc="Government debt")
    model.var("PSBR", desc="Government deficit")
