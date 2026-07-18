"""Commercial banks sector of the GROWTH model (Godley & Lavoie, ch. 11).

Registers the banking block: equations 11.87-11.109 (deposit and loan supply,
reserve requirements, the bank liquidity ratio and the band mechanism setting
the deposit rate, own-funds and capital-adequacy targets, bank profits,
dividends and the lending mark-up).
"""

from pysolve.model import Model

from economic_models.ground_truth.growth.modules.conventions import NOMINAL_YEAR_AGO
from economic_models.variables import Actions, Parameters, State

_DESC = {**State.describe(), **Parameters.describe(), **Actions.describe()}


def add_commercial_banks_equations(model: Model) -> None:
    """Register the commercial banks equations 11.87-11.109 onto ``model``."""
    model.add("Ms = Md")  # 11.87 : Bank deposits supplied on demand
    model.add("Lfs = Lfd")  # 11.88 : Loans to firms supplied on demand
    model.add("Lhs = Lhd")  # 11.89 : Personal loans supplied on demand
    model.add("Hbd = ro*Ms")  # 11.90 : Reserve requirements of banks
    model.add(
        "Bbs - Bbs(-1) = d(Bs) - d(Bhs) - d(Bcbs)"
    )  # 11.91 : Bills supplied to banks
    model.add(
        "Bbd = Ms + OFb - Lfs - Lhs - Hbd"
    )  # 11.92 : Balance sheet constraint of banks
    model.add("BLR = Bbd/Ms")  # 11.93 : Bank liquidity ratio
    model.add(
        "Rm - Rm(-1) = dt*(z1a*xim1 + z1b*xim2 - z2a*xim1 - z2b*xim2)"
    )  # 11.94 : Deposit interest rate
    model.add(
        "z2a = if_true(BLR(-1) > (top + BANDlr))"
    )  # 11.95-97 : Mechanism for determining changes to the interest rate on deposits
    model.add("z2b = if_true(BLR(-1) > top)")
    model.add("z1a = if_true(BLR(-1) <= bot)")
    model.add("z1b = if_true(BLR(-1) <= (bot - BANDlr))")
    model.add("Rl = Rm + ADDl")  # 11.98 : Loan interest rate
    model.add(
        f"OFbt = NCAR*(Lfs(-1) + Lhs(-1))*{NOMINAL_YEAR_AGO}"
    )  # 11.99 : Long-run own funds target (based on loans of one year ago)
    # 11.100 : Short-run own funds target, adjusting toward OFbt. The correction
    # factor on OFbt keeps the steady-state capital adequacy ratio independent of
    # dt (OFbt grows at the nominal rate (1+GRk(-1))*(1+PI(-1)) - 1); it equals 1
    # at dt=1.
    model.add(
        "OFbe = OFb(-1) + (1 - (1-betab)**dt)*(OFbt"
        "*(betab*(1 + GRk(-1))*(1 + PI(-1))/((1 + GRk(-1))*(1 + PI(-1)) - 1 + betab))"
        "*((((1 + GRk(-1))*(1 + PI(-1)))**dt - 1 + (1 - (1-betab)**dt))"
        "/((1 - (1-betab)**dt)*((1 + GRk(-1))*(1 + PI(-1)))**dt))"
        " - OFb(-1))"
    )
    model.add(
        f"FUbt = (OFbe - OFb(-1))/dt + NPLke*Lfs(-1)*{NOMINAL_YEAR_AGO}"
    )  # 11.101 : Target retained earnings of banks
    model.add(
        "NPLke = epsb**dt*NPLke(-1) + (1 - epsb**dt)*NPLk(-1)"
    )  # 11.102 : Expected proportion of non-performing loans
    model.add("FDb = Fb - FUb")  # 11.103 : Dividends of banks
    model.add(
        f"Fbt = lambdab*Y(-1)*{NOMINAL_YEAR_AGO} + ((OFbe - OFb(-1))/dt + NPLke*Lfs(-1)*{NOMINAL_YEAR_AGO})"
    )  # 11.104 : Target profits of banks
    model.add(
        f"Fb = Rl(-1)*((Lfs(-1) + Lhs(-1))*{NOMINAL_YEAR_AGO} - NPL*dt) + (Rb(-1)*Bbd(-1) - Rm(-1)*Ms(-1))*{NOMINAL_YEAR_AGO}"
    )  # 11.105 : Actual profits of banks (interest on stocks of one year ago)
    model.add(
        f"ADDl = (Fbt - Rb(-1)*Bbd(-1)*{NOMINAL_YEAR_AGO} + Rm*(Ms(-1) - (1 - NPLke*dt)*Lfs(-1) - Lhs(-1))*{NOMINAL_YEAR_AGO})/(((1 - NPLke*dt)*Lfs(-1) + Lhs(-1))*{NOMINAL_YEAR_AGO})"
    )  # 11.106 : Lending mark-up over deposit rate
    model.add(
        f"FUb = Fb - lambdab*Y(-1)*{NOMINAL_YEAR_AGO}"
    )  # 11.107 : Actual retained earnings
    model.add("OFb - OFb(-1) = dt*(FUb - NPL)")  # 11.108 : Own funds of banks
    model.add("CAR = OFb/(Lfs + Lhs)")  # 11.109 : Actual capital adequacy ratio


def add_commercial_banks_params(model: Model) -> None:
    """Register the commercial banks' exogenous parameters onto ``model``."""
    model.param("BANDlr", desc="Outer offset on the bank liquidity ratio band (top/bot)")
    model.param("betab", desc="Speed of adjustment of banks own funds")
    model.param("bot", desc="Bottom value for bank net liquidity ratio")
    model.param("epsb", desc="Speed of adjustment in expected proportion of non-performing loans")
    model.param("lambdab", desc="Parameter determining dividends of banks")
    model.param("NCAR", desc=_DESC["NCAR"])
    model.param("ro", desc=_DESC["ro"])
    model.param("top", desc="Top value for bank net liquidity ratio")
    model.param("xim1", desc="Parameter in the equation for setting interest rate on deposits")
    model.param("xim2", desc="Parameter in the equation for setting interest rate on deposits")


def add_commercial_banks_variables(model: Model) -> None:
    """Register the commercial banks' endogenous variables onto ``model``."""
    model.var("ADDl", desc="Spread between interest rate on loans and rate on deposits")
    model.var("Bbd", desc="Government bills demanded by commercial banks")
    model.var("Bbs", desc="Government bills supplied to commercial banks")
    model.var("BLR", desc="Gross bank liquidity ratio")
    model.var("CAR", desc="Capital adequacy ratio of banks")
    model.var("Fb", desc="Realized banks profits")
    model.var("Fbt", desc="Target profits of banks")
    model.var("FDb", desc="Dividends of banks")
    model.var("FUb", desc="Retained earnings of banks")
    model.var("FUbt", desc="Target retained earnings of banks")
    model.var("Hbd", desc="Cash required by banks")
    model.var("Lfs", desc="Supply of loans to firms")
    model.var("Lhs", desc="Loans supplied to households")
    model.var("Ms", desc="Deposits supplied by banks")
    model.var("NPLke", desc="Expected proportion of Non-Performing Loans")
    model.var("OFb", desc="Own funds of banks")
    model.var("OFbe", desc="Short-run target for banks own funds")
    model.var("OFbt", desc="Long-run target for banks own funds")
    model.var("Rl", desc=_DESC["Rl"])
    model.var("Rm", desc=_DESC["Rm"])
    model.var("z1a", desc="Is one if bank liquidity ratio is below bottom range")
    model.var("z1b", desc="Is one if bank liquidity ratio is below bottom range")
    model.var("z2a", desc="Is one if bank liquidity ratio is above top range")
    model.var("z2b", desc="Is one if bank liquidity ratio is above top range")
