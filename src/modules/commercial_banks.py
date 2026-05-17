from pysolve.model import Model


def add_commercial_banks_equations(model: Model) -> None:
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
        "Rm - Rm(-1) = z1a*xim1 + z1b*xim2 - z2a*xim1 - z2b*xim2"
    )  # 11.94 : Deposit interest rate
    model.add(
        "z2a = if_true(BLR(-1) > (top + .05))"
    )  # 11.95-97 : Mechanism for determining changes to the interest rate on deposits
    model.add("z2b = if_true(BLR(-1) > top)")
    model.add("z1a = if_true(BLR(-1) <= bot)")
    model.add("z1b = if_true(BLR(-1) <= (bot -.05))")
    model.add("Rl = Rm + ADDl")  # 11.98 : Loan interest rate
    model.add("OFbt = NCAR*(Lfs(-1) + Lhs(-1))")  # 11.99 : Long-run own funds target
    model.add(
        "OFbe = OFb(-1) + betab*(OFbt - OFb(-1))"
    )  # 11.100 : Short-run own funds target
    model.add(
        "FUbt = OFbe - OFb(-1) + NPLke*Lfs(-1)"
    )  # 11.101 : Target retained earnings of banks
    model.add(
        "NPLke = epsb*NPLke(-1) + (1 - epsb)*NPLk(-1)"
    )  # 11.102 : Expected proportion of non-performaing loans
    model.add("FDb = Fb - FUb")  # 11.103 : Dividends of banks
    model.add(
        "Fbt = lambdab*Y(-1) + (OFbe - OFb(-1) + NPLke*Lfs(-1))"
    )  # 11.104 : Target profits of banks
    model.add(
        "Fb = Rl(-1)*(Lfs(-1) + Lhs(-1) - NPL) + Rb(-1)*Bbd(-1) - Rm(-1)*Ms(-1)"
    )  # 11.105 : Actual profits of banks
    model.add(
        "ADDl = (Fbt - Rb(-1)*Bbd(-1) + Rm*(Ms(-1) - (1 - NPLke)*Lfs(-1) - Lhs(-1)))/((1 - NPLke)*Lfs(-1) + Lhs(-1))"
    )  # 11.106 : Lending mark-up over deposit rate
    model.add("FUb = Fb - lambdab*Y(-1)")  # 11.107 : Actual retained earnings
    model.add("OFb - OFb(-1) = FUb - NPL")  # 11.108 : Own funds of banks
    model.add("CAR = OFb/(Lfs + Lhs)")  # 11.109 : Actual capital capacity ratio


def add_commercial_banks_params(model: Model) -> None:
    model.param("betab", desc="Spped of adjustment of banks own funds")
    model.param("bot", desc="Bottom value for bank net liquidity ratio")
    model.param("epsb", desc="Speed of adjustment in expected proportion of non-performing loans")
    model.param("lambdab", desc="Parameter determining dividends of banks")
    model.param("NCAR", desc="Normal capital adequacy ratio of banks")
    model.param("ro", desc="Reserve requirement parameter")
    model.param("top", desc="Top value for bank net liquidity ratio")
    model.param("xim1", desc="Parameter in the equation for setting interest rate on deposits")
    model.param("xim2", desc="Parameter in the equation for setting interest rate on deposits")


def add_commercial_banks_variables(model: Model) -> None:
    model.var("ADDl", desc="Spread between interest rate on loans and rate on deposits")
    model.var("Bbd", desc="Government bills demanded by commercial banks")
    model.var("Bbs", desc="Government bills supplied to commercial banks")
    model.var("BLR", desc="Gross bank liquidity ratio")
    model.var("CAR", desc="Capital adequacy ratio of banks")
    model.var("Fb", desc="Realized banks profits")
    model.var("Fbt", desc="Target profits of banks")
    model.var("FDb", desc="Dividends of banks")
    model.var("FUb", desc="Retained earnings of banks")
    model.var("FUbt", desc="Targt retained earnings of banks")
    model.var("Hbd", desc="Cash required by banks")
    model.var("Lfs", desc="Supply of loans to firms")
    model.var("Lhs", desc="Loans supplied to households")
    model.var("Ms", desc="Deposits supplied by banks")
    model.var("NPLke", desc="Expected proportion of Non-Performing Loans")
    model.var("OFb", desc="Own funds of banks")
    model.var("OFbe", desc="Short-run target for banks own funds")
    model.var("OFbt", desc="Long-run target for banks own funds")
    model.var("Rl", desc="Interest rate on loans")
    model.var("Rm", desc="Interest rate on deposits")
    model.var("z1a", desc="Is one if bank liquidity ratio is below bottom range")
    model.var("z1b", desc="Is one if bank liquidity ratio is below bottom range")
    model.var("z2a", desc="Is one if bank liquidity ratio is above top range")
    model.var("z2b", desc="Is one if bank liquidity ratio is above top range")
