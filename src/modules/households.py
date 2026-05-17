from pysolve.model import Model


def add_households_equations(model: Model) -> None:
    model.add(
        "YP = WB + FDf + FDb + Rm(-1)*Md(-1) + Rb(-1)*Bhd(-1) + BLs(-1)"
    )  # 11.45 : Personal income
    model.add("T = theta*YP")  # 11.46 : Income taxes
    model.add("YDr = YP - T - Rl(-1)*Lhd(-1)")  # 11.47 : Regular disposable income
    model.add("YDhs = YDr + CG")  # 11.48 : Haig-Simons disposable income
    model.add("CG = d(Pbl)*BLd(-1) + d(Pe)*Ekd(-1) + d(OFb)")  # 11.49 : Capital gains
    model.add(
        "V - V(-1) = YDr - CONS + d(Pe)*Ekd(-1) + d(Pbl)*BLs(-1) + d(OFb)"
    )  # 11.50 : Wealth
    model.add("Vk = V/P")  # 11.51 : Real staock of wealth
    model.add("CONS = Ck*P")  # 11.52 : Consumption
    model.add("Ck = alpha1*(YDkre + NLk) + alpha2*Vk(-1)")  # 11.53 : Real consumption
    model.add(
        "YDkre = eps*YDkr + (1 - eps)*YDkr(-1)*(1 + GRpr)"
    )  # 11.54 : Expected real regular disposable income
    model.add("YDkr = YDr/P - d(P)*Vk(-1)/P")  # 11.55 : Real regular disposable income
    model.add("GL = eta*YDr")  # 11.56 : Gross amount of new personal loans
    model.add("eta = eta0 - etar*RRl")  # 11.57 : New loans to personal income ratio
    model.add("NL = GL - REP")  # 11.58 : Net amount of new personal loans
    model.add("REP = deltarep*Lhd(-1)")  # 11.59 : Personal loans repayments
    model.add("Lhd - Lhd(-1) = GL - REP")  # 11.60 : Demand for personal loans
    model.add("NLk = NL/P")  # 11.61 : Real amount of new personal loans
    model.add("BUR = (REP + Rl(-1)*Lhd(-1))/YDr(-1)")  # 11.62 : Burden of personal debt
    model.add(
        "Bhd = Vfma(-1)*(lambda20 + lambda22*Rb(-1) - lambda21*Rm(-1) - lambda24*Rk(-1) - lambda23*Rbl(-1) - lambda25*YDr/V)"
    )  # 11.64 : Demand for bills
    model.add(
        "BLd = Vfma(-1)*(lambda30 - lambda32*Rb(-1) - lambda31*Rm(-1) - lambda34*Rk(-1) + lambda33*Rbl(-1) - lambda35*YDr/V)/Pbl"
    )  # 11.65 : Demand for bonds
    model.add(
        "Pe = Vfma(-1)*(lambda40 - lambda42*Rb(-1) - lambda41*Rm(-1) + lambda44*Rk(-1) - lambda43*Rbl(-1) - lambda45*YDr/V)/Ekd"
    )  # 11.66 : Demand for equities - normalized to get the price of equitities
    model.add(
        "Md = Vfma - Bhd - Pe*Ekd - Pbl*BLd + Lhd"
    )  # 11.67 : Money deposits - as a residual
    model.add("Vfma = V - Hhd - OFb")  # 11.68 : Investible wealth
    model.add("Hhd = lambdac*CONS")  # 11.69 : Households' demand for cash
    model.add("Ekd = Eks")  # 11.70 : Stock market equilibrium


def add_households_params(model: Model) -> None:
    model.param("alpha1", desc="Propensity to consume out of income")
    model.param("alpha2", desc="Propensity to consume out of wealth")
    model.param("deltarep", desc="Ratio of personal loans repayments to stock of loans")
    model.param("eps", desc="Parameter in expectation formations on real disposable income")
    model.param("eta0", desc="Ratio of new loans to personal income - exogenous component")
    model.param("etar", desc="Relation between the ratio of new loans to personal income and the interest rate on loans")
    model.param("lambda20", desc="Parameter in households demand for bills")
    model.param("lambda21", desc="Parameter in households demand for bills")
    model.param("lambda22", desc="Parameter in households demand for bills")
    model.param("lambda23", desc="Parameter in households demand for bills")
    model.param("lambda24", desc="Parameter in households demand for bills")
    model.param("lambda25", desc="Parameter in households demand for bills")
    model.param("lambda30", desc="Parameter in households demand for bonds")
    model.param("lambda31", desc="Parameter in households demand for bonds")
    model.param("lambda32", desc="Parameter in households demand for bonds")
    model.param("lambda33", desc="Parameter in households demand for bonds")
    model.param("lambda34", desc="Parameter in households demand for bonds")
    model.param("lambda35", desc="Parameter in households demand for bonds")
    model.param("lambda40", desc="Parameter in households demand for equities")
    model.param("lambda41", desc="Parameter in households demand for equities")
    model.param("lambda42", desc="Parameter in households demand for equities")
    model.param("lambda43", desc="Parameter in households demand for equities")
    model.param("lambda44", desc="Parameter in households demand for equities")
    model.param("lambda45", desc="Parameter in households demand for equities")
    model.param("lambdac", desc="Parameter in households demand for cash")
    model.param("theta", desc="Income tax rate")


def add_households_variables(model: Model) -> None:
    model.var("Bhd", desc="Demand for government bills from households")
    model.var("BLd", desc="Demand for government bonds")
    model.var("BUR", desc="Burden of personal debt")
    model.var("Ck", desc="Real consumption")
    model.var("CG", desc="Capital gains on government bonds")
    model.var("CONS", desc="Consumption at current prices")
    model.var("Ekd", desc="Number of equities demanded")
    model.var("eta", desc="Ratio of new loans to personal income")
    model.var("GL", desc="Gross amount of new personal loans")
    model.var("Hhd", desc="Households demand for cash")
    model.var("Lhd", desc="Demand for loans by households")
    model.var("Md", desc="Deposits demanded by households")
    model.var("NL", desc="Net flow of new loans to the household sector")
    model.var("NLk", desc="Real flow of new loans to the household sector")
    model.var("Pe", desc="Price of equities")
    model.var("REP", desc="Personal loans repayments")
    model.var("T", desc="Taxes")
    model.var("V", desc="Wealth of households")
    model.var("Vk", desc="Real wealth of households")
    model.var("Vfma", desc="Investible wealth of households")
    model.var("YDhs", desc="Haig-Simons measure of disposable income")
    model.var("YDr", desc="Regular disposable income")
    model.var("YDkr", desc="Regular real disposable income")
    model.var("YDkre", desc="Expected regular real disposable income")
    model.var("YP", desc="Personal income")
