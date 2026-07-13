from pysolve.model import Model

NOMINAL_YEAR_AGO = "((1 + GRk(-1))*(1 + PI(-1)))**(dt - 1)"  # nominal stocks/flows
PRICE_YEAR_AGO = "(1 + PI(-1))**(dt - 1)"  # prices/unit costs
REAL_YEAR_AGO = "(1 + GRk(-1))**(dt - 1)"  # real stocks


def add_firms_equations(model: Model) -> None:
    model.add("Yk = Ske + (INke - INk(-1))/dt")  # 11.1 : Real output
    model.add(
        "Ske = (1 - (1-beta)**dt)*Sk + (1-beta)**dt*Sk(-1)*(1 + (GRpr + RA))**dt"
    )  # 11.2 : Expected real sales
    model.add("INkt = sigmat*Ske")  # 11.3 : Long-run inventory target
    model.add(
        "INke = INk(-1) + (1 - (1-gamma)**dt)*(INkt - INk(-1))"
    )  # 11.4 : Short-run inventory target
    model.add("INk - INk(-1) = dt*(Yk - Sk - NPL/UC)")  # 11.5 : Actual real inventories
    model.add("Kk = Kk(-1)*(1 + GRk)**dt")  # 11.6 : Real capital stock
    model.add(
        "GRk = gamma0 + gammau*U(-1) - gammar*RRl"
    )  # 11.7 : Growth of real capital stock
    model.add(
        "U = Yk/(Kk(-1)*(1 + GRk(-1))**(dt - 1))"
    )  # 11.8 : Capital utilization proxy (relative to capital one year ago)
    model.add("RRl = ((1 + Rl)/(1 + PI)) - 1")  # 11.9 : Real interest rate on loans
    # 11.10 : Price inflation -- the one-period price change annualized and
    # exponentially smoothed over roughly one year. Reduces to d(P)/P(-1) at dt=1.
    model.add("PI = dt*((P/P(-1))**(1/dt) - 1) + (1 - dt)*PI(-1)")
    # 11.11 : Real gross investment (net accumulation GRk*Kk plus depreciation
    # delta*Kk), both measured against the capital stock of one year ago.
    model.add("Ik = (GRk + delta)*Kk(-1)*(1 + GRk(-1))**(dt - 1)")
    model.add("Sk = Ck + Gk + Ik")  # 11.12 : Actual real sales
    model.add("S = Sk*P")  # 11.13 : Value of realized sales
    model.add("IN = INk*UC")  # 11.14 : Inventories valued at current cost
    model.add("INV = Ik*P")  # 11.15 : Nominal gross investment
    model.add("K = Kk*P")  # 11.16 : Nominal value of fixed capital
    model.add("Y = Sk*P + (d(INk)/dt)*UC")  # 11.17 : Nominal GDP
    model.add(
        "omegat = exp(omega0 + omega1*log(PR) + omega2*log(ER + z3*(1 - ER) - z4*BANDt + z5*BANDb))"
    )  # 11.18 : Real wage aspirations
    model.add("ER = N(-1)/Nfe(-1)")  # 11.19 : Employment rate
    model.add(
        "z3 = if_true(ER > (1-BANDb)) * if_true(ER <= (1+BANDt))"
    )  # 11.20 : Switch variables
    model.add("z4 = if_true(ER > (1+BANDt))")
    model.add("z5 = if_true(ER < (1-BANDb))")
    # 11.21 : Nominal wage, adjusting toward the real-wage target omegat. The
    # correction factor on the target keeps the steady-state target-to-wage
    # ratio omegat*P/W (and hence the wage-Phillips equilibrium) independent of
    # dt; it equals 1 at dt=1.
    model.add(
        "W - W(-1) = (1 - (1-omega3)**dt)"
        "*(omegat*P(-1)"
        "*(1 + (((1 + PI(-1))*(1 + GRpr))**dt - 1)/(1 - (1-omega3)**dt))"
        "*(1 + GRpr)**(1 - dt)"
        "/(1 + ((1 + PI(-1))*(1 + GRpr) - 1)/omega3)"
        " - W(-1))"
    )
    model.add("PR = PR(-1)*(1 + GRpr)**dt")  # 11.22 : Labor productivity
    model.add("Nt = Yk/PR")  # 11.23 : Desired employment
    model.add(
        "N - N(-1) = (1 - (1-etan)**dt)*(Nt - N(-1))"
    )  # 11.24 : Actual employment
    model.add("WB = N*W")  # 11.25 : Nominal wage bill
    model.add("UC = WB/Yk")  # 11.26 : Actual unit cost
    model.add("NUC = W/PR")  # 11.27 : Normal unit cost
    model.add(
        f"NHUC = (1 - sigman)*NUC + sigman*(1 + Rln(-1))*NUC(-1)*{PRICE_YEAR_AGO}"
    )  # 11.28 : Normal historic unit cost
    model.add("P = (1 + phi)*NHUC")  # 11.29 : Normal-cost pricing
    model.add(
        "phi - phi(-1) = (1 - (1-eps2)**dt)*(phit(-1) - phi(-1))"
    )  # 11.30 : Actual mark-up
    model.add(
        f"phit = (FDf + FUft + Rl(-1)*(Lfd(-1) - IN(-1))*{NOMINAL_YEAR_AGO})/((1 - sigmase)*Ske*UC + (1 + Rl(-1))*sigmase*Ske*UC(-1)*{PRICE_YEAR_AGO})"
    )  # 11.31 : Ideal mark-up (interest on loans/inventories of one year ago)
    model.add(
        f"HCe = (1 - sigmase)*Ske*UC + (1 + Rl(-1))*sigmase*Ske*UC(-1)*{PRICE_YEAR_AGO}"
    )  # 11.32 : Expected historical costs
    model.add(
        f"sigmase = INk(-1)*{REAL_YEAR_AGO}/Ske"
    )  # 11.33 : Opening (year-ago) inventories to expected sales ratio
    model.add(
        f"Fft = FUft + FDf + Rl(-1)*(Lfd(-1) - IN(-1))*{NOMINAL_YEAR_AGO}"
    )  # 11.34 : Planned entrepreneurial profits of firms
    model.add(
        f"FUft = psiu*INV(-1)*{NOMINAL_YEAR_AGO}"
    )  # 11.35 : Planned retained earnings of firms
    model.add(f"FDf = psid*Ff(-1)*{NOMINAL_YEAR_AGO}")  # 11.36 : Dividends of firms
    model.add(
        f"Ff = S - WB + d(IN)/dt - Rl(-1)*IN(-1)*{NOMINAL_YEAR_AGO}"
    )  # 11.37 : Realized entrepreneurial profits
    model.add(
        f"FUf = Ff - FDf - Rl(-1)*(Lfd(-1) - IN(-1))*{NOMINAL_YEAR_AGO} + Rl(-1)*NPL*dt"
    )  # 11.38 : Retained earnings of firms
    model.add(
        "Lfd - Lfd(-1) = dt*(INV - FUf - NPL) + d(IN) - d(Eks)*Pe"
    )  # 11.39 : Demand for loans by firms
    model.add(
        f"NPL = NPLk*Lfs(-1)*{NOMINAL_YEAR_AGO}"
    )  # 11.40 : Defaulted loans (proportion of loans of one year ago)
    model.add(
        f"Eks - Eks(-1) = dt*((1 - psiu)*INV(-1)*{NOMINAL_YEAR_AGO})/Pe"
    )  # 11.41 : Supply of equities issued by firms
    model.add(
        f"Rk = FDf/(Pe(-1)*Ekd(-1)*{NOMINAL_YEAR_AGO})"
    )  # 11.42 : Dividend yield of firms
    model.add("PE = Pe/(Ff/Eks(-1))")  # 11.43 : Price earnings ratio
    model.add("Q = (Eks*Pe + Lfd)/(K + IN)")  # 11.44 : Tobin's Q ratio


def add_firms_params(model: Model) -> None:
    model.param("beta", desc="Parameter in expectation formations on real sales")
    model.param("delta", desc="Rate of depreciation of fixed capital")
    model.param("eps2", desc="Speed of adjustment of mark-up")
    model.param(
        "etan", desc="Speed of adjustment of actual employment to desired employment"
    )
    model.param("gamma", desc="Speed of adjustment of inventories to the target level")
    model.param("gamma0", desc="Exogenous growth in the real stock of capital")
    model.param(
        "gammar",
        desc="Relation between the real interest rate and growth in the stock of capital",
    )
    model.param(
        "gammau",
        desc="Relation between the utilization rate and growth in the stock of capital",
    )
    model.param("psid", desc="Ratio of dividends to gross profits")
    model.param("psiu", desc="Ratio of retained earnings to investments")
    model.param("sigman", desc="Parameter of influencing normal historic unit costs")
    model.param("sigmas", desc="Realized inventories to sales ratio")
    model.param("sigmat", desc="Target inventories to sales ratio")
    model.param("omega0", desc="Parameter influencing the target real wage for workers")
    model.param("omega1", desc="Parameter influencing the target real wage for workers")
    model.param("omega2", desc="Parameter influencing the target real wage for workers")
    model.param("omega3", desc="Speed of adjustment of wages to target value")
    model.param("BANDb", desc="Lower range of the flat Phillips curve")
    model.param("BANDt", desc="Upper range of the flat Phillips curve")
    model.param("GRpr", desc="Growth rate of productivity")
    model.param("Nfe", desc="Full employment level")
    model.param("NPLk", desc="Proportion of Non-Performing loans")
    model.param("RA", desc="Random shock to expectations on real sales")
    model.param("Rln", desc="Normal interest rate on loans")


def add_firms_variables(model: Model) -> None:
    model.var("Eks", desc="Number of equities supplied by firms")
    model.var("ER", desc="Employment rate")
    model.var("Ff", desc="Realized entrepreneurial profits")
    model.var("Fft", desc="Planned entrepreneurial profits")
    model.var("FDf", desc="Dividends of firms")
    model.var("FUf", desc="Retained earnings of firms")
    model.var("FUft", desc="Planned retained earnings of firms")
    model.var("GRk", desc="Growth of real capital stock")
    model.var("HCe", desc="Expected historical costs")
    model.var("INV", desc="Gross investment")
    model.var("Ik", desc="Gross investment in real terms")
    model.var("IN", desc="Stock of inventories at current costs")
    model.var("INk", desc="Real inventories")
    model.var("INke", desc="Expected real inventories")
    model.var("INkt", desc="Target level of real inventories")
    model.var("K", desc="Capital stock")
    model.var("Kk", desc="Real capital stock")
    model.var("Lfd", desc="Demand for loans by firms")
    model.var("N", desc="Employment level")
    model.var("Nt", desc="Desired employment level")
    model.var("NHUC", desc="Normal historic unit cost")
    model.var("NPL", desc="Non-Performing loans")
    model.var("NUC", desc="Normal unit cost")
    model.var("omegat", desc="Target real wage for workers")
    model.var("P", desc="Price level")
    model.var("PE", desc="Price earnings ratio")
    model.var("PI", desc="Price inflation")
    model.var("PR", desc="Labor productivity")
    model.var("Q", desc="Tobin's Q")
    model.var("Rk", desc="Dividend yield of firms")
    model.var("RRl", desc="Real interest rate on loans")
    model.var("S", desc="Sales at current prices")
    model.var("Sk", desc="Real sales")
    model.var("Ske", desc="Expected real sales")
    model.var("U", desc="Capital utilization proxy")
    model.var("UC", desc="Unit costs")
    model.var("W", desc="Wage rate")
    model.var("WB", desc="The wage bill")
    model.var("Y", desc="Output at current prices (nominal GDP)")
    model.var("Yk", desc="Real output")
    model.var("phi", desc="Mark-up on unit costs")
    model.var("phit", desc="Ideal mark-up on unit costs")
    model.var("z3", desc="Parameter in wage aspiration equation")
    model.var("z4", desc="Parameter in wage aspiration equation")
    model.var("z5", desc="Parameter in wage aspiration equation")
    model.var("sigmase", desc="Opening inventories to expected sales ratio")
