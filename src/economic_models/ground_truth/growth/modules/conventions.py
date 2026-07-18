"""Shared backdating factors for the GROWTH model's annualized-flows setup.

Interest and coupon flows accrue on the stocks of one *year* ago (Godley &
Lavoie's timing), but ``X(-1)`` is one *period* ago. Multiplying a one-period lag
by the matching ``*_YEAR_AGO`` factor backdates it to its year-ago equivalent:
it divides by the growth compounded over the remaining ``1 - dt`` years,
``growth**(dt - 1)``. At ``dt = 1`` every factor is 1, since the lag already is
the year-ago value. Use the factor matching the quantity's growth trend:

* :data:`NOMINAL_YEAR_AGO` -- nominal stocks/flows, growing at
  ``(1 + GRk)(1 + PI)``;
* :data:`PRICE_YEAR_AGO` -- prices/unit costs, growing at ``1 + PI``;
* :data:`REAL_YEAR_AGO` -- real stocks, growing at ``1 + GRk``.
"""

NOMINAL_YEAR_AGO = "((1 + GRk(-1))*(1 + PI(-1)))**(dt - 1)"  # nominal stocks/flows
PRICE_YEAR_AGO = "(1 + PI(-1))**(dt - 1)"  # prices/unit costs
REAL_YEAR_AGO = "(1 + GRk(-1))**(dt - 1)"  # real stocks
