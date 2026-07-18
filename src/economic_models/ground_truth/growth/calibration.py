"""Baseline calibration of the GROWTH model (Godley & Lavoie, ch. 11).

A :class:`GrowthCalibration` partitions every starting value of a
:class:`~economic_models.ground_truth.growth.model.GrowthModel` by what it is:

* ``structural_params`` -- behavioural/structural ``model.param`` values (not
  part of the visible interface);
* ``exogenous_baselines`` -- baseline values of the exogenous inputs (the visible
  :class:`~economic_models.variables.Parameters` /
  :class:`~economic_models.variables.Actions` plus the hidden shock ``RA``);
* ``initial_state`` -- initial ``model.var`` values, *ordered* so string
  initialisers (e.g. ``('Hs', 'Hbd + Hhd')``) resolve against earlier ones.

:meth:`GrowthCalibration.seed` applies all three in the right order, so the
ordering contract lives in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from pysolve.model import Model


@dataclass(frozen=True)
class GrowthCalibration:
    """A complete set of starting values for a GROWTH model.

    Apply with :meth:`seed`; the three partitions are set in the order
    ``structural_params`` -> ``exogenous_baselines`` -> ``initial_state`` so
    that string cross-references in ``initial_state`` resolve correctly.
    """

    #: behavioural/structural parameters (name -> value)
    structural_params: Mapping[str, float]
    #: baseline values of the exogenous inputs, visible and hidden (name -> value)
    exogenous_baselines: Mapping[str, float]
    #: initial values of endogenous variables, in application order; a string
    #: value is a pysolve expression evaluated against previously-set values
    initial_state: tuple[tuple[str, float | str], ...]

    def seed(self, model: Model) -> None:
        """Apply the whole calibration onto a pysolve ``model``, in order."""
        model.set_values(dict(self.structural_params))
        model.set_values(dict(self.exogenous_baselines))
        model.set_values(list(self.initial_state))

    def baselines(self) -> dict[str, float]:
        """Baseline value of every model *parameter*, structural and exogenous.

        The mapping excitation processes draw their drift baselines from.
        """
        return {**self.structural_params, **self.exogenous_baselines}

    @classmethod
    def baseline(cls) -> "GrowthCalibration":
        """The book's baseline calibration (Godley & Lavoie, ch. 11)."""
        structural_params = {
            "alpha1": 0.75,  # propensity to consume out of income (11.53)
            "alpha2": 0.064,  # propensity to consume out of wealth (11.53)
            "beta": 0.5,  # sales-expectation adjustment (11.2)
            "betab": 0.4,  # banks' own-funds adjustment speed (11.100)
            "gamma": 0.15,  # inventory-target adjustment speed (11.4)
            "gamma0": 0.00122,  # investment animal spirits (11.7)
            "gammar": 0.1,  # real-rate effect on accumulation (11.7)
            "gammau": 0.05,  # utilization effect on accumulation (11.7)
            "delta": 0.10667,  # depreciation rate of fixed capital (11.11)
            "deltarep": 0.1,  # personal-loan repayment ratio (11.59)
            "eps": 0.5,  # disposable-income expectation adjustment (11.54)
            "eps2": 0.8,  # mark-up adjustment speed (11.30)
            "epsb": 0.25,  # expected-NPL adjustment speed (11.102)
            "eta0": 0.07416,  # new-loans-to-income ratio, exogenous part (11.57)
            "etan": 0.6,  # employment adjustment speed (11.24)
            "etar": 0.4,  # rate effect on new-loans ratio (11.57)
            "lambda20": 0.25,  # household demand for bills (11.64)
            "lambda21": 2.2,
            "lambda22": 6.6,
            "lambda23": 2.2,
            "lambda24": 2.2,
            "lambda25": 0.1,
            "lambda30": -0.04341,  # household demand for bonds (11.65)
            "lambda31": 2.2,
            "lambda32": 2.2,
            "lambda33": 6.6,
            "lambda34": 2.2,
            "lambda35": 0.1,
            "lambda40": 0.67132,  # household demand for equities (11.66)
            "lambda41": 2.2,
            "lambda42": 2.2,
            "lambda43": 2.2,
            "lambda44": 6.6,
            "lambda45": 0.1,
            "lambdab": 0.0153,  # bank dividend ratio (11.104)
            "lambdac": 0.05,  # household demand for cash (11.69)
            "xim1": 0.0008,  # deposit-rate step size (11.94)
            "xim2": 0.0007,  # deposit-rate step size (11.94)
            "sigman": 0.1666,  # normal historic unit cost weight (11.28)
            "sigmat": 0.2,  # target inventories-to-sales ratio (11.3)
            "psid": 0.15255,  # dividends-to-gross-profits ratio (11.36)
            "psiu": 0.92,  # retained-earnings-to-investment ratio (11.35)
            "omega0": -0.20594,  # target real wage (11.18)
            "omega1": 1,
            "omega2": 2,
            "omega3": 0.45621,  # wage adjustment speed (11.21)
            "BANDt": 0.01,  # upper range of the flat Phillips curve (11.20)
            "BANDb": 0.01,  # lower range of the flat Phillips curve (11.20)
            "bot": 0.05,  # bottom of the bank liquidity ratio band (11.95-97)
            "top": 0.12,  # top of the bank liquidity ratio band (11.95-97)
            "BANDlr": 0.05,  # outer offset on the liquidity band (11.95, 11.97)
        }
        exogenous_baselines = {
            # visible Parameters
            "ADDbl": 0.02,  # bond-bill spread (11.85)
            "GRg": 0.03,  # growth of real government expenditures (11.72)
            "GRpr": 0.03,  # productivity growth (11.22)
            "Nfe": 87.181,  # full-employment level (11.19)
            "NPLk": 0.02,  # proportion of non-performing loans (11.40)
            "Rln": 0.07,  # normal interest rate on loans (11.28)
            "theta": 0.22844,  # income tax rate (11.46)
            # visible Actions
            "Rbbar": 0.035,  # policy rate on bills (11.84)
            "NCAR": 0.1,  # normal capital adequacy ratio (11.99)
            "ro": 0.05,  # reserve requirement ratio (11.90)
            # hidden exogenous
            "RA": 0,  # random shock to sales expectations (11.2)
        }
        initial_state = (
            ("eta", 0.04918),  # new-loans-to-income ratio (11.57)
            ("sigmase", 0.16667),  # opening inventories to expected sales (11.33)
            ("phi", 0.26417),  # mark-up on unit costs (11.30)
            ("phit", 0.26417),  # ideal mark-up (11.31)
            ("ADDl", 0.04592),
            ("BLR", 0.1091),
            ("BUR", 0.06324),
            ("Ck", 7334240),
            ("CAR", 0.09245),
            ("CONS", 52603100),
            ("ER", 1),
            ("Fb", 1744130),
            ("Fbt", 1744140),
            ("Ff", 18081100),
            ("Fft", 18013600),
            ("FDb", 1325090),
            ("FDf", 2670970),
            ("FUb", 419039),
            ("FUf", 15153800),
            ("FUft", 15066200),
            ("G", 16755600),
            ("Gk", 2336160),
            ("GL", 2775900),
            ("GRk", 0.03001),
            ("INV", 16911600),
            ("Ik", 2357910),
            ("N", "Nfe"),
            ("Nt", "Nfe"),
            ("NHUC", 5.6735),
            ("NL", 683593),
            ("NLk", 95311),
            ("NPL", 309158),
            ("NPLke", 0.02),
            ("NUC", 5.6106),
            ("omegat", 112852),
            ("P", 7.1723),
            ("Pbl", 18.182),
            ("Pe", 17937),
            ("PE", 5.07185),
            ("PI", 0.0026),
            ("PR", 138659),
            ("PSBR", 1894780),
            ("Q", 0.77443),
            ("Rb", 0.035),
            ("Rbl", 0.055),
            ("Rk", 0.03008),
            ("Rl", 0.06522),
            ("Rm", 0.0193),
            ("REP", 2092310),
            ("RRl", 0.06246),
            ("S", 86270300),
            ("Sk", 12028300),
            ("Ske", "Sk"),
            ("T", 17024100),
            ("U", 0.70073),
            ("UC", 5.6106),
            ("W", 777968),
            ("WB", 67824000),
            ("Y", 86607700),
            ("Yk", 12088400),
            ("YDr", 56446400),
            ("YDkr", 7813270),
            ("YDkre", 7813290),
            ("YP", 73158700),
            ("z1a", 0),
            ("z1b", 0),
            ("z2a", 0),
            ("z2b", 0),
            ("Bbd", 4388930),
            ("Bbs", 4389790),
            ("Bcbd", 4655690),
            ("Bcbs", 4655690),
            ("Bhd", 33396900),
            ("Bhs", "Bhd"),
            ("Bs", 42484800),
            ("BLd", 840742),
            ("BLs", "BLd"),
            ("GD", 57728700),
            ("Ekd", 5112.6001),
            ("Eks", "Ekd"),
            ("Hbd", 2025540),
            ("Hbs", "Hbd"),
            ("Hhd", 2630150),
            ("Hhs", "Hhd"),
            ("Hs", "Hbd + Hhd"),
            ("IN", 11585400),
            ("INk", 2064890),
            ("INke", 2405660),
            ("INkt", "INk"),
            ("K", 127444000),
            ("Kk", 17768900),
            ("Lfd", 15962900),
            ("Lfs", "Lfd"),
            ("Lhd", 21606600),
            ("Lhs", "Lhd"),
            ("Md", 40510800),
            ("Ms", "Md"),
            ("OFb", 3473280),
            ("OFbe", 3782430),
            ("OFbt", 3638100),
            ("V", 165395000),
            ("Vfma", 159291000),
            ("Vk", 22576100),
        )
        return cls(
            structural_params=structural_params,
            exogenous_baselines=exogenous_baselines,
            initial_state=initial_state,
        )
