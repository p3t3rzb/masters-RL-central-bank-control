"""Model-agnostic process specs describing how one exogenous input is excited.

Each spec is a small frozen dataclass: an AR(1) drift, a log random walk, a
stochastic-volatility regime, rare crisis episodes, or a per-run climate draw.
Specs are self-contained (an :class:`AR1Spec` carries its own persistence
``phi``) and dt-aware: annual ``phi``/``sigma`` are discretized to sub-annual
steps by :func:`discretize_ar1` so the annual autocorrelation and stationary
variance are invariant to the timestep.

These are shared across every ground-truth model; a model's own excitation
(e.g. GROWTH's government-spending stabilizer) composes them and adds its
model-specific specs alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


def discretize_ar1(phi: float, sigma: float, dt: float) -> tuple[float, float]:
    """Discretize an annual AR(1) ``(phi, sigma)`` to a step of ``dt`` years.

    Returns ``(phi_dt, sigma_dt)`` with ``phi_dt = phi**dt`` and the matching
    innovation std ``sigma_dt = sigma*sqrt((1 - phi_dt**2)/(1 - phi**2))`` that
    keeps the *annual* autocorrelation and stationary variance invariant to
    ``dt``. At ``dt = 1`` this is exactly the annual process.

    ``phi`` must be strictly inside ``(-1, 1)`` for the process to be stationary;
    at ``|phi| = 1`` the variance-matching factor divides by zero (a random walk
    has no stationary variance -- use :class:`RandomWalkSpec` for that).
    """
    if not -1.0 < phi < 1.0:
        raise ValueError(
            f"AR(1) persistence phi={phi} must be strictly inside (-1, 1) to be "
            "stationary; use RandomWalkSpec for a unit root"
        )
    phi_dt = phi**dt
    sigma_dt = sigma * float(np.sqrt((1.0 - phi_dt**2) / (1.0 - phi**2)))
    return phi_dt, sigma_dt


@dataclass(frozen=True)
class AR1Spec:
    """A persistent, clipped AR(1) deviation of one exogenous input.

    The input drifts as ``dev_t = phi * dev_{t-1} + sigma * eps_t`` around its
    baseline calibration value, with both the deviation and the resulting level
    clipped into ``[lower, upper]`` -- persistent enough to read as slow regime
    drift, bounded so the economy stays inside its stable corridor.
    """

    sigma: float  #: innovation standard deviation
    lower: float  #: lower clip on the resulting level
    upper: float  #: upper clip on the resulting level
    phi: float = 0.9  #: annual AR(1) persistence

    def advance(
        self,
        dev: float,
        base: float,
        rng: np.random.Generator,
        sigma_scale: float = 1.0,
        extra: float = 0.0,
        dt: float = 1.0,
    ) -> tuple[float, float]:
        """Advance one step; return ``(clipped_deviation, clipped_level)``.

        ``sigma_scale`` multiplies the innovation size for this step (the
        stochastic-volatility regime); ``extra`` is an additive shock to the
        *level* only (a crisis impulse) -- it is not folded into the persistent
        AR(1) deviation, so it decays on its own schedule rather than through
        ``phi``. Both the AR(1) deviation and the final level are clipped, so the
        clips double as the hard solver-safety corridor.

        ``dt`` is the timestep in years: ``phi``/``sigma`` are annual and are
        discretized with :func:`discretize_ar1`.
        """
        phi_dt, sigma_dt = discretize_ar1(self.phi, self.sigma, dt)
        dev = phi_dt * dev + sigma_dt * sigma_scale * rng.standard_normal()
        dev = float(np.clip(dev, self.lower - base, self.upper - base))
        return dev, float(np.clip(base + dev + extra, self.lower, self.upper))


@dataclass(frozen=True)
class RandomWalkSpec:
    """A log random walk around a baseline level, clipped in log-deviation.

    Used for level (rather than rate) inputs -- e.g. GROWTH's full-employment
    labour force ``Nfe`` -- so the input drifts multiplicatively as
    ``base * exp(logdev)`` with ``logdev`` a bounded random walk.
    """

    sigma: float  #: innovation standard deviation of the log-deviation
    max_logdev: float  #: maximum absolute log-deviation from baseline

    def advance(
        self, logdev: float, rng: np.random.Generator, sigma_scale: float = 1.0,
        dt: float = 1.0,
    ) -> float:
        """Advance the log-deviation one step and return its clipped value.

        A random walk's increment variance scales with the timestep, so the
        per-step innovation std is ``sigma*sqrt(dt)`` -- the annual increment
        variance is invariant to ``dt``.
        """
        logdev = logdev + self.sigma * np.sqrt(dt) * sigma_scale * rng.standard_normal()
        return float(np.clip(logdev, -self.max_logdev, self.max_logdev))


@dataclass(frozen=True)
class StochasticVolatilitySpec:
    """A persistent multiplier on every input's innovation size.

    Real macro data is heteroskedastic: long calm stretches punctuated by
    turbulent ones, where *all* shocks get bigger at once. Log-volatility follows
    ``logv_t = rho * logv_{t-1} + xi * eps_t`` (clipped to ``+/-max_logvol``); the
    multiplier ``exp(logv_t)`` scales the ``sigma`` of every AR(1)/random-walk
    input for that step. So the same inputs drift slowly for years, then move
    fast for a while -- "sometimes variables change faster, sometimes slower" --
    without changing their central tendency.
    """

    rho: float  #: persistence of log-volatility (volatility clustering)
    xi: float  #: innovation std of log-volatility
    max_logvol: float  #: clip on the absolute log-multiplier

    def advance(self, logvol: float, rng: np.random.Generator, dt: float = 1.0) -> float:
        """Advance log-volatility one step and return its clipped value.

        Like :class:`AR1Spec`, ``rho``/``xi`` are annual and are discretized
        with :func:`discretize_ar1` so the annual persistence and variance of
        the volatility regime are invariant to ``dt``.
        """
        rho_dt, xi_dt = discretize_ar1(self.rho, self.xi, dt)
        logvol = rho_dt * logvol + xi_dt * rng.standard_normal()
        return float(np.clip(logvol, -self.max_logvol, self.max_logvol))

    def multiplier(self, logvol: float) -> float:
        """The innovation-size multiplier for the given log-volatility."""
        return float(np.exp(logvol))


@dataclass(frozen=True)
class CrisisSpec:
    """Rare, recoverable adverse shocks layered on top of the drift.

    Each step, at the per-step hazard implied by the annual ``prob`` (and only if
    at least ``min_gap`` years have passed since the last onset), a crisis
    *episode* erupts: a bundle of signed level shocks that fades geometrically,
    kept separate from the AR(1) state. Several episodes can overlap; an input's
    crisis deviation is the sum over the live episodes. Three things are drawn at
    onset so no two crises look alike:

    * **duration** -- the geometric decay, drawn uniformly from ``decay_range``
      (a sharp spike clearing in a couple of years to a decade-long slump);
    * **severity** -- a common scale on every impulse, drawn from
      ``severity_range``;
    * **character** -- every episode carries the ``impulses`` bundle, and with
      probability ``financial_prob`` also the ``financial_impulses`` bundle.

    A model's own stabilizer recovers most crises; a deep or long enough one can
    still tip the economy past its stable corridor, which surfaces as a truncated
    run rather than a bug.
    """

    prob: float  #: annual onset probability (converted to a per-step hazard by dt)
    impulses: Mapping[str, float]  #: level shock per input at onset
    decay_range: tuple[float, float]  #: per-onset *annual* geometric decay, drawn uniformly
    min_gap: int = 6  #: minimum years between successive onsets
    severity_range: tuple[float, float] = (1.0, 1.0)  #: per-onset magnitude scale
    financial_prob: float = 0.0  #: chance an onset also triggers the financial bundle
    #: extra level shocks added when the financial component fires
    financial_impulses: Mapping[str, float] = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        """Every input a crisis can shock (base and financial), stable order."""
        return tuple({**self.impulses, **self.financial_impulses})


@dataclass(frozen=True)
class ClimateSpec:
    """Per-run turbulence, drawn once so runs differ in character.

    Every run samples a climate ``c ~ Beta(a, b)`` in ``[0, 1]`` and holds it for
    its whole history. ``c`` biases the mean log-volatility by ``(2c-1)*vol_shift``
    and scales the crisis onset probability by
    ``crisis_lo + (crisis_hi-crisis_lo)*c**gamma``. The convex ``gamma`` keeps
    calm and middling climates nearly crisis-free while letting the stormiest
    erupt often, so a single generator stamps out a mix of run characters -- calm
    to crisis-prone -- rather than one archetype.
    """

    a: float = 1.3  #: Beta shape (``a < b`` leans the draw toward calm)
    b: float = 1.7  #: Beta shape toward stormy
    vol_shift: float = 0.35  #: max mean-log-volatility bias from climate
    crisis_lo: float = 0.0  #: crisis-probability multiplier at ``c = 0`` (calm)
    crisis_hi: float = 2.5  #: crisis-probability multiplier at ``c = 1`` (stormy)
    gamma: float = 1.8  #: convexity of the climate -> crisis-probability map

    def draw(self, rng: np.random.Generator) -> float:
        """Sample one run's climate in ``[0, 1]``."""
        return float(rng.beta(self.a, self.b))

    def vol_offset(self, climate: float) -> float:
        """Mean log-volatility bias for the given climate."""
        return (2.0 * climate - 1.0) * self.vol_shift

    def crisis_scale(self, climate: float) -> float:
        """Crisis-onset-probability multiplier for the given climate."""
        return self.crisis_lo + (self.crisis_hi - self.crisis_lo) * climate**self.gamma
