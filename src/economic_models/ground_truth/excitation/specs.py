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

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ExcitationJitter:
    """How far a *perturbed* copy of a spec may wander from the one it came from.

    A preset names one economy. Two banks of futures drawn from it differ in
    their shocks and in nothing else, so an agent trained across thousands of
    them and a proxy fitted on a history from the same generator have, between
    them, already seen that economy in full -- and a deployment onto a further
    draw of it is a test of sampling luck rather than of transfer. Perturbing the
    spec per deployment gives each one a *neighbouring* economy instead: the same
    variables under the same corridor, drifting at its own speeds, with crises of
    its own character.

    The perturbations are deliberately **difficulty-neutral in the median**.
    Scales are multiplied by a log-normal centred on one, so an economy is as
    likely to be calmer as stormier and the deployment bank is not quietly a
    harder bank; that is what makes this separable from
    :attr:`~control.world.WorldConfig.deploy_excitation`, which is the knob for
    deliberately deploying into something worse.

    What is *not* perturbed is as considered as what is. Every ``lower``/
    ``upper``/``max_*`` field is a hard clip that doubles as the solver-safety
    corridor, so widening one risks a collapsed run rather than a different
    economy, and none of them moves here. What moves is dynamics: how fast each
    input drifts, how persistent it is, how often crises arrive and which
    variables they hit hardest -- the structure a filter and a fitted proxy
    actually encode, and therefore the structure whose change they must be
    corrected for.
    """

    #: spread on the *operating point* each input drifts around, as a fraction of
    #: its corridor's half-width. This is the one that makes a perturbed economy
    #: a different economy rather than the same one shaken differently. Everything
    #: else here changes how an input wobbles; this changes where it wobbles
    #: *about* -- a different trend productivity growth, a different propensity to
    #: consume, a different payout ratio -- so the perturbed world has its own
    #: steady state and not merely its own noise. It is also, for exactly that
    #: reason, the perturbation a fitted proxy is worst at: a model estimated at
    #: one operating point is biased at another in a way that persists for the
    #: whole run rather than averaging out, which is precisely the systematic,
    #: slowly-varying residual an online correction exists to find.
    #:
    #: Applied to the *level* inputs only, and clipped into the same corridor
    #: everything else respects. Where a baseline already sits near one edge of
    #: its corridor the shift is effectively one-directional -- a smaller
    #: perturbation than the number suggests, not a broken one.
    center: float = 0.15
    scale: float = 0.35  #: log-normal spread on every innovation size
    persistence: float = 0.05  #: additive spread on annual AR(1)/decay persistence
    rate: float = 0.50  #: log-normal spread on crisis onset rate and severity
    impulse: float = 0.45  #: log-normal spread on each crisis impulse, drawn per input
    feedback: float = 0.35  #: log-normal spread on a model's own feedback gains
    #: the range an AR(1) persistence is kept inside. The lower bound stops a
    #: perturbation turning slow structural drift into noise; the upper keeps
    #: :func:`discretize_ar1` away from the unit root it cannot discretize.
    phi_bounds: tuple[float, float] = (0.5, 0.98)

    def scaled(self, strength: float) -> "ExcitationJitter":
        """The same jitter with every spread multiplied by ``strength``.

        One dial over the whole perturbation, so a deployment can be swept from
        "the economy it trained in" to "a distant cousin of it" without choosing
        five numbers -- and so the correction's value can be reported as a curve
        in that dial rather than a single number at one arbitrary point on it.
        """
        if strength < 0.0:
            raise ValueError(f"jitter strength must be non-negative, got {strength}")
        return replace(
            self,
            center=self.center * strength,
            scale=self.scale * strength,
            persistence=self.persistence * strength,
            rate=self.rate * strength,
            impulse=self.impulse * strength,
            feedback=self.feedback * strength,
        )


def lognormal_factor(rng: np.random.Generator, spread: float) -> float:
    """A positive multiplier with median one and log-spread ``spread``.

    The median rather than the mean, because these multiply *scale* parameters,
    where the neutral point is the one that leaves a doubling and a halving
    equally likely.
    """
    return 1.0 if spread <= 0.0 else float(np.exp(rng.normal(0.0, spread)))


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
    #: shift of the level the drift mean-reverts to, away from the model's own
    #: baseline and inside the same corridor. Zero -- the calibrated economy --
    #: everywhere except in a config redrawn by :meth:`perturbed`, where it is
    #: what gives the perturbed world its own steady state rather than only its
    #: own noise. The AR(1) reverts to ``base + center``, not to ``base``.
    center: float = 0.0

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

        The deviation is taken around ``base + center``, itself held inside the
        corridor, so a :meth:`perturbed` spec drifts about its own operating
        point. At the default ``center = 0`` this is the baseline exactly.
        """
        anchor = float(np.clip(base + self.center, self.lower, self.upper))
        phi_dt, sigma_dt = discretize_ar1(self.phi, self.sigma, dt)
        dev = phi_dt * dev + sigma_dt * sigma_scale * rng.standard_normal()
        dev = float(np.clip(dev, self.lower - anchor, self.upper - anchor))
        return dev, float(np.clip(anchor + dev + extra, self.lower, self.upper))

    def perturbed(
        self, rng: np.random.Generator, jitter: ExcitationJitter
    ) -> "AR1Spec":
        """A neighbouring drift: same corridor, its own centre, speed and memory.

        The centre shift is scaled by the corridor's own half-width, which is the
        only scale-free unit available: these inputs are growth rates, ratios and
        propensities whose natural sizes differ by four orders of magnitude, and
        the corridor is the one statement the calibration makes about how far
        each may sensibly move.
        """
        half = 0.5 * (self.upper - self.lower)
        return replace(
            self,
            sigma=self.sigma * lognormal_factor(rng, jitter.scale),
            phi=float(
                np.clip(
                    self.phi + rng.normal(0.0, jitter.persistence), *jitter.phi_bounds
                )
            ),
            center=self.center + float(rng.normal(0.0, jitter.center * half)),
        )


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

    def perturbed(
        self, rng: np.random.Generator, jitter: ExcitationJitter
    ) -> "RandomWalkSpec":
        """A neighbouring walk: same bound, its own step size."""
        return replace(self, sigma=self.sigma * lognormal_factor(rng, jitter.scale))


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

    def perturbed(
        self, rng: np.random.Generator, jitter: ExcitationJitter
    ) -> "StochasticVolatilitySpec":
        """A neighbouring regime: same band, its own stickiness and swing."""
        return replace(
            self,
            rho=float(
                np.clip(
                    self.rho + rng.normal(0.0, jitter.persistence), *jitter.phi_bounds
                )
            ),
            xi=self.xi * lognormal_factor(rng, jitter.scale),
        )


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

    def perturbed(
        self, rng: np.random.Generator, jitter: ExcitationJitter
    ) -> "CrisisSpec":
        """Crises of a different character: same kind of event, its own anatomy.

        Each impulse is drawn its **own** multiplier rather than the bundle
        sharing one. A common scale would only make crises deeper or shallower,
        which ``severity_range`` already does at every onset and which the
        history has therefore already shown the proxy. Independent draws change
        *which* variables a crisis hits hardest -- credit against demand against
        expectations -- and that is a covariance the fitted model has no way to
        have learned.

        ``min_gap`` is left alone: it is a structural guard against overlapping
        onsets rather than a description of the economy.
        """
        rate = lognormal_factor(rng, jitter.rate)
        severity = lognormal_factor(rng, jitter.rate)
        decay = rng.normal(0.0, jitter.persistence)
        return replace(
            self,
            prob=float(np.clip(self.prob * rate, 0.0, 1.0)),
            severity_range=(
                self.severity_range[0] * severity,
                self.severity_range[1] * severity,
            ),
            decay_range=(
                float(np.clip(self.decay_range[0] + decay, 0.5, 0.95)),
                float(np.clip(self.decay_range[1] + decay, 0.5, 0.95)),
            ),
            impulses={
                name: value * lognormal_factor(rng, jitter.impulse)
                for name, value in self.impulses.items()
            },
            financial_prob=float(
                np.clip(self.financial_prob * lognormal_factor(rng, jitter.rate), 0.0, 1.0)
            ),
            financial_impulses={
                name: value * lognormal_factor(rng, jitter.impulse)
                for name, value in self.financial_impulses.items()
            },
        )


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

    def perturbed(
        self, rng: np.random.Generator, jitter: ExcitationJitter
    ) -> "ClimateSpec":
        """A neighbouring mix of run characters: same axis, different weather map.

        ``a`` and ``b`` move so the calm/stormy balance differs, and the two
        gains so a given climate means something different. ``gamma`` holds: its
        convexity is what keeps middling climates nearly crisis-free, and a
        perturbation of it would change the shape of the mix rather than the mix.
        """
        return replace(
            self,
            a=self.a * lognormal_factor(rng, jitter.rate),
            b=self.b * lognormal_factor(rng, jitter.rate),
            vol_shift=self.vol_shift * lognormal_factor(rng, jitter.scale),
            crisis_hi=self.crisis_hi * lognormal_factor(rng, jitter.rate),
        )
