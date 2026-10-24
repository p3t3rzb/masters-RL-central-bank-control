"""The excitation building blocks: AR(1) drift, random walks, volatility, jitter.

Two properties carry the whole dataset design and are pinned here. The clips
double as the solver's safety corridor, so a level may never leave them however
large the shock; and the annual ``(phi, sigma)`` are discretized so the *annual*
persistence and stationary variance are the same at every ``dt`` (see the
GROWTH ``dt`` convention).
"""

from __future__ import annotations

import numpy as np
import pytest

from economic_models.ground_truth.excitation.specs import (
    AR1Spec,
    ExcitationJitter,
    RandomWalkSpec,
    StochasticVolatilitySpec,
    discretize_ar1,
    lognormal_factor,
)
from economic_models.ground_truth.models.growth.excitation.specs import GovSpendingSpec


# -- discretization ----------------------------------------------------------


def test_annual_step_is_the_annual_process():
    """At ``dt = 1`` discretization is the identity."""
    assert discretize_ar1(0.9, 0.01, 1.0) == pytest.approx((0.9, 0.01))


@pytest.mark.parametrize("dt", [1.0, 0.5, 0.25, 1 / 12])
def test_discretization_preserves_annual_moments(dt):
    """``dt`` steps compound to the annual persistence; the stationary variance holds."""
    phi, sigma = 0.8, 0.02
    phi_dt, sigma_dt = discretize_ar1(phi, sigma, dt)
    assert phi_dt ** (1.0 / dt) == pytest.approx(phi)
    assert sigma_dt**2 / (1.0 - phi_dt**2) == pytest.approx(sigma**2 / (1.0 - phi**2))


@pytest.mark.parametrize("phi", [1.0, -1.0, 1.2])
def test_unit_root_is_rejected(phi):
    """A random walk has no stationary variance to match; it must use RandomWalkSpec."""
    with pytest.raises(ValueError, match="RandomWalkSpec"):
        discretize_ar1(phi, 0.01, 0.25)


# -- AR(1) drift -------------------------------------------------------------


def test_ar1_level_never_leaves_its_corridor():
    """However violent the shocks, the clipped level stays inside ``[lower, upper]``."""
    spec = AR1Spec(sigma=1.0, lower=0.01, upper=0.05, phi=0.9)
    rng = np.random.default_rng(0)
    dev, levels = 0.0, []
    for _ in range(500):
        dev, level = spec.advance(dev, 0.03, rng, sigma_scale=3.0, dt=0.25)
        levels.append(level)
    assert min(levels) >= spec.lower
    assert max(levels) <= spec.upper
    assert min(levels) == pytest.approx(spec.lower) and max(levels) == pytest.approx(spec.upper)


def test_ar1_without_noise_rests_at_its_anchor():
    """No innovation and no deviation leaves the input at ``base + center``."""
    spec = AR1Spec(sigma=0.0, lower=0.0, upper=1.0, center=0.1)
    dev, level = spec.advance(0.0, 0.4, np.random.default_rng(0))
    assert dev == 0.0
    assert level == pytest.approx(0.5)


def test_crisis_impulse_moves_the_level_not_the_memory():
    """``extra`` is a level shock only -- it must not leak into the AR(1) deviation."""
    spec = AR1Spec(sigma=0.01, lower=-1.0, upper=1.0)
    calm = spec.advance(0.0, 0.0, np.random.default_rng(3))
    hit = spec.advance(0.0, 0.0, np.random.default_rng(3), extra=0.2)
    assert hit[0] == calm[0]
    assert hit[1] == pytest.approx(calm[1] + 0.2)


def test_ar1_is_reproducible_per_stream():
    """The same generator state produces the same draw."""
    spec = AR1Spec(sigma=0.05, lower=-1.0, upper=1.0)
    a = spec.advance(0.1, 0.0, np.random.default_rng(7), dt=0.25)
    b = spec.advance(0.1, 0.0, np.random.default_rng(7), dt=0.25)
    assert a == b


def test_perturbed_ar1_keeps_the_corridor():
    """A neighbouring economy moves dynamics, never the solver-safety clips."""
    spec = AR1Spec(sigma=0.01, lower=0.0, upper=0.1, phi=0.9)
    jitter = ExcitationJitter()
    for seed in range(20):
        other = spec.perturbed(np.random.default_rng(seed), jitter)
        assert (other.lower, other.upper) == (spec.lower, spec.upper)
        assert jitter.phi_bounds[0] <= other.phi <= jitter.phi_bounds[1]
        assert other.sigma > 0.0


# -- the other processes -----------------------------------------------------


def test_random_walk_is_bounded_in_log_deviation():
    """The level input drifts multiplicatively but within ``max_logdev``."""
    spec = RandomWalkSpec(sigma=1.0, max_logdev=0.2)
    rng = np.random.default_rng(0)
    logdev = 0.0
    for _ in range(200):
        logdev = spec.advance(logdev, rng, dt=0.25)
        assert abs(logdev) <= spec.max_logdev


def test_volatility_multiplier_is_positive_and_bounded():
    """Log-volatility is clipped, so the innovation multiplier is too."""
    spec = StochasticVolatilitySpec(rho=0.9, xi=2.0, max_logvol=0.5)
    rng = np.random.default_rng(0)
    logvol = 0.0
    for _ in range(200):
        logvol = spec.advance(logvol, rng, dt=0.25)
        assert np.exp(-0.5) - 1e-12 <= spec.multiplier(logvol) <= np.exp(0.5) + 1e-12


def test_lognormal_factor_is_neutral_at_zero_spread():
    """No spread means no perturbation at all, not a draw near one."""
    assert lognormal_factor(np.random.default_rng(0), 0.0) == 1.0


def test_jitter_scales_every_spread_together():
    """One dial over the whole perturbation; zero is the preset itself."""
    jitter = ExcitationJitter()
    zero = jitter.scaled(0.0)
    assert (zero.center, zero.scale, zero.persistence, zero.rate, zero.impulse, zero.feedback) == (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    assert jitter.scaled(2.0).scale == pytest.approx(2.0 * jitter.scale)
    with pytest.raises(ValueError):
        jitter.scaled(-1.0)


# -- GROWTH's fiscal stabilizer ----------------------------------------------


def test_stabilizer_is_silent_at_full_employment():
    """A scenario is drawn at ``ER = 1``, where the response contributes nothing."""
    spec = GovSpendingSpec(gap_sigma=0.01, gap_clip=0.02, stabilizer=0.5, bounds=(-0.02, 0.08))
    assert spec.support(1.0) == 0.0


def test_stabilizer_leans_against_the_employment_gap():
    """Spending rises in a slump and falls in a boom, in proportion to the gap."""
    spec = GovSpendingSpec(gap_sigma=0.01, gap_clip=0.02, stabilizer=0.5, bounds=(-0.02, 0.08))
    assert spec.support(0.9) == pytest.approx(0.05)
    assert spec.support(1.1) == pytest.approx(-0.05)
    assert spec.target == "GRg"


def test_spending_gap_is_symmetric_around_zero():
    """Trend demand tracks trend supply: the gap drifts around zero inside its clip."""
    gap = GovSpendingSpec(gap_sigma=0.01, gap_clip=0.02, stabilizer=0.5, bounds=(0.0, 0.1)).gap_spec()
    assert (gap.lower, gap.upper, gap.center) == (-0.02, 0.02, 0.0)
