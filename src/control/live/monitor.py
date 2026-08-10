"""The out-of-distribution monitor: when to stop trusting the corrected model.

The classic model-based failure is a loop: the policy adapts, so it visits states
the correction was not fit on, so the correction extrapolates, so the model
becomes optimistic exactly where the policy is heading. Here it is sharper than
usual, because there is no reset to recover from it -- an over-confident update in
a live economy is simply the outcome.

The monitor is the tripwire on that loop. It watches three quantities, chosen
because they fail in three different ways and a single one of them can be quiet
while the run is going wrong:

* **the correction's predictive variance** at the contexts actually visited. This
  is the *epistemic* signal, and the earliest: it rises as soon as the design
  moves away from where the recursion accumulated its information, before any
  prediction has been seen to be wrong.
* **the realised one-step residual**, against its own recent scale. The
  *empirical* signal: the corrected model is being wrong by more than it has been
  wrong lately, whatever it believes about itself.
* **the fraction of observation features hitting the standardisation clip**. The
  observer's statistics were frozen on the history, so heavy clipping means the
  live economy has left the region the whole apparatus -- correction, critic,
  actor, all of it -- was calibrated in. That one is not about the world model at
  all, and nothing else would catch it.

Each is compared against its own threshold in units of its own recent behaviour,
so none of them needs a number chosen in advance for this particular economy.
Tripping is deliberately *sticky* over a short window: a single wild quarter is
not evidence of a regime the correction cannot handle, and a monitor that
un-trips the moment one reading recovers would flap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MonitorReading:
    """One step's worth of what the monitor watched, for the record and the plot."""

    step: int
    variance: float  #: mean predictive variance of the correction at this context
    residual: float  #: norm of the realised one-step residual
    residual_z: float  #: that norm in units of its own recent scale
    clipped: float  #: fraction of observation features at the standardisation clip
    tripped: bool  #: whether any threshold was over at this step


@dataclass
class OODMonitor:
    """Watches three signals and says when the policy should stop being updated.

    ``variance_factor`` and ``residual_z`` are thresholds in units of each
    signal's own recent behaviour; ``clip_fraction`` is absolute, because the
    clip rate has a meaningful zero. ``patience`` is how many of the last
    ``window`` steps must be over threshold before the monitor trips, and
    ``fallback_after`` how many trips in a row hand control back to the frozen
    policy altogether.

    ``warmup`` is the number of readings the monitor takes before it will trip at
    all, and it is not a nicety. Two of the three signals are measured *against
    their own recent behaviour*, so before that behaviour has been observed there
    is nothing to be unusual with respect to: at the second reading the variance
    test is comparing one number to the one before it, which is noise. Left
    unguarded, a transient in the opening quarters trips the monitor, the sticky
    ``window`` holds it tripped for ``window`` steps, that clears
    ``fallback_after``, and the deployment hands back control before it has taken
    a single update -- killing the run it was meant to protect. The warmup is set
    inside the policy's own warmup, so it costs nothing: no update would have been
    taken during it anyway.
    """

    variance_factor: float = 4.0
    residual_z: float = 4.0
    clip_fraction: float = 0.25
    window: int = 8
    patience: int = 3
    fallback_after: int = 6
    warmup: int = 16
    kappa: float = 0.1  #: EWMA rate of the reference scales

    variance_ref_: float | None = field(default=None, init=False)
    residual_mean_: float | None = field(default=None, init=False)
    residual_var_: float = field(default=0.0, init=False)
    readings_: list[MonitorReading] = field(default_factory=list, init=False)
    trips_: int = field(default=0, init=False)

    def update(
        self, step: int, variance: np.ndarray, residual: np.ndarray, obs: np.ndarray,
        *, clip: float = 10.0,
    ) -> MonitorReading:
        """Fold in one live step and return what it read.

        The reference scales are exponentially weighted and are updated **from
        every step, including tripped ones**. That is deliberate: the alternative
        -- freezing the reference while the monitor is tripped -- makes a
        genuinely new regime look permanently anomalous, and the monitor would
        never let the deployment adapt to a world that had simply moved.
        """
        v = float(np.mean(variance)) if variance.size else 0.0
        r = float(np.linalg.norm(residual))
        clipped = float(np.mean(np.abs(np.asarray(obs)) >= clip - 1e-9))

        over_variance = (
            self.variance_ref_ is not None and v > self.variance_factor * self.variance_ref_
        )
        z = 0.0
        if self.residual_mean_ is not None:
            scale = max(np.sqrt(self.residual_var_), 1e-12)
            z = (r - self.residual_mean_) / scale
        over_residual = z > self.residual_z
        over_clip = clipped > self.clip_fraction

        self._track(v, r)
        reading = MonitorReading(
            step=step,
            variance=v,
            residual=r,
            residual_z=z,
            clipped=clipped,
            tripped=bool(over_variance or over_residual or over_clip),
        )
        self.readings_.append(reading)
        if self.tripped():
            self.trips_ += 1
        else:
            self.trips_ = 0
        return reading

    def tripped(self) -> bool:
        """Whether enough of the recent window was over threshold.

        Always ``False`` until :attr:`warmup` readings have accumulated -- see the
        class docstring for why a monitor that can fire before it has a reference
        is worse than no monitor at all.
        """
        if len(self.readings_) < self.warmup:
            return False
        recent = self.readings_[-self.window :]
        return sum(r.tripped for r in recent) >= self.patience

    def failed(self) -> bool:
        """Whether it has been tripped long enough to hand control back."""
        return self.trips_ >= self.fallback_after

    def _track(self, variance: float, residual: float) -> None:
        """Advance the exponentially weighted reference scales."""
        k = self.kappa
        self.variance_ref_ = (
            variance if self.variance_ref_ is None
            else (1.0 - k) * self.variance_ref_ + k * variance
        )
        if self.residual_mean_ is None:
            # Variance zero, not ``residual**2``: one observation carries a level,
            # not a spread, and seeding the variance with the square of the level
            # makes the first several z-scores a statement about how large the
            # residual is rather than about how unusual it is.
            self.residual_mean_, self.residual_var_ = residual, 0.0
            return
        delta = residual - self.residual_mean_
        self.residual_mean_ += k * delta
        self.residual_var_ = (1.0 - k) * (self.residual_var_ + k * delta**2)
