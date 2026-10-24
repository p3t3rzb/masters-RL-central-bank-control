"""The live deployment's out-of-distribution monitor.

Its two guarantees are the ones the deployment's safety rests on: it can never
fire before it has a reference to be unusual against (the warmup), and a
sustained anomaly -- not a single spike -- is what trips it and, held long
enough, hands control back.
"""

from __future__ import annotations

import numpy as np

from control.live.monitor import OODMonitor

OBS = np.zeros(8)


def _feed(monitor: OODMonitor, n: int, *, residual: float = 1.0, start: int = 0,
          obs: np.ndarray = OBS, rng_seed: int = 0) -> None:
    """Feed ``n`` readings with a mildly noisy residual of the given size."""
    rng = np.random.default_rng(rng_seed)
    for t in range(start, start + n):
        noise = 1.0 + 0.05 * rng.standard_normal()
        monitor.update(t, np.ones(4), np.full(3, residual * noise), obs)


def test_never_trips_inside_the_warmup():
    """Huge signals before a reference exists are not anomalies.

    The guard is "fewer than ``warmup`` readings", so the first ``warmup - 1``
    updates can never trip however extreme they are.
    """
    monitor = OODMonitor(warmup=10, patience=1)
    for t in range(monitor.warmup - 1):
        monitor.update(t, np.full(4, 1e6 * (t + 1)), np.full(3, 1e6 * (t + 1)), np.full(8, 10.0))
        assert not monitor.tripped()
    assert not monitor.failed()


def test_opening_transient_ages_out_before_the_warmup_ends():
    """At the defaults a stationary world never trips the monitor.

    The second reading's residual z-score is enormous -- one prior reading is a
    level, not a spread -- which is exactly why the default warmup is longer than
    the window: the transient has left the window before the guard lifts.
    """
    monitor = OODMonitor()
    assert monitor.warmup > monitor.window
    rng = np.random.default_rng(0)
    for t in range(200):
        noise = 1.0 + 0.05 * rng.standard_normal()
        monitor.update(t, np.ones(4), np.full(3, noise), OBS)
        assert not monitor.tripped(), t


def test_sustained_residual_spike_trips_and_then_fails():
    """Past patience it trips; past ``fallback_after`` trips it hands control back."""
    monitor = OODMonitor(warmup=5, patience=3, window=8, fallback_after=4, kappa=0.01)
    _feed(monitor, 30)
    assert not monitor.tripped()
    _feed(monitor, 2, residual=1e3, start=30)
    assert not monitor.tripped()
    _feed(monitor, 1, residual=1e3, start=32)
    assert monitor.tripped()
    _feed(monitor, 3, residual=1e3, start=33)
    assert monitor.failed()


def test_clip_fraction_is_an_absolute_threshold():
    """An observation pinned at the standardisation clip trips with no history needed."""
    monitor = OODMonitor(warmup=0, patience=1, clip_fraction=0.25)
    reading = monitor.update(0, np.ones(4), np.ones(3), np.full(8, 10.0), clip=10.0)
    assert reading.clipped == 1.0
    assert reading.tripped and monitor.tripped()


def test_readings_are_recorded_every_step():
    """Every update is kept for the record and the plot."""
    monitor = OODMonitor()
    _feed(monitor, 7)
    assert [r.step for r in monitor.readings_] == list(range(7))
