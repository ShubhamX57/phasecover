"""Simulated observations: low-Earth-orbit visibility gaps and photon events."""

import astropy.units as u
import numpy as np

from ._units import to_seconds

__all__ = ["leo_gtis", "simulate_events"]


def leo_gtis(
    duration,
    orbit_period=96 * u.min,
    visible_fraction=0.55,
    start=0.0,
    jitter=60 * u.s,
    rng=None,
):
    """Good time intervals for a target observed from low Earth orbit.

    A satellite in low Earth orbit loses sight of most targets for part
    of every orbit (Earth occultation, passages through the South
    Atlantic Anomaly). This produces regular gaps whose spacing can
    interact with a source's period and make coverage of its cycle uneven.

    Parameters
    ----------
    duration : float or `~astropy.units.Quantity`
        Total span of the observation (seconds if a float).
    orbit_period : float or `~astropy.units.Quantity`, optional
        Satellite orbital period. Defaults to 96 minutes.
    visible_fraction : float, optional
        Fraction of each orbit during which the target is visible.
    start : float or `~astropy.units.Quantity`, optional
        Start time of the first orbit, in seconds.
    jitter : float or `~astropy.units.Quantity`, optional
        Standard deviation of random shifts applied to each window's
        start and stop, to avoid unrealistically perfect regularity.
    rng : `numpy.random.Generator` or int, optional
        Random generator or seed.

    Returns
    -------
    gtis : (N, 2) ndarray
        Start and stop times in seconds.
    """
    if not 0 < visible_fraction <= 1:
        raise ValueError("visible_fraction must be in (0, 1]")
    rng = np.random.default_rng(rng)
    duration = to_seconds(duration)
    orbit = to_seconds(orbit_period)
    start = to_seconds(start)
    jitter = to_seconds(jitter)

    orbit_starts = start + np.arange(0.0, duration, orbit)
    visible = visible_fraction * orbit
    gti_start = orbit_starts + rng.normal(0.0, jitter, orbit_starts.size)
    gti_stop = gti_start + visible + rng.normal(0.0, jitter, orbit_starts.size)

    gti_start = np.clip(gti_start, start, start + duration)
    gti_stop = np.clip(gti_stop, start, start + duration)
    gtis = np.column_stack([gti_start, gti_stop])
    return gtis[gtis[:, 1] > gtis[:, 0]]


def simulate_events(
    gtis,
    rate,
    period=None,
    pulsed_fraction=0.0,
    epoch=0.0,
    rng=None,
):
    """Draw photon arrival times inside ``gtis``.

    The count rate is ``rate * (1 + pulsed_fraction * cos(2*pi*phase))``,
    so ``pulsed_fraction=0`` gives a constant source with no modulation.
    Events are drawn by thinning a homogeneous Poisson process.

    Parameters
    ----------
    gtis : (N, 2) array-like
        Good time intervals in seconds.
    rate : float or `~astropy.units.Quantity`
        Mean count rate (counts per second if a float).
    period : float or `~astropy.units.Quantity`, optional
        Modulation period. Required if ``pulsed_fraction`` is non-zero.
    pulsed_fraction : float, optional
        Modulation amplitude, between 0 and 1.
    epoch : float or `~astropy.units.Quantity`, optional
        Time of maximum brightness, in seconds.
    rng : `numpy.random.Generator` or int, optional
        Random generator or seed.

    Returns
    -------
    times : ndarray
        Sorted event times in seconds.
    """
    if not 0 <= pulsed_fraction <= 1:
        raise ValueError("pulsed_fraction must be between 0 and 1")
    if pulsed_fraction > 0 and period is None:
        raise ValueError("period is required for a modulated source")
    rng = np.random.default_rng(rng)
    if isinstance(rate, u.Quantity):
        rate = rate.to_value(1 / u.s)
    rate = float(rate)
    peak = rate * (1.0 + pulsed_fraction)

    chunks = []
    for start, stop in np.atleast_2d(np.asarray(gtis, dtype=float)):
        n = rng.poisson(peak * (stop - start))
        chunks.append(rng.uniform(start, stop, n))
    times = np.sort(np.concatenate(chunks)) if chunks else np.array([])

    if pulsed_fraction > 0:
        period = to_seconds(period)
        epoch = to_seconds(epoch)
        phase = (times - epoch) / period
        keep_prob = (1.0 + pulsed_fraction * np.cos(2 * np.pi * phase)) / (1.0 + pulsed_fraction)
        times = times[rng.uniform(size=times.size) < keep_prob]

    return times
