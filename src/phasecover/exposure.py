"""Phase coverage (exposure) of a set of good time intervals.

When a periodic source is observed with gaps, some phases of the cycle
are watched more often than others. A test for modulation that ignores
this will mistake the observing pattern for a signal. The functions here
fold the good time intervals (GTIs) onto one cycle using
:func:`astropy.stats.fold_intervals` and turn the result into the
quantities needed for an exposure-corrected test.
"""

import numpy as np
from astropy.stats import cdf_from_intervals, fold_intervals, histogram_intervals

from ._units import to_seconds

__all__ = ["exposure_cdf", "exposure_profile", "fold_gtis", "fold_phases"]


def _fold_intervals_drops_boundary_period():
    """Detect the upstream boundary bug in ``fold_intervals``.

    Astropy releases before the fix submitted in astropy/astropy#20671
    subtract one full period of weight from every phase cell for each
    interval whose upper limit is an exact whole number of cycles (so a
    single full cycle ``(0, 1)`` gets zero coverage). We probe for that
    behaviour once at import rather than relying on version numbers.
    """
    _, totals = fold_intervals([(0.0, 1.0, 1.0)])
    return not np.isclose(totals[0], 1.0)


_NEEDS_BOUNDARY_WORKAROUND = _fold_intervals_drops_boundary_period()


def fold_phases(times, period, epoch=0.0):
    """Phase of each event time, in the range [0, 1).

    Parameters
    ----------
    times : array-like
        Event times in seconds (same reference as ``epoch``).
    period : float or `~astropy.units.Quantity`
        Trial period; floats are taken to be seconds.
    epoch : float or `~astropy.units.Quantity`, optional
        Time of phase zero, in seconds.
    """
    period = to_seconds(period)
    epoch = to_seconds(epoch)
    return np.mod((np.asarray(times, dtype=float) - epoch) / period, 1.0)


def fold_gtis(gtis, period, epoch=0.0):
    """Fold good time intervals onto one cycle.

    Parameters
    ----------
    gtis : (N, 2) array-like
        Start and stop times of each good time interval, in seconds.
    period, epoch : float or `~astropy.units.Quantity`
        As for :func:`fold_phases`.

    Returns
    -------
    breaks : ndarray
        Phase boundaries, from 0 to 1.
    coverage : ndarray
        Number of times each phase cell ``breaks[i]..breaks[i+1]`` was
        observed.
    """
    period = to_seconds(period)
    epoch = to_seconds(epoch)
    gtis = np.atleast_2d(np.asarray(gtis, dtype=float))
    if gtis.shape[1] != 2:
        raise ValueError("gtis must have shape (N, 2)")

    cycles = (gtis - epoch) / period
    intervals = [(a, b, 1.0) for a, b in cycles if b > a]
    if not intervals:
        raise ValueError("no good time intervals with positive length")

    breaks, coverage = fold_intervals(intervals)

    if _NEEDS_BOUNDARY_WORKAROUND:
        # Older astropy removes one full cycle of weight for each interval
        # ending exactly on a cycle boundary; add it back.
        n_on_boundary = sum(1 for _, b, _ in intervals if b % 1 == 0)
        coverage = coverage + n_on_boundary

    return breaks, coverage


def exposure_cdf(gtis, period, epoch=0.0):
    """Expected phase CDF for an unmodulated source observed during ``gtis``.

    The returned callable maps phase in [0, 1] to the fraction of the
    total exposure that falls below that phase. It is the correct null
    hypothesis for :func:`astropy.stats.kuiper` when the coverage of the
    cycle is uneven.
    """
    return cdf_from_intervals(*fold_gtis(gtis, period, epoch))


def exposure_profile(gtis, period, nbins, epoch=0.0):
    """Exposure time (seconds) in each of ``nbins`` equal phase bins.

    The values sum to the total length of the good time intervals.
    """
    period = to_seconds(period)
    breaks, coverage = fold_gtis(gtis, period, epoch)
    mean_coverage = histogram_intervals(nbins, breaks, coverage)
    return mean_coverage * period / nbins
