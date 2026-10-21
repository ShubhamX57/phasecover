"""Period search and modulation tests."""

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.stats import kuiper, rayleightest
from astropy.timeseries import BinnedTimeSeries, LombScargle

from ._units import to_seconds
from .exposure import exposure_cdf, exposure_profile, fold_phases
from .io import DEFAULT_MJDREF

__all__ = [
    "ModulationResult",
    "binned_lightcurve",
    "folded_profile",
    "modulation_test",
    "period_search",
]


@dataclass
class ModulationResult:
    """Outcome of testing event phases for modulation at one period.

    ``*_fpp`` values are false-positive probabilities: the chance of a
    result at least this extreme from an unmodulated source. Small values
    mean significant modulation.
    """

    period: float
    n_events: int
    naive_kuiper_d: float
    naive_kuiper_fpp: float
    corrected_kuiper_d: float
    corrected_kuiper_fpp: float
    naive_rayleigh_fpp: float

    def is_significant(self, threshold=1e-3):
        """True if the exposure-corrected test rejects no modulation."""
        return self.corrected_kuiper_fpp < threshold

    def summary(self):
        return "\n".join(
            [
                f"Trial period          : {self.period:.6g} s",
                f"Events                : {self.n_events}",
                f"Naive Kuiper   D, fpp : {self.naive_kuiper_d:.4f}, {self.naive_kuiper_fpp:.3g}",
                f"Naive Rayleigh fpp    : {self.naive_rayleigh_fpp:.3g}",
                f"Corrected Kuiper D,fpp: {self.corrected_kuiper_d:.4f}, {self.corrected_kuiper_fpp:.3g}",
            ]
        )


def modulation_test(times, gtis, period, epoch=0.0):
    """Test event phases for modulation, with and without exposure correction.

    The naive tests (Kuiper against a uniform distribution, and the
    Rayleigh test) assume every phase was observed equally. The corrected
    Kuiper test compares the phases with the distribution expected from
    the actual phase coverage of ``gtis`` (see :func:`exposure_cdf`).
    """
    period = to_seconds(period)
    phases = fold_phases(times, period, epoch)

    naive_d, naive_fpp = kuiper(phases)
    corrected_d, corrected_fpp = kuiper(phases, exposure_cdf(gtis, period, epoch))
    rayleigh_fpp = float(rayleightest(2 * np.pi * phases))

    return ModulationResult(
        period=period,
        n_events=int(phases.size),
        naive_kuiper_d=float(naive_d),
        naive_kuiper_fpp=float(naive_fpp),
        corrected_kuiper_d=float(corrected_d),
        corrected_kuiper_fpp=float(corrected_fpp),
        naive_rayleigh_fpp=rayleigh_fpp,
    )


def folded_profile(times, gtis, period, nbins=16, epoch=0.0):
    """Exposure-corrected folded light curve.

    Returns
    -------
    phase_centres : ndarray
    rate : ndarray
        Counts per second in each phase bin (counts / exposure).
    rate_err : ndarray
        Poisson uncertainty on ``rate``.
    exposure : ndarray
        Exposure time (s) in each phase bin.
    """
    phases = fold_phases(times, period, epoch)
    counts, edges = np.histogram(phases, bins=nbins, range=(0.0, 1.0))
    exposure = exposure_profile(gtis, period, nbins, epoch)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(exposure > 0, counts / exposure, np.nan)
        rate_err = np.where(exposure > 0, np.sqrt(counts) / exposure, np.nan)
    return 0.5 * (edges[:-1] + edges[1:]), rate, rate_err, exposure


def binned_lightcurve(times, gtis, binsize, mjdref=DEFAULT_MJDREF):
    """Count-rate light curve using only bins that lie fully inside a GTI.

    Returns
    -------
    `~astropy.timeseries.BinnedTimeSeries`
        Columns ``counts``, ``rate`` and ``rate_err`` (counts per second).
        ``time_bin_start`` is absolute time referenced to ``mjdref``.
    """
    binsize = to_seconds(binsize)
    times = np.asarray(times, dtype=float)
    starts = []
    for start, stop in np.atleast_2d(np.asarray(gtis, dtype=float)):
        n_bins = int(np.floor((stop - start) / binsize))
        starts.append(start + binsize * np.arange(n_bins))
    starts = np.concatenate(starts) if starts else np.array([])
    if starts.size == 0:
        raise ValueError("binsize is longer than every good time interval")

    idx = np.searchsorted(starts, times, side="right") - 1
    inside = (idx >= 0) & (times < starts[np.clip(idx, 0, None)] + binsize)
    counts = np.bincount(idx[inside], minlength=starts.size).astype(float)

    return BinnedTimeSeries(
        time_bin_start=mjdref + starts * u.s,
        time_bin_size=binsize * u.s,
        data={
            "counts": counts,
            "rate": counts / binsize / u.s,
            "rate_err": np.sqrt(np.maximum(counts, 1.0)) / binsize / u.s,
        },
    )


def period_search(lightcurve, min_period, max_period, samples_per_peak=10):
    """Lomb-Scargle periodogram of a binned light curve.

    Parameters
    ----------
    lightcurve : `~astropy.timeseries.BinnedTimeSeries`
        Output of :func:`binned_lightcurve`.
    min_period, max_period : float or `~astropy.units.Quantity`
        Search range (seconds if floats).

    Returns
    -------
    periods : `~astropy.units.Quantity`
    power : ndarray
    best_period : `~astropy.units.Quantity`
    false_alarm_probability : float
        For the highest peak, from :meth:`LombScargle.false_alarm_probability`.
    """
    min_f = 1.0 / to_seconds(max_period) / u.s
    max_f = 1.0 / to_seconds(min_period) / u.s

    t = lightcurve.time_bin_center
    t = (t - t[0]).to(u.s)
    ls = LombScargle(t, lightcurve["rate"], lightcurve["rate_err"])
    frequency, power = ls.autopower(
        minimum_frequency=min_f,
        maximum_frequency=max_f,
        samples_per_peak=samples_per_peak,
    )
    best = np.argmax(power)
    fap = ls.false_alarm_probability(power[best], minimum_frequency=min_f, maximum_frequency=max_f)
    return (1 / frequency).to(u.s), np.asarray(power), (1 / frequency[best]).to(u.s), float(fap)
