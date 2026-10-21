"""Reading and writing OGIP-style photon event files.

High-energy missions (NICER, XMM-Newton, Chandra, Swift, ...) distribute
photon lists as FITS files with an ``EVENTS`` table holding a ``TIME``
column and a ``GTI`` table holding ``START``/``STOP`` columns. Times are
seconds since a mission reference epoch given by ``MJDREFI``/``MJDREFF``
(or ``MJDREF``) in the header. This module reads and writes that layout.
"""

from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

__all__ = ["DEFAULT_MJDREF", "EventList", "read_events", "write_events"]

#: NICER mission reference epoch, used when no other reference is given.
DEFAULT_MJDREF = Time(56658, format="mjd", scale="tt")


@dataclass
class EventList:
    """Photon arrival times and good time intervals.

    Attributes
    ----------
    times : ndarray
        Event times in seconds since ``mjdref``.
    gtis : (N, 2) ndarray
        Good time intervals in seconds since ``mjdref``.
    mjdref : `~astropy.time.Time`
        Reference epoch for ``times`` and ``gtis``.
    meta : dict
        Selected header keywords (telescope, instrument, object).
    """

    times: np.ndarray
    gtis: np.ndarray
    mjdref: Time
    meta: dict = field(default_factory=dict)

    @property
    def exposure(self):
        """Total good time as a `~astropy.units.Quantity`."""
        return np.sum(self.gtis[:, 1] - self.gtis[:, 0]) * u.s

    @property
    def start_time(self):
        """Absolute time of the first good time interval."""
        return self.mjdref + self.gtis[0, 0] * u.s

    def to_time(self, seconds):
        """Convert seconds since ``mjdref`` to absolute `~astropy.time.Time`."""
        return self.mjdref + np.asarray(seconds) * u.s


def _mjdref_from_header(header):
    timesys = header.get("TIMESYS", "TT").strip().lower()
    scale = timesys if timesys in Time.SCALES else "tt"
    if "MJDREFI" in header:
        return Time(header["MJDREFI"], header.get("MJDREFF", 0.0), format="mjd", scale=scale)
    if "MJDREF" in header:
        return Time(header["MJDREF"], format="mjd", scale=scale)
    raise KeyError("event file header has no MJDREF/MJDREFI keyword")


def read_events(path):
    """Read an OGIP-style event file into an `EventList`.

    ``TIMEZERO`` (if present) is added to event and GTI times.
    """
    with fits.open(path) as hdul:
        events_hdu = hdul["EVENTS"]
        header = events_hdu.header
        mjdref = _mjdref_from_header(header)
        timezero = header.get("TIMEZERO", 0.0)

        times = np.asarray(events_hdu.data["TIME"], dtype=float) + timezero

        if "GTI" in hdul:
            gti = hdul["GTI"].data
            gtis = np.column_stack([gti["START"], gti["STOP"]]).astype(float) + timezero
        else:
            gtis = np.array([[times.min(), times.max()]])

        meta = {k: header[k] for k in ("TELESCOP", "INSTRUME", "OBJECT") if k in header}

    return EventList(times=np.sort(times), gtis=gtis, mjdref=mjdref, meta=meta)


def write_events(path, times, gtis, mjdref=DEFAULT_MJDREF, **meta):
    """Write event times and GTIs as an OGIP-style FITS file.

    Parameters
    ----------
    path : str or path-like
        Output file name (overwritten if it exists).
    times : array-like
        Event times in seconds since ``mjdref``.
    gtis : (N, 2) array-like
        Good time intervals in seconds since ``mjdref``.
    mjdref : `~astropy.time.Time`, optional
        Reference epoch. Defaults to MJD 56658 (TT), the NICER reference.
    **meta
        Extra header keywords, e.g. ``TELESCOP="SIMULATED"``.
    """
    gtis = np.atleast_2d(np.asarray(gtis, dtype=float))
    mjdref = mjdref.tt
    mjdrefi = int(np.floor(mjdref.mjd))
    mjdreff = float((mjdref - Time(mjdrefi, format="mjd", scale="tt")).to_value(u.day))

    header = fits.Header()
    header["MJDREFI"] = (mjdrefi, "integer part of reference MJD")
    header["MJDREFF"] = (mjdreff, "fractional part of reference MJD")
    header["TIMESYS"] = ("TT", "time system")
    header["TIMEUNIT"] = ("s", "unit of time keywords")
    header["TSTART"] = float(gtis[:, 0].min())
    header["TSTOP"] = float(gtis[:, 1].max())
    for key, value in meta.items():
        header[key.upper()] = value

    events = Table({"TIME": np.asarray(times, dtype=float)})
    events.meta.update(header)
    gti = Table({"START": gtis[:, 0], "STOP": gtis[:, 1]})

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.table_to_hdu(events),
            fits.table_to_hdu(gti),
        ]
    )
    hdul[1].name = "EVENTS"
    hdul[2].name = "GTI"
    hdul[2].header.update({k: header[k] for k in ("MJDREFI", "MJDREFF", "TIMESYS", "TIMEUNIT")})
    hdul.writeto(path, overwrite=True)
