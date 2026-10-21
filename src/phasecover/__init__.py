"""Exposure-corrected searches for periodic modulation in gappy event data."""

from .exposure import exposure_cdf, exposure_profile, fold_gtis, fold_phases
from .io import EventList, read_events, write_events
from .search import (
    ModulationResult,
    binned_lightcurve,
    folded_profile,
    modulation_test,
    period_search,
)
from .simulate import leo_gtis, simulate_events

__version__ = "0.1.0"

__all__ = [
    "EventList",
    "ModulationResult",
    "binned_lightcurve",
    "exposure_cdf",
    "exposure_profile",
    "fold_gtis",
    "fold_phases",
    "folded_profile",
    "leo_gtis",
    "modulation_test",
    "period_search",
    "read_events",
    "simulate_events",
    "write_events",
]
