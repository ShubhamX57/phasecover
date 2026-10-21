import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from numpy.testing import assert_allclose

from phasecover import leo_gtis, read_events, simulate_events, write_events


def test_leo_gtis_shape_and_duty_cycle():
    gtis = leo_gtis(24 * u.hour, orbit_period=96 * u.min, visible_fraction=0.5, jitter=0, rng=0)
    assert gtis.shape == (15, 2)
    assert np.all(np.diff(gtis, axis=1) > 0)
    assert_allclose(np.diff(gtis, axis=1), 48 * 60)


def test_events_lie_inside_gtis_and_match_rate():
    rng = np.random.default_rng(3)
    gtis = leo_gtis(20 * u.hour, rng=rng)
    times = simulate_events(gtis, rate=5.0, rng=rng)
    inside = np.zeros(times.size, bool)
    for start, stop in gtis:
        inside |= (times >= start) & (times < stop)
    assert inside.all()
    expected = 5.0 * np.sum(gtis[:, 1] - gtis[:, 0])
    assert abs(times.size - expected) < 5 * np.sqrt(expected)


def test_modulated_source_needs_period():
    with pytest.raises(ValueError):
        simulate_events([[0, 10]], 1.0, pulsed_fraction=0.5)


def test_event_file_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    gtis = leo_gtis(3 * u.hour, rng=rng)
    times = simulate_events(gtis, 1.0, rng=rng)
    mjdref = Time(58000.25, format="mjd", scale="tt")
    path = tmp_path / "events.fits"
    write_events(path, times, gtis, mjdref=mjdref, TELESCOP="SIMULATED")

    ev = read_events(path)
    assert_allclose(ev.times, times)
    assert_allclose(ev.gtis, gtis)
    assert abs((ev.mjdref - mjdref).to(u.us)) < 1 * u.us
    assert ev.meta["TELESCOP"] == "SIMULATED"
    assert_allclose(ev.exposure.to_value(u.s), np.sum(gtis[:, 1] - gtis[:, 0]))
