import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose

from phasecover import (
    binned_lightcurve,
    folded_profile,
    leo_gtis,
    modulation_test,
    period_search,
    simulate_events,
)
from phasecover.cli import main


def _observation(seed, pulsed_fraction, period=5 * u.hour):
    rng = np.random.default_rng(seed)
    gtis = leo_gtis(30 * u.hour, rng=rng)
    times = simulate_events(gtis, 2.0, period=period, pulsed_fraction=pulsed_fraction, rng=rng)
    return times, gtis


def test_gaps_fool_naive_test_but_not_corrected_test():
    # A constant source looks strongly "modulated" to tests that assume
    # uniform phase coverage, purely because of the orbital gaps.
    times, gtis = _observation(seed=11, pulsed_fraction=0.0)
    result = modulation_test(times, gtis, 5 * u.hour)
    assert result.naive_kuiper_fpp < 1e-10
    assert result.corrected_kuiper_fpp > 1e-3
    assert not result.is_significant()


def test_real_modulation_is_detected():
    times, gtis = _observation(seed=12, pulsed_fraction=0.05)
    result = modulation_test(times, gtis, 5 * u.hour)
    assert result.is_significant()


def test_corrected_test_false_positive_rate_is_reasonable():
    # Under the null hypothesis the corrected test should rarely fire.
    fpps = [modulation_test(*_observation(seed, 0.0), 5 * u.hour).corrected_kuiper_fpp for seed in range(100, 130)]
    assert np.mean(np.array(fpps) < 0.01) <= 0.1


def test_folded_profile_recovers_constant_rate():
    times, gtis = _observation(seed=13, pulsed_fraction=0.0)
    _, rate, err, exposure = folded_profile(times, gtis, 5 * u.hour, nbins=8)
    assert_allclose(exposure.sum(), np.sum(gtis[:, 1] - gtis[:, 0]))
    assert np.all(np.abs(rate - 2.0) < 5 * err)


def test_lightcurve_and_period_search():
    period = 40 * u.min
    rng = np.random.default_rng(14)
    gtis = leo_gtis(12 * u.hour, rng=rng)
    times = simulate_events(gtis, 5.0, period=period, pulsed_fraction=0.3, rng=rng)
    lc = binned_lightcurve(times, gtis, 60 * u.s)
    assert lc["counts"].sum() <= times.size
    _, _, best, fap = period_search(lc, 20 * u.min, 80 * u.min)
    assert abs(best - period) < 1 * u.min
    assert fap < 1e-6


def test_cli_round_trip(tmp_path, capsys):
    path = tmp_path / "sim.fits"
    main(["simulate", str(path), "--pulsed-fraction", "0.1", "--seed", "1"])
    main(["analyze", str(path), "--period", "5", "--plot", str(tmp_path / "p.png")])
    out = capsys.readouterr().out
    assert "modulation detected" in out
    assert (tmp_path / "p.png").exists()
