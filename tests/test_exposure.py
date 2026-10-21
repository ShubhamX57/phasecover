import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from phasecover import exposure_cdf, exposure_profile, fold_gtis, fold_phases


def test_fold_phases_range_and_units():
    t = np.array([0.0, 2.5, 10.0, 12.5])
    assert_allclose(fold_phases(t, 10.0), [0.0, 0.25, 0.0, 0.25])
    assert_allclose(fold_phases(t, 10 * u.s, epoch=2.5), [0.75, 0.0, 0.75, 0.0])
    assert_allclose(fold_phases([3600.0], 1 * u.hour), [0.0])


def test_single_full_cycle_has_unit_coverage():
    # One GTI spanning exactly one period must cover every phase once.
    # Released astropy (8.0.x) returns zero here; phasecover corrects it.
    breaks, coverage = fold_gtis([[0.0, 100.0]], period=100.0)
    assert_allclose(breaks, [0, 1])
    assert_allclose(coverage, [1])


@pytest.mark.parametrize(
    "gti,expected",
    [
        ([[50.0, 200.0]], [1, 2]),  # ends on a cycle boundary
        ([[50.0, 175.0]], [1, 2, 1]),  # ends mid-cycle
        ([[0.0, 300.0]], [3]),  # whole cycles only
    ],
)
def test_fold_gtis_known(gti, expected):
    _, coverage = fold_gtis(gti, period=100.0)
    assert_allclose(coverage, expected)


def test_exposure_profile_sums_to_gti_length():
    rng = np.random.default_rng(1)
    starts = np.sort(rng.uniform(0, 1e5, 30))
    gtis = np.column_stack([starts, starts + rng.uniform(100, 2000, 30)])
    total = np.sum(gtis[:, 1] - gtis[:, 0])
    for nbins in (1, 7, 32):
        assert_allclose(exposure_profile(gtis, 3456.7, nbins).sum(), total)


def test_exposure_cdf_is_uniform_for_whole_cycles():
    cdf = exposure_cdf([[0.0, 400.0]], period=100.0)
    x = np.linspace(0, 1, 11)
    assert_allclose(cdf(x), x)


def test_exposure_cdf_matches_partial_coverage():
    # Phases 0-0.5 observed twice, 0.5-1 once: 2/3 of exposure below 0.5.
    cdf = exposure_cdf([[0.0, 150.0]], period=100.0)
    assert_allclose(cdf(0.5), 2 / 3)
    assert_allclose(cdf(1.0), 1.0)


def test_invalid_gtis():
    with pytest.raises(ValueError):
        fold_gtis([[1.0, 2.0, 3.0]], 10.0)
    with pytest.raises(ValueError):
        fold_gtis([[5.0, 5.0]], 10.0)
