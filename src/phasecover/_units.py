"""Small helpers for accepting either plain floats (seconds) or Quantities."""

import astropy.units as u


def to_seconds(value):
    """Return ``value`` as a float in seconds.

    Plain numbers are assumed to already be in seconds; Quantities are
    converted, so ``5.5 * u.hour`` and ``19800.0`` are equivalent.
    """
    if isinstance(value, u.Quantity):
        return value.to_value(u.s)
    return float(value)
